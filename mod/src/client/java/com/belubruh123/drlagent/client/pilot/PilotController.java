package com.belubruh123.drlagent.client.pilot;

import com.belubruh123.drlagent.bridge.BridgeConfig;
import com.belubruh123.drlagent.env.ObsProjector;
import net.minecraft.client.Minecraft;
import net.minecraft.client.player.AbstractClientPlayer;
import net.minecraft.client.player.LocalPlayer;
import net.minecraft.network.chat.Component;
import net.minecraft.util.Mth;
import net.minecraft.world.InteractionHand;
import net.minecraft.world.level.ClipContext;
import net.minecraft.world.phys.AABB;
import net.minecraft.world.phys.HitResult;
import net.minecraft.world.phys.Vec3;

import java.io.IOException;
import java.net.SocketTimeoutException;
import java.util.Arrays;
import java.util.Optional;

/**
 * Pilot mode: the trained aim/swing/move policies drive the local player.
 * Timing mirrors training — the action applied at the start of tick T was
 * computed from the observation sent at the end of tick T-1. A one-tick
 * pipeline instead of lock-step, so a slow inference server can never freeze
 * the game: a late action just leaves last tick's inputs held.
 *
 * <p>The observation is built with the same code and semantics the agent was
 * trained on (fixed 70° FOV mask via {@link ObsProjector}, PROTOCOL.md scalar
 * order); actions go through vanilla input paths (key mappings, the normal
 * attack packet), so the pilot can do nothing a human at the keyboard could
 * not.
 */
public final class PilotController {
	/** Matches Arena.MAX_TURN_PER_TICK — the trained action space. */
	private static final float MAX_TURN_PER_TICK = 15.0f;
	private static final double TARGET_RANGE = 48.0;
	/** ~3 s of inference silence before giving control back. */
	private static final int MAX_CONSECUTIVE_LATE = 60;
	/** Client ticks to wait for the server to confirm a hit (hurtTime jump). */
	private static final int REACH_CONFIRM_TICKS = 10;

	private PilotBridge bridge;
	private int port = PilotBridge.DEFAULT_PORT;
	private AbstractClientPlayer target;
	private boolean obsPending;
	private int lateActions;
	/** Consumed by KeyboardInputMixin: KeyMapping.setDown is overwritten by
	 * KeyMapping.setAll() every in-game tick, so movement keys go via the mixin. */
	private boolean wantForward;
	private boolean wantJump;
	private boolean wantSprint;

	private final byte[] mask = new byte[(BridgeConfig.OBS_WIDTH * BridgeConfig.OBS_HEIGHT + 7) / 8];
	private double prevX, prevZ;
	private double lastGroundY;
	private float lastReach;
	private float pendingReach;
	private int pendingReachTicks;
	private int prevTargetHurt;

	/** Fight stats for the HUD panel and /pilot status. Hits are counted when
	 * the server confirms them (target hurtTime jump), same as lastReach; the
	 * combo rules mirror training (Arena.stepCombo): hits <= 40 ticks apart
	 * with nothing taken in between chain, a chain of 2+ is a combo. */
	public record Stats(int hits, int sprintHits, int crits, int chain,
			int bestChain, int combos, int taken) {
	}

	private static final int STAT_CHAIN_WINDOW = 40;
	private int statHits;
	private int statSprintHits;
	private int statCrits;
	private int statChain;
	private int statBestChain;
	private int statCombos;
	private int statTaken;
	private int statTick;
	private int lastHitStatTick;
	private boolean takenSinceLastHit;
	private boolean pendingSprint;
	private boolean pendingCrit;
	private int prevSelfHurt;

	public Stats stats() {
		return new Stats(statHits, statSprintHits, statCrits, statChain,
				statBestChain, statCombos, statTaken);
	}

	public boolean isEngaged() {
		return bridge != null;
	}

	public int getPort() {
		return port;
	}

	/** Takes effect on the next engage; /pilot port while engaged reconnects. */
	public void setPort(int port) {
		this.port = port;
	}

	/** The player the pilot is currently fighting, or null. */
	public String targetName() {
		return target != null ? target.getName().getString() : null;
	}

	public boolean wantsForward() {
		return wantForward;
	}

	public boolean wantsJump() {
		return wantJump;
	}

	public boolean wantsSprint() {
		return wantSprint;
	}

	public void toggle(Minecraft mc) {
		if (isEngaged()) {
			disengage(mc, "released");
		} else {
			engage(mc);
		}
	}

	public void engage(Minecraft mc) {
		LocalPlayer player = mc.player;
		if (player == null || mc.level == null) {
			return;
		}
		PilotBridge b;
		try {
			b = new PilotBridge(port, 300, 45);
			b.handshake(BridgeConfig.OBS_WIDTH, BridgeConfig.OBS_HEIGHT);
		} catch (IOException e) {
			msg(mc, "pilot server unreachable on :" + port
					+ " — start trainer/pilot.py first (" + e.getMessage() + ")");
			return;
		}
		bridge = b;
		target = null;
		obsPending = false;
		lateActions = 0;
		prevX = player.getX();
		prevZ = player.getZ();
		lastGroundY = player.getY();
		lastReach = 0;
		pendingReach = 0;
		pendingReachTicks = 0;
		prevTargetHurt = 0;
		statHits = statSprintHits = statCrits = 0;
		statChain = statBestChain = statCombos = statTaken = 0;
		statTick = 0;
		lastHitStatTick = -STAT_CHAIN_WINDOW - 1;
		takenSinceLastHit = false;
		pendingSprint = pendingCrit = false;
		prevSelfHurt = player.hurtTime;
		msg(mc, "engaged — press the toggle again to take back control");
	}

	public void disengage(Minecraft mc, String reason) {
		if (bridge != null) {
			bridge.close();
			bridge = null;
		}
		target = null;
		obsPending = false;
		wantForward = false;
		wantJump = false;
		wantSprint = false;
		msg(mc, reason);
	}

	/** Before the player ticks: receive and apply the pending action. */
	public void startTick(Minecraft mc) {
		if (bridge == null) {
			return;
		}
		LocalPlayer player = mc.player;
		if (player == null || mc.level == null) {
			disengage(mc, "left the world");
			return;
		}
		if (!obsPending) {
			return; // first tick after engaging: nothing requested yet
		}
		PilotBridge.Action a;
		try {
			a = bridge.readAction();
		} catch (SocketTimeoutException e) {
			// reply still in flight; keep last inputs held and stay in sync
			if (++lateActions >= MAX_CONSECUTIVE_LATE) {
				disengage(mc, "inference server stopped answering");
			}
			return;
		} catch (IOException e) {
			disengage(mc, "connection lost: " + e.getMessage());
			return;
		}
		obsPending = false;
		lateActions = 0;

		player.setYRot(Mth.wrapDegrees(player.getYRot()
				+ Mth.clamp(a.dyaw(), -MAX_TURN_PER_TICK, MAX_TURN_PER_TICK)));
		player.setXRot(Mth.clamp(player.getXRot()
				+ Mth.clamp(a.dpitch(), -MAX_TURN_PER_TICK, MAX_TURN_PER_TICK), -90, 90));
		wantForward = a.forward();
		wantJump = a.jump();
		wantSprint = a.sprint();
		if (a.attack()) {
			attack(mc, player);
		}
	}

	/**
	 * Mirrors Arena.applyAction: raycast our own view line against the target
	 * box within vanilla reach; hit → real attack, miss → air swing (which
	 * costs the cooldown, exactly the mechanic the swing policy trained on).
	 */
	private void attack(Minecraft mc, LocalPlayer player) {
		// sample BEFORE the swing resets the ticker; same gates as
		// Player.attack (26.1 bytecode) and Arena.applyAction
		float charge = player.getAttackStrengthScale(0.5f);
		boolean critCandidate = charge > 0.9f && player.fallDistance > 0
				&& !player.onGround() && !player.onClimbable() && !player.isInWater()
				&& !player.isPassenger() && !player.isSprinting();
		boolean sprintCandidate = charge > 0.9f && player.isSprinting();
		if (target != null) {
			Vec3 eye = player.getEyePosition();
			Vec3 end = eye.add(player.getViewVector(1.0f).scale(player.entityInteractionRange()));
			Optional<Vec3> hit = target.getBoundingBox().clip(eye, end);
			if (hit.isPresent() && !blockedByBlocks(mc, player, eye, hit.get())) {
				pendingReach = (float) eye.distanceTo(hit.get());
				pendingReachTicks = REACH_CONFIRM_TICKS;
				pendingCrit = critCandidate;
				pendingSprint = sprintCandidate;
				mc.gameMode.attack(player, target);
				player.swing(InteractionHand.MAIN_HAND);
				return;
			}
		}
		player.swing(InteractionHand.MAIN_HAND);
	}

	/** After the world ticked: update bookkeeping, pick a target, send obs. */
	public void endTick(Minecraft mc) {
		if (bridge == null) {
			return;
		}
		LocalPlayer player = mc.player;
		if (player == null || mc.level == null) {
			disengage(mc, "left the world");
			return;
		}
		if (player.onGround()) {
			lastGroundY = player.getY();
		}
		retarget(mc, player);
		statTick++;

		// getting hit breaks the combo chain, mirroring training
		if (player.hurtTime > prevSelfHurt) {
			statTaken++;
			takenSinceLastHit = true;
			statChain = 0;
		}
		prevSelfHurt = player.hurtTime;

		// last-hit reach: the server confirms a hit as a hurtTime jump, which
		// arrives a few ticks after our attack on a remote server
		if (pendingReachTicks > 0) {
			if (target != null && target.hurtTime > prevTargetHurt) {
				lastReach = pendingReach;
				pendingReachTicks = 0;
				statHits++;
				if (pendingSprint) {
					statSprintHits++;
				}
				if (pendingCrit) {
					statCrits++;
				}
				if (statTick - lastHitStatTick <= STAT_CHAIN_WINDOW && !takenSinceLastHit) {
					statChain++;
				} else {
					statChain = 1;
				}
				if (statChain == 2) {
					statCombos++;
				}
				statBestChain = Math.max(statBestChain, statChain);
				lastHitStatTick = statTick;
				takenSinceLastHit = false;
			} else {
				pendingReachTicks--;
			}
		}
		prevTargetHurt = target != null ? target.hurtTime : 0;

		if (obsPending) {
			return; // previous action still in flight; strict alternation
		}
		buildAndSendObs(mc, player);
	}

	private void retarget(Minecraft mc, LocalPlayer player) {
		if (target != null && (target.isRemoved() || target.isDeadOrDying()
				|| target.isSpectator() || target.distanceTo(player) > TARGET_RANGE)) {
			target = null;
		}
		if (target == null) {
			double best = TARGET_RANGE;
			for (AbstractClientPlayer p : mc.level.players()) {
				if (p == player || p.isSpectator() || p.isDeadOrDying()) {
					continue;
				}
				double d = p.distanceTo(player);
				if (d < best) {
					best = d;
					target = p;
				}
			}
			if (target != null) {
				msg(mc, "target: " + target.getName().getString());
			}
		}
	}

	private void buildAndSendObs(Minecraft mc, LocalPlayer player) {
		Arrays.fill(mask, (byte) 0);
		if (target != null && isTargetVisible(mc, player)) {
			Vec3 eye = player.getEyePosition();
			AABB box = target.getBoundingBox();
			ObsProjector.render(mask, BridgeConfig.OBS_WIDTH, BridgeConfig.OBS_HEIGHT,
					eye.x, eye.y, eye.z, player.getYRot(), player.getXRot(),
					box.minX, box.minY, box.minZ, box.maxX, box.maxY, box.maxZ);
		}

		double dx = player.getX() - prevX;
		double dz = player.getZ() - prevZ;
		prevX = player.getX();
		prevZ = player.getZ();
		double yawRad = Math.toRadians(player.getYRot());
		float[] scalars = {
				(float) Math.sqrt(dx * dx + dz * dz),
				(float) Math.sin(yawRad),
				(float) Math.cos(yawRad),
				player.getXRot() / 90.0f,
				(float) (player.getY() - lastGroundY),
				player.onGround() ? 1 : 0,
				player.getAttackStrengthScale(1.0f),
				player.getHealth() / player.getMaxHealth(),
				lastReach,
		};
		try {
			bridge.sendObs(mask, scalars);
			obsPending = true;
		} catch (IOException e) {
			disengage(mc, "connection lost: " + e.getMessage());
		}
	}

	private boolean isTargetVisible(Minecraft mc, LocalPlayer player) {
		Vec3 eye = player.getEyePosition();
		return !blockedByBlocks(mc, player, eye, target.getEyePosition())
				|| !blockedByBlocks(mc, player, eye, target.getBoundingBox().getCenter());
	}

	private static boolean blockedByBlocks(Minecraft mc, LocalPlayer player, Vec3 from, Vec3 to) {
		return mc.level.clip(new ClipContext(from, to, ClipContext.Block.COLLIDER,
				ClipContext.Fluid.NONE, player)).getType() != HitResult.Type.MISS;
	}

	private static void msg(Minecraft mc, String text) {
		if (mc.player != null) {
			mc.player.sendSystemMessage(Component.literal("[pilot] " + text));
		}
	}
}
