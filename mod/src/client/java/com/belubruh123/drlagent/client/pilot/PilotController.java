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
	/** Matches Arena.MAX_TURN_PER_TICK — the trained action space (restored to
	 * the original 15 deg/tick; the 7.5 nerf made the pilot too slow to track). */
	private static final float MAX_TURN_PER_TICK = 15.0f;
	private static final double TARGET_RANGE = 48.0;
	/** ~3 s of inference silence before giving control back. */
	private static final int MAX_CONSECUTIVE_LATE = 60;
	/** Client ticks to wait for the server to confirm a hit (hurtTime jump). */
	private static final int REACH_CONFIRM_TICKS = 10;

	// --- Online training (train mode) -------------------------------------
	/** Live-play episode length before a timeout "done" (matches stage8/7b). */
	private static final int TRAIN_EPISODE_TICKS = 1200;
	// Combat reward constants MIRRORED from Arena.java stepCombo — keep in sync
	// (a shared env/ComboReward helper is the clean follow-up). The dominant
	// terms are reproduced; minor per-tick shaping (strafe/jump/hesitation
	// costs, chain-drop-on-lapse) is intentionally omitted for the live path.
	private static final float R_WHIFF = 1.0f;
	private static final float R_HIT_TAKEN = 0.75f;
	private static final float R_CRIT_SCALE = 0.35f;
	private static final float R_SPRINT_SCALE = 1.0f;
	private static final float R_CHAIN_BONUS = 0.35f;
	private static final int R_CHAIN_CAP = 8;
	private static final float R_CHAIN_BREAK = 0.2f;
	private static final float R_PURSUIT = 0.15f;
	private static final float R_CROWD = 0.03f;
	private static final double R_CROWD_DIST = 1.6;
	private static final float R_KILL = 5.0f;
	private static final float R_DEATH = 5.0f;
	private static final float R_MORTAL_TICK = 0.02f;
	private static final float R_FAR = 0.01f;
	// info bit flags — same set/values as vec_env.py + Arena.INFO_*
	private static final int INFO_HIT_LANDED = 2;
	private static final int INFO_HIT_TAKEN = 4;
	private static final int INFO_WHIFF = 16;
	private static final int INFO_CRIT = 32;
	private static final int INFO_SPRINT_HIT = 64;
	private static final int INFO_CHAIN_HIT = 128;

	private PilotBridge bridge;
	private int port = PilotBridge.DEFAULT_PORT;
	private AbstractClientPlayer target;
	private boolean obsPending;
	private int lateActions;
	/** Consumed by KeyboardInputMixin: KeyMapping.setDown is overwritten by
	 * KeyMapping.setAll() every in-game tick, so movement keys go via the mixin.
	 * move: 0 none, 1 = W, 2 = S; strafe: 0 none, 1 = A, 2 = D. */
	private int wantMove;
	private int wantStrafe;
	private boolean wantJump;
	private boolean wantSprint;

	private final byte[] mask = new byte[(BridgeConfig.OBS_WIDTH * BridgeConfig.OBS_HEIGHT + 7) / 8];
	private double prevX, prevZ;
	private double lastGroundY;
	private float lastReach;
	private float pendingReach;
	private int pendingReachTicks;
	private int prevTargetHurt;

	// Online training: when the connected server asked for train mode, the
	// controller also computes a per-tick combat reward + done and sends them
	// with the obs, so trainer/pilot_train.py can run PPO from live play.
	private boolean trainMode;
	private float trainReward;
	private int trainInfo;
	private boolean airWhiffThisTick;
	private float pendingCharge;
	private double chasePhiBefore;
	private boolean haveChasePhi;
	private boolean hadTarget;
	private int episodeStartTick;

	/** Fight stats for the HUD panel and /pilot status. Hits are counted when
	 * the server confirms them (target hurtTime jump), same as lastReach; the
	 * combo rules mirror training (Arena.stepCombo): hits <= 30 ticks apart
	 * with nothing taken in between chain, a chain of 2+ is a combo. */
	public record Stats(int hits, int sprintHits, int crits, int chain,
			int bestChain, int combos, int taken) {
	}

	private static final int STAT_CHAIN_WINDOW = 30;
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

	public int wantsMove() {
		return wantMove;
	}

	public int wantsStrafe() {
		return wantStrafe;
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
		boolean train;
		try {
			b = new PilotBridge(port, 300, 45);
			train = b.handshake(BridgeConfig.OBS_WIDTH, BridgeConfig.OBS_HEIGHT);
		} catch (IOException e) {
			msg(mc, "pilot server unreachable on :" + port
					+ " — start trainer/pilot.py first (" + e.getMessage() + ")");
			return;
		}
		bridge = b;
		trainMode = train;
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
		trainReward = 0;
		airWhiffThisTick = false;
		pendingCharge = 0;
		chasePhiBefore = 0;
		haveChasePhi = false;
		hadTarget = false;
		episodeStartTick = 0;
		msg(mc, trainMode
				? "engaged (TRAINING) — the policy is learning from this fight"
				: "engaged — press the toggle again to take back control");
	}

	public void disengage(Minecraft mc, String reason) {
		if (bridge != null) {
			bridge.close();
			bridge = null;
		}
		target = null;
		obsPending = false;
		wantMove = 0;
		wantStrafe = 0;
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

		// sub-degree deadzone mirrors Arena.applyAction: the policy trained
		// with "hold still" available, so the pilot must honor it too
		float dyaw = Mth.clamp(a.dyaw(), -MAX_TURN_PER_TICK, MAX_TURN_PER_TICK);
		float dpitch = Mth.clamp(a.dpitch(), -MAX_TURN_PER_TICK, MAX_TURN_PER_TICK);
		if (Math.abs(dyaw) < 5.0f) dyaw = 0;
		if (Math.abs(dpitch) < 5.0f) dpitch = 0;
		player.setYRot(Mth.wrapDegrees(player.getYRot() + dyaw));
		player.setXRot(Mth.clamp(player.getXRot() + dpitch, -90, 90));
		wantMove = a.move();
		wantStrafe = a.strafe();
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
				pendingCharge = charge;
				pendingCrit = critCandidate;
				pendingSprint = sprintCandidate;
				mc.gameMode.attack(player, target);
				player.swing(InteractionHand.MAIN_HAND);
				return;
			}
		}
		// swung with no target in reach: an air swing that burns the cooldown
		airWhiffThisTick = true;
		player.swing(InteractionHand.MAIN_HAND);
	}

	/** After the world ticked: update bookkeeping, pick a target, compute the
	 * training reward (train mode only), and send the obs. */
	public void endTick(Minecraft mc) {
		if (bridge == null) {
			return;
		}
		LocalPlayer player = mc.player;
		if (player == null || mc.level == null) {
			disengage(mc, "left the world");
			return;
		}
		// snapshot terminal events BEFORE retarget clears a dead/removed target
		boolean targetKilled = target != null
				&& (target.isDeadOrDying() || target.isRemoved());
		boolean selfDead = player.isDeadOrDying() || player.getHealth() <= 0;

		if (player.onGround()) {
			lastGroundY = player.getY();
		}
		retarget(mc, player);
		statTick++;
		trainReward = 0;
		trainInfo = 0;

		// getting hit breaks the combo chain, mirroring training
		if (player.hurtTime > prevSelfHurt) {
			if (trainMode) {
				trainReward -= R_HIT_TAKEN;
				if (!takenSinceLastHit && statTick - lastHitStatTick <= STAT_CHAIN_WINDOW) {
					trainReward -= R_CHAIN_BREAK
							* Math.min(Math.max(statChain - 1, 0), R_CHAIN_CAP);
				}
			}
			trainInfo |= INFO_HIT_TAKEN;
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
				trainInfo |= INFO_HIT_LANDED;
				if (pendingCrit) trainInfo |= INFO_CRIT;
				if (pendingSprint) trainInfo |= INFO_SPRINT_HIT;
				if (statChain >= 2) trainInfo |= INFO_CHAIN_HIT;
				if (trainMode) {
					trainReward += pendingCharge;
					if (pendingCrit) trainReward += R_CRIT_SCALE * pendingCharge;
					if (pendingSprint) trainReward += R_SPRINT_SCALE * pendingCharge;
					trainReward += R_CHAIN_BONUS * Math.min(statChain - 1, R_CHAIN_CAP);
				}
			} else if (--pendingReachTicks == 0) {
				// confirmation window lapsed with no hurtTime jump: swing missed
				trainInfo |= INFO_WHIFF;
				if (trainMode) trainReward -= R_WHIFF;
			}
		}
		prevTargetHurt = target != null ? target.hurtTime : 0;

		// an air swing (no target in reach) is an immediate whiff
		if (airWhiffThisTick) {
			trainInfo |= INFO_WHIFF;
			if (trainMode) trainReward -= R_WHIFF;
		}
		airWhiffThisTick = false;

		boolean done = false;
		if (trainMode) {
			// spacing / mortal shaping (dominant stepCombo terms; the minor
			// per-tick strafe/jump/hesitation costs are omitted for live play).
			// Only accrue while a target is engaged — idling before acquisition
			// must not bleed reward or spawn spurious 1-tick episodes.
			if (target != null) {
				double dist = player.distanceTo(target);
				double chasePhi = -Math.max(0.0, dist - 2.8);
				if (!haveChasePhi) {
					chasePhiBefore = chasePhi;
					haveChasePhi = true;
				}
				trainReward += (float) (R_PURSUIT * (chasePhi - chasePhiBefore));
				chasePhiBefore = chasePhi;
				trainReward -= R_MORTAL_TICK;
				trainReward -= (float) (R_FAR * Math.min(4.0, Math.max(0.0, dist - 3.0)));
				if (dist < R_CROWD_DIST) {
					trainReward -= R_CROWD;
				}
			}
			if (targetKilled) trainReward += R_KILL;
			if (selfDead) trainReward -= R_DEATH;
			// episode ends on a kill, a death, losing an engaged target, or the
			// timeout; a fresh connection with no target yet does NOT end.
			boolean lostTarget = hadTarget && target == null;
			done = targetKilled || selfDead || lostTarget
					|| statTick - episodeStartTick >= TRAIN_EPISODE_TICKS;
			if (done) {
				statChain = 0;
				takenSinceLastHit = false;
				haveChasePhi = false;
				episodeStartTick = statTick;
			}
		}
		hadTarget = target != null;

		if (obsPending) {
			return; // previous action still in flight; strict alternation
		}
		buildAndSendObs(mc, player, done);
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

	private void buildAndSendObs(Minecraft mc, LocalPlayer player, boolean done) {
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
			if (trainMode) {
				bridge.sendObs(mask, scalars, trainReward, done, trainInfo);
			} else {
				bridge.sendObs(mask, scalars);
			}
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
