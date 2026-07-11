package com.belubruh123.drlagent.env;

import com.belubruh123.drlagent.bridge.BridgeConfig;

import com.mojang.authlib.GameProfile;

import net.fabricmc.fabric.api.entity.FakePlayer;

import net.minecraft.core.BlockPos;
import net.minecraft.server.level.ServerLevel;
import net.minecraft.util.Mth;
import net.minecraft.world.InteractionHand;
import net.minecraft.world.entity.EquipmentSlot;
import net.minecraft.world.item.ItemStack;
import net.minecraft.world.item.Items;
import net.minecraft.world.level.ClipContext;
import net.minecraft.world.level.GameType;
import net.minecraft.world.level.block.Blocks;
import net.minecraft.world.phys.AABB;
import net.minecraft.world.phys.HitResult;
import net.minecraft.world.phys.Vec3;

import java.nio.ByteBuffer;
import java.util.Optional;
import java.util.Random;
import java.util.UUID;

/**
 * One training arena: a flat platform with an agent fake player and an
 * opponent. Owns per-episode state and the stage reward computation.
 */
public final class Arena {
	private static final int PLATFORM_RADIUS = 20;
	private static final float MAX_TURN_PER_TICK = 15.0f;

	// Stage-1 aim reward constants
	private static final float ON_TARGET_REWARD = 0.3f;
	// Cap on paid on-target ticks per episode: unbounded, the per-tick reward
	// outpays the lock bonus and the policy farms it by never completing a lock.
	private static final int MAX_PAID_ON_TARGET = 10;
	private static final float SHAPING_SCALE = 1.5f;
	private static final float TIME_PENALTY = 0.02f;
	private static final float LOCK_BONUS = 3.0f;
	private static final float LOCK_SPEED_BONUS = 2.0f;
	private static final float TIMEOUT_PENALTY = 1.0f;

	// Swing-stage rewards: a landed hit pays the attack-cooldown charge at the
	// moment of the swing (full charge = +1), any swing that fails to damage
	// (out of reach, occluded, or inside the invulnerability window) is a whiff.
	private static final float WHIFF_PENALTY = 0.3f;

	// Move-stage rewards: sword-PvP spacing band (just inside reach), a small
	// per-tick bonus for holding it plus potential shaping toward it, hits
	// taken hurt. Swing outcomes pass through — positioning enables them.
	private static final double BAND_MIN = 2.0;
	private static final double BAND_MAX = 2.9;
	private static final float IN_BAND_REWARD = 0.02f;
	private static final float BAND_SHAPING = 0.2f;
	private static final float HIT_TAKEN_PENALTY = 0.5f;

	public static final int INFO_ON_TARGET = 1;
	public static final int INFO_HIT_LANDED = 2;
	public static final int INFO_HIT_TAKEN = 4;
	public static final int INFO_ELEVATED = 8;
	public static final int INFO_WHIFF = 16;

	private final int index;
	private final ServerLevel level;
	private final int centerX;
	private final int centerZ;
	private final int floorY;
	private final Random rng;
	private final byte[] mask = new byte[(BridgeConfig.OBS_WIDTH * BridgeConfig.OBS_HEIGHT + 7) / 8];

	private FakePlayer agent;
	private FakePlayer opponent;

	private EnvConfig cfg = new EnvConfig();
	private int episodeTick;
	private int consecOnTarget;
	private int paidOnTargetTicks;
	private double prevAngErr;
	private float lastReach;
	private boolean attackPressed;
	private boolean hitLanded;
	private float lastAttackCharge;
	private boolean forceReset;
	private boolean elevatedEpisode;
	private float opponentHealthBefore;
	private int strafeDir;
	private int strafeFlipTicks;
	private int agentHurtTimeBefore;
	private double bandDistBefore;

	public Arena(int index, ServerLevel level, Random rng) {
		this.index = index;
		this.level = level;
		this.rng = rng;
		this.centerX = index * 1024;
		this.centerZ = 0;
		this.floorY = 200;
	}

	public void build() {
		int chunkR = (PLATFORM_RADIUS >> 4) + 1;
		int ccx = centerX >> 4, ccz = centerZ >> 4;
		for (int cx = ccx - chunkR; cx <= ccx + chunkR; cx++) {
			for (int cz = ccz - chunkR; cz <= ccz + chunkR; cz++) {
				level.setChunkForced(cx, cz, true);
			}
		}
		for (int x = -PLATFORM_RADIUS; x <= PLATFORM_RADIUS; x++) {
			for (int z = -PLATFORM_RADIUS; z <= PLATFORM_RADIUS; z++) {
				level.setBlockAndUpdate(new BlockPos(centerX + x, floorY - 1, centerZ + z),
						Blocks.SMOOTH_STONE.defaultBlockState());
				// clear any terrain above the platform
				for (int y = 0; y < 4; y++) {
					BlockPos p = new BlockPos(centerX + x, floorY + y, centerZ + z);
					if (!level.getBlockState(p).isAir()) {
						level.setBlockAndUpdate(p, Blocks.AIR.defaultBlockState());
					}
				}
			}
		}
	}

	public void spawnPlayers() {
		agent = createPlayer("Agent" + index);
		opponent = createPlayer("Enemy" + index);
	}

	private FakePlayer createPlayer(String name) {
		UUID uuid = UUID.nameUUIDFromBytes(("drlagent:" + name).getBytes());
		FakePlayer p = new AgentPlayer(level, new GameProfile(uuid, name));
		p.setPos(centerX + 0.5, floorY, centerZ + 0.5);
		level.addNewPlayer(p);
		p.setGameMode(GameType.SURVIVAL);
		p.setItemSlot(EquipmentSlot.HEAD, new ItemStack(Items.DIAMOND_HELMET));
		p.setItemSlot(EquipmentSlot.CHEST, new ItemStack(Items.DIAMOND_CHESTPLATE));
		p.setItemSlot(EquipmentSlot.LEGS, new ItemStack(Items.DIAMOND_LEGGINGS));
		p.setItemSlot(EquipmentSlot.FEET, new ItemStack(Items.DIAMOND_BOOTS));
		p.setItemSlot(EquipmentSlot.MAINHAND, new ItemStack(Items.DIAMOND_SWORD));
		return p;
	}

	public void setConfig(EnvConfig cfg) {
		this.cfg = cfg;
	}

	public void reset() {
		episodeTick = 0;
		consecOnTarget = 0;
		paidOnTargetTicks = 0;
		lastReach = 0;
		forceReset = false;

		float agentYaw = (float) (rng.nextDouble() * 360.0 - 180.0);
		place(agent, centerX + 0.5, floorY, centerZ + 0.5, agentYaw, 0);
		agent.setNoGravity(false);

		boolean horizontal = rng.nextDouble() < cfg.horizontalProb || cfg.maxElev <= 0;
		elevatedEpisode = !horizontal;
		double dist = cfg.distMin + rng.nextDouble() * (cfg.distMax - cfg.distMin);
		double angle = Math.toRadians(agentYaw + (rng.nextDouble() * 2 - 1) * cfg.yawRange);
		// Direction the agent looks at yaw a: (-sin a, 0, cos a)
		double ox = centerX + 0.5 - Math.sin(angle) * dist;
		double oz = centerZ + 0.5 + Math.cos(angle) * dist;
		double oy = floorY;
		if (!horizontal) {
			oy += (rng.nextDouble() * 2 - 1) * cfg.maxElev;
			oy = Math.max(oy, floorY - 0.5);
		}
		float opponentYaw = (float) Mth.wrapDegrees(Math.toDegrees(angle) + 180);
		place(opponent, ox, oy, oz, opponentYaw, 0);
		// A hovering opponent is a stationary aim dummy; grounded ones use physics.
		opponent.setNoGravity(!horizontal || "static".equals(cfg.opponent));
		strafeDir = rng.nextBoolean() ? 1 : -1;
		strafeFlipTicks = 20 + rng.nextInt(40);

		opponentHealthBefore = opponent.getHealth();
		prevAngErr = currentAngleError();
		agentHurtTimeBefore = 0;
		bandDistBefore = bandDistance();
	}

	/** How far the opponent sits outside the ideal spacing band (0 inside it). */
	private double bandDistance() {
		double d = opponent.distanceTo(agent);
		return Math.min(Math.max(0, Math.max(BAND_MIN - d, d - BAND_MAX)), 5.0);
	}

	private void place(FakePlayer p, double x, double y, double z, float yaw, float pitch) {
		p.teleportTo(x, y, z);
		p.setYRot(yaw);
		p.setYHeadRot(yaw);
		p.setXRot(pitch);
		p.setOldPosAndRot();
		p.setDeltaMovement(Vec3.ZERO);
		p.setHealth(p.getMaxHealth());
		p.getFoodData().setFoodLevel(20);
		p.resetAttackStrengthTicker();
		p.invulnerableTime = 0;
		p.hurtTime = 0;
		p.zza = 0;
		p.xxa = 0;
		p.setSprinting(false);
	}

	public void applyAction(float dyaw, float dpitch, boolean attack, boolean forward, boolean reset) {
		if (reset) {
			forceReset = true;
		}
		attackPressed = attack;
		hitLanded = false;

		agent.setOldPosAndRot();
		opponent.setOldPosAndRot();

		agent.setYRot(Mth.wrapDegrees(agent.getYRot() + Mth.clamp(dyaw, -MAX_TURN_PER_TICK, MAX_TURN_PER_TICK)));
		agent.setYHeadRot(agent.getYRot());
		agent.setXRot(Mth.clamp(agent.getXRot() + Mth.clamp(dpitch, -MAX_TURN_PER_TICK, MAX_TURN_PER_TICK), -90, 90));

		agent.zza = forward ? 1.0f : 0.0f;
		agent.xxa = 0;
		agent.setSprinting(forward);

		if (attack) {
			// 26.1: swing() itself resets the attack ticker (even air swings),
			// so sample the charge before anything else
			lastAttackCharge = agent.getAttackStrengthScale(0.5f);
			agent.swing(InteractionHand.MAIN_HAND);
			Vec3 eye = agent.getEyePosition();
			Vec3 end = eye.add(agent.getViewVector(1.0f).scale(agent.entityInteractionRange()));
			Optional<Vec3> hit = opponent.getBoundingBox().clip(eye, end);
			if (hit.isPresent() && !blockedByBlocks(eye, hit.get())) {
				int hurtBefore = opponent.hurtTime;
				// vanilla applies full damage + knockback only when the
				// invulnerability window is at half or below (<=10 of 20);
				// above that, escalating spam swings register difference
				// damage with no knockback — the policy farms those, so
				// they count as whiffs, mirroring vanilla's full-hit boundary
				boolean clean = opponent.invulnerableTime <= 10;
				agent.attack(opponent);
				hitLanded = clean && opponent.hurtTime > hurtBefore;
				if (hitLanded) {
					lastReach = (float) eye.distanceTo(hit.get());
				}
			}
		}

		if ("strafe".equals(cfg.opponent)) {
			tickStrafeOpponent();
		} else if ("fight".equals(cfg.opponent)) {
			tickStrafeOpponent();
			tickOpponentAttack();
		}
		// No manual agent/opponent tick: the server level ticks entities in
		// force-loaded chunks itself — adding one here double-ticks physics
		// and the attack-cooldown ticker.
	}

	/** Fighting opponent: swings at the agent with perfect discipline —
	 * full charge, in reach, outside the agent's invulnerability window. */
	private void tickOpponentAttack() {
		if (opponent.getAttackStrengthScale(0.5f) < 0.9f) {
			return;
		}
		Vec3 eye = opponent.getEyePosition();
		Vec3 dir = agent.getEyePosition().subtract(eye).normalize();
		Vec3 end = eye.add(dir.scale(opponent.entityInteractionRange()));
		Optional<Vec3> hit = agent.getBoundingBox().clip(eye, end);
		if (hit.isPresent() && !blockedByBlocks(eye, hit.get())
				&& agent.invulnerableTime <= 10) {
			opponent.swing(InteractionHand.MAIN_HAND);
			opponent.attack(agent);
		}
	}

	/** Circle-strafe around the agent, flipping direction at random intervals
	 * and steering back toward sword range when knockback pushes it out. */
	private void tickStrafeOpponent() {
		double dx = agent.getX() - opponent.getX();
		double dz = agent.getZ() - opponent.getZ();
		float yawToAgent = (float) Math.toDegrees(Math.atan2(-dx, dz));
		opponent.setYRot(yawToAgent);
		opponent.setYHeadRot(yawToAgent);
		if (--strafeFlipTicks <= 0) {
			strafeDir = -strafeDir;
			strafeFlipTicks = 20 + rng.nextInt(40);
		}
		opponent.xxa = 0.98f * strafeDir;
		// oscillate around sword reach: walk back in after knockback, give
		// ground when crowded — keeps in-range/out-of-range transitions coming
		// so the swing model sees both sides of the reach boundary. Center
		// distance 3.0 ≈ ray-to-hitbox 2.7 (box half-width 0.3), inside reach.
		double dist = opponent.distanceTo(agent);
		opponent.zza = dist > 3.0 ? 0.7f : (dist < 2.0 ? -0.5f : 0.0f);
	}

	public record StepResult(float reward, boolean done, int info) {
	}

	/** Called after the world tick: computes reward, handles episode end + auto-reset. */
	public StepResult step() {
		episodeTick++;

		if ("swing".equals(cfg.stage)) {
			return stepSwing();
		}
		if ("move".equals(cfg.stage)) {
			return stepMove();
		}

		boolean onTarget = isCrosshairOnTarget();
		double angErr = currentAngleError();

		float reward = 0;
		reward += (float) (SHAPING_SCALE * (prevAngErr - angErr) / 180.0);
		if (onTarget) {
			if (paidOnTargetTicks < MAX_PAID_ON_TARGET) {
				reward += ON_TARGET_REWARD;
				paidOnTargetTicks++;
			}
			consecOnTarget++;
		} else {
			consecOnTarget = 0;
		}
		reward -= TIME_PENALTY;
		prevAngErr = angErr;

		int info = 0;
		if (onTarget) info |= INFO_ON_TARGET;
		if (hitLanded) info |= INFO_HIT_LANDED;
		if (elevatedEpisode) info |= INFO_ELEVATED;
		if (opponent.getHealth() < opponentHealthBefore - 1e-4) {
			// reserved for later stages (opponent fights back)
		}
		if (agent.hurtTime > 0) info |= INFO_HIT_TAKEN;
		opponentHealthBefore = opponent.getHealth();

		boolean done = false;
		if (consecOnTarget >= cfg.lockTicks) {
			reward += LOCK_BONUS + LOCK_SPEED_BONUS * (1.0f - (float) episodeTick / cfg.episodeTicks);
			done = true;
		} else if (episodeTick >= cfg.episodeTicks) {
			reward -= TIMEOUT_PENALTY;
			done = true;
		} else if (forceReset) {
			done = true;
		}

		if (done) {
			reset();
		}
		return new StepResult(reward, done, info);
	}

	/** Swing stage: the (frozen) aim model steers, this reward only judges the
	 * attack button. Fixed-length episodes; the opponent never dies (healed each
	 * tick) so knockback keeps varying the range instead of ending the fight. */
	private StepResult stepSwing() {
		float reward = 0;
		if (attackPressed) {
			reward += hitLanded ? lastAttackCharge : -WHIFF_PENALTY;
		}

		int info = 0;
		if (isCrosshairOnTarget()) info |= INFO_ON_TARGET;
		if (hitLanded) info |= INFO_HIT_LANDED;
		else if (attackPressed) info |= INFO_WHIFF;
		if (elevatedEpisode) info |= INFO_ELEVATED;
		if (agent.hurtTime > 0) info |= INFO_HIT_TAKEN;

		opponent.setHealth(opponent.getMaxHealth());
		opponent.getFoodData().setFoodLevel(20);

		boolean fellOff = opponent.getY() < floorY - 2;
		boolean done = episodeTick >= cfg.episodeTicks || fellOff || forceReset;
		if (done) {
			reset();
		}
		return new StepResult(reward, done, info);
	}

	/** Move stage: frozen aim + swing act, the W key learns spacing. Swing
	 * outcomes pass through, fresh hits taken are punished, and the spacing
	 * band pays a small hold bonus plus potential shaping toward it. */
	private StepResult stepMove() {
		float reward = 0;
		if (attackPressed) {
			reward += hitLanded ? lastAttackCharge : -WHIFF_PENALTY;
		}
		boolean freshHitTaken = agent.hurtTime > agentHurtTimeBefore;
		agentHurtTimeBefore = agent.hurtTime;
		if (freshHitTaken) {
			reward -= HIT_TAKEN_PENALTY;
		}
		double bandDist = bandDistance();
		reward += (float) (BAND_SHAPING * (bandDistBefore - bandDist));
		if (bandDist == 0) {
			reward += IN_BAND_REWARD;
		}
		bandDistBefore = bandDist;

		int info = 0;
		if (isCrosshairOnTarget()) info |= INFO_ON_TARGET;
		if (hitLanded) info |= INFO_HIT_LANDED;
		else if (attackPressed) info |= INFO_WHIFF;
		if (elevatedEpisode) info |= INFO_ELEVATED;
		if (freshHitTaken) info |= INFO_HIT_TAKEN;

		opponent.setHealth(opponent.getMaxHealth());
		opponent.getFoodData().setFoodLevel(20);
		agent.setHealth(agent.getMaxHealth());
		agent.getFoodData().setFoodLevel(20);

		boolean fellOff = opponent.getY() < floorY - 2 || agent.getY() < floorY - 2;
		boolean done = episodeTick >= cfg.episodeTicks || fellOff || forceReset;
		if (done) {
			reset();
		}
		return new StepResult(reward, done, info);
	}

	private boolean isCrosshairOnTarget() {
		Vec3 eye = agent.getEyePosition();
		Vec3 view = agent.getViewVector(1.0f);
		AABB box = opponent.getBoundingBox();
		return ObsProjector.rayHitsBox(view.x, view.y, view.z,
				box.minX - eye.x, box.minY - eye.y, box.minZ - eye.z,
				box.maxX - eye.x, box.maxY - eye.y, box.maxZ - eye.z);
	}

	private double currentAngleError() {
		Vec3 eye = agent.getEyePosition();
		Vec3 target = opponent.getBoundingBox().getCenter();
		return ObsProjector.angleToTarget(agent.getYRot(), agent.getXRot(),
				eye.x, eye.y, eye.z, target.x, target.y, target.z);
	}

	private boolean blockedByBlocks(Vec3 from, Vec3 to) {
		return level.clip(new ClipContext(from, to, ClipContext.Block.COLLIDER,
				ClipContext.Fluid.NONE, agent)).getType() != HitResult.Type.MISS;
	}

	/** Writes mask + scalars for the current post-tick state (PROTOCOL.md order). */
	public void writeObs(ByteBuffer buf) {
		java.util.Arrays.fill(mask, (byte) 0);
		if (opponent.isAlive() && isTargetVisible()) {
			Vec3 eye = agent.getEyePosition();
			AABB box = opponent.getBoundingBox();
			ObsProjector.render(mask, BridgeConfig.OBS_WIDTH, BridgeConfig.OBS_HEIGHT,
					eye.x, eye.y, eye.z, agent.getYRot(), agent.getXRot(),
					box.minX, box.minY, box.minZ, box.maxX, box.maxY, box.maxZ);
		}
		buf.put(mask);

		// actual displacement this tick, not getDeltaMovement(): that reads
		// post-friction velocity (x0.546 on ground) and under-reports speed
		double dx = agent.getX() - agent.xOld;
		double dz = agent.getZ() - agent.zOld;
		buf.putFloat((float) Math.sqrt(dx * dx + dz * dz));
		double yawRad = Math.toRadians(agent.getYRot());
		buf.putFloat((float) Math.sin(yawRad));
		buf.putFloat((float) Math.cos(yawRad));
		buf.putFloat(agent.getXRot() / 90.0f);
		buf.putFloat((float) (agent.getY() - floorY));
		buf.putFloat(agent.onGround() ? 1 : 0);
		buf.putFloat(agent.getAttackStrengthScale(1.0f));
		buf.putFloat(agent.getHealth() / agent.getMaxHealth());
		buf.putFloat(lastReach);
	}

	private boolean isTargetVisible() {
		Vec3 eye = agent.getEyePosition();
		return !blockedByBlocks(eye, opponent.getEyePosition())
				|| !blockedByBlocks(eye, opponent.getBoundingBox().getCenter());
	}

	public int getIndex() {
		return index;
	}
}
