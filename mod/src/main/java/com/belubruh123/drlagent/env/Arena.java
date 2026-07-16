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
	// Sized for LONG combos: each hit knocks the target ~3 blocks back and it
	// walks partway back during the cooldown; a 20-hit chain still drifts a
	// few dozen blocks, and the fight must not ring-out before the combo can.
	private static final int PLATFORM_RADIUS = 32;
	// Soft wall: real flat-ground PvP has no void edge, but our platform does,
	// and sprint-knockback juggles the opponent off it before the kill lands.
	// Recorded duels confirmed it — ~half ended with the opponent walking past
	// x/z=+-32 and falling (oppY -> -1.6, no KILL reward), capping "success" at
	// 0.5. Clamping both fighters just inside the edge each tick makes the arena
	// behave like an unbounded field: nobody rings out, the combo must actually
	// kill. 2-block margin keeps the 0.6-wide hitbox fully on the stone.
	private static final double IN_BOUNDS_RADIUS = PLATFORM_RADIUS - 2;
	// Camera turn cap, restored to the original 15 deg/tick (2x) — the 7.5 nerf
	// (commit 0d0f489) made the aim too slow to keep up with a moving target.
	// The net saturates this cap on large corrections, so raising it makes
	// flicks/tracking 2x faster while sub-deadzone adjustments are unchanged.
	// Usable per-tick band with the 5 deg deadzone is [5, 15].
	private static final float MAX_TURN_PER_TICK = 15.0f;

	// Stage-1 aim reward constants
	private static final float ON_TARGET_REWARD = 0.3f;
	// Smooth-aim fine-tune: the cost is on JERK — the tick-to-tick CHANGE in
	// the aim command — not on turn magnitude. Tracking a moving target
	// correctly needs sustained turning (smooth pursuit ~ the target's
	// angular speed), which is jerk-free; bang-bang alternation (±15° flips,
	// the camera shake) pays the full cost every tick; a flick pays once.
	// First attempt penalized |dyaw| while on target and lock success fell
	// 96%->73% by 250k — it was punishing correct pursuit.
	private static final float AIM_JERK_COST = 0.2f;
	// User's aim doctrine: a move must be worth >= 5 degrees or not happen at
	// all — deliberate flicks, zero chatter. Commands under the deadzone snap
	// to 0 (a Gaussian head can never output exact 0, so this IS the "hold
	// still" action). Anywhere on the white blob counts as aimed (no
	// centering pay while on target), and moving while aimed costs flat.
	private static final float AIM_DEADZONE = 5.0f;
	private static final float ON_TARGET_MOVE_PENALTY = 0.10f;
	// Bonus for holding PERFECTLY STILL while aimed (dyaw==dpitch==0, i.e. the
	// command sits inside the deadzone). Once the crosshair is on target the
	// ideal is to freeze — this pays for it on top of avoiding the move penalty,
	// so "aim then stop" strictly beats micro-adjusting on a settled target.
	// removed: AIMED_STILL_BONUS (caused mouse freezing)
	// Cap on paid on-target ticks per episode: unbounded, the per-tick reward
	// outpays the lock bonus and the policy farms it by never completing a lock.
	private static final int MAX_PAID_ON_TARGET = 10;
	private static final float SHAPING_SCALE = 1.5f;
	private static final float TIME_PENALTY = 0.02f;
	private static final float LOCK_BONUS = 3.0f;
	private static final float LOCK_SPEED_BONUS = 2.0f;
	private static final float TIMEOUT_PENALTY = 1.0f;

	// Vertical-aim recovery rewards: three signals that together give the policy
	// a clear gradient to find a lost target by correcting extreme pitch.
	//
	// (1) EXTREME_PITCH_COST: per-tick penalty for looking beyond ±35° from level.
	//     Ramps linearly to full cost at ±90°. At ±45° the cost ≈ 0.05/tick, mild
	//     enough that aiming at a legitimately elevated target still pays, but
	//     sustaining 60°+ without a target on-screen is ruinous.
	private static final float EXTREME_PITCH_COST = 0.75f;
	private static final float PITCH_COMFORT_DEG = 35.0f;
	// (2) LOST_TARGET_PENALTY: flat bleed when the target is fully off-screen.
	//     Creates urgency beyond the regular time penalty — the agent must DO
	//     SOMETHING to get the target back, not just hold still.
	private static final float LOST_TARGET_PENALTY = 0.10f;
	// (3) PITCH_RETURN_SHAPING: potential-based shaping (phi = -|pitch|/90) that
	//     rewards pitch corrections TOWARD level when the mask is blank. Potential-
	//     based so it doesn't shift the optimal policy; just accelerates learning
	//     the "return to level when lost" behavior. At 15°/tick max turn speed
	//     one tick of correction earns ~0.05, which with the extreme-pitch relief
	//     beats the jerk cost and time penalty — so correcting IS net positive.
	private static final float PITCH_RETURN_SHAPING = 0.90f;
	// (4) LEVEL_PITCH_BONUS: user-requested flat reward for keeping pitch near 0.
	//     Gives a tiny fraction of encouragement for just returning to center.
	//     Max +0.02 at 0°, dropping to 0 at ±30°.
	private static final float LEVEL_PITCH_BONUS = 0.02f;

	// Swing-stage rewards: a landed hit pays the attack-cooldown charge at the
	// moment of the swing (full charge = +1), any swing that fails to damage
	// (out of reach, occluded, or inside the invulnerability window) is a whiff.
	// Whiffs burn the 12.5-tick cooldown — inside a combo that IS the combo
	// dying. The user's target is <5 whiffs/ep: at 1.0 a speculative click
	// only pays if the hit is likelier than not, so the policy must actually
	// judge range and charge before pressing. The fighter2 head owns the
	// attack button, so missed swings are its own choice. (Was 0.3 while the
	// frozen swing net clicked; 0.6 briefly.)
	private static final float WHIFF_PENALTY = 0.1f;
	// Spamming swings before the cooldown refills ruins damage/knockback and
	// wastes the opponent's invulnerability frames.
	private static final float SPAM_PENALTY = 0.50f;
	// The user's rule: cooldown done + in range = swing NOW. Every tick where
	// a swing would land (full charge, crosshair on the target within reach,
	// target vulnerable) and the policy does not press costs this much —
	// hesitating 5 ticks burns 0.4, nearly half a hit's base pay.
	private static final float HESITATION_PENALTY = 0.08f;
	// Held strafe costs a trickle: an A-tap into a hit (3-4 ticks) is
	// pennies, orbiting the opponent all episode is ruinous. Only an aimbot
	// can fight while circling — combos should run mostly straight lines.
	private static final float STRAFE_COST = 0.015f;
	// Mortal fights (cfg.mortal): the episode IS the duel — winning it pays
	// like a deep combo, losing it costs the same. Dying must never be the
	// cheap way out of a bad position, and stalling a duel out must never
	// be free (the tick cost also makes farming a live victim strictly
	// worse than finishing them).
	private static final float KILL_REWARD = 5.0f;
	private static final float DEATH_PENALTY = 5.0f;
	private static final float MORTAL_TICK_COST = 0.02f;
	// Hovering out of reach must bleed, or kiting at constant radius is a
	// near-free equilibrium (potential-based pursuit pays nothing when the
	// distance is not CHANGING). Position cost, not potential: per tick,
	// scaled by blocks beyond sword range, capped at 4 blocks.
	private static final float FAR_COST = 0.01f;

	// Move-stage rewards: sword-PvP spacing band (just inside reach), a small
	// per-tick bonus for holding it plus potential shaping toward it, hits
	// taken hurt. Swing outcomes pass through — positioning enables them.
	private static final double BAND_MIN = 2.0;
	private static final double BAND_MAX = 2.9;
	private static final float IN_BAND_REWARD = 0.02f;
	private static final float BAND_SHAPING = 0.2f;
	private static final float HIT_TAKEN_PENALTY = 0.75f;

	// Combo-stage rewards: chained knockback hits and crits. A chained hit is
	// a clean hit within CHAIN_WINDOW ticks of the previous one with no hit
	// taken in between; a crit pays half again its charge (vanilla deals 1.5x).
	// Sprint hits pay their own bonus: knockback is utility, not damage, so
	// without it the policy rationally drops sprint for crit-fishing (observed
	// at 1.75M steps: sprint hits collapsed to 0). Sprint must outpay crit:
	// exploring the sprint bit mid-air voids crits (the crit gate requires
	// !isSprinting), so at 0.35 the bonus lost to the crit's 0.5 everywhere
	// and sprint stalled at ~1/ep. The two apply in disjoint states (grounded
	// vs airborne), so both skills survive. The window enforces the combo's
	// cadence — hit again the moment the 12.5-tick cooldown refills, with only
	// the knockback-chase (~15 ticks of sprint) in between. 30 ticks fits one
	// clean chase; it does NOT fit a whiffed cycle plus a re-approach, which a
	// looser window (the old 40) wrongly credited as still comboing.
	private static final int CHAIN_WINDOW = 30;
	private static final float CHAIN_BONUS = 0.35f;
	// The user's target metric is the combo — hits landed on cooldown without
	// being hit in between — so every chained hit must outpay a crit: at 0.35
	// the 2nd clean hit already matches the crit bonus and the 9th earns +2.8
	// on top of its base pay. Taking a hit both breaks the chain and costs
	// HIT_TAKEN_PENALTY, so vs the humanized opponent (where avoidance is
	// actually possible) protecting a live chain by spacing/strafing is worth
	// more than any single trade.
	private static final int CHAIN_CAP = 8;
	// Getting knocked out of a LIVE combo pays extra, scaled like the bonus:
	// losing a 5-chain costs 0.75 + 0.8, a neutral trade just 0.75. The deeper
	// the chain, the more disengaging (S-tap, strafe out) beats trading.
	private static final float CHAIN_BREAK_PENALTY = 0.2f;
	// After a hit the ONLY correct plan is to close and hit again the moment
	// the cooldown refills. While the combo is live, closing toward reach pays
	// and retreating charges (potential on distance beyond 2.8 blocks), and
	// letting the window lapse without the follow-up hit is punished like a
	// break — running away to "find a safer opening" is never free.
	private static final float PURSUIT_SHAPING = 0.15f;
	// The target is a chain on essentially EVERY hit: letting the window
	// lapse costs a good chunk of what continuing it would have paid.
	private static final float CHAIN_DROP_PENALTY = 0.25f;
	private static final float CRIT_BONUS_SCALE = 0.35f;
	// A sprint hit pays DOUBLE a plain hit: without it the policy discovers
	// crowding — stand inside the target, skip the sprint knockback so the
	// victim never leaves reach, and farm plain hits at perfect cadence
	// (observed: sprint hits slid 10.5 -> 7/ep as it optimized the school).
	private static final float SPRINT_HIT_BONUS_SCALE = 1.0f;
	// Standing closer than a sword-fight ever needs also costs directly:
	// vs a real opponent, chest-to-chest is where you get hit back.
	// Raised 1.6 -> 2.4 (user: "getting too close"). Reach is ~3 blocks, so
	// penalizing under 2.4 parks the fight in a 2.4-3.0 band that still lands
	// hits AND keeps the aim on the neck: at <2 blocks the target's body fills
	// the view and even a neck-locked crosshair pitches down (this is the real
	// cause of the "aim down when near" seen at point-blank), so holding spacing
	// is what keeps the camera level.
	// Reverted to the original mild anti-crowd: the user chose to KEEP the
	// aggressive knockback-juggle combo (and accept the down-aim it causes via
	// the sprint-chase to point-blank). Anti-crowd was proven not to decrowd —
	// the crowding is a chase transient, not a resting spot — so a strong value
	// only disrupts the warm-started fighters without moving the spacing.
	private static final double CROWD_DIST = 1.6;
	private static final float CROWD_COST = 0.03f;
	// Every takeoff costs a little: jumping is only worth it when it converts
	// to a crit (net +0.3), so the policy stops hopping around aimlessly.
	private static final float JUMP_COST = 0.05f;
	// Don't stare at the feet. Two real reasons: (1) aiming at the AABB center
	// (~0.9 blocks, half height) pitches the crosshair steeply DOWN as the
	// opponent closes (at 1.5 blocks ~26 deg down = looking at the ground), which
	// looks like an aimbot; (2) reach is measured ALONG THE VIEW RAY, so aiming
	// down theta costs horizontal reach = 3*cos(theta): at 30 deg down a swing at
	// 2.8 blocks WHIFFS even though the target is "in range" (a^2+b^2=c^2). So
	// level/head aim both looks human AND maximizes effective reach. On-target is
	// full-body (hits still land anywhere), so nothing pulls the aim up during
	// tracking; this penalizes only DOWNWARD deviation below the neck line —
	// level and upward aim are free.
	// Strong AND steep on purpose. First attempt (0.20, ramped over 30 deg) was
	// far too gentle: a realistic 6-10 deg dip below the neck cost ~0.04/tick,
	// trivial vs the 0.3 on-target reward, and the aim net still sat ~10 deg down
	// up close. Now 0.30 (== the full ON_TARGET_REWARD) ramped to full over just
	// LOW_AIM_FULL_DEG below the neck — at close range the chest is already ~6
	// deg below the neck, so the whole lower body fully cancels the on-target
	// pay: holding there is net <=0 while the neck keeps the full 0.3. Aiming at
	// the head (ABOVE the neck) is never charged, so the net rests neck-to-head.
	private static final float LOW_AIM_COST = 0.30f;
	private static final float LOW_AIM_MARGIN = 2.0f;  // degrees of slack below the neck
	private static final double LOW_AIM_FULL_DEG = 6.0; // deg below neck for full penalty
	// Fraction up the hitbox that counts as the neck/upper-chest aim point.
	private static final double NECK_FRACTION = 0.65;
	// For the AIM stage, "on target" (reward + lock) is only the UPPER body from
	// here up (neck-ish to head). The soft aim-down penalty alone kept losing to
	// the deep center-aim habit + the hold-still penalty; making the lower body
	// pay NOTHING removes the plateau entirely — the net cannot lock on the feet,
	// so it must ride the head/neck. Hits still use the full box, so this only
	// moves where the camera rests, not where swings land. At 2 blocks the zone
	// bottom is ~+8 deg (upper chest); at range it is near level.
	private static final double UPPER_TARGET_FRAC = 0.50;

	public static final int INFO_ON_TARGET = 1;
	public static final int INFO_HIT_LANDED = 2;
	public static final int INFO_HIT_TAKEN = 4;
	public static final int INFO_ELEVATED = 8;
	public static final int INFO_WHIFF = 16;
	public static final int INFO_CRIT = 32;
	public static final int INFO_SPRINT_HIT = 64;
	/** Landed hit extended a combo (chain length >= 2 at this hit). */
	public static final int INFO_CHAIN_HIT = 128;

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
	private boolean hitWasCrit;
	private boolean hitWasSprint;
	private boolean jumpHeld;
	private boolean preTickOnGround;
	private int comboChain;
	private int lastHitTick;
	private boolean takenSinceLastHit;
	/** Previous tick's pursuit potential: -(distance beyond reach). */
	private double chasePhiBefore;
	/** This tick, a swing would have landed but the policy did not press. */
	private boolean missedOpening;
	/** Clamped aim command applied this tick (for the smooth-aim cost). */
	private float lastDyaw;
	private float lastDpitch;
	/** Agent's pitch at the start of the tick (for pitch-return shaping). */
	private float prevPitch;
	/** Normalized tick-to-tick change of the aim command, in [0, 1]. */
	private float aimJerk;
	/** A strafe key was held this tick (for the orbit cost). */
	private boolean strafeHeld;
	/** Humanized opponent state: true = this episode uses the perfect bot. */
	private boolean oppPerfectEpisode;
	private int oppReactTicks;
	private boolean oppWasDisrupted;
	/** Opponent swung its sword this tick (telemetry for the 3D replay). */
	private boolean oppSwung;

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
		equipDiamondKit(p);
		return p;
	}

	/** Fresh kit every episode: durability persists on ItemStacks, and after
	 * ~500 hits (≈50k training steps) the armor SHATTERS — kills suddenly
	 * take 2 hits instead of 11, the value function's world model breaks,
	 * and the policy gets destroyed by the advantage shock. Three duel runs
	 * collapsed at the same ~50k mark before this was found. */
	private static void equipDiamondKit(FakePlayer p) {
		p.setItemSlot(EquipmentSlot.HEAD, new ItemStack(Items.DIAMOND_HELMET));
		p.setItemSlot(EquipmentSlot.CHEST, new ItemStack(Items.DIAMOND_CHESTPLATE));
		p.setItemSlot(EquipmentSlot.LEGS, new ItemStack(Items.DIAMOND_LEGGINGS));
		p.setItemSlot(EquipmentSlot.FEET, new ItemStack(Items.DIAMOND_BOOTS));
		p.setItemSlot(EquipmentSlot.MAINHAND, new ItemStack(Items.DIAMOND_SWORD));
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
		equipDiamondKit(agent);
		equipDiamondKit(opponent);
		comboChain = 0;
		lastHitTick = -CHAIN_WINDOW - 1;
		takenSinceLastHit = false;
		// seeded with the REAL spawn distance below (after the opponent is
		// placed) — a zero here made every mortal episode start with a
		// spurious ~-1 pursuit penalty on its first tick
		lastDyaw = 0;
		lastDpitch = 0;
		prevPitch = agent.getXRot();
		aimJerk = 0;

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
		} else if (cfg.vJitter > 0) {
			// Horizontal aim stage: jitter the target's height so the white blob is
			// not always on the same screen row. Pitch is locked, so on-target is
			// graded on YAW alone (isCrosshairAlignedYaw) — a level crosshair still
			// locks. This stops the policy from memorizing "white in the center band
			// = target" and spinning until it appears; it must actually center the
			// target horizontally no matter how high it sits. Floor-clamped so the
			// blob stays visible (a below-floor target gets occluded by the stone).
			oy += (rng.nextDouble() * 2 - 1) * cfg.vJitter;
			oy = Math.max(oy, floorY - 0.4);
		}
		float opponentYaw = (float) Mth.wrapDegrees(Math.toDegrees(angle) + 180);
		place(opponent, ox, oy, oz, opponentYaw, 0);
		chasePhiBefore = -Math.max(0.0, agent.distanceTo(opponent) - 2.8);
		// A hovering opponent is a stationary aim dummy; grounded ones use physics.
		opponent.setNoGravity(!horizontal || "static".equals(cfg.opponent));
		strafeDir = rng.nextBoolean() ? 1 : -1;
		strafeFlipTicks = 20 + rng.nextInt(40);

		opponentHealthBefore = opponent.getHealth();
		prevAngErr = currentAngleError();
		agentHurtTimeBefore = 0;
		bandDistBefore = bandDistance();

		oppPerfectEpisode = !"human".equals(cfg.opponent)
				|| rng.nextDouble() < cfg.oppFightProb;
		oppReactTicks = "theobald".equals(cfg.opponent)
				? theobaldReaction() : sampleReaction();
		oppWasDisrupted = false;
	}

	private int sampleReaction() {
		return cfg.oppReactionMin
				+ rng.nextInt(Math.max(1, cfg.oppReactionMax - cfg.oppReactionMin + 1));
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
		p.setJumping(false);
		p.resetFallDistance();
	}

	public void applyAction(float dyaw, float dpitch, boolean attack, int move,
			int strafe, boolean jump, boolean sprint, boolean reset) {
		if (reset) {
			forceReset = true;
		}
		attackPressed = attack;
		hitLanded = false;
		hitWasCrit = false;
		hitWasSprint = false;
		oppSwung = false;

		agent.setOldPosAndRot();
		opponent.setOldPosAndRot();

		float newDyaw = Mth.clamp(dyaw, -MAX_TURN_PER_TICK, MAX_TURN_PER_TICK);
		float newDpitch = Mth.clamp(dpitch, -MAX_TURN_PER_TICK, MAX_TURN_PER_TICK);
		if (Math.abs(newDyaw) < AIM_DEADZONE) newDyaw = 0;
		if (Math.abs(newDpitch) < AIM_DEADZONE) newDpitch = 0;
		// horizontal-only stage: hold pitch level so the eye-level target can
		// never leave the vertical view (the vertical stage unlocks this).
		if (cfg.lockPitch) newDpitch = 0;
		// pure-vertical stage: hold yaw fixed so the agent can ONLY move up/down
		// (target spawns dead ahead) — it physically cannot spin off-target.
		if (cfg.lockYaw) newDyaw = 0;
		aimJerk = (Math.abs(newDyaw - lastDyaw) + Math.abs(newDpitch - lastDpitch))
				/ (2 * MAX_TURN_PER_TICK);
		lastDyaw = newDyaw;
		lastDpitch = newDpitch;
		agent.setYRot(Mth.wrapDegrees(agent.getYRot() + lastDyaw));
		agent.setYHeadRot(agent.getYRot());
		agent.setXRot(Mth.clamp(agent.getXRot() + lastDpitch, -90, 90));

		// move: 0 none, 1 = W, 2 = S; strafe: 0 none, 1 = A (left), 2 = D (right)
		agent.zza = move == 1 ? 1.0f : (move == 2 ? -1.0f : 0.0f);
		agent.xxa = strafe == 1 ? 1.0f : (strafe == 2 ? -1.0f : 0.0f);
		strafeHeld = strafe != 0;
		// vanilla can only sprint while moving forward; a sprinting attack gets
		// bonus knockback but is barred from critting (Player.attack)
		agent.setSprinting(move == 1 && sprint);
		boolean jumpIn = jump && cfg.allowJump;
		agent.setJumping(jumpIn);
		jumpHeld = jumpIn;
		preTickOnGround = agent.onGround();

		// hesitation check BEFORE the swing consumes the charge: would a
		// swing land right now? (same raycast + gates as the real attack)
		missedOpening = false;
		if (!attack && agent.getAttackStrengthScale(0.5f) > 0.9f) {
			Vec3 hEye = agent.getEyePosition();
			Vec3 hEnd = hEye.add(agent.getViewVector(1.0f).scale(agent.entityInteractionRange()));
			Optional<Vec3> hHit = opponent.getBoundingBox().clip(hEye, hEnd);
			missedOpening = hHit.isPresent() && !blockedByBlocks(hEye, hHit.get())
					&& opponent.invulnerableTime <= 10;
		}

		if (attack) {
			// sample the charge BEFORE attack/swing reset the ticker
			lastAttackCharge = agent.getAttackStrengthScale(0.5f);
			// vanilla crit gate at this instant (Player.attack, 26.1 bytecode):
			// >=90% charge, falling, airborne, dry, unmounted, NOT sprinting
			boolean critCandidate = lastAttackCharge > 0.9f && agent.fallDistance > 0
					&& !agent.onGround() && !agent.onClimbable() && !agent.isInWater()
					&& !agent.isPassenger() && !agent.isSprinting();
			boolean sprintCandidate = lastAttackCharge > 0.9f && agent.isSprinting();
			// ORDER MATTERS (26.1): attack() computes damage as
			// 0.2 + charge^2 * 0.8 at CALL time, and swing() resets the
			// ticker — swinging first made every hit ever landed deal the
			// 0.2x floor (7 -> ~0.3 through diamond armor). Attack first,
			// animate after.
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
					hitWasCrit = critCandidate;
					hitWasSprint = sprintCandidate;
					applyPlayerKnockback(agent, opponent, sprintCandidate);
				}
			}
			agent.swing(InteractionHand.MAIN_HAND);
		}

		if ("strafe".equals(cfg.opponent)) {
			tickStrafeOpponent();
		} else if ("fight".equals(cfg.opponent)
				|| ("human".equals(cfg.opponent) && oppPerfectEpisode)) {
			tickStrafeOpponent();
			tickOpponentAttack();
		} else if ("human".equals(cfg.opponent)) {
			tickHumanOpponent(true);
		} else if ("passive".equals(cfg.opponent)) {
			// combo school: human movement + hitstun/airborne disruption
			// (so knockback actually carries it), but it never swings back
			tickHumanOpponent(false);
		} else if ("theobald".equals(cfg.opponent)) {
			// TheoBaldTheBird-style practice bot on a 0..5 difficulty ladder
			tickTheobaldOpponent(cfg.opponentDifficulty);
		}
		// No manual agent/opponent tick: the server level ticks entities in
		// force-loaded chunks itself — adding one here double-ticks physics
		// and the attack-cooldown ticker.
	}

	/** Opponent turn rate while humanized (degrees/tick); a mouse flick is
	 * fast, so this mostly matters while recovering from hitstun. */
	private static final float OPP_TURN_RATE = 20.0f;

	// TheoBaldTheBird-style difficulty ladder (index 0..5). Reaction windows in
	// ticks (20/s): ~0.1 s at the top, ~0.7 s at the bottom. Higher difficulties
	// additionally add circle-strafing, invuln-window discipline, sprint
	// knockback and low-health retreat (see tickTheobaldOpponent).
	private static final int[] THEO_REACT_MIN = {8, 6, 5, 4, 3, 2};
	private static final int[] THEO_REACT_MAX = {14, 11, 9, 7, 5, 4};

	/**
	 * Humanized opponent: same movement instincts as the scripted bot, but
	 * with the limitations that make real players beatable — a sampled
	 * reaction delay before each attack, no awareness of the agent's
	 * invulnerability window (early swings waste into it), and full
	 * disruption while in hitstun or knocked airborne: no attacks, frozen
	 * aim, no intentional movement, plus a fresh reaction delay on recovery.
	 * This is what makes juggling, spacing and hit-selection pay off.
	 */
	private void tickHumanOpponent(boolean canAttack) {
		boolean disrupted = opponent.hurtTime > 0 || !opponent.onGround();
		if (disrupted) {
			oppWasDisrupted = true;
			opponent.zza = 0;
			opponent.xxa = 0;
			opponent.setSprinting(false);
			return;
		}
		if (oppWasDisrupted) {
			oppWasDisrupted = false;
			oppReactTicks = sampleReaction();
		}

		double dx = agent.getX() - opponent.getX();
		double dz = agent.getZ() - opponent.getZ();
		float yawToAgent = (float) Math.toDegrees(Math.atan2(-dx, dz));
		float dyaw = Mth.wrapDegrees(yawToAgent - opponent.getYRot());
		float yaw = opponent.getYRot() + Mth.clamp(dyaw, -OPP_TURN_RATE, OPP_TURN_RATE);
		opponent.setYRot(yaw);
		opponent.setYHeadRot(yaw);

		// no dodging (user: the opponent should only face the agent and walk
		// forward): straight-line chase, sprinting while far. Strafing
		// opponents can return in a later stage once 20-hit combos hold.
		opponent.xxa = 0;
		double dist = opponent.distanceTo(agent);
		opponent.zza = dist > 3.0 ? 1.0f : (dist < 2.0 ? -0.5f : 0.0f);
		opponent.setSprinting(dist > 4.0 && opponent.onGround());

		if (!canAttack) {
			return;
		}
		if (opponent.getAttackStrengthScale(0.5f) < 0.9f) {
			return;
		}
		Vec3 eye = opponent.getEyePosition();
		Vec3 dir = agent.getEyePosition().subtract(eye).normalize();
		Vec3 end = eye.add(dir.scale(opponent.entityInteractionRange()));
		Optional<Vec3> hit = agent.getBoundingBox().clip(eye, end);
		if (hit.isEmpty() || blockedByBlocks(eye, hit.get())) {
			return;
		}
		// the reaction delay counts down only while the opportunity (charged
		// + in reach) actually exists — a human reacts to the opening, they
		// don't pre-compute it during the cooldown. Ducking out of reach
		// before the delay elapses wastes the opening entirely.
		if (oppReactTicks > 0) {
			oppReactTicks--;
			return;
		}
		// no invulnerableTime check: a human can't see the window, so
		// early swings land into it and do nothing (wasted timing).
		// attack BEFORE swing: swing resets the ticker and floors the damage.
		int hurtBefore = agent.hurtTime;
		opponent.attack(agent);
		opponent.swing(InteractionHand.MAIN_HAND);
		if (agent.hurtTime > hurtBefore) {
			applyPlayerKnockback(opponent, agent, false);
		}
		oppSwung = true;
		oppReactTicks = sampleReaction();
	}

	/** Reaction sample for the Theobald bot at the current difficulty. */
	private int theobaldReaction() {
		int d = Mth.clamp(cfg.opponentDifficulty, 0, 5);
		int lo = THEO_REACT_MIN[d], hi = THEO_REACT_MAX[d];
		return lo + rng.nextInt(Math.max(1, hi - lo + 1));
	}

	/**
	 * TheoBaldTheBird-style practice bot: a 0..5 difficulty ladder built from
	 * the same movement/attack instincts as {@link #tickHumanOpponent}, adding
	 * circle-strafing, invuln-window discipline, sprint knockback and
	 * low-health retreat as the difficulty rises. Faithful in behavior to
	 * Theobald's sword bot; it is NOT the literal HeroBot mod (which targets a
	 * different MC version and so cannot load alongside this one) — that bot is
	 * reachable only through the live pilot-train path against a real client.
	 *
	 * <p>Like the humanized opponent it is fully disrupted while in hitstun or
	 * knocked airborne (so the agent's juggles/combos still pay), and it never
	 * dodges backward except to reset at low health — it faces the agent and
	 * pressures, which is what Theobald's sword bot does.
	 */
	private void tickTheobaldOpponent(int difficulty) {
		int d = Mth.clamp(difficulty, 0, 5);
		boolean strafe = d >= 2;
		float strafeMag = d >= 3 ? 0.9f : (d == 2 ? 0.5f : 0.0f);
		boolean respectInvuln = d >= 4;
		boolean retreatLow = d >= 5;

		boolean disrupted = opponent.hurtTime > 0 || !opponent.onGround();
		if (disrupted) {
			oppWasDisrupted = true;
			opponent.zza = 0;
			opponent.xxa = 0;
			opponent.setSprinting(false);
			return;
		}
		if (oppWasDisrupted) {
			oppWasDisrupted = false;
			oppReactTicks = theobaldReaction();
		}

		// face the agent (fast but not instant — mirrors a mouse flick)
		double dx = agent.getX() - opponent.getX();
		double dz = agent.getZ() - opponent.getZ();
		float yawToAgent = (float) Math.toDegrees(Math.atan2(-dx, dz));
		float dyaw = Mth.wrapDegrees(yawToAgent - opponent.getYRot());
		float yaw = opponent.getYRot() + Mth.clamp(dyaw, -OPP_TURN_RATE, OPP_TURN_RATE);
		opponent.setYRot(yaw);
		opponent.setYHeadRot(yaw);

		double dist = opponent.distanceTo(agent);
		boolean low = retreatLow && opponent.getHealth() < 0.3f * opponent.getMaxHealth();

		// circle-strafe at the higher difficulties, flipping on the shared timer
		if (strafe && !low) {
			if (--strafeFlipTicks <= 0) {
				strafeDir = -strafeDir;
				strafeFlipTicks = 20 + rng.nextInt(40);
			}
			opponent.xxa = strafeMag * strafeDir;
		} else {
			opponent.xxa = 0;
		}

		// approach and hold sword range; back off to reset when low on health
		if (low) {
			opponent.zza = -1.0f;
			opponent.setSprinting(opponent.onGround());
		} else {
			opponent.zza = dist > 3.0 ? 1.0f : (dist < 2.0 ? -0.5f : 0.0f);
			opponent.setSprinting(dist > 4.0 && opponent.onGround());
		}

		if (low) {
			return; // defensive reset: don't press swings while retreating
		}
		if (opponent.getAttackStrengthScale(0.5f) < 0.9f) {
			return;
		}
		Vec3 eye = opponent.getEyePosition();
		Vec3 dir = agent.getEyePosition().subtract(eye).normalize();
		Vec3 end = eye.add(dir.scale(opponent.entityInteractionRange()));
		Optional<Vec3> hit = agent.getBoundingBox().clip(eye, end);
		if (hit.isEmpty() || blockedByBlocks(eye, hit.get())) {
			return;
		}
		// high difficulty is disciplined about the agent's i-frames (like the
		// perfect "fight" bot); low difficulty wastes early swings into them.
		if (respectInvuln && agent.invulnerableTime > 10) {
			return;
		}
		// the reaction delay counts down only while the opening actually exists
		if (oppReactTicks > 0) {
			oppReactTicks--;
			return;
		}
		// attack BEFORE swing: swing resets the ticker and floors the damage.
		int hurtBefore = agent.hurtTime;
		boolean sprintHit = opponent.isSprinting();
		opponent.attack(agent);
		opponent.swing(InteractionHand.MAIN_HAND);
		if (agent.hurtTime > hurtBefore) {
			applyPlayerKnockback(opponent, agent, sprintHit);
		}
		oppSwung = true;
		oppReactTicks = theobaldReaction();
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
			// attack BEFORE swing: swing resets the ticker and would floor
			// the damage at the 0.2x multiplier (26.1 baseDamageScaleFactor)
			int hurtBefore = agent.hurtTime;
			opponent.attack(agent);
			opponent.swing(InteractionHand.MAIN_HAND);
			if (agent.hurtTime > hurtBefore) {
				applyPlayerKnockback(opponent, agent, false);
			}
			oppSwung = true;
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

	/** Vanilla applies player-victim knockback on the victim's CLIENT: the
	 * server computes it, sends a motion packet, and restores the server-side
	 * velocity (the victim's client simulates the shove). Fake players have
	 * no client, so every hit's knockback silently evaporated — measured as
	 * 0.00 blocks of launch over 115 recorded hits. Re-apply it server-side
	 * with vanilla's own numbers: hurt knockback 0.4 away from the attacker
	 * (LivingEntity.hurtServer), plus 0.5 along the attacker's facing for
	 * sprint hits (Player.attack -> causeExtraKnockback, 26.1 bytecode). */
	private static void applyPlayerKnockback(FakePlayer attacker, FakePlayer victim,
			boolean sprint) {
		victim.knockback(0.4,
				attacker.getX() - victim.getX(), attacker.getZ() - victim.getZ());
		if (sprint) {
			float yawRad = attacker.getYRot() * ((float) Math.PI / 180f);
			victim.knockback(0.5, Mth.sin(yawRad), -Mth.cos(yawRad));
		}
	}

	/** Called after the world tick: computes reward, handles episode end + auto-reset. */
	public StepResult step() {
		episodeTick++;

		// soft wall (post-tick): keep both fighters on the platform so nobody
		// rings out. Physics has already run this tick; clamp before any reward
		// or obs reads their positions.
		keepInBounds(agent);
		keepInBounds(opponent);

		if ("swing".equals(cfg.stage)) {
			return stepSwing();
		}
		if ("move".equals(cfg.stage)) {
			return stepMove();
		}
		if ("combo".equals(cfg.stage) || cfg.stage.startsWith("fighter")) {
			return stepCombo(); // fighter shares the combo reward structure
		}

		// aim stage: on-target = UPPER body only, so the camera cannot rest on
		// the feet (hits elsewhere still use the full box). When pitch is LOCKED
		// (horizontal stage) the target's height is jittered, so grade YAW alone —
		// a level crosshair can never satisfy the 2D upper-body test off eye level.
		boolean onTarget = cfg.lockPitch ? isCrosshairAlignedYaw() : isCrosshairOnUpperBody();
		double angErr = currentAngleError();

		float reward = 0;
		if (onTarget) {
			if (paidOnTargetTicks < MAX_PAID_ON_TARGET) {
				reward += ON_TARGET_REWARD;
				paidOnTargetTicks++;
			}
			consecOnTarget++;
			// Don't rest on the feet: while the crosshair is on the body, aiming
			// below the neck bleeds (the aim net is the policy that owns the
			// camera for EVERY downstream stage; aiming down also shortens reach
			// to 3*cos(pitch)). At the neck the cost is ~0 so a settled aim pays
			// full. Only DOWNWARD deviation is charged; head aim is free.
			// Skipped while pitch is locked: pitch is frozen level and the target
			// height is jittered, so this would charge an aim the policy cannot fix.
			if (!cfg.lockPitch) {
				double low = lowAimError();
				reward -= (float) (LOW_AIM_COST * Math.min(1.0, low / LOW_AIM_FULL_DEG));
			}
			// Hold still whenever aimed: on-target is now the UPPER BODY only, so
			// "aimed" already excludes the feet — there is no feet-trap to dodge,
			// and the always-on rule matches the user's doctrine (aimed at the
			// enemy => don't move => moving is punished). A position gate here was
			// worse: it made moving FREE below the neck, so on a moving target the
			// aim drifted DOWN into that haven (near-range pitch regressed +4->+10
			// deg over 40k). The movement penalty applied *during* target lock if crosshair slips
			if (lastDyaw != 0 || lastDpitch != 0) {
				reward -= 0.1f;
			}
		} else {
			// centering shaping only while OFF target — on the blob every
			// pixel is equally good, there is nothing to creep toward
			reward += (float) (SHAPING_SCALE * (prevAngErr - angErr) / 180.0);
			consecOnTarget = 0;
		}
		// jerk cost: rest and steady turns are free, bang-bang alternation
		// pays two-thirds of the on-target reward every tick, a flick once
		reward -= AIM_JERK_COST * aimJerk;
		reward -= TIME_PENALTY;
		prevAngErr = angErr;

		// --- Vertical-aim recovery signals (stage 2+) ---
		// Skipped while pitch is locked: the agent can't control pitch, so
		// penalizing it or rewarding corrections is meaningless.
		if (!cfg.lockPitch) {
			// (1) Extreme pitch penalty
			float absPitch = Math.abs(agent.getXRot());
			if (absPitch > PITCH_COMFORT_DEG) {
				float excess = (absPitch - PITCH_COMFORT_DEG)
						/ (90.0f - PITCH_COMFORT_DEG);
				reward -= EXTREME_PITCH_COST * excess;
			}
			// (2+3) Lost target: off-screen bleed + pitch-return shaping
			if (!isTargetVisible()) {
				reward -= LOST_TARGET_PENALTY;
				float prevAbsPitch = Math.abs(prevPitch);
				float currAbsPitch = Math.abs(agent.getXRot());
				reward += PITCH_RETURN_SHAPING * (prevAbsPitch - currAbsPitch) / 90.0f;
			}
			// (4) Level pitch bonus
			if (absPitch < 30.0f) {
				reward += LEVEL_PITCH_BONUS * (1.0f - absPitch / 30.0f);
			}
		}
		prevPitch = agent.getXRot();

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

	/** Combo stage: frozen aim + swing act; forward/jump/sprint learn to chain
	 * knockback hits and land crits. Chained hits pay an escalating bonus, a
	 * crit pays half again its charge (vanilla deals 1.5x), taking a hit
	 * breaks the chain and is punished. No spacing shaping — positioning is
	 * judged purely by what it enables. */
	private StepResult stepCombo() {
		float reward = 0;
		if (attackPressed) {
			if (hitLanded) {
				if (hitWasSprint) {
					reward += lastAttackCharge;
					reward += SPRINT_HIT_BONUS_SCALE * lastAttackCharge;
					if (hitWasCrit) {
						reward += CRIT_BONUS_SCALE * lastAttackCharge;
					}
					// Only sprint hits count for combo chain
					if (episodeTick - lastHitTick <= CHAIN_WINDOW && !takenSinceLastHit) {
						comboChain++;
					} else {
						comboChain = 1;
					}
					reward += CHAIN_BONUS * Math.min(comboChain - 1, CHAIN_CAP);
				} else {
					// "little few reward" for non-sprint hit
					reward += 0.1f * lastAttackCharge;
					comboChain = 0; // break chain if it's not a sprint hit combo
				}
				lastHitTick = episodeTick;
				takenSinceLastHit = false;
			} else {
				reward -= WHIFF_PENALTY;
			}
			
			if (lastAttackCharge < 0.9f) {
				reward -= SPAM_PENALTY;
			}
		}
		boolean freshHitTaken = agent.hurtTime > agentHurtTimeBefore;
		agentHurtTimeBefore = agent.hurtTime;
		if (freshHitTaken) {
			reward -= HIT_TAKEN_PENALTY;
			if (!takenSinceLastHit && episodeTick - lastHitTick <= CHAIN_WINDOW) {
				reward -= CHAIN_BREAK_PENALTY * Math.min(comboChain - 1, CHAIN_CAP);
			}
			takenSinceLastHit = true;
		}
		// cooldown done + in range = swing NOW; every hesitating tick bleeds
		if (missedOpening) {
			reward -= HESITATION_PENALTY;
		}
		// Strafing while the crosshair already sits on the target is orbiting
		// and costs. Strafing while it is slightly OFF is the fine-adjust
		// tool the mouse no longer has (flicks are >=5°): sidestep with A/D
		// to walk the target back under the crosshair — free, and paid for
		// by the hits it enables.
		if (strafeHeld && isCrosshairOnTarget()) {
			reward -= STRAFE_COST;
		}
		// chest-to-chest is not a sword fight (mild flat penalty, original)
		if (agent.distanceTo(opponent) < CROWD_DIST) {
			reward -= CROWD_COST;
		}
		// (feet-aiming is fixed in the AIM net, not here: the fighter policy
		// only moves and attacks — a frozen aim net owns the camera. The
		// low-aim penalty lives in stepAim where the aim net can respond.)
		// pursue while the combo is live: closing toward reach pays, backing
		// off charges, and a window lapsing without its follow-up hit drops
		// the combo for a depth-scaled penalty (fires once, on the lapse tick).
		// In MORTAL duels the shaping applies at ALL times — the collapsed 7b
		// run proved that without a dense engage gradient, one bad stretch
		// teaches permanent kiting (0 hits, 0 taken, 1200-tick timeouts).
		boolean chainLive = comboChain >= 1 && !takenSinceLastHit
				&& episodeTick - lastHitTick <= CHAIN_WINDOW;
		
		float currentCharge = agent.getAttackStrengthScale(1.0f);
		// If cooldown is ready, we want to engage (2.8 reach). If not, we keep distance to "stop" and wait.
		double idealDist = (currentCharge > 0.95f) ? 2.8 : 3.5;
		// use Math.abs to penalize both being too far and being too close, enforcing the spacing explicitly
		double chasePhi = -Math.abs(agent.distanceTo(opponent) - idealDist);
		
		reward += (float) (PURSUIT_SHAPING * (chasePhi - chasePhiBefore));
		chasePhiBefore = chasePhi;
		if (comboChain >= 1 && !takenSinceLastHit
				&& episodeTick - lastHitTick == CHAIN_WINDOW + 1) {
			reward -= CHAIN_DROP_PENALTY * Math.min(comboChain, CHAIN_CAP);
		}
		// took off this tick (was grounded, jump held, now airborne)
		if (jumpHeld && preTickOnGround && !agent.onGround()) {
			reward -= JUMP_COST;
		}

		int info = 0;
		if (isCrosshairOnTarget()) info |= INFO_ON_TARGET;
		if (hitLanded) info |= INFO_HIT_LANDED;
		else if (attackPressed) info |= INFO_WHIFF;
		if (hitWasCrit) info |= INFO_CRIT;
		if (hitWasSprint) info |= INFO_SPRINT_HIT;
		if (hitLanded && comboChain >= 2) info |= INFO_CHAIN_HIT;
		if (elevatedEpisode) info |= INFO_ELEVATED;
		if (freshHitTaken) info |= INFO_HIT_TAKEN;

		boolean opponentDead = false;
		boolean agentDead = false;
		if (cfg.mortal) {
			reward -= MORTAL_TICK_COST;
			reward -= (float) (FAR_COST
					* Math.min(4.0, Math.max(0.0, agent.distanceTo(opponent) - 3.0)));
			// real fight: health persists (vanilla food regen only), the
			// duel ends when someone drops
			opponentDead = opponent.getHealth() <= 0;
			agentDead = agent.getHealth() <= 0;
			if (opponentDead) {
				reward += KILL_REWARD;
			}
			if (agentDead) {
				reward -= DEATH_PENALTY;
			}
		} else {
			opponent.setHealth(opponent.getMaxHealth());
			agent.setHealth(agent.getMaxHealth());
		}
		opponent.getFoodData().setFoodLevel(20);
		agent.getFoodData().setFoodLevel(20);

		boolean fellOff = opponent.getY() < floorY - 2 || agent.getY() < floorY - 2;
		boolean done = episodeTick >= cfg.episodeTicks || fellOff || forceReset
				|| opponentDead || agentDead;
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

	/** Crosshair on the UPPER body only (see UPPER_TARGET_FRAC). This is the aim
	 * stage's "on target": rewarding/locking only the head-neck band forces the
	 * camera to rest there instead of the feet. Hit detection still uses the full
	 * box, so nothing about where swings land changes. */
	private boolean isCrosshairOnUpperBody() {
		Vec3 eye = agent.getEyePosition();
		Vec3 view = agent.getViewVector(1.0f);
		AABB box = opponent.getBoundingBox();
		double upMinY = box.minY + UPPER_TARGET_FRAC * (box.maxY - box.minY);
		return ObsProjector.rayHitsBox(view.x, view.y, view.z,
				box.minX - eye.x, upMinY - eye.y, box.minZ - eye.z,
				box.maxX - eye.x, box.maxY - eye.y, box.maxZ - eye.z);
	}

	/** Crosshair aligned in YAW with the target, ignoring pitch/height. This is
	 * the horizontal aim stage's "on target": pitch is locked level and the
	 * target's height is jittered, so the vertical view plane (the crosshair's
	 * yaw) passing through the target's horizontal extent is the whole task —
	 * a level crosshair can never satisfy the full 2D upper-body test off eye
	 * level. On target when the yaw error is within the target's angular
	 * half-width (player box is 0.6 wide), matching the horizontal tightness of
	 * the full box test. Hit detection is unchanged. */
	private boolean isCrosshairAlignedYaw() {
		double dx = opponent.getX() - agent.getX();
		double dz = opponent.getZ() - agent.getZ();
		double horiz = Math.sqrt(dx * dx + dz * dz);
		double yawToTarget = Math.toDegrees(Math.atan2(-dx, dz));
		double yawErr = Math.abs(Mth.wrapDegrees(yawToTarget - agent.getYRot()));
		double halfWidth = Math.toDegrees(Math.atan2(0.3, Math.max(0.5, horiz)));
		return yawErr <= halfWidth;
	}

	private double currentAngleError() {
		Vec3 eye = agent.getEyePosition();
		AABB box = opponent.getBoundingBox();
		Vec3 c = box.getCenter();
		// aim at the neck/upper chest, NOT the AABB center: the center sits at
		// half height (~0.9 blocks) so as the opponent closes the crosshair
		// pitches down into the feet. The neck is near eye level on flat ground,
		// keeping the acquisition sweep natural and level.
		double neckY = box.minY + NECK_FRACTION * (box.maxY - box.minY);
		return ObsProjector.angleToTarget(agent.getYRot(), agent.getXRot(),
				eye.x, eye.y, eye.z, c.x, neckY, c.z);
	}

	/** Degrees the crosshair points BELOW the neck line (0 if level or above).
	 * Positive MC pitch = looking down; the neck-pitch is how far down the eye
	 * must tilt to sit on the neck, so anything beyond that (plus a margin) is
	 * staring at the lower body/feet. */
	private double lowAimError() {
		Vec3 eye = agent.getEyePosition();
		AABB box = opponent.getBoundingBox();
		Vec3 c = box.getCenter();
		double neckY = box.minY + NECK_FRACTION * (box.maxY - box.minY);
		double dx = c.x - eye.x, dz = c.z - eye.z;
		double horiz = Math.sqrt(dx * dx + dz * dz);
		double neckPitch = Math.toDegrees(Math.atan2(eye.y - neckY, horiz));
		return Math.max(0.0, agent.getXRot() - neckPitch - LOW_AIM_MARGIN);
	}

	/** Soft wall: keep a fighter on the platform (see IN_BOUNDS_RADIUS). Clamps
	 * horizontal position just inside the edge and kills the outward velocity so
	 * knockback stops pushing instead of ringing the target out. */
	private void keepInBounds(FakePlayer p) {
		double cx = centerX + 0.5, cz = centerZ + 0.5;
		double x = p.getX(), z = p.getZ();
		Vec3 v = p.getDeltaMovement();
		double nx = x, nz = z, vx = v.x, vz = v.z;
		boolean clamped = false;
		if (x - cx > IN_BOUNDS_RADIUS) { nx = cx + IN_BOUNDS_RADIUS; vx = Math.min(vx, 0); clamped = true; }
		else if (cx - x > IN_BOUNDS_RADIUS) { nx = cx - IN_BOUNDS_RADIUS; vx = Math.max(vx, 0); clamped = true; }
		if (z - cz > IN_BOUNDS_RADIUS) { nz = cz + IN_BOUNDS_RADIUS; vz = Math.min(vz, 0); clamped = true; }
		else if (cz - z > IN_BOUNDS_RADIUS) { nz = cz - IN_BOUNDS_RADIUS; vz = Math.max(vz, 0); clamped = true; }
		if (clamped) {
			p.setPos(nx, p.getY(), nz);
			p.setDeltaMovement(vx, v.y, vz);
		}
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
		
		double odx = opponent.getX() - opponent.xOld;
		double ody = opponent.getY() - opponent.yOld;
		double odz = opponent.getZ() - opponent.zOld;
		double dy = agent.getY() - agent.yOld;
		
		buf.putFloat((float) (odx - dx));
		buf.putFloat((float) (ody - dy));
		buf.putFloat((float) (odz - dz));
	}

	/** Debug/replay telemetry (PROTOCOL.md obs frame, 12 f32): both fighters'
	 * poses and swing/hurt states. Never fed to the policy — only recorded
	 * for the dashboard's top-down and reconstructed-3D replay views, so the
	 * human-fair observation contract is untouched. First 7 floats predate
	 * the 3D view; keep their order. */
	public void writeTelemetry(ByteBuffer buf) {
		buf.putFloat((float) (agent.getX() - centerX));
		buf.putFloat((float) (agent.getZ() - centerZ));
		buf.putFloat((float) (agent.getY() - floorY));
		buf.putFloat(agent.getYRot());
		buf.putFloat((float) (opponent.getX() - centerX));
		buf.putFloat((float) (opponent.getZ() - centerZ));
		buf.putFloat((float) (opponent.getY() - floorY));
		buf.putFloat(agent.getXRot());
		buf.putFloat(opponent.getYRot());
		buf.putFloat(attackPressed ? 1 : 0);
		buf.putFloat(oppSwung ? 1 : 0);
		buf.putFloat(opponent.hurtTime / 10.0f);
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
