package com.belubruh123.drlagent.env;

import com.google.gson.JsonObject;

/** Curriculum parameters sent by the trainer; applied at each episode reset. */
public final class EnvConfig {
	public String stage = "aim";
	public long seed = 0;
	public int recordArena = 0;

	/** Probability an episode spawns the opponent at agent eye level (rehearsal mixing). */
	public double horizontalProb = 1.0;
	/** Max vertical offset (blocks) for non-horizontal spawns; opponent hovers. */
	public double maxElev = 0.0;
	public double distMin = 3.0;
	public double distMax = 8.0;
	/** Spawn direction spread (degrees) relative to the agent's random facing. */
	public double yawRange = 180.0;
	public int episodeTicks = 200;
	/** Consecutive on-target ticks required to count as an aim lock. */
	public int lockTicks = 3;
	public String opponent = "static";
	/** TheoBaldTheBird-style opponent ("theobald"): difficulty 0 (easiest) .. 5
	 * (hardest). Governs reaction speed, strafing, invuln-window discipline,
	 * sprint knockback and low-health retreat (see Arena.tickTheobaldOpponent). */
	public int opponentDifficulty = 3;
	/** Humanized opponent: reaction delay range in ticks (~150-350 ms). */
	public int oppReactionMin = 3;
	public int oppReactionMax = 7;
	/** With opponent "human": probability an episode uses the perfect
	 * "fight" bot instead (rehearsal so the policy keeps a worst-case answer). */
	public double oppFightProb = 0.0;
	/** 0 makes the agent's jump input a no-op (no-jump curriculum: grounded
	 * combos only; no crits, which require falling). */
	public boolean allowJump = true;
	/** 1 = real fight: no per-tick healing (vanilla food regen only), the
	 * episode ends when someone dies, a kill pays and dying is punished. */
	public boolean mortal = false;
	/** 1 = horizontal-only aim stage: hold the agent's pitch level (ignore the
	 * dpitch command) so a from-scratch policy can never look up/down and lose
	 * the eye-level target off the top of the view — where no yaw spin recovers
	 * it and the mask goes blank. The vertical stage unlocks it (lock_pitch 0).
	 * While locked, "on target" is judged on YAW alignment only (the target's
	 * height is jittered by vJitter, so a level crosshair could never satisfy the
	 * full 2D test — and grading a pitch the agent cannot move is meaningless). */
	public boolean lockPitch = false;
	/** Horizontal aim stage: max vertical offset (blocks) applied to the target on
	 * EVERY episode (even eye-level ones), so the white blob does not always land
	 * on the same screen row. Forces the policy to center the target horizontally
	 * whatever its height instead of memorizing "white in the center band = lock".
	 * Kept small + floor-clamped so the blob stays inside the vertical FOV at
	 * dist_min. 0 = no jitter (target dead level). Distinct from maxElev, which
	 * marks large ELEVATED episodes for the pitch-unlocked vertical stage. */
	public double vJitter = 0.0;
	/** 1 = pure-vertical aim stage: hold the agent's yaw fixed (ignore the dyaw
	 * command) so it can ONLY move up/down. The opponent spawns dead ahead
	 * (yaw_range 0) at varying elevation, so the agent physically cannot spin
	 * off looking for it — it just learns pitch. Mirror of lockPitch. */
	public boolean lockYaw = false;

	public static EnvConfig from(JsonObject config) {
		EnvConfig c = new EnvConfig();
		if (config == null) {
			return c;
		}
		if (config.has("stage")) c.stage = config.get("stage").getAsString();
		if (config.has("seed")) c.seed = config.get("seed").getAsLong();
		if (config.has("record_arena")) c.recordArena = config.get("record_arena").getAsInt();
		JsonObject cur = config.has("curriculum") ? config.getAsJsonObject("curriculum") : null;
		if (cur != null) {
			if (cur.has("horizontal_prob")) c.horizontalProb = cur.get("horizontal_prob").getAsDouble();
			if (cur.has("max_elev")) c.maxElev = cur.get("max_elev").getAsDouble();
			if (cur.has("dist_min")) c.distMin = cur.get("dist_min").getAsDouble();
			if (cur.has("dist_max")) c.distMax = cur.get("dist_max").getAsDouble();
			if (cur.has("yaw_range")) c.yawRange = cur.get("yaw_range").getAsDouble();
			if (cur.has("episode_ticks")) c.episodeTicks = cur.get("episode_ticks").getAsInt();
			if (cur.has("lock_ticks")) c.lockTicks = cur.get("lock_ticks").getAsInt();
			if (cur.has("opponent")) c.opponent = cur.get("opponent").getAsString();
			if (cur.has("opp_difficulty")) c.opponentDifficulty = cur.get("opp_difficulty").getAsInt();
			if (cur.has("opp_reaction_min")) c.oppReactionMin = cur.get("opp_reaction_min").getAsInt();
			if (cur.has("opp_reaction_max")) c.oppReactionMax = cur.get("opp_reaction_max").getAsInt();
			if (cur.has("opp_fight_prob")) c.oppFightProb = cur.get("opp_fight_prob").getAsDouble();
			if (cur.has("allow_jump")) c.allowJump = cur.get("allow_jump").getAsInt() != 0;
			if (cur.has("mortal")) c.mortal = cur.get("mortal").getAsInt() != 0;
			if (cur.has("lock_pitch")) c.lockPitch = cur.get("lock_pitch").getAsInt() != 0;
			if (cur.has("lock_yaw")) c.lockYaw = cur.get("lock_yaw").getAsInt() != 0;
			if (cur.has("v_jitter")) c.vJitter = cur.get("v_jitter").getAsDouble();
		}
		return c;
	}
}
