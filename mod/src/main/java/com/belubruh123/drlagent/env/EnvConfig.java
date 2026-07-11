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
	/** Humanized opponent: reaction delay range in ticks (~150-350 ms). */
	public int oppReactionMin = 3;
	public int oppReactionMax = 7;
	/** With opponent "human": probability an episode uses the perfect
	 * "fight" bot instead (rehearsal so the policy keeps a worst-case answer). */
	public double oppFightProb = 0.0;

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
			if (cur.has("opp_reaction_min")) c.oppReactionMin = cur.get("opp_reaction_min").getAsInt();
			if (cur.has("opp_reaction_max")) c.oppReactionMax = cur.get("opp_reaction_max").getAsInt();
			if (cur.has("opp_fight_prob")) c.oppFightProb = cur.get("opp_fight_prob").getAsDouble();
		}
		return c;
	}
}
