package com.belubruh123.drlagent.env;

import com.belubruh123.drlagent.DrlAgentMod;

import net.minecraft.server.MinecraftServer;
import net.minecraft.server.level.ServerLevel;

import java.util.ArrayList;
import java.util.List;
import java.util.Random;

/** Builds and owns the N parallel arenas in the overworld. */
public final class ArenaManager {
	private final List<Arena> arenas = new ArrayList<>();
	private final Random rng = new Random();
	private boolean initialized;

	public void initIfNeeded(MinecraftServer server, int count) {
		if (initialized) {
			return;
		}
		ServerLevel level = server.overworld();
		String[] rules = {
				"gamerule spawn_mobs false",
				"gamerule advance_time false",
				"gamerule advance_weather false",
				"gamerule immediate_respawn true",
		};
		for (String rule : rules) {
			server.getCommands().performPrefixedCommand(server.createCommandSourceStack(), rule);
		}
		long t0 = System.currentTimeMillis();
		for (int i = 0; i < count; i++) {
			Arena arena = new Arena(i, level, rng);
			arena.build();
			arena.spawnPlayers();
			arenas.add(arena);
		}
		initialized = true;
		DrlAgentMod.LOGGER.info("Built {} arenas in {} ms", count, System.currentTimeMillis() - t0);
	}

	public void resetAll(EnvConfig cfg) {
		rng.setSeed(cfg.seed);
		for (Arena arena : arenas) {
			arena.setConfig(cfg);
			arena.reset();
		}
	}

	public void applyConfig(EnvConfig cfg) {
		for (Arena arena : arenas) {
			arena.setConfig(cfg);
		}
	}

	public List<Arena> getArenas() {
		return arenas;
	}
}
