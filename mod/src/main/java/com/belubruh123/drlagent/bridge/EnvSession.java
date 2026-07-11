package com.belubruh123.drlagent.bridge;

import com.belubruh123.drlagent.DrlAgentMod;
import com.belubruh123.drlagent.env.Arena;
import com.belubruh123.drlagent.env.ArenaManager;
import com.belubruh123.drlagent.env.EnvConfig;

import com.google.gson.JsonObject;

import net.minecraft.server.MinecraftServer;

import java.io.IOException;
import java.nio.ByteBuffer;
import java.util.List;

/**
 * Drives the lock-step exchange from the server tick thread: at tick start,
 * block for the trainer's action batch and apply it (fake players tick
 * there); at tick end, send observations + rewards. While a trainer is
 * attached the tick rate is raised so training runs as fast as the CPU and
 * round-trip allow.
 */
public final class EnvSession {
	private static final float TRAINING_TICK_RATE = 10000.0f;
	/** v3: dyaw f32, dpitch f32, attack u8, move u8, strafe u8, jump u8,
	 * sprint u8, flags u8. */
	private static final int ACTION_BYTES = 4 + 4 + 1 + 1 + 1 + 1 + 1 + 1;
	/** Telemetry floats appended per arena to each obs frame (PROTOCOL.md). */
	private static final int TELEMETRY_FLOATS = 12;

	private final BridgeServer bridge;
	private final ArenaManager arenas = new ArenaManager();

	private BridgeServer.Connection current;
	private JsonObject appliedConfig;
	private boolean primed;
	private boolean actionsAppliedThisTick;
	private int tickCounter;
	private final Arena.StepResult[] results;

	public EnvSession(BridgeServer bridge) {
		this.bridge = bridge;
		this.results = new Arena.StepResult[BridgeConfig.arenas()];
	}

	public void onTickStart(MinecraftServer server) {
		BridgeServer.Connection conn = bridge.getConnection();
		if (conn == null) {
			if (current != null) {
				detach(server);
			}
			return;
		}
		if (conn != current) {
			current = conn;
			primed = false;
		}

		actionsAppliedThisTick = false;
		try {
			if (!primed) {
				prime(server);
				return; // initial obs goes out at tick end with zero rewards
			}
			maybeApplyNewConfig();
			byte[] frame = current.readActionFrame();
			decodeAndApply(frame);
			actionsAppliedThisTick = true;
		} catch (IOException e) {
			DrlAgentMod.LOGGER.warn("Trainer connection lost: {}", e.toString());
			bridge.dropConnection();
			detach(server);
		}
	}

	public void onTickEnd(MinecraftServer server) {
		if (current == null || !primed) {
			return;
		}
		if (actionsAppliedThisTick) {
			stepArenas();
		}
		List<Arena> list = arenas.getArenas();
		ByteBuffer buf = ByteBuffer.allocate(obsFrameSize(list.size()));
		buf.putInt(tickCounter++);
		for (int i = 0; i < list.size(); i++) {
			Arena.StepResult r = results[i];
			list.get(i).writeObs(buf);
			buf.putFloat(r == null ? 0 : r.reward());
			buf.put((byte) (r != null && r.done() ? 1 : 0));
			buf.put((byte) (r == null ? 0 : r.info()));
			list.get(i).writeTelemetry(buf);
			results[i] = null;
		}
		try {
			current.sendObsFrame(buf.array());
		} catch (IOException e) {
			DrlAgentMod.LOGGER.warn("Trainer connection lost on send: {}", e.toString());
			bridge.dropConnection();
			detach(server);
		}
	}

	private void prime(MinecraftServer server) {
		arenas.initIfNeeded(server, BridgeConfig.arenas());
		appliedConfig = bridge.getLatestConfig();
		arenas.resetAll(EnvConfig.from(appliedConfig));
		server.tickRateManager().setTickRate(TRAINING_TICK_RATE);
		tickCounter = 0;
		primed = true;
		DrlAgentMod.LOGGER.info("Training session primed: {} arenas, tick rate {}",
				BridgeConfig.arenas(), TRAINING_TICK_RATE);
	}

	private void detach(MinecraftServer server) {
		current = null;
		primed = false;
		server.tickRateManager().setTickRate(20.0f);
		DrlAgentMod.LOGGER.info("Training session detached; tick rate restored to 20");
	}

	private void maybeApplyNewConfig() {
		JsonObject latest = bridge.getLatestConfig();
		if (latest != appliedConfig) {
			appliedConfig = latest;
			if (latest.has("reseed") && latest.get("reseed").getAsBoolean()) {
				// reproducible eval: re-seed the rng and restart every episode
				arenas.resetAll(EnvConfig.from(latest));
				DrlAgentMod.LOGGER.info("Applied new curriculum config with reseed");
			} else {
				arenas.applyConfig(EnvConfig.from(latest));
				DrlAgentMod.LOGGER.info("Applied new curriculum config");
			}
		}
	}

	private void decodeAndApply(byte[] frame) throws IOException {
		List<Arena> list = arenas.getArenas();
		if (frame.length != list.size() * ACTION_BYTES) {
			throw new IOException("bad action frame size " + frame.length);
		}
		ByteBuffer buf = ByteBuffer.wrap(frame);
		for (int i = 0; i < list.size(); i++) {
			float dyaw = buf.getFloat();
			float dpitch = buf.getFloat();
			boolean attack = buf.get() != 0;
			int move = buf.get() & 0xFF;
			int strafe = buf.get() & 0xFF;
			boolean jump = buf.get() != 0;
			boolean sprint = buf.get() != 0;
			boolean reset = (buf.get() & 1) != 0;
			list.get(i).applyAction(dyaw, dpitch, attack, move, strafe, jump, sprint, reset);
		}
	}

	/** Rewards/dones are produced by stepping arenas after the world tick ran. */
	private void stepArenas() {
		List<Arena> list = arenas.getArenas();
		for (int i = 0; i < list.size(); i++) {
			results[i] = list.get(i).step();
		}
	}

	private static int obsFrameSize(int arenaCount) {
		int maskBytes = (BridgeConfig.OBS_WIDTH * BridgeConfig.OBS_HEIGHT + 7) / 8;
		return 4 + arenaCount * (maskBytes + 4 * BridgeConfig.SCALARS.length + 4 + 1 + 1
				+ 4 * TELEMETRY_FLOATS);
	}

	public boolean isPrimed() {
		return primed;
	}
}
