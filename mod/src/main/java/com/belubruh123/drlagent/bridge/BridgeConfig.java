package com.belubruh123.drlagent.bridge;

/**
 * Training-bridge settings. All values come from env vars (usable with
 * `DRLAGENT_TRAIN=1 ./gradlew runServer`) with system-property overrides.
 * When {@link #trainingEnabled()} is false the mod does nothing, so it is
 * safe to have installed during normal play.
 */
public final class BridgeConfig {
	public static final int PROTOCOL_VERSION = 2;
	public static final int OBS_WIDTH = 426;
	public static final int OBS_HEIGHT = 240;
	public static final String[] SCALARS = {
			"speed", "yaw_sin", "yaw_cos", "pitch", "y", "on_ground",
			"cooldown", "health", "last_reach"
	};

	private BridgeConfig() {
	}

	private static String get(String name, String def) {
		String prop = System.getProperty("drlagent." + name);
		if (prop != null) return prop;
		String env = System.getenv("DRLAGENT_" + name.toUpperCase());
		return env != null ? env : def;
	}

	public static boolean trainingEnabled() {
		String v = get("train", "0");
		return v.equals("1") || v.equalsIgnoreCase("true");
	}

	public static int port() {
		return Integer.parseInt(get("port", "36565"));
	}

	public static int arenas() {
		return Integer.parseInt(get("arenas", "8"));
	}
}
