package com.belubruh123.drlagent;

import com.belubruh123.drlagent.bridge.BridgeConfig;
import com.belubruh123.drlagent.bridge.BridgeServer;

import net.fabricmc.api.ModInitializer;
import net.fabricmc.fabric.api.event.lifecycle.v1.ServerLifecycleEvents;

import net.minecraft.resources.Identifier;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.io.IOException;

public class DrlAgentMod implements ModInitializer {
	public static final String MOD_ID = "drlagent";

	public static final Logger LOGGER = LoggerFactory.getLogger(MOD_ID);

	private static BridgeServer bridge;

	@Override
	public void onInitialize() {
		if (!BridgeConfig.trainingEnabled()) {
			LOGGER.info("DRL Agent mod loaded (training disabled; set DRLAGENT_TRAIN=1 to enable)");
			return;
		}

		ServerLifecycleEvents.SERVER_STARTED.register(server -> {
			try {
				bridge = new BridgeServer(BridgeConfig.port());
				bridge.start();
			} catch (IOException e) {
				throw new RuntimeException("Failed to start DRL training bridge", e);
			}
		});

		ServerLifecycleEvents.SERVER_STOPPING.register(server -> {
			if (bridge != null) {
				bridge.stop();
				bridge = null;
			}
		});
	}

	public static BridgeServer getBridge() {
		return bridge;
	}

	public static Identifier id(String path) {
		return Identifier.fromNamespaceAndPath(MOD_ID, path);
	}
}
