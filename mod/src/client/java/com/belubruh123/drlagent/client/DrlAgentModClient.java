package com.belubruh123.drlagent.client;

import com.belubruh123.drlagent.client.pilot.PilotController;
import com.mojang.blaze3d.platform.InputConstants;
import net.fabricmc.api.ClientModInitializer;
import net.fabricmc.fabric.api.client.event.lifecycle.v1.ClientTickEvents;
import net.fabricmc.fabric.api.client.keymapping.v1.KeyMappingHelper;
import net.minecraft.client.KeyMapping;

public class DrlAgentModClient implements ClientModInitializer {
	public static final PilotController PILOT = new PilotController();

	@Override
	public void onInitializeClient() {
		PilotCommand.register();
		KeyMapping toggle = KeyMappingHelper.registerKeyMapping(new KeyMapping(
				"key.drlagent.pilot_toggle", InputConstants.Type.KEYSYM,
				InputConstants.KEY_G, KeyMapping.Category.MISC));
		ClientTickEvents.START_CLIENT_TICK.register(mc -> {
			while (toggle.consumeClick()) {
				PILOT.toggle(mc);
			}
			PILOT.startTick(mc);
		});
		ClientTickEvents.END_CLIENT_TICK.register(PILOT::endTick);
	}
}
