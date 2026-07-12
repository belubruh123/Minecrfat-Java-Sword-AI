package com.belubruh123.drlagent.client;

import com.mojang.brigadier.arguments.StringArgumentType;
import net.fabricmc.fabric.api.client.command.v2.ClientCommandRegistrationCallback;
import net.fabricmc.fabric.api.client.command.v2.ClientCommands;
import net.fabricmc.fabric.api.client.command.v2.FabricClientCommandSource;
import net.minecraft.network.chat.Component;

/**
 * Replay theater control:
 *
 * <pre>
 *   /replay follow [run]   — keep re-enacting the newest recorded episode
 *                            (default run: stage6_fighter)
 *   /replay play [run]     — play the newest episode once
 *   /replay stop           — stop and release the camera
 *   /replay api <url>      — dashboard API base (default http://127.0.0.1:8080)
 * </pre>
 */
final class ReplayCommand {
	private static final String DEFAULT_RUN = "stage6_fighter";

	private ReplayCommand() {
	}

	static void register() {
		ClientCommandRegistrationCallback.EVENT.register((dispatcher, ctx) -> dispatcher.register(
				ClientCommands.literal("replay")
						.then(ClientCommands.literal("follow")
								.executes(c -> start(c.getSource(), DEFAULT_RUN, true))
								.then(ClientCommands.argument("run", StringArgumentType.word())
										.executes(c -> start(c.getSource(),
												StringArgumentType.getString(c, "run"), true))))
						.then(ClientCommands.literal("play")
								.executes(c -> start(c.getSource(), DEFAULT_RUN, false))
								.then(ClientCommands.argument("run", StringArgumentType.word())
										.executes(c -> start(c.getSource(),
												StringArgumentType.getString(c, "run"), false))))
						.then(ClientCommands.literal("stop")
								.executes(c -> {
									DrlAgentModClient.REPLAY.stop(c.getSource().getClient(), "command");
									return 1;
								}))
						.then(ClientCommands.literal("api")
								.then(ClientCommands.argument("url", StringArgumentType.greedyString())
										.executes(c -> {
											DrlAgentModClient.REPLAY.setApi(
													StringArgumentType.getString(c, "url"));
											c.getSource().sendFeedback(Component.literal(
													"[replay] api = " + DrlAgentModClient.REPLAY.getApi()));
											return 1;
										})))));
	}

	private static int start(FabricClientCommandSource src, String run, boolean follow) {
		if (DrlAgentModClient.PILOT.isEngaged()) {
			src.sendFeedback(Component.literal("[replay] disengage the pilot first (/pilot off)"));
			return 0;
		}
		DrlAgentModClient.REPLAY.start(src.getClient(), run, follow);
		return 1;
	}
}
