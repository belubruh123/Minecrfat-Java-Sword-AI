package com.belubruh123.drlagent.client;

import com.belubruh123.drlagent.client.pilot.PilotController;
import com.mojang.brigadier.arguments.IntegerArgumentType;
import net.fabricmc.fabric.api.client.command.v2.ClientCommandRegistrationCallback;
import net.fabricmc.fabric.api.client.command.v2.ClientCommands;
import net.fabricmc.fabric.api.client.command.v2.FabricClientCommandSource;
import net.minecraft.network.chat.Component;

/**
 * Command-line control of pilot mode, equivalent to (and alongside) the
 * toggle key:
 *
 * <pre>
 *   /pilot            — status
 *   /pilot on         — connect to trainer/pilot.py and hand over control
 *   /pilot off        — take back control
 *   /pilot status     — engaged?, port, current target
 *   /pilot port 12345 — pick the inference-server port (default 36566)
 * </pre>
 */
final class PilotCommand {
	private PilotCommand() {
	}

	static void register() {
		ClientCommandRegistrationCallback.EVENT.register((dispatcher, buildContext) -> dispatcher.register(
				ClientCommands.literal("pilot")
						.executes(ctx -> status(ctx.getSource()))
						.then(ClientCommands.literal("on").executes(ctx -> on(ctx.getSource())))
						.then(ClientCommands.literal("off").executes(ctx -> off(ctx.getSource())))
						.then(ClientCommands.literal("status").executes(ctx -> status(ctx.getSource())))
						.then(ClientCommands.literal("port")
								.then(ClientCommands.argument("port", IntegerArgumentType.integer(1, 65535))
										.executes(ctx -> port(ctx.getSource(),
												IntegerArgumentType.getInteger(ctx, "port")))))));
	}

	private static int on(FabricClientCommandSource src) {
		PilotController pilot = DrlAgentModClient.PILOT;
		if (pilot.isEngaged()) {
			src.sendFeedback(Component.literal("[pilot] already engaged — /pilot off to release"));
			return 0;
		}
		pilot.engage(src.getClient()); // reports success/failure in chat itself
		return 1;
	}

	private static int off(FabricClientCommandSource src) {
		PilotController pilot = DrlAgentModClient.PILOT;
		if (!pilot.isEngaged()) {
			src.sendFeedback(Component.literal("[pilot] not engaged"));
			return 0;
		}
		pilot.disengage(src.getClient(), "released");
		return 1;
	}

	private static int status(FabricClientCommandSource src) {
		PilotController pilot = DrlAgentModClient.PILOT;
		String target = pilot.targetName();
		src.sendFeedback(Component.literal("[pilot] " + (pilot.isEngaged()
				? "ENGAGED — target: " + (target != null ? target : "searching…")
				: "off") + " | server 127.0.0.1:" + pilot.getPort()));
		if (pilot.isEngaged()) {
			PilotController.Stats s = pilot.stats();
			src.sendFeedback(Component.literal("[pilot] hits " + s.hits()
					+ " (sprint " + s.sprintHits() + ", crit " + s.crits()
					+ ") | combo x" + s.chain() + " best x" + s.bestChain()
					+ " combos " + s.combos() + " | taken " + s.taken()));
		}
		return 1;
	}

	private static int port(FabricClientCommandSource src, int port) {
		PilotController pilot = DrlAgentModClient.PILOT;
		pilot.setPort(port);
		if (pilot.isEngaged()) {
			pilot.disengage(src.getClient(), "reconnecting on :" + port);
			pilot.engage(src.getClient());
		} else {
			src.sendFeedback(Component.literal("[pilot] port set to " + port
					+ " — takes effect on the next /pilot on"));
		}
		return 1;
	}
}
