package com.belubruh123.drlagent.client;

import com.belubruh123.drlagent.client.pilot.PilotController;

import net.fabricmc.fabric.api.client.rendering.v1.hud.HudElement;
import net.fabricmc.fabric.api.client.rendering.v1.hud.HudElementRegistry;
import net.minecraft.client.DeltaTracker;
import net.minecraft.client.Minecraft;
import net.minecraft.client.gui.Font;
import net.minecraft.client.gui.GuiGraphicsExtractor;
import net.minecraft.resources.Identifier;

/**
 * Top-left fight panel, visible while the pilot is engaged: confirmed hits
 * (split into sprint hits and crits), the live combo chain with best/total,
 * and hits taken. Counters reset each time the pilot engages.
 */
public final class PilotHud implements HudElement {
	private static final int PAD = 4;
	private static final int LINE_H = 10;
	private static final int BG = 0xA0101018;
	private static final int TITLE = 0xFFFFD54A;
	private static final int TEXT = 0xFFFFFFFF;
	private static final int GOOD = 0xFF7CE87C;
	private static final int BAD = 0xFFFF7A7A;

	static void register() {
		HudElementRegistry.addLast(
				Identifier.fromNamespaceAndPath("drlagent", "pilot_stats"), new PilotHud());
	}

	@Override
	public void extractRenderState(GuiGraphicsExtractor g, DeltaTracker delta) {
		PilotController pilot = DrlAgentModClient.PILOT;
		if (!pilot.isEngaged()) {
			return;
		}
		PilotController.Stats s = pilot.stats();
		String target = pilot.targetName();

		String title = "DRL PILOT" + (target != null ? " > " + target : " (no target)");
		String hits = "Hits " + s.hits() + "  (sprint " + s.sprintHits() + ", crit " + s.crits() + ")";
		String combo = "Combo x" + s.chain() + "  best x" + s.bestChain() + "  combos " + s.combos();
		String taken = "Taken " + s.taken();

		Font font = Minecraft.getInstance().font;
		int w = 0;
		for (String line : new String[] {title, hits, combo, taken}) {
			w = Math.max(w, font.width(line));
		}
		int x = 4;
		int y = 4;
		g.fill(x, y, x + w + 2 * PAD, y + 4 * LINE_H + 2 * PAD, BG);
		g.text(font, title, x + PAD, y + PAD, TITLE, true);
		g.text(font, hits, x + PAD, y + PAD + LINE_H, TEXT, true);
		g.text(font, combo, x + PAD, y + PAD + 2 * LINE_H,
				s.chain() >= 2 ? GOOD : TEXT, true);
		g.text(font, taken, x + PAD, y + PAD + 3 * LINE_H,
				s.taken() > 0 ? BAD : TEXT, true);
	}
}
