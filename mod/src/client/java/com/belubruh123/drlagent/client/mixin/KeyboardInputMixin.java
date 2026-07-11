package com.belubruh123.drlagent.client.mixin;

import com.belubruh123.drlagent.client.DrlAgentModClient;
import com.belubruh123.drlagent.client.pilot.PilotController;

import net.minecraft.client.player.ClientInput;
import net.minecraft.client.player.KeyboardInput;
import net.minecraft.world.entity.player.Input;
import net.minecraft.world.phys.Vec2;

import org.spongepowered.asm.mixin.Mixin;
import org.spongepowered.asm.mixin.injection.At;
import org.spongepowered.asm.mixin.injection.Inject;
import org.spongepowered.asm.mixin.injection.callback.CallbackInfo;

/**
 * Feeds the pilot's forward/sprint choice into the movement input. Key
 * mappings are useless for this: Minecraft calls KeyMapping.setAll() every
 * tick while in game, re-reading physical key state and discarding anything
 * set programmatically — so the override happens here, after the vanilla
 * tick fills keyPresses from the (unpressed) keyboard.
 */
@Mixin(KeyboardInput.class)
public abstract class KeyboardInputMixin extends ClientInput {
	@Inject(method = "tick", at = @At("TAIL"))
	private void drlagent$applyPilotInput(CallbackInfo ci) {
		PilotController pilot = DrlAgentModClient.PILOT;
		if (!pilot.isEngaged()) {
			return;
		}
		boolean forward = pilot.wantsForward();
		this.keyPresses = new Input(forward, false, false, false,
				pilot.wantsJump(), this.keyPresses.shift(), pilot.wantsSprint());
		this.moveVector = new Vec2(0.0f, forward ? 1.0f : 0.0f);
	}
}
