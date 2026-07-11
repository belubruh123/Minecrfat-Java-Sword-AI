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
		int move = pilot.wantsMove();     // 0 none, 1 W, 2 S
		int strafe = pilot.wantsStrafe(); // 0 none, 1 A (left), 2 D (right)
		this.keyPresses = new Input(move == 1, move == 2, strafe == 1, strafe == 2,
				pilot.wantsJump(), this.keyPresses.shift(), pilot.wantsSprint());
		// moveVector is (x: left+, y: forward+), matching KeyboardInput.tick
		this.moveVector = new Vec2(strafe == 1 ? 1.0f : (strafe == 2 ? -1.0f : 0.0f),
				move == 1 ? 1.0f : (move == 2 ? -1.0f : 0.0f));
	}
}
