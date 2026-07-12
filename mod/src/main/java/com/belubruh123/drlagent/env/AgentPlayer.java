package com.belubruh123.drlagent.env;

import com.mojang.authlib.GameProfile;

import net.fabricmc.fabric.api.entity.FakePlayer;

import net.minecraft.server.level.ServerLevel;
import net.minecraft.world.damagesource.DamageSource;

/**
 * Fake player that actually simulates. Two vanilla obstacles:
 * real players are client-authoritative, so the server skips their
 * {@code travel()}/gravity (fixed by {@link #isClientAuthoritative()}), and
 * Fabric's {@link FakePlayer#tick()} is a deliberate no-op, so the vanilla
 * player tick can never run (fixed by rebuilding the needed parts of the
 * tick from public vanilla pieces here — timers, physics, attack cooldown).
 */
public final class AgentPlayer extends FakePlayer {
	public AgentPlayer(ServerLevel level, GameProfile profile) {
		super(level, profile);
	}

	@Override
	public boolean isClientAuthoritative() {
		return false;
	}

	public int ticker() {
		return this.attackStrengthTicker;
	}

	// FakePlayer hardcodes blanket invulnerability (bytecode: iconst_1),
	// which silently swallows sword hits — agents must take damage.
	@Override
	public boolean isInvulnerableTo(ServerLevel level, DamageSource source) {
		return false;
	}

	// Mortal episodes end at 0 HP, but the vanilla death pipeline must never
	// run for arena players: die() would broadcast a death message, drop the
	// diamond kit as item entities and schedule removal. The arena detects
	// health <= 0 itself, ends the episode, and reset() revives the player.
	@Override
	public void die(DamageSource source) {
	}

	@Override
	public void tick() {
		this.baseTick();               // hurtTime/fire/air timers
		// baseTick skips the invulnerability countdown for ServerPlayers
		// (bytecode: instanceof guard) — the connection-driven tick path we
		// bypass normally does it, so do it here or one hit blocks all others
		if (this.invulnerableTime > 0) {
			this.invulnerableTime--;
		}
		this.aiStep();                 // travel() from xxa/zza: gravity, collision, sprint
		this.attackStrengthTicker++;   // attack cooldown recharge (vanilla: Player.tick)
		syncEquipmentAttributes();     // apply sword/armor attribute modifiers
	}

	// Vanilla applies equipment attribute modifiers (sword attack speed/damage,
	// armor points) in LivingEntity.tick() via private detectEquipmentUpdates();
	// our composite tick skips it, so invoke it reflectively (dev env runs
	// unobfuscated, so the mojmap name resolves at runtime).
	private static java.lang.reflect.Method detectEquipment;

	private void syncEquipmentAttributes() {
		try {
			if (detectEquipment == null) {
				detectEquipment = net.minecraft.world.entity.LivingEntity.class
						.getDeclaredMethod("detectEquipmentUpdates");
				detectEquipment.setAccessible(true);
			}
			detectEquipment.invoke(this);
		} catch (ReflectiveOperationException e) {
			throw new IllegalStateException("detectEquipmentUpdates reflection failed", e);
		}
	}
}
