package com.belubruh123.drlagent.env;

import com.mojang.authlib.GameProfile;

import net.fabricmc.fabric.api.entity.FakePlayer;

import net.minecraft.server.level.ServerLevel;

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

	@Override
	public void tick() {
		this.baseTick();               // hurtTime/invulnerability/fire/air timers
		this.aiStep();                 // travel() from xxa/zza: gravity, collision, sprint
		this.attackStrengthTicker++;   // attack cooldown recharge (vanilla: Player.tick)
	}
}
