package com.belubruh123.drlagent.client.replay;

import com.google.gson.JsonArray;
import com.google.gson.JsonObject;
import com.google.gson.JsonParser;
import com.mojang.authlib.GameProfile;

import net.minecraft.client.Minecraft;
import net.minecraft.client.player.RemotePlayer;
import net.minecraft.network.chat.Component;
import net.minecraft.world.InteractionHand;
import net.minecraft.world.entity.Entity;
import net.minecraft.world.entity.EquipmentSlot;
import net.minecraft.world.item.ItemStack;
import net.minecraft.world.item.Items;
import net.minecraft.world.phys.Vec3;

import java.net.URI;
import java.net.http.HttpClient;
import java.net.http.HttpRequest;
import java.net.http.HttpResponse;
import java.time.Duration;
import java.util.UUID;
import java.util.concurrent.atomic.AtomicReference;

/**
 * Replay theater: re-enacts recorded training episodes in the real game so
 * the fight can be watched with actual Minecraft rendering. The camera rides
 * the agent's recorded eye path; the opponent is a client-side player model
 * driven by the episode telemetry (position, facing, sword swings, hurt
 * flashes) at real 20 ticks/s. In follow mode it keeps fetching the newest
 * recorded episode from the dashboard API — a live TV channel of training.
 *
 * Telemetry rows (PROTOCOL.md): ax, az, ay, ayaw, ox, oz, oy, apitch, oyaw,
 * agent_swung, opp_swung, opp_hurt. Coordinates are relative to the arena
 * center/floor; arena 0 sits at (0, 200, 0) on the training server.
 */
public final class ReplayDirector {
	private static final int ARENA_X = 0, FLOOR_Y = 200, ARENA_Z = 0;
	private static final int DUMMY_ID = 1_888_000_001;

	private final HttpClient http = HttpClient.newBuilder()
			.connectTimeout(Duration.ofSeconds(3)).build();
	private String api = "http://127.0.0.1:8080";
	private String run;
	private boolean follow;
	private String lastEpisode = "";

	private float[][] telemetry;
	private int[] infos;
	private String episodeName;
	private int tick;
	private RemotePlayer dummy;
	private final AtomicReference<JsonObject> fetched = new AtomicReference<>();
	private Thread worker;

	public boolean isActive() {
		return run != null;
	}

	public void setApi(String api) {
		this.api = api.endsWith("/") ? api.substring(0, api.length() - 1) : api;
	}

	public String getApi() {
		return api;
	}

	public void start(Minecraft mc, String run, boolean follow) {
		this.run = run;
		this.follow = follow;
		this.lastEpisode = "";
		this.telemetry = null;
		if (mc.player != null) {
			mc.player.connection.sendCommand("gamemode creative");
			mc.player.connection.sendCommand(
					"tp @s " + ARENA_X + " " + (FLOOR_Y + 4) + " " + ARENA_Z);
			mc.player.setItemSlot(EquipmentSlot.MAINHAND, new ItemStack(Items.DIAMOND_SWORD));
			msg(mc, "fetching episodes of '" + run + "' from " + api + " …");
		}
		startWorker();
	}

	public void stop(Minecraft mc, String reason) {
		run = null;
		telemetry = null;
		if (worker != null) {
			worker.interrupt();
			worker = null;
		}
		removeDummy(mc);
		msg(mc, "stopped (" + reason + ")");
	}

	/** Fetches the newest episode (meta + light payload) off-thread. */
	private void startWorker() {
		String wantRun = run;
		worker = new Thread(() -> {
			while (!Thread.currentThread().isInterrupted() && wantRun.equals(run)) {
				try {
					JsonArray eps = JsonParser.parseString(
							get("/api/run/" + wantRun + "/episodes")).getAsJsonArray();
					if (!eps.isEmpty()) {
						String name = eps.get(0).getAsJsonObject().get("name").getAsString();
						if (!name.equals(lastEpisode) && fetched.get() == null) {
							JsonObject ep = JsonParser.parseString(
									get("/api/run/" + wantRun + "/episode/" + name
											+ "?light=1")).getAsJsonObject();
							ep.addProperty("__name", name);
							fetched.set(ep);
						}
					}
					Thread.sleep(2000);
				} catch (InterruptedException e) {
					return;
				} catch (Exception e) {
					// dashboard briefly unreachable: retry
					try {
						Thread.sleep(3000);
					} catch (InterruptedException e2) {
						return;
					}
				}
			}
		}, "drlagent-replay-fetch");
		worker.setDaemon(true);
		worker.start();
	}

	private String get(String path) throws Exception {
		HttpResponse<String> r = http.send(HttpRequest.newBuilder()
						.uri(URI.create(api + path)).timeout(Duration.ofSeconds(8)).build(),
				HttpResponse.BodyHandlers.ofString());
		if (r.statusCode() != 200) {
			throw new IllegalStateException("HTTP " + r.statusCode() + " for " + path);
		}
		return r.body();
	}

	/** Called at the end of every client tick. */
	public void tick(Minecraft mc) {
		if (run == null || mc.player == null || mc.level == null) {
			return;
		}
		if (telemetry == null) {
			JsonObject ep = fetched.getAndSet(null);
			if (ep == null) {
				return; // still fetching
			}
			loadEpisode(mc, ep);
			return;
		}

		float[] t = telemetry[tick];
		int inf = infos[tick];

		// camera = the agent's recorded eyes
		mc.player.getAbilities().flying = true;
		mc.player.setDeltaMovement(Vec3.ZERO);
		mc.player.setPos(ARENA_X + t[0], FLOOR_Y + t[2], ARENA_Z + t[1]);
		mc.player.setYRot(t[3]);
		mc.player.setYHeadRot(t[3]);
		mc.player.setXRot(t[7]);
		if (t[9] > 0.5f) {
			mc.player.swing(InteractionHand.MAIN_HAND);
		}

		if (dummy != null) {
			dummy.setOldPosAndRot(); // renderer interpolates old -> new
			dummy.setPos(ARENA_X + t[4], FLOOR_Y + t[6], ARENA_Z + t[5]);
			dummy.setYRot(t[8]);
			dummy.setYBodyRot(t[8]);
			dummy.setYHeadRot(t[8]);
			if (t[10] > 0.5f) {
				dummy.swing(InteractionHand.MAIN_HAND);
			}
			dummy.hurtTime = Math.max(dummy.hurtTime, Math.round(t[11] * 10));
			if ((inf & 2) != 0) {
				dummy.hurtTime = 9; // our hit landed this tick
			}
		}

		if (tick % 40 == 0) {
			mc.player.sendOverlayMessage(Component.literal(
					"[replay] " + episodeName + "  tick " + tick + "/" + telemetry.length
							+ (follow ? "  (following " + run + ")" : "")));
		}

		tick++;
		if (tick >= telemetry.length) {
			telemetry = null;
			if (!follow) {
				stop(mc, "episode finished");
			}
		}
	}

	private void loadEpisode(Minecraft mc, JsonObject ep) {
		JsonArray tel = ep.getAsJsonArray("telemetry");
		if (tel == null || tel.isEmpty() || tel.get(0).getAsJsonArray().size() < 12) {
			msg(mc, "episode has no full telemetry (recorded before the 3D upgrade); waiting for a newer one");
			lastEpisode = ep.get("__name").getAsString();
			return;
		}
		JsonArray inf = ep.getAsJsonArray("infos");
		telemetry = new float[tel.size()][12];
		infos = new int[tel.size()];
		for (int i = 0; i < tel.size(); i++) {
			JsonArray row = tel.get(i).getAsJsonArray();
			for (int j = 0; j < 12; j++) {
				telemetry[i][j] = row.get(j).getAsFloat();
			}
			infos[i] = inf.get(i).getAsInt();
		}
		episodeName = ep.get("__name").getAsString();
		lastEpisode = episodeName;
		tick = 0;
		ensureDummy(mc);
		msg(mc, "playing " + episodeName + " (" + telemetry.length + " ticks)");
	}

	private void ensureDummy(Minecraft mc) {
		if (dummy != null && !dummy.isRemoved()) {
			return;
		}
		GameProfile profile = new GameProfile(
				UUID.nameUUIDFromBytes("drlagent:replay-foe".getBytes()), "Enemy");
		dummy = new RemotePlayer(mc.level, profile);
		dummy.setId(DUMMY_ID);
		dummy.setNoGravity(true);
		dummy.setItemSlot(EquipmentSlot.HEAD, new ItemStack(Items.DIAMOND_HELMET));
		dummy.setItemSlot(EquipmentSlot.CHEST, new ItemStack(Items.DIAMOND_CHESTPLATE));
		dummy.setItemSlot(EquipmentSlot.LEGS, new ItemStack(Items.DIAMOND_LEGGINGS));
		dummy.setItemSlot(EquipmentSlot.FEET, new ItemStack(Items.DIAMOND_BOOTS));
		dummy.setItemSlot(EquipmentSlot.MAINHAND, new ItemStack(Items.DIAMOND_SWORD));
		dummy.setPos(ARENA_X + 0.5, FLOOR_Y + 1, ARENA_Z + 0.5);
		mc.level.addEntity(dummy);
	}

	private void removeDummy(Minecraft mc) {
		if (dummy != null && mc.level != null) {
			mc.level.removeEntity(DUMMY_ID, Entity.RemovalReason.DISCARDED);
		}
		dummy = null;
	}

	private static void msg(Minecraft mc, String text) {
		if (mc.player != null) {
			mc.player.sendSystemMessage(Component.literal("[replay] " + text));
		}
	}
}
