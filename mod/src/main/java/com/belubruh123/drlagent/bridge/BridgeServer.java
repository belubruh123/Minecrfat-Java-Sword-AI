package com.belubruh123.drlagent.bridge;

import com.belubruh123.drlagent.DrlAgentMod;
import com.google.gson.Gson;
import com.google.gson.JsonObject;

import java.io.DataInputStream;
import java.io.DataOutputStream;
import java.io.IOException;
import java.net.ServerSocket;
import java.net.Socket;
import java.nio.charset.StandardCharsets;

/**
 * TCP side of the lock-step training bridge (see PROTOCOL.md).
 *
 * <p>Owns the listen socket and the single trainer connection. Framing is
 * u32 length (big endian) + u8 type + payload; {@link DataInputStream} /
 * {@link DataOutputStream} are big-endian already.
 *
 * <p>Handshake runs on the accept thread. The per-tick step exchange is
 * driven by the server tick thread through {@link #readActionFrame()} and
 * {@link #sendObsFrame(byte[])} once a trainer is attached.
 */
public final class BridgeServer {
	public static final int TYPE_JSON = 0;
	public static final int TYPE_ACTIONS = 1;
	public static final int TYPE_OBS = 2;

	private static final Gson GSON = new Gson();

	private final int port;
	private ServerSocket listener;
	private Thread acceptThread;

	private volatile Connection connection;
	private volatile JsonObject latestConfig;

	public BridgeServer(int port) {
		this.port = port;
	}

	public void start() throws IOException {
		listener = new ServerSocket(port);
		acceptThread = new Thread(this::acceptLoop, "drlagent-bridge-accept");
		acceptThread.setDaemon(true);
		acceptThread.start();
		DrlAgentMod.LOGGER.info("Training bridge listening on port {}", port);
	}

	private void acceptLoop() {
		while (!listener.isClosed()) {
			try {
				Socket socket = listener.accept();
				socket.setTcpNoDelay(true);
				Connection conn = new Connection(socket);
				conn.handshake();
				connection = conn;
				DrlAgentMod.LOGGER.info("Trainer connected from {}", socket.getRemoteSocketAddress());
			} catch (IOException e) {
				if (!listener.isClosed()) {
					DrlAgentMod.LOGGER.warn("Bridge accept/handshake failed", e);
				}
			}
		}
	}

	/** Non-null once a trainer has completed the handshake. */
	public Connection getConnection() {
		Connection conn = connection;
		if (conn != null && conn.socket.isClosed()) {
			connection = null;
			return null;
		}
		return conn;
	}

	/** Latest config JSON sent by the trainer (handshake or mid-run reconfigure). */
	public JsonObject getLatestConfig() {
		return latestConfig;
	}

	/** Drops the current trainer connection (e.g. after an I/O error mid-step). */
	public void dropConnection() {
		Connection conn = connection;
		connection = null;
		if (conn != null) conn.close();
	}

	public void stop() {
		try {
			if (listener != null) listener.close();
		} catch (IOException ignored) {
		}
		Connection conn = connection;
		if (conn != null) conn.close();
	}

	public final class Connection {
		private final Socket socket;
		private final DataInputStream in;
		private final DataOutputStream out;

		private Connection(Socket socket) throws IOException {
			this.socket = socket;
			this.in = new DataInputStream(socket.getInputStream());
			this.out = new DataOutputStream(socket.getOutputStream());
		}

		private void handshake() throws IOException {
			JsonObject hello = new JsonObject();
			hello.addProperty("msg", "hello");
			hello.addProperty("protocol", BridgeConfig.PROTOCOL_VERSION);
			hello.addProperty("mc_version", "26.1.2");
			hello.addProperty("arenas", BridgeConfig.arenas());
			hello.addProperty("width", BridgeConfig.OBS_WIDTH);
			hello.addProperty("height", BridgeConfig.OBS_HEIGHT);
			hello.add("scalars", GSON.toJsonTree(BridgeConfig.SCALARS));
			sendJson(hello);

			JsonObject config = readJson();
			if (!"config".equals(config.get("msg").getAsString())) {
				throw new IOException("expected config message, got: " + config);
			}
			latestConfig = config;
			DrlAgentMod.LOGGER.info("Trainer config: {}", config);

			JsonObject ready = new JsonObject();
			ready.addProperty("msg", "ready");
			sendJson(ready);
		}

		public synchronized void sendJson(JsonObject obj) throws IOException {
			byte[] payload = GSON.toJson(obj).getBytes(StandardCharsets.UTF_8);
			out.writeInt(payload.length + 1);
			out.writeByte(TYPE_JSON);
			out.write(payload);
			out.flush();
		}

		public synchronized void sendObsFrame(byte[] payload) throws IOException {
			out.writeInt(payload.length + 1);
			out.writeByte(TYPE_OBS);
			out.write(payload);
			out.flush();
		}

		private JsonObject readJson() throws IOException {
			byte[] payload = readFrameOfType(TYPE_JSON);
			return GSON.fromJson(new String(payload, StandardCharsets.UTF_8), JsonObject.class);
		}

		/**
		 * Blocks until the next action frame arrives. JSON control frames that
		 * arrive in between (curriculum changes) are applied to
		 * {@link #getLatestConfig()} and skipped.
		 */
		public byte[] readActionFrame() throws IOException {
			return readFrameOfType(TYPE_ACTIONS);
		}

		private byte[] readFrameOfType(int wanted) throws IOException {
			while (true) {
				int length = in.readInt();
				int type = in.readUnsignedByte();
				byte[] payload = new byte[length - 1];
				in.readFully(payload);
				if (type == wanted) {
					return payload;
				}
				if (type == TYPE_JSON) {
					latestConfig = GSON.fromJson(new String(payload, StandardCharsets.UTF_8), JsonObject.class);
					DrlAgentMod.LOGGER.info("Trainer reconfigured: {}", latestConfig);
				} else {
					throw new IOException("unexpected frame type " + type);
				}
			}
		}

		public void close() {
			try {
				socket.close();
			} catch (IOException ignored) {
			}
		}
	}
}
