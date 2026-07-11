package com.belubruh123.drlagent.client.pilot;

import java.io.Closeable;
import java.io.DataInputStream;
import java.io.DataOutputStream;
import java.io.IOException;
import java.net.InetSocketAddress;
import java.net.Socket;
import java.nio.charset.StandardCharsets;

/**
 * Connection to the pilot inference server (trainer/pilot.py). Framing is
 * PROTOCOL.md's (u32 big-endian length + u8 type + payload) with the roles
 * reversed: the game connects out, and there is exactly one "arena" — the
 * local player. Strict alternation: one obs out, one action back.
 */
public final class PilotBridge implements Closeable {
	public static final int DEFAULT_PORT = 36566;
	private static final int TYPE_JSON = 0;
	private static final int TYPE_ACTION = 1;
	private static final int TYPE_OBS = 2;

	public record Action(float dyaw, float dpitch, boolean attack, boolean forward) {
	}

	private final Socket socket;
	private final DataInputStream in;
	private final DataOutputStream out;

	public PilotBridge(int port, int connectTimeoutMs, int readTimeoutMs) throws IOException {
		socket = new Socket();
		socket.connect(new InetSocketAddress("127.0.0.1", port), connectTimeoutMs);
		socket.setTcpNoDelay(true);
		socket.setSoTimeout(readTimeoutMs);
		in = new DataInputStream(socket.getInputStream());
		out = new DataOutputStream(socket.getOutputStream());
	}

	/** Sends the hello and waits for the server's ready reply. */
	public void handshake(int width, int height) throws IOException {
		String hello = "{\"msg\":\"pilot_hello\",\"protocol\":1,\"width\":" + width
				+ ",\"height\":" + height + "}";
		byte[] payload = hello.getBytes(StandardCharsets.UTF_8);
		out.writeInt(payload.length + 1);
		out.writeByte(TYPE_JSON);
		out.write(payload);
		out.flush();
		String reply = new String(readFrame(TYPE_JSON), StandardCharsets.UTF_8);
		if (!reply.contains("\"ready\"")) {
			throw new IOException("unexpected pilot server reply: " + reply);
		}
	}

	public void sendObs(byte[] mask, float[] scalars) throws IOException {
		out.writeInt(1 + mask.length + scalars.length * 4);
		out.writeByte(TYPE_OBS);
		out.write(mask);
		for (float s : scalars) {
			out.writeFloat(s);
		}
		out.flush();
	}

	/** Blocks up to the read timeout; throws SocketTimeoutException when late. */
	public Action readAction() throws IOException {
		byte[] p = readFrame(TYPE_ACTION);
		if (p.length < 10) {
			throw new IOException("short action frame: " + p.length + " bytes");
		}
		float dyaw = Float.intBitsToFloat(intAt(p, 0));
		float dpitch = Float.intBitsToFloat(intAt(p, 4));
		return new Action(dyaw, dpitch, p[8] != 0, p[9] != 0);
	}

	private static int intAt(byte[] b, int off) {
		return ((b[off] & 0xff) << 24) | ((b[off + 1] & 0xff) << 16)
				| ((b[off + 2] & 0xff) << 8) | (b[off + 3] & 0xff);
	}

	private byte[] readFrame(int expectedType) throws IOException {
		int len = in.readInt();
		int type = in.readUnsignedByte();
		if (len < 1 || len > 1 << 20) {
			throw new IOException("bad frame length " + len);
		}
		byte[] payload = new byte[len - 1];
		in.readFully(payload);
		if (type != expectedType) {
			throw new IOException("expected frame type " + expectedType + ", got " + type);
		}
		return payload;
	}

	@Override
	public void close() {
		try {
			socket.close();
		} catch (IOException ignored) {
		}
	}
}
