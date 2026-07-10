package com.belubruh123.drlagent.env;

/**
 * Renders the agent's-eye binary observation: projects a target AABB through
 * a perspective camera (vanilla 70° vertical FOV) and sets a bit for every
 * pixel whose view ray intersects the box. Pure math — no Minecraft types —
 * so it is unit-testable and reusable client-side for pilot mode.
 *
 * <p>Pixel layout: row-major from the top-left, MSB-first bit packing, i.e.
 * bit index {@code py * width + px}, matching PROTOCOL.md.
 */
public final class ObsProjector {
	public static final double FOV_Y_DEG = 70.0;

	private ObsProjector() {
	}

	/**
	 * Fills {@code out} (zeroed by the caller) with the target silhouette.
	 * Yaw/pitch in Minecraft degrees (yaw 0 = +Z, pitch + = down).
	 */
	public static void render(byte[] out, int width, int height,
			double eyeX, double eyeY, double eyeZ,
			float yawDeg, float pitchDeg,
			double minX, double minY, double minZ,
			double maxX, double maxY, double maxZ) {
		double yaw = Math.toRadians(yawDeg);
		double pitch = Math.toRadians(pitchDeg);

		// Camera basis. Facing +Z (yaw 0): forward (0,0,1), right (-1,0,0), up (0,1,0).
		double cosYaw = Math.cos(yaw), sinYaw = Math.sin(yaw);
		double cosPit = Math.cos(pitch), sinPit = Math.sin(pitch);
		double fx = -sinYaw * cosPit, fy = -sinPit, fz = cosYaw * cosPit;
		double rx = -cosYaw, ry = 0, rz = -sinYaw;
		// up = right x forward
		double ux = ry * fz - rz * fy;
		double uy = rz * fx - rx * fz;
		double uz = rx * fy - ry * fx;

		double tanHalfY = Math.tan(Math.toRadians(FOV_Y_DEG) / 2.0);
		double tanHalfX = tanHalfY * ((double) width / height);

		// Project the 8 corners to find the candidate pixel rectangle.
		double sxMin = Double.POSITIVE_INFINITY, sxMax = Double.NEGATIVE_INFINITY;
		double syMin = Double.POSITIVE_INFINITY, syMax = Double.NEGATIVE_INFINITY;
		boolean anyInFront = false, anyBehind = false;
		for (int c = 0; c < 8; c++) {
			double wx = ((c & 1) == 0 ? minX : maxX) - eyeX;
			double wy = ((c & 2) == 0 ? minY : maxY) - eyeY;
			double wz = ((c & 4) == 0 ? minZ : maxZ) - eyeZ;
			double cz = wx * fx + wy * fy + wz * fz;
			if (cz <= 1e-6) {
				anyBehind = true;
				continue;
			}
			anyInFront = true;
			double sx = (wx * rx + wy * ry + wz * rz) / (cz * tanHalfX);
			double sy = (wx * ux + wy * uy + wz * uz) / (cz * tanHalfY);
			sxMin = Math.min(sxMin, sx);
			sxMax = Math.max(sxMax, sx);
			syMin = Math.min(syMin, sy);
			syMax = Math.max(syMax, sy);
		}
		if (!anyInFront) {
			return;
		}
		if (anyBehind) {
			// Box straddles the camera plane; the projected rect is unbounded.
			sxMin = -1;
			sxMax = 1;
			syMin = -1;
			syMax = 1;
		}

		int pxMin = clampPx((int) Math.floor((sxMin + 1) * 0.5 * width), width);
		int pxMax = clampPx((int) Math.ceil((sxMax + 1) * 0.5 * width), width);
		// screen y is flipped: sy +1 = top row 0
		int pyMin = clampPx((int) Math.floor((1 - syMax) * 0.5 * height), height);
		int pyMax = clampPx((int) Math.ceil((1 - syMin) * 0.5 * height), height);

		double rminX = minX - eyeX, rminY = minY - eyeY, rminZ = minZ - eyeZ;
		double rmaxX = maxX - eyeX, rmaxY = maxY - eyeY, rmaxZ = maxZ - eyeZ;

		for (int py = pyMin; py <= pyMax; py++) {
			double ndcY = 1 - 2 * (py + 0.5) / height;
			double vy = ndcY * tanHalfY;
			int rowBase = py * width;
			for (int px = pxMin; px <= pxMax; px++) {
				double ndcX = 2 * (px + 0.5) / width - 1;
				double vx = ndcX * tanHalfX;
				double dx = fx + rx * vx + ux * vy;
				double dy = fy + ry * vx + uy * vy;
				double dz = fz + rz * vx + uz * vy;
				if (rayHitsBox(dx, dy, dz, rminX, rminY, rminZ, rmaxX, rmaxY, rmaxZ)) {
					int bit = rowBase + px;
					out[bit >> 3] |= (byte) (0x80 >>> (bit & 7));
				}
			}
		}
	}

	private static int clampPx(int v, int size) {
		return v < 0 ? 0 : Math.min(v, size - 1);
	}

	/** Slab test: does the ray from the origin along (dx,dy,dz) hit the box (t > 0)? */
	public static boolean rayHitsBox(double dx, double dy, double dz,
			double minX, double minY, double minZ,
			double maxX, double maxY, double maxZ) {
		double tMin = 0, tMax = Double.POSITIVE_INFINITY;

		double inv = 1.0 / dx;
		double t1 = minX * inv, t2 = maxX * inv;
		tMin = Math.max(tMin, Math.min(t1, t2));
		tMax = Math.min(tMax, Math.max(t1, t2));

		inv = 1.0 / dy;
		t1 = minY * inv;
		t2 = maxY * inv;
		tMin = Math.max(tMin, Math.min(t1, t2));
		tMax = Math.min(tMax, Math.max(t1, t2));

		inv = 1.0 / dz;
		t1 = minZ * inv;
		t2 = maxZ * inv;
		tMin = Math.max(tMin, Math.min(t1, t2));
		tMax = Math.min(tMax, Math.max(t1, t2));

		return tMax >= tMin && !Double.isNaN(tMin) && !Double.isNaN(tMax);
	}

	/** Angle in degrees between the view direction for (yaw, pitch) and the vector to a point. */
	public static double angleToTarget(float yawDeg, float pitchDeg,
			double eyeX, double eyeY, double eyeZ,
			double tx, double ty, double tz) {
		double yaw = Math.toRadians(yawDeg);
		double pitch = Math.toRadians(pitchDeg);
		double fx = -Math.sin(yaw) * Math.cos(pitch);
		double fy = -Math.sin(pitch);
		double fz = Math.cos(yaw) * Math.cos(pitch);
		double dx = tx - eyeX, dy = ty - eyeY, dz = tz - eyeZ;
		double len = Math.sqrt(dx * dx + dy * dy + dz * dz);
		if (len < 1e-9) {
			return 0;
		}
		double dot = (fx * dx + fy * dy + fz * dz) / len;
		return Math.toDegrees(Math.acos(Math.max(-1, Math.min(1, dot))));
	}
}
