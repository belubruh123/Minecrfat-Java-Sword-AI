# DRL Agent Bridge Protocol (v3)

TCP on localhost, default port **36565**. The **mod listens**, the trainer connects.
The connection is lock-step: the server does not tick until it has an action
batch, and the trainer does not act until it has an observation batch.

## Framing

Every message: `u32 length (big-endian)` + `u8 type` + payload (`length` counts
the type byte + payload).

| type | direction | payload |
|------|-----------|---------|
| 0    | both      | UTF-8 JSON control message |
| 1    | trainer → mod | action batch (binary) |
| 2    | mod → trainer | observation batch (binary) |

All binary numbers are **big-endian**; floats are IEEE-754 f32 (matches Java
`DataOutputStream`).

## Handshake

1. On accept, mod sends JSON `hello`:
```json
{"msg": "hello", "protocol": 1, "mc_version": "26.1.2",
 "arenas": 8, "width": 426, "height": 240,
 "scalars": ["speed", "yaw_sin", "yaw_cos", "pitch", "y", "on_ground",
              "cooldown", "health", "last_reach"]}
```
2. Trainer replies JSON `config`:
```json
{"msg": "config", "stage": "aim", "seed": 1234,
 "curriculum": {"horizontal_prob": 1.0, "max_elev": 0.0,
                 "opponent": "static", "yaw_range": 180.0},
 "record_arena": 0}
```
3. Mod responds JSON `ready`, then immediately sends the first observation
   batch (type 2) for tick 0. From then on strict alternation:
   trainer sends type 1, mod replies type 2.

`config` may be re-sent at any point between steps (curriculum changes apply
at the next episode reset of each arena). If a re-sent `config` carries
`"reseed": true`, the mod re-seeds its rng from `seed` and force-resets every
arena immediately — used for reproducible evaluation suites.

Curriculum opponents: `static`, `strafe`, `fight` (scripted, superhuman:
perfect full-charge timing, instant aim, aware of the agent's invulnerability
window — an upper-bound stress test), `human` (humanized: reaction delay
`opp_reaction_min..max` ticks, no invulnerability awareness, attacks and aim
disrupted while in hitstun or knocked airborne — so combos and spacing have
real defensive value). With `opponent: "human"`, `opp_fight_prob` mixes in
perfect-bot episodes at that probability (anti-overfitting rehearsal).

## Action batch (type 1)

Repeated `arenas` times (arena index order):

| field | type | meaning |
|-------|------|---------|
| dyaw  | f32  | yaw delta, degrees/tick (mod clamps to ±15) |
| dpitch| f32  | pitch delta, degrees/tick (mod clamps to ±15) |
| attack| u8   | 1 = press attack this tick |
| move  | u8   | 0 = none, 1 = W (forward), 2 = S (backward) |
| strafe| u8   | 0 = none, 1 = A (left), 2 = D (right) |
| jump  | u8   | 1 = jump held this tick |
| sprint| u8   | 1 = sprint held (only effective while move is forward) |
| flags | u8   | bit 0: force episode reset |

## Observation batch (type 2)

Header: `u32 tick_counter`, then repeated `arenas` times:

| field | type | meaning |
|-------|------|---------|
| mask  | `ceil(W*H/8)` bytes | row-major, MSB-first bit-packed; 1 = opponent pixel |
| scalars | f32 × 9 | order as in `hello.scalars` |
| reward | f32 | reward earned by the tick just simulated |
| done  | u8  | 1 = episode ended this tick (obs is the first of the new episode) |
| info  | u8  | bit 0: crosshair on target; bit 1: hit landed; bit 2: hit taken; bit 3: episode has an elevated (non-horizontal) spawn; bit 4: whiff (attack pressed, no clean hit); bit 5: landed hit was a critical; bit 6: landed hit had the sprint-knockback bonus |
| telemetry | f32 × 12 | replay/debug only, never policy input: agent x, z, y (relative to arena center/floor), agent yaw (deg), opponent x, z, y, agent pitch (deg), opponent yaw (deg), agent swung (0/1), opponent swung (0/1), opponent hurtTime/10 |

Mask size for 426×240 = 12780 bytes.

## Scalar definitions

- `speed`: horizontal velocity magnitude, blocks/tick
- `yaw_sin`, `yaw_cos`: of agent yaw (absolute)
- `pitch`: degrees / 90 (−1 = up)
- `y`: agent Y − arena floor Y, blocks
- `on_ground`: 0/1
- `cooldown`: attack-cooldown progress 0..1 (as shown on the vanilla HUD)
- `health`: 0..1 (fraction of max, as shown by hearts)
- `last_reach`: eye-to-target distance measured at the last landed hit, blocks
  (0 until the first hit of the episode)
