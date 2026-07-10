"""Smoke test: connect to the mod bridge, complete the handshake, print HELLO.

Usage: python handshake_test.py [port]
"""

import sys

from drlagent.protocol import DEFAULT_PORT, BridgeConnection


def main() -> None:
    port = int(sys.argv[1]) if len(sys.argv) > 1 else DEFAULT_PORT
    conn = BridgeConnection(port=port)
    h = conn.hello
    print(f"HELLO ok: protocol={h.protocol} mc={h.mc_version} "
          f"arenas={h.arenas} obs={h.width}x{h.height} scalars={h.scalars}")
    conn.configure(stage="aim", curriculum={"horizontal_prob": 1.0}, seed=1)
    print("CONFIG acknowledged (ready received)")
    conn.close()
    print("handshake test PASSED")


if __name__ == "__main__":
    main()
