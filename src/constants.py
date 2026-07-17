"""Shared constants that define the particle package format."""

import struct
from typing import Final, Tuple


FIELD_ALIASES: Final = {
    "id": ("id", "particle_id", "pid"),
    "x": ("x", "posx", "position_x"),
    "y": ("y", "posy", "position_y"),
    "z": ("z", "posz", "position_z"),
    "vx": ("vx", "velx", "velocity_x"),
    "vy": ("vy", "vely", "velocity_y"),
    "vz": ("vz", "velz", "velocity_z"),
}

LOGICAL_ORDER: Final[Tuple[str, ...]] = ("id", "x", "y", "z", "vx", "vy", "vz")
POSITION_FIELDS: Final[Tuple[str, str, str]] = ("x", "y", "z")
VELOCITY_FIELDS: Final[Tuple[str, str, str]] = ("vx", "vy", "vz")

MIN_CODEC_VALUES: Final = 10_000

LCP_CHUNK_MAGIC: Final = b"LCPCHK2\0"
LCP_CHUNK_CONTAINER: Final = "chunked_lcp_v2"
LCP_CHUNK_HEADER: Final = struct.Struct("<8sQQQ")
LCP_CHUNK_ENTRY: Final = struct.Struct("<QQ")
LCP_CHUNK_BATCH_VALUES: Final = 1 << 20
MAX_INT32_ORDER_VALUES: Final = 2**31

