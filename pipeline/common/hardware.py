"""Which cameras a car records, the order they are indexed in, and where calibration starts from.

Every stage refers to a camera by its position in one list. That list was six hardcoded names, the dashcam of
a Model 3 with HW4. It now comes from the footage: HW4 records six cameras, HW3 four (front, back and the two
repeaters, all at 1280x960). Standard library only, so the ingest and every stage can import it.

Only the HW4 layout has been run end to end. The HW3 lens and angle numbers below are rounded from the
calibration of one HW3 car (a single drive); they are where the solver starts, not a rig to use as it is.
"""

import json
import os
from pathlib import Path

# Index order shared by calibration, poses, road and photometry. A roster is always a subsequence of it, so
# "front" is index 0 and "back" index 1 on every supported car.
ORDER = (
    "front",
    "back",
    "left_pillar",
    "right_pillar",
    "left_repeater",
    "right_repeater",
)

# Per camera: native dashcam size in pixels; equidistant focal length in native pixels (image half-width over
# the half field of view in radians); mount yaw (left positive) and pitch (up positive) in degrees; position in
# metres from the rear axle at ground level (x forward, y left, z up). Positions are those of a Model 3. All of
# it is a soft prior that the solver refines from the footage.
_SIDE = dict(yaw=148, pitch=-5, pos=(2.53, 0.88, 0.79))
PROFILES = {
    "hw4": dict(
        label="HW4, six cameras",
        cameras={
            "front": dict(size=(2896, 1876), focal=3639, yaw=0, pitch=-3, pos=(1.94, 0, 1.31)),
            "back": dict(size=(1448, 938), focal=460, yaw=180, pitch=-32, pos=(-0.89, 0, 0.82)),
            "left_pillar": dict(size=(1448, 938), focal=810, yaw=64, pitch=-5, pos=(1.12, 0.85, 1.15)),
            "right_pillar": dict(size=(1448, 938), focal=810, yaw=-64, pitch=-5, pos=(1.12, -0.85, 1.15)),
            "left_repeater": dict(_SIDE, size=(1448, 938), focal=820),
            "right_repeater": dict(_SIDE, size=(1448, 938), focal=820, yaw=-148, pos=(2.53, -0.88, 0.79)),
        },
    ),
    # Front is the main forward camera (46 degrees across), the repeaters 75 degrees, the rear a fisheye of
    # 148 degrees: focal lengths and angles as one HW3 car calibrated, rounded. The mount positions are left
    # at the HW4 Model 3's starting values, which that solve moved by up to 18 cm.
    "hw3": dict(
        label="HW3, four cameras",
        # The back camera counts its own frames, at a rate that differs from the others' by up to a frame a
        # minute (either way, drive to drive) and falls from 36 to 30 fps at a standstill.
        own_counter=("back",),
        cameras={
            "front": dict(size=(1280, 960), focal=1620, yaw=0, pitch=-1, pos=(1.94, 0, 1.31)),
            "back": dict(size=(1280, 960), focal=500, yaw=180, pitch=-33, pos=(-0.89, 0, 0.82)),
            "left_repeater": dict(size=(1280, 960), focal=1005, yaw=146, pitch=2, pos=(2.53, 0.88, 0.79)),
            "right_repeater": dict(size=(1280, 960), focal=1005, yaw=-146, pitch=2, pos=(2.53, -0.88, 0.79)),
        },
    ),
}

# Lens pairs that can see the same thing, for feature matching. Calibration tries each pair at the same instant
# and a few keyframes either side.
ADJACENT = (
    ("front", "left_pillar"),
    ("front", "right_pillar"),
    ("left_pillar", "left_repeater"),
    ("right_pillar", "right_repeater"),
    ("back", "left_repeater"),
    ("back", "right_repeater"),
    ("back", "left_pillar"),
    ("back", "right_pillar"),
    ("left_pillar", "right_pillar"),
    ("left_repeater", "right_repeater"),
)

# The same for the whole-drive pose solve: (earlier lens, later lens, keyframe offsets). Something the first
# lens sees now is seen by the second that many keyframes later.
CROSS = (
    ("front", "left_pillar", (0, 1, 3, 6, 10, 15)),
    ("front", "right_pillar", (0, 1, 3, 6, 10, 15)),
    ("front", "left_repeater", (2, 5, 9, 14)),
    ("front", "right_repeater", (2, 5, 9, 14)),
    ("left_pillar", "left_repeater", (-1, 0, 1, 3, 6)),
    ("right_pillar", "right_repeater", (-1, 0, 1, 3, 6)),
    ("left_repeater", "back", (-1, 0, 1, 3, 7)),
    ("right_repeater", "back", (-1, 0, 1, 3, 7)),
    ("left_pillar", "back", (2, 5, 9)),
    ("right_pillar", "back", (2, 5, 9)),
)

# Without pillar cameras nothing overlaps the front lens at one instant: the repeaters look backwards. The only
# visual tie between the front and the rest is something the front saw that a repeater sees once the car has
# passed it, this many calibration keyframes later.
FRONT_TO_REPEATER = (5, 10, 15, 20)


def reduction(width, target):
    """How far (4, 2 or 1) a JPEG `width` pixels across can be decoded reduced and still be `target` across.

    Stages read frames no larger than they need. On HW4 that halves the 2896 px front image relative to the
    1448 px others; HW3's 1280 px frames are mostly read whole.
    """
    return 4 if width >= 4 * target else 2 if width >= 2 * target else 1


def ordered(names):
    """Names in index order: ORDER first, anything unknown after it, alphabetically."""
    names = list(dict.fromkeys(names))
    return [n for n in ORDER if n in names] + sorted(n for n in names if n not in ORDER)


def roster(root=None, footage=False):
    """Camera names of the drive in the workspace `root` (default $RS, else the current directory).

    Read from the calibrated rig when there is one, else from the ingested footage; `footage=True` asks the
    footage first, for calibration, which is what writes the rig. Outside a workspace it is the six HW4 names,
    which is what every stage assumed before.
    """
    root = Path(root or os.environ.get("RS") or ".")
    files = [("calib/rig.json", "cameras"), ("data/drive/meta.json", "cams")]
    for path, key in reversed(files) if footage else files:
        try:
            names = ordered(json.loads((root / path).read_text())[key])
        except (OSError, ValueError, KeyError):
            continue
        if "front" not in names:
            raise ValueError(f"{root / path}: no front camera; it is the master clock of every stage")
        return names
    return list(ORDER)


def identify(names, sizes=None):
    """Name of the profile with exactly these cameras, or None. `sizes` ({camera: (w, h)}) narrows the match."""
    names = set(names)
    found = [k for k, p in PROFILES.items() if set(p["cameras"]) == names]
    if sizes and len(found) > 1:
        exact = [
            k
            for k in found
            if all(tuple(sizes[c]) == tuple(q["size"]) for c, q in PROFILES[k]["cameras"].items())
        ]
        found = exact or found
    return found[0] if found else None


def own_counter(names, sizes=None):
    """Cameras of this layout known to keep a frame counter of their own instead of the front camera's, so
    that their frames have to be timed from file timestamps (pipeline/ingest)."""
    key = identify(names, sizes)
    return tuple(PROFILES[key].get("own_counter", ())) if key else ()


def priors(names, sizes):
    """Starting lens and mount values for calibration, one entry per camera in `names` order.

    Returns (profile name, focal lengths, yaws, pitches, positions). The focal length follows the footage's
    actual width, so a resolution the profile does not list still starts from the right field of view.
    """
    key = identify(names, sizes)
    if key is None:
        known = "; ".join(f"{k}: {', '.join(p['cameras'])}" for k, p in PROFILES.items())
        raise ValueError(
            f"no calibration starting values for the cameras {', '.join(names)}. Known layouts: {known}. "
            "Add a profile to pipeline/common/hardware.py."
        )
    cams = [PROFILES[key]["cameras"][n] for n in names]
    return (
        key,
        [c["focal"] * sizes[n][0] / c["size"][0] for n, c in zip(names, cams)],
        [c["yaw"] for c in cams],
        [c["pitch"] for c in cams],
        [list(c["pos"]) for c in cams],
    )


def adjacent(names):
    """ADJACENT as index pairs into `names`, keeping the pairs whose two lenses both exist."""
    names = list(names)
    return [(names.index(a), names.index(b)) for a, b in ADJACENT if a in names and b in names]


def cross(names):
    """CROSS as (index, index, offsets) into `names`, keeping the pairs whose two lenses both exist."""
    names = list(names)
    return [(names.index(a), names.index(b), list(o)) for a, b, o in CROSS if a in names and b in names]


def front_to_repeater(names):
    """(front index, repeater index) for each side that has a repeater but no pillar camera."""
    names = list(names)
    return [
        (names.index("front"), names.index(f"{side}_repeater"))
        for side in ("left", "right")
        if f"{side}_repeater" in names and f"{side}_pillar" not in names
    ]


def mirrored(names):
    """Index pairs of the left and right camera of each mirrored mount."""
    names = list(names)
    return [
        (names.index(n), names.index("right" + n[4:]))
        for n in names
        if n.startswith("left_") and "right" + n[4:] in names
    ]
