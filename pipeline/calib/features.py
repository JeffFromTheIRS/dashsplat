"""Select synchronized keyframes, apply measured body masks, and extract/match RootSIFT."""

import os, sys, json, itertools
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
import cv2 as cv
import numpy as np

cv.setNumThreads(1)
sys.path.append(str(Path(__file__).resolve().parents[2]))  # run.sh starts these scripts by path
from pipeline.common import hardware  # noqa: E402

RS = Path(os.getenv("RS", ".")).resolve()
DATA = RS / "data/drive"
W = RS / "work/calib"
OUT = RS / "calib"
# The cameras of this footage, in the index order every array below uses: six on HW4, four on HW3.
CAMS = hardware.roster(RS, footage=True)
NC = len(CAMS)


def native_sizes():
    """{camera: [width, height]} of the footage; the HW4 sizes when nothing has been ingested here."""
    try:
        sizes = json.load(open(DATA / "meta.json"))["sizes"]
        return {c: sizes[c] for c in CAMS}
    except (OSError, ValueError, KeyError):
        return {c: list(hardware.PROFILES["hw4"]["cameras"][c]["size"]) for c in CAMS}


# World-valid polygons of the HW4 Model 3 this was built on, drawn on its 600x389 temporal median contact
# sheets: "keep" polygons are the only valid area, "drop" polygons are bodywork. At the back both boundaries
# are car/camera housing, not the limits of a rectangular sensor.
HW4_POLYGONS = {
    "back": dict(
        keep=[
            [
                (4, 177),
                (12, 146),
                (28, 119),
                (50, 91),
                (80, 63),
                (115, 40),
                (154, 21),
                (204, 7),
                (234, 3),
                (360, 3),
                (400, 10),
                (443, 27),
                (486, 50),
                (524, 78),
                (551, 106),
                (574, 138),
                (590, 177),
                (595, 208),
                (595, 245),
                (578, 270),
                (546, 288),
                (512, 301),
                (460, 312),
                (404, 322),
                (347, 326),
                (285, 328),
                (230, 324),
                (175, 316),
                (125, 305),
                (76, 293),
                (43, 281),
                (20, 267),
                (5, 247),
            ]
        ]
    ),
    "left_repeater": dict(
        drop=[
            [
                (0, 0),
                (151, 0),
                (136, 8),
                (46, 7),
                (8, 44),
                (19, 84),
                (37, 150),
                (39, 240),
                (52, 316),
                (68, 389),
                (0, 389),
            ]
        ]
    ),
    "right_repeater": dict(
        drop=[
            [
                (600, 0),
                (447, 0),
                (460, 17),
                (479, 24),
                (545, 14),
                (569, 12),
                (594, 39),
                (594, 70),
                (569, 148),
                (568, 237),
                (549, 389),
                (600, 389),
            ]
        ]
    ),
    "left_pillar": dict(drop=[[(600, 328), (595, 351), (580, 373), (556, 389), (600, 389)]]),
    "right_pillar": dict(drop=[[(0, 325), (8, 350), (21, 370), (44, 389), (0, 389)]]),
    "front": dict(
        drop=[[(0, 380), (115, 384), (260, 387), (440, 385), (600, 378), (600, 389), (0, 389)]]
    ),
}


def polygons(sizes):
    """Body-mask polygons per camera and the (width, height) they were drawn in.

    calib/mask-polygons.json wins when it exists: {"space": [w, h], "<camera>": {"keep": [[[x, y], ...]],
    "drop": [...]}}, with "space" defaulting to [1, 1] (image fractions). Draw them on the sheets that
    `features.py --median` writes. Otherwise HW4 footage gets the polygons of the car this was built on, and
    anything else only loses its three-pixel border.
    """
    custom = OUT / "mask-polygons.json"
    if custom.exists():
        doc = json.load(open(custom))
        return doc, doc.get("space", [1, 1])
    if hardware.identify(CAMS, sizes) == "hw4":
        return HW4_POLYGONS, (600, 389)
    print(
        f"no {custom}: masks cover the image border only, so this car's own bodywork stays in view of "
        "the solver and of every later stage. Run features.py --median and draw polygons.",
        flush=True,
    )
    return {}, (1, 1)


def masks():
    sizes = native_sizes()
    polys, (sw, sh) = polygons(sizes)
    (OUT / "masks").mkdir(exist_ok=True, parents=True)
    for cam in CAMS:
        w, h = sizes[cam]
        spec = polys.get(cam, {})
        m = np.full((h, w), 0 if spec.get("keep") else 255, np.uint8)
        for key, color in (("keep", 255), ("drop", 0)):
            for ps in spec.get(key, []):
                cv.fillPoly(m, [np.rint(np.array(ps) * [w / sw, h / sh]).astype(np.int32)], color)
        m[:3] = 0
        m[-3:] = 0
        m[:, :3] = 0
        m[:, -3:] = 0
        cv.imwrite(str(OUT / "masks" / f"{cam}.png"), m)


def median_sheets(count=60):
    """Temporal median of moving frames per camera, 600 px wide, in calib/debug: the world blurs away and
    the car's own bodywork stays sharp, which is what the body-mask polygons are drawn around."""
    tl = np.load(DATA / "timeline.npz")
    moving = tl["t"][tl["speed"] > 3]
    if len(moving) < count:
        raise ValueError("not enough footage above 3 m/s for a temporal median")
    picks = moving[np.linspace(0, len(moving) - 1, count).astype(int)]
    (OUT / "debug").mkdir(exist_ok=True, parents=True)
    for cam in CAMS:
        rows = json.load(open(DATA / "cams" / f"{cam}.json"))
        ts = np.array([r["t"] for r in rows])
        stack = []
        for t in picks:
            im = cv.imread(str(DATA / "frames" / cam / f"{rows[int(np.argmin(abs(ts - t)))]['n']:06d}.jpg"))
            stack.append(cv.resize(im, (600, round(600 * im.shape[0] / im.shape[1])), interpolation=cv.INTER_AREA))
        dest = OUT / "debug" / f"median-{cam}.jpg"
        cv.imwrite(str(dest), np.median(stack, axis=0).astype(np.uint8))
        print(dest, stack[0].shape[1], "x", stack[0].shape[0], flush=True)


# Hand-picked windows of the drive this was built on: (start s, end s, keyframe spacing m). The last one is
# the held-out scale check; the others train the rig.
AUTHOR_SEGS = [
    (0, 4, 0.7),
    (82, 94, 0.85),
    (95, 104, 1.05),
    (104.1, 111.4, 1.15),
    (318.5, 325, 1.7),
    (231.6, 235.4, 0.85),
]


def auto_segments(tl, length=60.0, hold=16.0, spacing=0.85):
    """Pick the windows from telemetry alone: the straightest `hold` metres as the scale check, then up to
    five disjoint `length`-metre stretches that turn the most (turns are what locate the mounts)."""
    t = tl["t"]
    yaw = np.unwrap(tl["yaw"])
    speed = tl["speed"]
    ds = np.r_[0, np.cumsum(np.hypot(np.diff(tl["x"]), np.diff(tl["y"])))]
    slow = np.r_[0, np.cumsum(speed <= 0.5)]

    def windows(metres):
        step = max(1, len(t) // 4000)
        a = np.arange(0, len(t), step)
        b = np.searchsorted(ds, ds[a] + metres)
        ok = b < len(t)
        a, b = a[ok], b[ok]
        ok = (slow[b + 1] - slow[a] == 0) & (t[b] - t[a] < 60)
        return a[ok], b[ok]

    a, b = windows(hold)
    if not len(a):
        raise ValueError(f"no {hold:.0f} m of unbroken forward driving for the scale check")
    turn = np.array([np.ptp(yaw[i : j + 1]) for i, j in zip(a, b)])
    # Above about 13 m/s the 16 m are over in 1.2 s and the lenses share too few landmarks to measure them
    # (a stretch at 22 m/s gave none): take the straightest of the slower stretches when there are any.
    calm = (ds[b] - ds[a]) / (t[b] - t[a]) <= 13.0
    if calm.any():
        a, b, turn = a[calm], b[calm], turn[calm]
    i = int(np.argmin(turn))
    held = (float(t[a[i]]), float(t[b[i]]))
    a, b = windows(length)
    turn = np.array([np.abs(np.diff(yaw[i : j + 1])).sum() for i, j in zip(a, b)])
    train = []
    for i in np.argsort(-turn, kind="stable"):
        lo, hi = float(t[a[i]]), float(t[b[i]])
        if all(hi < x - 0.4 or lo > y + 0.4 for x, y in train + [held]):
            train.append((lo, hi))
        if len(train) == 5:
            break
    if not train:
        raise ValueError(f"no {length:.0f} m of unbroken forward driving away from the scale check")
    return [(lo, hi, spacing) for lo, hi in sorted(train)] + [(*held, spacing)]


def segments(tl):
    """CALIB_SEGS picks the keyframe windows: unset keeps the hand-picked ones on HW4 footage and is "auto"
    on anything else; "auto" selects from telemetry; or a JSON list of [start s, end s, spacing m], scale
    check last, with at most five before it."""
    spec = os.getenv("CALIB_SEGS", "" if hardware.identify(CAMS, native_sizes()) == "hw4" else "auto")
    if not spec:
        return AUTHOR_SEGS
    segs = auto_segments(tl) if spec == "auto" else [tuple(s) for s in json.loads(spec)]
    if not 2 <= len(segs) <= 6:
        raise ValueError("CALIB_SEGS needs one to five training windows and then the scale check")
    return segs


def select():
    tl = np.load(DATA / "timeline.npz")
    rows = [json.load(open(DATA / "cams" / f"{c}.json")) for c in CAMS]
    ts = [np.array([r["t"] for r in rr]) for rr in rows]
    frames = []
    segs = segments(tl)
    print("keyframe windows", segs, flush=True)
    # The solver trains on seg < 5 and holds seg 5 out, however many training windows there are.
    for s, (a, b, spacing) in zip([*range(len(segs) - 1), 5], segs):
        last = None
        for i in np.flatnonzero((tl["t"] >= a) & (tl["t"] < b) & (tl["speed"] > 0.5)):
            xy = np.array([tl["x"][i], tl["y"][i]])
            t = tl["t"][i]
            if last is not None and np.linalg.norm(xy - last) < spacing:
                continue
            inds = [np.argmin(abs(tt - t)) for tt in ts]
            offsets = [float(tt[j] - t) for tt, j in zip(ts, inds)]
            if max(abs(x) for x in offsets) > 0.02:
                continue
            frames.append(
                dict(
                    t=float(t),
                    seg=s,
                    xy=xy.tolist(),
                    yaw=float(tl["yaw"][i]),
                    ns=[r[j]["n"] for r, j in zip(rows, inds)],
                    dt=offsets,
                )
            )
            last = xy
    W.mkdir(exist_ok=True, parents=True)
    json.dump(frames, open(W / "frames.json", "w"))
    return frames


def extract(item):
    k, c, n = item
    dest = W / "features" / f"{k:03d}_{c}.npz"
    if dest.exists():
        return
    im = cv.imread(str(DATA / "frames" / CAMS[c] / f"{n:06d}.jpg"), 0)
    mask = cv.imread(str(OUT / "masks" / f"{CAMS[c]}.png"), 0)
    # The 2896 px HW4 front image is reduced so every lens yields features at a similar pixel scale.
    scale = 0.72 if im.shape[1] > 2000 else 1.0
    im = cv.resize(im, None, fx=scale, fy=scale, interpolation=cv.INTER_AREA) if scale != 1 else im
    mask = cv.resize(mask, (im.shape[1], im.shape[0]), interpolation=cv.INTER_NEAREST)
    sift = cv.SIFT_create(
        nfeatures=3000, contrastThreshold=0.016, edgeThreshold=12, enable_precise_upscale=True
    )
    kp, ds = sift.detectAndCompute(im, mask)
    p = (np.array([k.pt for k in kp]) + 0.5) / scale  # COLMAP pixel centres
    uniq = np.sort(np.unique(np.round(p * 2).astype(int), return_index=True, axis=0)[1])
    ds = ds[uniq]
    ds = np.rint(255 * np.sqrt(ds / (ds.sum(1, keepdims=True) + 1e-9))).astype(np.uint8)
    np.savez(dest, p=p[uniq].astype("f"), d=ds)


def pairs(fr):
    # Candidate lens pairs with a shared view, among the cameras this car has. Image id = keyframe * NC + lens.
    adjacent = hardware.adjacent(CAMS)
    lagged = hardware.front_to_repeater(CAMS)
    out = set()
    for k, f in enumerate(fr):
        for step in [1, 2, 3, 5, 8]:
            j = k + step
            if j >= len(fr) or fr[j]["seg"] != f["seg"]:
                continue
            for c in range(NC):
                out.add((k * NC + c, j * NC + c))
        for l, r in adjacent:
            for step in [-10, -5, -2, 0, 2, 5, 10]:
                j = k + step
                if not 0 <= j < len(fr) or fr[j]["seg"] != f["seg"]:
                    continue
                out.add(tuple(sorted((k * NC + l, j * NC + r))))
        # No pillar camera on this side: the front lens and the repeater never see the same thing at once.
        for l, r in lagged:
            for step in hardware.FRONT_TO_REPEATER:
                j = k + step
                if j < len(fr) and fr[j]["seg"] == f["seg"]:
                    out.add((k * NC + l, j * NC + r))
    return sorted(out)


def match(fr, split=False):
    # Process one target at a time so each FLANN index is built only once.
    allpairs = pairs(fr)
    dest = W / ("matches-split" if split else "matches")
    if split:
        needed = set()
        for parity in (0, 1):
            ix = [k for k in range(parity, len(fr), 2) if fr[k]["seg"] < 5]
            ff = [fr[k] for k in ix]
            needed.update((ix[i // NC] * NC + i % NC, ix[j // NC] * NC + j % NC) for i, j in pairs(ff))
        allpairs = sorted(needed.difference(allpairs))
    groups = {}
    for i, j in allpairs:
        groups.setdefault(j, []).append(i)
    F = []
    for k in range(len(fr)):
        for c in range(NC):
            f = np.load(W / "features" / f"{k:03d}_{c}.npz")
            F.append((f["p"], f["d"].astype("f")))

    def group(item):
        j, ii = item
        fp = dest / f"{j:05d}.npz"
        if fp.exists():
            return
        idx = cv.flann_Index(F[j][1], dict(algorithm=1, trees=3))
        dic = {}
        for i in ii:
            matches, dist = idx.knnSearch(F[i][1], 2, params={"checks": 30})
            ids = np.flatnonzero(dist[:, 0] < (0.75 if i % NC == j % NC else 0.77) ** 2 * dist[:, 1])
            v = matches[ids, 0]
            if len(ids) < 9:
                continue
            # Local epipolar test on raw coordinates is intentionally permissive for fisheye; true geometry is tested after unprojection.
            _, mask = cv.findFundamentalMat(
                F[i][0][ids], F[j][0][v], cv.USAC_MAGSAC, 3.5, 0.997, 600
            )
            if mask is not None and mask.sum() >= 9:
                good = mask.ravel().astype(bool)
                dic[str(i)] = np.column_stack((ids[good], v[good])).astype(np.int32)
        np.savez(fp, **dic)

    dest.mkdir(exist_ok=True)
    list(ThreadPoolExecutor(14).map(group, groups.items()))


if __name__ == "__main__":
    if "--median" in sys.argv:
        median_sheets()
        sys.exit(0)
    if "--splits-only" in sys.argv:
        match(json.load(open(W / "frames.json")), True)
        sys.exit(0)
    masks()
    fr = select()
    print("keyframes per window", np.bincount([r["seg"] for r in fr]), flush=True)
    (W / "features").mkdir(exist_ok=True)
    list(
        ThreadPoolExecutor(12).map(
            extract, [(k, c, n) for k, f in enumerate(fr) for c, n in enumerate(f["ns"])]
        )
    )
    print("features complete", flush=True)
    match(fr)
    match(fr, True)
    print("matching complete", flush=True)
