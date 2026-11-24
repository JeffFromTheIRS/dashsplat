"""Static photographic ground. Coarse grid is exclusively ownership/evidence; colour is always resampled from native JPEG."""

import argparse, hashlib, json, os, time
from pathlib import Path
from functools import lru_cache
from concurrent.futures import ThreadPoolExecutor
import cv2 as cv
import numpy as np
from PIL import Image
from scipy.spatial import cKDTree
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import maximum_flow, breadth_first_order
from pipeline.common.hardware import roster
from pipeline.common.rig import Trajectory, load_rig, frame_table
from pipeline.photo.model import Photometry, basis
from pipeline.road.geometry import ground_to_pixels
from pipeline.road.scene import reader

C = 0.04
N = 256
M = C * N
G = 8
W = Path(os.getenv("GT_WORK", "work/groundtex"))
OUT = Path(os.getenv("GT_OUT", "scene/stitched-opt-v4/ground"))
ROAD = Path("scene/road-stitched-v2")
# Source order, among the cameras this car has.
CAMS = [
    c
    for c in (
        "front",
        "left_pillar",
        "right_pillar",
        "back",
        "left_repeater",
        "right_repeater",
    )
    if c in roster()
]
cv.setNumThreads(1)


def dump(p, v):
    tmp = p.with_suffix(".tmp")
    tmp.write_text(json.dumps(v, separators=(",", ":"), allow_nan=False))
    tmp.replace(p)


def read(p):
    return json.loads(Path(p).read_text())


@lru_cache(1)
def inputs():
    tr = Trajectory()
    rig = load_rig()
    ph = Photometry("stitched-ground-v2")
    return tr, rig, ph


@lru_cache(6)
def table(cam):
    return frame_table(cam)


@lru_cache(32)
def files(cam, kind):
    path = (
        Path("maps/movers-v1/remove") if kind == "remove" else Path("masks") / kind
    ) / cam
    return set(int(p.stem) for p in path.glob("*.png"))


def schedule(t0, t1):
    tr, rig, _ = inputs()
    arc = np.r_[0, np.cumsum(np.linalg.norm(np.diff(tr.p[:, :2], axis=0), axis=1))]
    ex = read("work/stitch-v2/exposures.json")
    rows = []
    for cam in CAMS:
        hold = {e["n"] for e in ex[cam] if e["hold"]}
        road = files(cam, "road")
        mov = files(cam, "remove")
        ds = 1.5 if cam == "front" else 3.5
        pool = {}
        for e in table(cam):
            n, t = e["n"], e["t"]
            if not t0 <= t <= t1 or n in hold or n % 29 == 7:
                continue
            s = np.interp(t, tr.t, arc)
            k = int(s / ds)
            qual = (
                (n in road) * 3
                + (n in mov) * 0.45
                - abs(s / ds - k - 0.5)
                - 0.01 * abs(e.get("ax", 0))
            )
            if k not in pool or qual > pool[k][0]:
                pool[k] = (qual, dict(cam=cam, n=n, t=t, arc=round(s, 4)))
        rows.extend(v[1] for v in pool.values())
    return rows


def slices(box):
    x, y, w, h = box
    for j in range(y // N, (y + h - 1) // N + 1):
        for i in range(x // N, (x + w - 1) // N + 1):
            X, Y = max(x, i * N), max(y, j * N)
            u, v = min(x + w, (i + 1) * N), min(y + h, (j + 1) * N)
            yield (
                (i, j),
                np.s_[Y - j * N : v - j * N, X - i * N : u - i * N],
                np.s_[Y - y : v - y, X - x : u - x],
            )


D = np.dtype([("rgb", "u1", 3), ("q", "f2"), ("id", "i4"), ("ok", "u1")])


class Canvas:
    def __init__(self):
        self.tiles = {}
        self.dirty = set()

    def get(self, box):
        a = np.zeros((box[3], box[2]), D)
        for k, s, b in slices(box):
            if k in self.tiles:
                a[b] = self.tiles[k][s]
        return a

    def put(self, box, a):
        for k, s, b in slices(box):
            if not a[b]["id"].any():
                continue
            if k not in self.tiles:
                self.tiles[k] = np.zeros((N, N), D)
            if not np.array_equal(self.tiles[k][s], a[b]):
                self.tiles[k][s] = a[b]
                self.dirty.add(k)


@lru_cache(1)
def height_model():
    tr = inputs()[0]
    arc = np.r_[0, np.cumsum(np.linalg.norm(np.diff(tr.p[:, :2], axis=0), axis=1))]
    ts = np.interp(np.arange(0, arc[-1], 0.55), arc, tr.t)
    T = tr.T_world_ego(ts)
    n, h = tr.ground_plane(ts)
    norm = np.einsum("nij,nj->ni", T[:, :3, :3], n)
    p = T[:, :3, 3] - norm * h[:, None]
    return cKDTree(p[:, :2]), p, norm


def height(xy):
    tree, p, n = height_model()
    d, k = tree.query(xy, k=6)
    z = p[k, 2] - np.sum((xy[..., None, :] - p[k, :2]) * n[k, :2], axis=-1) / n[k, 2]
    wt = 1 / (0.15 + d * d)
    return (z * wt).sum(-1) / wt.sum(-1)


@lru_cache(1)
def roadtiles():
    return read(ROAD / "map.json")["tiles"]


@lru_cache(100)
def support(k):
    # Geometry prior only. Use small, near-horizontal discs validated by the ground builder;
    # reject coarse distant guesses. Never fetch their RGB into the atlas.
    i, j = k
    x = i * M
    y = j * M
    pad = 5
    h = np.zeros((N + 2 * pad, N + 2 * pad), "B")
    for e in roadtiles():
        a, b, _, A, B, _ = e["bbox"]
        if a > x + M + 0.2 or b > y + M + 0.2 or A < x - 0.2 or B < y - 0.2:
            continue
        q = reader(ROAD / e["file"])
        sel = (
            (abs(q[:, 0] - x - M / 2) < M / 2 + 0.20)
            & (abs(q[:, 1] - y - M / 2) < M / 2 + 0.20)
            & (q[:, 5] > 0.91)
            & (q[:, 10] < np.log(0.17))
            & (q[:, 11] < np.log(0.20))
        )
        xy = q[sel, :2]
        ix = np.floor((xy - [x, y]) / C).astype(int) + pad
        ix = ix.clip(0, len(h) - 1)
        h[ix[:, 1], ix[:, 0]] = 255
    h = cv.dilate(h, cv.getStructuringElement(cv.MORPH_ELLIPSE, (7, 7)))
    return h[pad:-pad, pad:-pad]


def footprint(r):
    tr, rig, _ = inputs()
    cam = rig[r["cam"]]
    T = tr.T_world_ego(r["t"]) @ cam.T_ego_cam
    xy = T[:2, 3]
    f = T[:2, 2]
    f = f / np.linalg.norm(f)
    s = np.array([-f[1], f[0]])
    ra = 18.0 if cam.name == "front" else (8.5 if cam.name == "back" else 14.0)
    half = 0.45 if cam.name == "front" else (1.8 if cam.name == "back" else 0.90)
    pts = np.array(
        [
            xy + s * b + a * f
            for a, b in [
                (0, -2),
                (0, 2),
                (ra, -min(ra * half, 13)),
                (ra, min(ra * half, 13)),
            ]
        ]
    )
    lo = np.floor(pts.min(0) / (0.16)).astype(int) * 4 - 4
    hi = np.ceil(pts.max(0) / 0.16).astype(int) * 4 + 4
    return tuple(map(int, [*lo, *(hi - lo)])), xy, ra


@lru_cache(48)
def field(path):
    f = np.load(path)
    return f["xy"].copy(), f["d"].astype("f"), f["gain"].astype("f")


def localfield(r, xy, kind):
    origin, delta, gain = field(r["field"])
    return cv.remap(
        delta if kind == "d" else gain,
        ((xy - origin) / 0.32).astype("f"),
        None,
        cv.INTER_LINEAR,
        borderMode=cv.BORDER_REPLICATE,
    )


@lru_cache(2)
def worldfield(path):
    with np.load(path) as f:
        return f["origin"], f["delta"].astype("f"), float(f["pitch"])


def corrected_xy(r, xy):
    if "worldfit" in r:
        origin, delta, pitch = worldfield(r["worldfit"])
        xy = xy + cv.remap(
            delta, ((xy - origin) / pitch).astype("f"), None, cv.INTER_LINEAR
        )
    for region in r.get("patch_shifts", []):
        centre, axis, half, delta = map(np.asarray, region)
        d = np.max(abs((xy - centre) @ axis) / half, axis=-1)
        v = np.clip((1.4 - d) / 0.35, 0, 1)
        xy = xy + delta * (v * v * (3 - 2 * v))[..., None]
    if "field" in r:
        xy = xy + localfield(r, xy, "d")
    A = np.array(r.get("A", [[1, 0, 0], [0, 1, 0]]))
    pivot = np.array(r.get("pivot", [0, 0]))
    return (xy - pivot) @ A[:, :2].T + pivot + A[:, 2]


def uvgrid(r, origin, size, cell, step):
    # The outer projection lattice sits on actual texel centres (including a halo),
    # so interpolation does not clamp near block boundaries.
    tr, rig, ph = inputs()
    cam = rig[r["cam"]]
    w, h = size
    xx = np.arange(-step, w + step, step)
    yy = np.arange(-step, h + step, step)
    x, y = np.meshgrid(xx, yy)
    xy = origin + (np.stack((x, y), -1) + 0.5) * cell
    xy = corrected_xy(r, xy)
    uv, _, _ = ground_to_pixels(cam, tr, r["t"], xy, 40)
    uu = uv.astype("f")
    xx, yy = np.meshgrid(
        np.arange(w, dtype="f") / step + 1, np.arange(h, dtype="f") / step + 1
    )
    uv = cv.remap(uu, xx, yy, cv.INTER_LINEAR)
    return uv


def sam(im, uv, cam, near=False):
    uv = uv * np.array([im.shape[1] / cam.width, im.shape[0] / cam.height], "f") - 0.5
    return cv.remap(
        im,
        uv,
        None,
        cv.INTER_NEAREST if near else cv.INTER_LINEAR,
        borderMode=cv.BORDER_CONSTANT,
    )


@lru_cache(20)
def image(c, n, half):
    return cv.imread(
        f"data/drive/frames/{c}/{n:06d}.jpg",
        cv.IMREAD_REDUCED_COLOR_2 if half else cv.IMREAD_COLOR,
    )


@lru_cache(24)
def gainmap(c, n):
    ph = inputs()[2]
    i, g, v = ph.params(c, n)
    x, y = np.meshgrid((np.arange(65) + 0.5) / 65, (np.arange(49) + 0.5) / 49)
    gain = np.exp(
        np.clip(g + basis(np.stack([x, y], -1).reshape(-1, 2)) @ v, -0.65, 0.65)
    ).reshape(49, 65, 3)
    return gain[:, :, ::-1].astype("f")


def photo(r, uv, half):
    c = r["cam"]
    n = r["n"]
    cam = inputs()[1][c]
    raw = sam(image(c, n, half), uv, cam)
    gain = sam(
        gainmap(c, n),
        uv.clip(
            [cam.width / 130, cam.height / 98],
            [cam.width * (1 - 1 / 130), cam.height * (1 - 1 / 98)],
        ).astype("f"),
        cam,
    )
    return np.uint8(
        np.clip(raw * gain * np.array(r.get("gain", [1, 1, 1])) + 0.5, 0, 255)
    )


def coarse(r):
    box, pos, ra = footprint(r)
    x, y, w, h = box
    c = r["cam"]
    cam = inputs()[1][c]
    origin = np.array([x, y]) * C
    uv = uvgrid(r, origin, (w, h), C, 4)
    bgr = photo(r, uv, True)
    valid = cam.mask_at(uv)
    n = r["n"]
    sem = np.zeros((h, w), "B")
    for kind in ["road", "sky", "dyn", "remove"]:
        if n not in files(c, kind):
            continue
        root = (
            Path("maps/movers-v1/remove") if kind == "remove" else Path("masks") / kind
        )
        raw = cv.imread(str(root / c / f"{n:06d}.png"), -1)
        m = sam(np.uint8(raw > 0), uv, cam, True)
        if kind == "road":
            sem = m
        else:
            valid &= cv.dilate(m, np.ones((5, 5), "B")) == 0
    xx, yy = np.meshgrid(
        (np.arange(w) + 0.5) * C + origin[0], (np.arange(h) + 0.5) * C + origin[1]
    )
    d = np.hypot(xx - pos[0], yy - pos[1])
    valid &= (d < ra) & (d > 1.5)
    prior = np.zeros((h, w), "B")
    for k, s, b in slices(box):
        prior[b] = support(k)[s]
    valid &= (prior > 0) | (sem > 0)
    edge = cv.distanceTransform(valid.astype("B"), cv.DIST_L2, 3)
    valid &= edge > 2
    # Smallest singular value measures pixels per ground metre in the worst direction.
    dx = np.gradient(uv, axis=1) / C
    dy = np.gradient(uv, axis=0) / C
    a = (dx**2).sum(2)
    b = (dy**2).sum(2)
    off = (dx * dy).sum(2)
    smin = np.sqrt(np.maximum(0, (a + b - np.hypot(a - b, 2 * off)) * 0.5))
    grey = cv.cvtColor(bgr, cv.COLOR_BGR2GRAY)
    hf = abs(grey.astype("f") - cv.GaussianBlur(grey, (5, 5), 1.1))
    sharp = np.percentile(hf[valid], 85) if valid.any() else 0
    # Native FOV, foreshortening, actual detail, masks, and distance from invalid pixels.
    q = (
        (4.8 if c == "front" else 1.0)
        * np.clip((smin / 95) ** 0.55, 0.14, 1.9)
        * np.clip(edge / 18, 0.3, 1)
        * (0.88 + 0.12 * sem)
        * np.clip(0.93 + 0.025 * sharp, 0.93, 1.12)
    )
    q = np.where(valid, q, 0).astype("f")
    return box, bgr, q, sem


def registration(old, rgb, valid):
    # Track nominal-ground corners between the two ortho strips, with a backward check.
    h, w = valid.shape
    mask = cv.erode(valid.astype("B"), np.ones((11, 11), "B")) * 255
    null = np.eye(2, 3, dtype="f")
    if (mask > 0).sum() < 160:
        return null, dict(matches=0)
    a = cv.cvtColor(old, cv.COLOR_BGR2GRAY)
    b = cv.cvtColor(rgb, cv.COLOR_BGR2GRAY)
    p = cv.goodFeaturesToTrack(a, 190, 0.012, 8, mask=mask, blockSize=5)
    if p is None or len(p) < 5:
        return null, dict(matches=0)
    kw = dict(
        winSize=(19, 19),
        maxLevel=2,
        criteria=(cv.TERM_CRITERIA_EPS | cv.TERM_CRITERIA_COUNT, 18, 0.03),
    )
    q, ok, _ = cv.calcOpticalFlowPyrLK(a, b, p, None, **kw)
    rev, ok2, _ = cv.calcOpticalFlowPyrLK(b, a, q, None, **kw)
    sel = (
        (ok[:, 0] > 0)
        & (ok2[:, 0] > 0)
        & (np.linalg.norm(rev - p, axis=2)[:, 0] < 0.85)
        & (np.linalg.norm(q - p, axis=2)[:, 0] < 11)
    )
    p = p[sel, 0]
    q = q[sel, 0]
    stats = dict(matches=len(p))
    if len(p) < 5:
        return null, stats
    A, m = cv.estimateAffinePartial2D(
        p, q, method=cv.RANSAC, ransacReprojThreshold=1.20, maxIters=300
    )
    if A is None:
        return null, stats
    inl = m[:, 0] > 0
    pivot = np.array([w / 2, h / 2])
    shift = A[:, :2] @ pivot + A[:, 2] - pivot
    if inl.sum() < 5 or inl.mean() < 0.32:
        return null, stats
    if (
        np.linalg.norm(A[:, :2] - np.eye(2)) > 0.032
        or np.ptp(p[inl], axis=0).min() < 12
    ):
        delta = np.median(q[inl] - p[inl], axis=0)
        A = null.copy()
        A[:, 2] = delta
        shift = delta
    if np.linalg.norm(shift) > 10:
        return null, stats
    residual = np.linalg.norm(p[inl] @ A[:, :2].T + A[:, 2] - q[inl], axis=1)
    stats.update(
        inliers=int(inl.sum()),
        before_cm=round(
            float(np.median(np.linalg.norm(p[inl] - q[inl], axis=1)) * 4), 3
        ),
        after_cm=round(float(np.median(residual) * 4), 3),
    )
    return A.astype("f"), stats


def seam(rgb, old, q, oq):
    # Binary s/t cut at 8 cm. Unary cost is relative resolution. Pairwise cut cost is
    # high over paint, kerbs and unmatched edges; near-identical asphalt is inexpensive.
    f = 2
    H, W = q.shape
    sm = lambda x: cv.resize(x, (W // f, H // f), interpolation=cv.INTER_AREA)
    qn, qo = sm(q), sm(oq)
    new = qn > qo
    overlap = (qn > 0) & (qo > 0)
    if not overlap.any():
        return q > oq
    ratio = (qn + 0.0001) / (qo + 0.0001)
    band = overlap & (ratio > 0.40) & (ratio < 2.55)
    idx = np.full(new.shape, -1, "i")
    yy, xx = np.nonzero(band)
    n = len(xx)
    if n:
        idx[yy, xx] = np.arange(n)
        a = sm(rgb).astype("f")
        b = sm(old).astype("f")
        gray = lambda im: cv.cvtColor(im, cv.COLOR_BGR2GRAY)
        g = np.maximum(
            cv.morphologyEx(gray(a), cv.MORPH_GRADIENT, np.ones((3, 3), "B")),
            cv.morphologyEx(gray(b), cv.MORPH_GRADIENT, np.ones((3, 3), "B")),
        )
        diff = np.mean(abs(a - b), 2)
        penalty = 25 + 2.2 * g + 3 * diff
        ids = np.arange(n)
        unary = 55 * np.log(ratio[yy, xx])
        src = np.maximum(0, unary)
        sink = np.maximum(0, -unary)
        ii = []
        jj = []
        vv = []
        for dy, dx in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
            Y = np.clip(yy + dy, 0, new.shape[0] - 1)
            X = np.clip(xx + dx, 0, new.shape[1] - 1)
            other = idx[Y, X]
            weight = (penalty[yy, xx] + penalty[Y, X]) * 0.5
            free = other >= 0
            ii.append(ids[free])
            jj.append(other[free])
            vv.append(weight[free])
            src += weight * (~free & new[Y, X])
            sink += weight * (~free & ~new[Y, X] & (qo[Y, X] > 0))
        ii.extend([np.full(n, n), ids])
        jj.extend([ids, np.full(n, n + 1)])
        vv.extend([src, sink])
        mat = coo_matrix(
            (np.concatenate(vv).astype("i"), (np.concatenate(ii), np.concatenate(jj))),
            shape=(n + 2, n + 2),
        ).tocsr()
        flow = maximum_flow(mat, n, n + 1).flow
        res = mat - flow
        res.eliminate_zeros()
        reach = breadth_first_order(res, n, return_predecessors=False)
        keep = np.zeros(n, bool)
        keep[reach[reach < n]] = 1
        new[yy, xx] = keep
    return (
        (cv.resize(new.astype("B"), (W, H), interpolation=cv.INTER_NEAREST) > 0)
        | (oq == 0)
    ) & (q > 0)


def integrate(canvas, obs, r, chunk):
    box, rgb, q, sem = chunk
    x, y, w, h = box
    old = canvas.get(box)
    oq = old["q"].astype("f")
    ov = (oq > 0) & (q > 0)
    A, stats = registration(old["rgb"], rgb, ov)
    pivot = np.array([w / 2, h / 2])
    shift = A[:, :2] @ pivot + A[:, 2] - pivot
    r["pivot"] = list((np.array([x, y]) + pivot + 0.5) * C)
    r["A"] = np.c_[A[:, :2], shift * C].tolist()
    r["registration"] = stats
    warp = lambda im, flag: cv.warpAffine(
        im, A, (w, h), flags=flag | cv.WARP_INVERSE_MAP
    )
    rgb = warp(rgb, cv.INTER_LINEAR)
    q = warp(q, cv.INTER_NEAREST)
    sem = warp(sem, cv.INTER_NEAREST)
    ov = (oq > 0) & (q > 0)
    gray = cv.cvtColor(rgb, cv.COLOR_BGR2GRAY)
    valid = (
        ov
        & (gray > 35)
        & (gray < 190)
        & (cv.morphologyEx(gray, cv.MORPH_GRADIENT, np.ones((3, 3), "B")) < 14)
    )
    if valid.sum() > 75:
        gain = np.clip(
            np.median((old["rgb"][valid][::4] + 1.0) / (rgb[valid][::4] + 1.0), axis=0),
            0.90,
            1.10,
        )
        r["gain"] = (0.82 * gain + 0.18).tolist()
        rgb = np.uint8(np.clip(rgb * np.array(r["gain"]) + 0.5, 0, 255))
    # Agreeing second observation also admits flat non-road regions. Semantic road
    # already carries the fitted ground prior or comes from the actual exposed pixels.
    agreement = ov & (np.max(abs(old["rgb"].astype("i2") - rgb), 2) < 15)
    agreement = cv.blur(agreement.astype("f"), (5, 5)) > 0.72
    sel = seam(rgb, old["rgb"], q, oq)
    sok = (sem > 0) | agreement
    existing = old["ok"].astype(bool)
    sel &= sok | ~existing
    old["ok"][agreement] = 1
    old["id"][sel] = len(obs) + 1
    old["rgb"][sel] = rgb[sel]
    old["q"][sel] = q[sel]
    old["ok"][sel] = sok[sel]
    canvas.put(box, old)
    obs.append(r)


def export_tile(k, canvas, obs, edge=None, guard=None, grade=None):
    i, j = k  # Lower detail for tiles with no narrow-camera photograph.
    ids = np.unique(canvas.tiles[k]["id"])
    fine = any(obs[id - 1]["cam"] == "front" for id in ids if id)
    z = 4 if fine else 2
    cell = C / z
    pix = N * z
    g = G
    origin = np.array([i, j]) * M - g * cell
    size = pix + g * 2
    lc = int(np.ceil(g / z)) + (24 if edge else 4)
    a = canvas.get((i * N - lc, j * N - lc, N + 2 * lc, N + 2 * lc))
    valid = (a["ok"] > 0) & (a["id"] > 0)
    inside = valid[lc : lc + N, lc : lc + N]
    if not inside.any():
        return None
    # A shared coarse signed-distance field softens only the verified support edge.
    di = cv.distanceTransform(valid.astype("B"), cv.DIST_L2, 5)
    do = cv.distanceTransform((~valid).astype("B"), cv.DIST_L2, 5)
    alpha = np.clip((di - do - 0.5) / 2, 0, 1)
    if edge is not None:
        alpha = edge(a, valid, (i * N - lc, j * N - lc), di, do)
    crop = lc * z - g
    up = lambda a, method: cv.resize(
        a, (a.shape[1] * z, a.shape[0] * z), interpolation=method
    )[crop : crop + size, crop : crop + size]
    alpha = np.uint8(255 * up(alpha, cv.INTER_LINEAR))
    labels = up(a["id"].astype("f"), cv.INTER_NEAREST).astype("i")
    alpha[labels == 0] = 0
    out = np.zeros((size, size, 4), "B")
    out[:, :, 3] = alpha
    for id in np.unique(labels[alpha > 0]):
        ys, xs = np.nonzero((labels == id) & (alpha > 0))
        x, y = xs.min(), ys.min()
        ww, hh = xs.max() - x + 1, ys.max() - y + 1
        r = obs[id - 1]
        uv = uvgrid(
            r, origin + np.array([x, y]) * cell, (ww, hh), cell, 4 if fine else 2
        )
        rgb = photo(r, uv, False)
        if "field" in r:
            Y, X = np.mgrid[y : y + hh, x : x + ww]
            xy = origin + (np.stack((X, Y), -1) + 0.5) * cell
            rgb = np.uint8(np.clip(rgb * localfield(r, xy, "gain") + 0.5, 0, 255))
        out[ys, xs, :3] = rgb[ys - y, xs - x, ::-1]
        if guard is not None:
            accepted = guard(r, uv, ys - y, xs - x, k)
            out[ys[~accepted], xs[~accepted]] = 0
    if grade is not None:
        grade(out, k, origin, cell)
    import io

    bio = io.BytesIO()
    Image.fromarray(out).save(bio, "WEBP", quality=92, method=3, exact=True)
    data = bio.getvalue()
    name = f"tiles/{i}_{j}_{hashlib.sha256(data).hexdigest()[:12]}.webp"
    path = OUT / name
    if not path.exists():
        path.write_bytes(data)
    axis = np.linspace(0, M, 9)
    xx, yy = np.meshgrid(i * M + axis, j * M + axis)
    zs = height(np.stack((xx, yy), -1))
    return dict(
        id=f"{i}_{j}",
        bounds=np.round([i * M, j * M, (i + 1) * M, (j + 1) * M], 5).tolist(),
        pixels=pix,
        gutter=g,
        texelSize=cell,
        texture=name,
        bytes=len(data),
        grid=dict(rows=9, cols=9, z=zs.round(5).ravel().tolist()),
    )


def run(t0, t1, workers):
    W.mkdir(parents=True, exist_ok=True)
    (W / "state").mkdir(exist_ok=True)
    (OUT / "tiles").mkdir(parents=True, exist_ok=True)
    canvas = Canvas()
    obs = []
    manifest = {}
    phases = []
    if (W / "observations.json").exists():
        obs = read(W / "observations.json")
        prior = read(OUT / "ground.json")
        phases = prior["intervals"]
        manifest = {v["id"]: v for v in prior["tiles"]}
        for p in (W / "state").glob("*.npy"):
            canvas.tiles[tuple(map(int, p.stem.split("_")))] = np.load(p)
    seen = {(r["cam"], r["n"]) for r in obs}
    rows = [r for r in schedule(t0, t1) if (r["cam"], r["n"]) not in seen]
    print("start", t0, t1, "sources", len(rows), flush=True)
    ts = time.monotonic()
    with ThreadPoolExecutor(workers) as ex:
        for j in range(0, len(rows), workers * 2):
            batch = rows[j : j + workers * 2]
            for r, chunk in zip(batch, ex.map(coarse, batch)):
                integrate(canvas, obs, r, chunk)
            if j % (workers * 24) == 0:
                print(
                    "source",
                    j,
                    rows[j]["cam"],
                    len(canvas.tiles),
                    "elapsed",
                    round(time.monotonic() - ts),
                    flush=True,
                )
    dirty = {
        k
        for i, j in canvas.dirty
        for k in [(i + dx, j + dy) for dx in (-1, 0, 1) for dy in (-1, 0, 1)]
        if k in canvas.tiles
    }
    print("export", len(dirty), flush=True)
    with ThreadPoolExecutor(workers) as ex:
        for item in ex.map(lambda k: export_tile(k, canvas, obs), sorted(dirty)):
            if item:
                manifest[item["id"]] = item
    for i, j in canvas.dirty:
        np.save(W / "state" / f"{i}_{j}.npy", canvas.tiles[i, j])
    dump(W / "observations.json", obs)
    phases.append([t0, t1])
    tiles = sorted(manifest.values(), key=lambda e: e["id"])
    dump(
        OUT / "ground.json",
        dict(
            format="reality-ground",
            version=1,
            colour="camera-rgb-code",
            alpha="straight-coverage",
            mips="viewer-generated",
            hideGroundSplats=dict(
                layer="ground", opaqueAlpha=0.98, heightTolerance=0.20
            ),
            intervals=phases,
            tiles=tiles,
        ),
    )
    print(
        "published",
        len(tiles),
        sum(t["bytes"] for t in tiles),
        "elapsed",
        round(time.monotonic() - ts),
        flush=True,
    )


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("start", type=float)
    p.add_argument("end", type=float)
    p.add_argument("--workers", type=int, default=4)
    a = p.parse_args()
    run(a.start, a.end, a.workers)
