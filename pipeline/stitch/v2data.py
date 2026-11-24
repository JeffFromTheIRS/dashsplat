"""Physical offline exposures, calibrated ranges, high-resolution rectified targets and free space."""

from .views import (
    RS,
    START,
    END,
    CAMS,
    rig,
    traj,
    photo,
    J,
    savej,
    remap,
    designs,
    T_world_cam,
    undistort_maps,
)
from pipeline.common.hardware import reduction
from functools import lru_cache
from pathlib import Path
import numpy as np, cv2 as cv, os, json
from pipeline.road.scene import reader, binary_header
from scipy.spatial import cKDTree

V = Path(os.getenv("STITCH_V2_WORK", RS / "work/stitch-v2")).resolve()
V.mkdir(exist_ok=True, parents=True)
GROUND = RS / "scene/road-stitched-v2"
cv.setNumThreads(1)


def schedule():
    p = V / "exposures.json"
    if p.exists():
        return J(p)
    raw = J(RS / "depth/keyframes.json")["frames"]
    out = {}
    counts = {}
    for c in CAMS:
        ee = [e for e in raw[c] if START <= e["t_key"] <= END]
        hold = {e["n"] for e in ee[7::8]}
        dirs = [RS / "maps/object-v3/stitched/depth" / c, RS / "depth" / c]
        idx = [
            J(d / "index.json") if (d / "index.json").is_file() else {} for d in dirs
        ]
        rows = []
        bad = 0
        avail = 0
        for i, e in enumerate(ee):
            n = e["n"]
            depth = None
            seen = False
            for d, l in zip(dirs, idx):
                f = d / ("%06d.npy" % n)
                r = l.get(str(n), {})
                if not f.is_file():
                    continue
                seen = True
                if (
                    n not in hold
                    and r.get("posed")
                    and r.get("model") == "da3-large-1.1"
                    and r.get("window")
                    and not (set(r["window"]) & hold)
                ):
                    depth = str(f)
                    break
            avail += seen
            bad += seen and not depth
            rows.append(dict(**e, c=c, i=i, hold=n in hold, depth=depth))
        out[c] = rows
        counts[c] = dict(
            keyframes=len(rows),
            available=int(avail),
            depth_used=sum(bool(e["depth"]) for e in rows),
            held=len(hold),
            excluded_depth=int(bad),
        )
    savej(p, out)
    savej(V / "input-counts.json", counts)
    return out


@lru_cache(None)
def calibration(c):
    aud = J(RS / "maps/object-v3/stitched/audit.json")[c]
    m = J(Path(os.getenv("STITCH_DEPTH_MEASUREMENTS", RS / "pipeline/livemap/measurements.json")))["depth"]["stitched"][c]
    return aud["a"], aud["b"], m


@lru_cache(None)
def rays(c):
    C = rig[c]
    y, x = np.mgrid[:469, :724]
    uv = np.stack([(x + 0.5) * C.width / 724, (y + 0.5) * C.height / 469], 2)
    return np.nan_to_num(C.unproject(uv)).astype("f"), C.mask_at(uv)


def physical_range(c, z):
    ray, ok = rays(c)
    ang = np.rad2deg(np.arccos(np.clip(ray[:, :, 2], -1, 1)))
    a, b, m = calibration(c)
    r = z / np.maximum(ray[:, :, 2], 0.04)
    den = np.clip(a + b * np.minimum(r, 40), 0.55, 1.55)
    r = r / den
    # Measured periphery under-range is severe but uncertain; smoothly correct, reduce confidence rather than discard the rear rim.
    rim = float(m["outer_rim_frac"].get("median", 0))
    start = m["rim_start_deg"]
    fac = np.clip((ang - start + 4) / 12, 0, 1)
    r /= np.clip(1 + min(0, rim) * fac, 0.53, 1)
    w = np.clip(1 / (1 + (r / 40) ** 1.3), 0.09, 1) * (
        1 - 0.65 * fac if c == "back" else 1
    )
    w = w * np.where(r > 10, 0.23, 0.55) if c == "right_repeater" else w
    good = (z > 0) & ok & (ray[:, :, 2] > 0.036) & (r >= 1.0) & (r <= 210)
    return np.where(good, r, 0).astype("f"), np.where(good, w, 0).astype("f")


def masks(c, n):
    return [
        cv.imread(str(Path(os.getenv("STITCH_MASKS", RS / "masks")) / k / c / ("%06d.png" % n)), cv.IMREAD_UNCHANGED)
        for k in ["sky", "dyn", "road"]
    ]


@lru_cache(36)
def rgb(c, n, hi=False):
    # About 724 px across for every lens; `hi` gives the front lens up to 1448 px from its full frame.
    small = {1: cv.IMREAD_COLOR, 2: cv.IMREAD_REDUCED_COLOR_2, 4: cv.IMREAD_REDUCED_COLOR_4}[
        reduction(rig[c].width, 724)
    ]
    flag = cv.IMREAD_COLOR if hi and c == "front" else small
    a = cv.imread(str(RS / "data/drive/frames" / c / ("%06d.jpg" % n)), flag)
    assert a is not None
    if hi and c == "front" and a.shape[1] > 1448:
        a = cv.resize(a, (1448, round(a.shape[0] * 1448 / a.shape[1])), interpolation=cv.INTER_AREA)
    return photo.apply(c, n, a[:, :, ::-1])


@lru_cache(None)
def pin(c, yaw):
    w, h = (1152, 744) if c == "front" else (640, 416)
    r = np.deg2rad(yaw)
    R = np.array([[np.cos(r), 0, np.sin(r)], [0, 1, 0], [-np.sin(r), 0, np.cos(r)]])
    f = w / (2 * np.tan(np.deg2rad({"front": 44, "back": 83}.get(c, 80)) / 2))
    K = np.array([[f, 0, w / 2], [0, f, h / 2], [0, 0, 1]])
    C = rig[c]
    mx, my, row = undistort_maps(C, K, (w, h), R)
    valid = np.isfinite(row) & C.mask_at(np.stack([mx + 0.5, my + 0.5], 2))
    bands = np.where(
        valid, np.clip(np.nan_to_num(row) * 8 / C.height, 0, 7).astype("i"), -1
    )
    ys, xs = np.mgrid[:h, :w]
    v = np.stack([(xs + 0.5 - w / 2) / f, (ys + 0.5 - h / 2) / f, np.ones((h, w))], 2)
    rz = np.linalg.norm(v, axis=2)
    return w, h, f, R, mx, my, bands.astype("i1"), rz.astype("f")


@lru_cache(28)
def target(c, n, yaw, dep):
    w, h, f, R, mx, my, bands, rz = pin(c, yaw)
    C = rig[c]
    im = remap(rgb(c, n, True), mx, my, C, cv.INTER_LINEAR)
    sky, dyn, road = masks(c, n)
    valid = (bands >= 0) & (
        remap(cv.dilate((dyn != 0).astype("B"), np.ones((5, 5), "B")), mx, my, C, 0)
        == 0
    )
    sk = remap((sky != 0).astype("B"), mx, my, C, 0) != 0
    ro = remap((road != 0).astype("B"), mx, my, C, 0) != 0
    label = np.where(valid, np.where(sk, 2, np.where(ro, 3, 1)), 0).astype("B")
    z = wt = None
    if dep:
        rr, ww = physical_range(c, np.load(dep).astype("f"))
        z = remap(rr, mx, my, C, 0) / rz
        wt = remap(ww, mx, my, C, 0)
        wt[(label != 1) | (z <= 0)] = 0
    return im, label, z, wt


def poses(c, t):
    return np.ascontiguousarray(
        T_world_cam(rig[c], t, (np.arange(8) + 0.5) * rig[c].height / 8, traj), "d"
    )


def tiles_ground():
    # Match v1's ground coverage for the slice, but link the immutable *original* mosaic.
    files = {
        t["file"]
        for t in J(Path(os.getenv("STITCH_GROUND_SELECTION", RS / "scene/stitched-opt-v1/map.json")))["tiles"]
        if t["layer"] == "ground"
    }
    return [
        t
        for t in J(GROUND / "map.json")["tiles"]
        if t["layer"] == "ground" and t["file"] in files
    ]


class Volume:
    def __init__(self):
        p = V / "free-grid.npz"
        step = 0.4
        if not p.exists():
            T = traj.T_world_ego(np.arange(START - 2, END + 2, 0.05))
            xy = T[:, :2, 3]
            origin = np.floor((xy.min(0) - 230) / step) * step
            dims = np.ceil((xy.max(0) + 230 - origin) / step).astype(int)
            ys, xs = np.mgrid[: dims[1], : dims[0]]
            q = origin + (np.stack([xs, ys], 2) + 0.5) * step
            ds, k = cKDTree(xy).query(q.reshape(-1, 2))
            zz = T[k, 2, 3].reshape(xs.shape).astype("f")
            owned = np.zeros_like(zz, "B")
            hull = np.full_like(zz, -1e5)
            for t in tiles_ground():
                a = reader(GROUND / t["file"])
                ij = np.floor((a[:, :2] - origin) / step).astype(int)
                good = (
                    (ij[:, 0] >= 0)
                    & (ij[:, 1] >= 0)
                    & (ij[:, 0] < dims[0])
                    & (ij[:, 1] < dims[1])
                )
                i, j = ij[good].T
                zz[j, i] = a[good, 2]
                owned[j, i] = 1
            for M in T:
                ij = np.floor((M[:2, 3] - origin) / step).astype(int)
                b = 13
                sl = np.s_[ij[1] - b : ij[1] + b + 1, ij[0] - b : ij[0] + b + 1]
                v = q[sl] - M[:2, 3]
                dx = v @ M[:2, 0]
                dy = v @ M[:2, 1]
                sel = (dx >= -1.0) & (dx <= 3.85) & (abs(dy) < 1.13)
                hull[sl][sel] = np.maximum(hull[sl][sel], M[2, 3] + 1.72)
            np.savez(
                p,
                origin=origin.astype("f"),
                step=step,
                ground=zz,
                hull=hull,
                owned=owned,
            )
        a = np.load(p)
        self.origin = a["origin"]
        self.step = float(a["step"])
        self.g = a["ground"]
        self.h = a["hull"]
        self.owned = a["owned"]

    def lookup(self, p):
        ij = np.clip(
            np.floor((p[:, :2] - self.origin) / self.step).astype("i"),
            [0, 0],
            [self.g.shape[1] - 1, self.g.shape[0] - 1],
        )
        i, j = ij.T
        return self.g[j, i], self.h[j, i], self.owned[j, i]

    def floor(self, p):
        g, h, o = self.lookup(p)
        return np.maximum(g + np.where(o, 0.085, 0.015), h)


@lru_cache(24)
def cached_range(c, n):
    """Recover range only on observed pinhole pixels from the original immutable training targets."""
    ray, ok = rays(c)
    rr = np.zeros(ok.shape, "f")
    ww = rr.copy()
    for yaw in designs(c):
        p = V / "targets" / c / f"{n:06d}_{yaw}.npz"
        w, h, f, R, _, _, _, _ = pin(c, yaw)
        v = ray @ R
        rz = v[:, :, 2]
        mx = (w / 2 + f * v[:, :, 0] / np.maximum(rz, 0.01) - 0.5).astype("f")
        my = (h / 2 + f * v[:, :, 1] / np.maximum(rz, 0.01) - 0.5).astype("f")
        with np.load(p) as a:
            z = a["z"]
            wt = a["wt"]
        if not z.size:
            continue
        d = cv.remap(z.astype("f"), mx, my, cv.INTER_NEAREST) / np.maximum(rz, 0.01)
        q = cv.remap(wt.astype("f") / 255, mx, my, cv.INTER_NEAREST)
        take = (q > ww) & ok & (rz > 0.08)
        rr[take] = d[take]
        ww[take] = q[take]
    return rr, ww


def exposure_range(e):
    path = Path(e["depth"])
    if not (V / "train.json").exists():
        return physical_range(e["c"], np.load(path).astype("f"))
    rr, ww = cached_range(e["c"], e["n"])
    return rr.copy(), ww.copy()
