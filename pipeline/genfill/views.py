"""Pinhole render sets for the generative fixer, with reference photographs.

real:  held-out exposures (the live-map hold set, which live-v4 never fused and whose depth
       windows it skipped; frames in the ground schedule are dropped), rendered as virtual
       pinhole crops with eight rolling-shutter row bands. The undistorted,
       photometry-corrected photo is ground truth and never a reference.
chase: the viewer's default chase camera, a pure function of drive time
       (viewer/src/camera/chasePose.ts): pivot 1.45 m ahead of and 0.8 m above the car, camera
       16.2 m from it (+0.4 % per m/s of smoothed speed, capped at 25 m/s) directly behind, 16.9 deg
       up, looking at the pivot and tilted up 5.3 deg; vertical field of view 50 deg. The az = 0
       poses are read from viewer/tools/chase-track.ts output (--track, the source of truth); orbits
       of the azimuth (+ = camera swung counter-clockwise seen from above, towards the car's right
       rear) use a Python replica of placeChase, checked against the track over the whole slice.
       One station every STEP metres of travel. Rendered at 1408x896 (1440x900 rounded to the
       64-pixel grid Difix needs).
Each view is rendered from the static tiles (ground + objects; moving-object tiles skipped) at a
playback time: the station or photo time for live scenes (what the viewer shows then), a fixed
time otherwise. The scene's sky panorama, if any, is drawn behind the splats as the viewer does.
Every view gets the training photographs (the stitch fit's views, never held-out ones) that see
most of what it renders, scored on rendered surface points.

Set layout (all paths relative to the set folder):
  views.json            views: id, kind, t, t_map, T_world_cam (4x4, OpenCV axes), K, size, refs
  render/<id>.png       RGB render, render/<id>.npz alpha (splat coverage) + z (metres, 0 = none)
  ref/<cam>_<n>.jpg     reference photo on that camera's pinhole design at the set size
  gt/<id>.png           real views only; gt/<id>-mask.png 1 static, 2 sky, 0 invalid/mover/body
"""

import argparse
import ctypes as ct
import json
import os
import time
from concurrent.futures import ThreadPoolExecutor
from functools import lru_cache
from pathlib import Path

import cv2 as cv
import numpy as np

from pipeline.common.hardware import roster
from pipeline.common.rig import load_rig, Trajectory, T_world_cam, undistort_maps
from pipeline.photo.model import Photometry
from pipeline.road.scene import reader

RS = Path(os.environ.get("RS", Path.home() / "dev/reality-splat")).resolve()
START, END = 85.0, 140.0
REAL_SIZE = (1024, 576)
CHASE_SIZE = (1408, 896)
CAMS = roster(RS)  # the cameras of the calibrated rig: six on HW4, four on HW3
FOV = {"front": 44.0, "back": 83.0}  # horizontal degrees, others 80 (as pipeline/stitch)
BANDS = 8
BG = np.array([22, 26, 30], "f")
CHASE = dict(dist=16.2, elev=16.9, tilt=5.3, pan=0.0, pivot_x=1.45, pivot_z=0.8, vfov=50.0, min_hfov=64.0)
KINDS = {"mid": 0.0, "orb-60": -60.0, "orb-30": -30.0, "orb+30": 30.0, "orb+60": 60.0}
SPLIT = RS / "work/stitch/views-85-140/manifest.json"
rig = load_rig(RS / "calib/rig.json")
cv.setNumThreads(2)


def J(p):
    return json.loads(Path(p).read_text())


def savej(p, a):
    Path(p).parent.mkdir(parents=True, exist_ok=True)
    tmp = Path(str(p) + ".tmp")
    tmp.write_text(json.dumps(a, indent=1))
    tmp.replace(p)


def unit(v):
    return v / np.linalg.norm(v, axis=-1, keepdims=True)


def K_of(fov, size):
    w, h = size
    f = w / 2 / np.tan(np.radians(fov) / 2)
    return np.array([[f, 0, w / 2], [0, f, h / 2], [0, 0, 1.0]])


def K_vfov(vfov, size):
    w, h = size
    f = h / 2 / np.tan(np.radians(vfov) / 2)
    return np.array([[f, 0, w / 2], [0, f, h / 2], [0, 0, 1.0]])


@lru_cache(None)
def design(c, size):
    C = rig[c]
    K = K_of(FOV.get(c, 80.0), size)
    mx, my, row = undistort_maps(C, K, size)
    valid = np.isfinite(row) & C.mask_at(np.stack([mx + 0.5, my + 0.5], -1))
    band = np.where(
        valid, np.minimum(BANDS - 1, np.floor(np.nan_to_num(row) * BANDS / C.height)), -1
    ).astype("i1")
    return K, mx, my, valid, band


def remap(im, mx, my, C, interp):
    h, w = im.shape[:2]
    return cv.remap(im, (mx + 0.5) * w / C.width - 0.5, (my + 0.5) * h / C.height - 0.5, interp,
                    borderMode=cv.BORDER_CONSTANT)


@lru_cache(None)
def photometry():
    return Photometry("stitched-ground-v2", RS)


def photo(c, n, size):
    """Undistorted, photometry-corrected RGB uint8 photo on the camera's pinhole design."""
    C = rig[c]
    im = cv.imread(str(RS / "data/drive/frames" / c / f"{n:06d}.jpg"),
                   cv.IMREAD_REDUCED_COLOR_2 if C.width >= 2896 else cv.IMREAD_COLOR)
    assert im is not None, (c, n)
    rgb = np.clip(photometry().apply(c, n, im[:, :, ::-1].astype("f")), 0, 255).astype("B")
    K, mx, my, valid, _ = design(c, size)
    out = remap(np.ascontiguousarray(rgb), mx, my, C, cv.INTER_CUBIC)
    out[~valid] = 0
    return out, valid


def photo_mask(c, n, size):
    """1 static, 2 sky (eroded), 0 invalid, body, sky edge or mover (dilated)."""
    C = rig[c]
    K, mx, my, valid, _ = design(c, size)
    sky, dyn = [cv.imread(str(RS / "masks" / k / c / f"{n:06d}.png"), cv.IMREAD_UNCHANGED) != 0
                for k in ("sky", "dyn")]
    dyn = remap(cv.dilate(dyn.astype("B"), np.ones((5, 5), "B")), mx, my, C, 0) > 0
    core = remap(cv.erode(sky.astype("B"), np.ones((5, 5), "B")), mx, my, C, 0) > 0
    edge = remap(cv.dilate(sky.astype("B"), np.ones((3, 3), "B")), mx, my, C, 0) > 0
    return np.where(valid & ~dyn, np.where(core, 2, np.where(edge, 0, 1)), 0).astype("B")


def rays(K, size):
    w, h = size
    y, x = np.mgrid[:h, :w]
    return np.stack([(x + 0.5 - K[0, 2]) / K[0, 0], (y + 0.5 - K[1, 2]) / K[1, 1], np.ones((h, w))], -1)


class Environment:
    """A scene's environment map (map.json "environment", pipeline/envmap): the sky at infinity, drawn as the
    viewer draws it (render/sky.ts): equirectangular, read from 0.9 deg up, under the horizon band of the
    haze colour (the map's horizon). Also its provenance (255 observed, 170 harmonised, 85 filled)."""

    def __init__(self, path):
        path = Path(path)
        e = J(path)
        self.img = cv.imread(str(path.parent / e["map"]))[:, :, ::-1].astype("f")
        self.prov = cv.imread(str(path.parent / e["provenance"]), cv.IMREAD_GRAYSCALE)
        self.horizon = np.array(e["horizon"], "f")
        self.band = float(e.get("band", 0.06))

    def canvas(self, T, K, size, prov=False):
        d = unit(rays(K, size) @ T[:3, :3].T)
        H, W = self.img.shape[:2]
        u = (0.5 - np.arctan2(d[..., 1], d[..., 0]) / (2 * np.pi)) % 1.0
        v = np.maximum(0.5 + np.arcsin(np.clip(d[..., 2], -1, 1)) / np.pi, 0.505)
        mx, my = (u * W - 0.5).astype("f"), ((1 - v) * H - 0.5).astype("f")
        img = cv.remap(self.img, mx, my, cv.INTER_LINEAR, borderMode=cv.BORDER_WRAP)
        s = np.clip(d[..., 2] / self.band, 0, 1)
        img = self.horizon + (s * s * (3 - 2 * s))[..., None] * (img - self.horizon)
        if not prov:
            return img
        return img, cv.remap(self.prov, mx, my, cv.INTER_NEAREST, borderMode=cv.BORDER_WRAP)


class Scene:
    """Static splats of a contract scene near the slice; rendered at any playback time."""

    def __init__(self, root, t_map=None, margin=140.0, t_range=(START, END)):
        self.root = (RS / root).resolve()
        self.live = "live" in self.root.name
        self.t_map = t_map  # None: per-view playback time
        meta = J(self.root / "map.json")
        self.drive = J(self.root / "drive.json")
        self.traj = Trajectory(RS / "poses" / ("ego_causal.npz" if self.live else "ego.npz"))
        lib = ct.CDLL(str(RS / "maps/object-v3/raster.dylib"))
        self.rast = lib.raster
        P, I = ct.c_void_p, ct.c_int
        self.rast.argtypes = [P, I, ct.c_float, I, P, P, I, I, I, P]
        self.sky = None
        if meta.get("sky"):
            pano = cv.imread(str(self.root / meta["sky"]))[:, :, ::-1].astype("f")
            self.sky = cv.GaussianBlur(pano, (0, 0), 3.0)  # the viewer reads it mip-blurred
        # an environment map replaces the panorama behind the splats (self.sky stays the crown-colour reference)
        self.env = Environment(self.root / meta["environment"]) if meta.get("environment") else None
        p = self.traj.T_world_ego(np.arange(t_range[0], t_range[1] + 0.01, 0.5))[:, :2, 3]
        parts = []
        for tile in meta["tiles"]:
            if tile["kind"] != "static":
                continue
            bb = np.array(tile["bbox"])
            d = np.maximum(np.maximum(bb[:2] - p, p - bb[3:5]), 0)
            if (d * d).sum(1).min() > margin**2:
                continue
            a = np.asarray(reader(self.root / tile["file"]))
            a = a[(a[:, 17] <= t_range[1] + 1) & (a[:, 18] > t_range[0] - 1)]
            layer = float(tile.get("layer") == "objects")
            parts.append(np.column_stack([a, np.full(len(a), layer, "f")]))
        pts = np.concatenate(parts).astype("f")
        cell = np.floor(pts[:, :2] / 10).astype(int)
        order = np.lexsort((cell[:, 1], cell[:, 0]))
        self.pts = np.ascontiguousarray(pts[order])
        keys, start, count = np.unique(cell[order], axis=0, return_index=True, return_counts=True)
        self.keys, self.span = keys, np.stack([start, start + count], 1)

    def when(self, t):
        return float(t if self.t_map is None else self.t_map)

    def state(self, t):
        d = self.drive
        yaw = np.unwrap(np.asarray(d["yaw"]))
        return [float(np.interp(t, d["t"], d[k])) for k in ("x", "y", "z")] + [
            float(np.interp(t, d["t"], yaw)), float(np.interp(t, d["t"], d["speed"]))]

    def local(self, o, radius=135.0):
        d = self.keys * 10 + 5 - o[:2]
        ok = (d * d).sum(1) < (radius + 8) ** 2
        return np.ascontiguousarray(np.concatenate([self.pts[a:b] for a, b in self.span[ok]]))

    def sky_canvas(self, T, K, size):
        if self.env is not None:
            return self.env.canvas(T, K, size)
        d = unit(rays(K, size) @ T[:3, :3].T)
        H, W = self.sky.shape[:2]
        u = (0.5 - np.arctan2(d[..., 1], d[..., 0]) / (2 * np.pi)) % 1.0
        v = np.maximum(0.5 + np.arcsin(np.clip(d[..., 2], -1, 1)) / np.pi, 0.505)
        return cv.remap(self.sky, (u * W - 0.5).astype("f"), ((1 - v) * H - 0.5).astype("f"),
                        cv.INTER_LINEAR, borderMode=cv.BORDER_WRAP)

    def render(self, T, K, size, t, pts=None, canvas=None):
        """Pinhole render from T_world_cam at playback time t. Returns rgb f32, alpha, z (0 = none).
        canvas: initial RGB (default: sky panorama or background); zeros give premultiplied colour."""
        w, h = size
        rows = np.ascontiguousarray(np.repeat(np.asarray(T, "d")[None], 33, 0))
        if pts is None:
            pts = self.local(rows[16, :3, 3])
        out = np.zeros((h, w, 6), "f")
        if canvas is not None:
            out[:, :, :3] = canvas
        else:
            out[:, :, :3] = BG if self.sky is None else self.sky_canvas(T, K, size)
        Kv = np.ascontiguousarray(np.r_[K[0, 0], K[1, 1], K[0, 2], K[1, 2], 0, 0, 0, 0], "d")
        self.rast(pts.ctypes.data, len(pts), self.when(t), int(self.live), rows.ctypes.data,
                  Kv.ctypes.data, 0, w, h, out.ctypes.data)
        alpha = out[:, :, 3]
        rng = np.where(alpha > 1e-3, out[:, :, 4] / np.maximum(alpha, 1e-3), 0)
        norm = np.linalg.norm(rays(K, size), axis=-1)
        return out[:, :, :3], alpha, (rng / norm).astype("f")

    def chase_inputs(self, T):
        """chasePose.ts chaseInputs for an array of drive times: car x, y; Gaussian-smoothed z and
        heading (lead 0.05 s); trailing exponential speed."""
        d = self.drive
        t = np.asarray(d["t"], "f8")
        end = float(d.get("duration", t[-1]))
        f = lambda k, q: np.interp(q, t, np.unwrap(d[k]) if k == "yaw" else np.asarray(d[k], "f8"))
        T = np.atleast_1d(np.asarray(T, "f8"))[:, None]
        dt = (np.arange(25) - 12) * 0.05
        wh, wz = np.exp(-0.5 * (dt / 0.25) ** 2), np.exp(-0.5 * (dt / 0.2) ** 2)
        h = (wh * f("yaw", np.clip(T + 0.05 + dt, 0, end))).sum(1) / wh.sum()
        z = (wz * f("z", np.clip(T + dt, 0, end))).sum(1) / wz.sum()
        lag = np.arange(9) * 0.3
        wv = np.exp(-lag / 1.2)
        v = (wv * np.maximum(0, f("speed", np.maximum(0, T - lag)))).sum(1) / wv.sum()
        return f("x", T[:, 0]), f("y", T[:, 0]), z, h, v

    def chase_T(self, t, az_deg=0.0, aspect=CHASE_SIZE[0] / CHASE_SIZE[1]):
        """placeChase (chasePose.ts) at drive time t with an azimuth offset; T_world_cam, OpenCV axes."""
        x, y, z, h, v = [a[0] for a in self.chase_inputs(t)]
        pivot = np.array([x + np.cos(h) * CHASE["pivot_x"], y + np.sin(h) * CHASE["pivot_x"], z + CHASE["pivot_z"]])
        a, e = h + np.pi + np.radians(az_deg), np.radians(CHASE["elev"])
        D = CHASE["dist"] * (1 if aspect >= 1 else 0.78) * (1 + 0.004 * min(25.0, v))
        pos = pivot + D * np.array([np.cos(a) * np.cos(e), np.sin(a) * np.cos(e), np.sin(e)])
        Z = unit(pos - pivot)
        X = unit(np.cross([0, 0, 1.0], Z))
        R = np.stack([X, np.cross(Z, X), Z], 1)  # three.js camera axes: x right, y up, z back
        p, q = np.radians(CHASE["tilt"]), np.radians(-CHASE["pan"])
        Rx = np.array([[1, 0, 0], [0, np.cos(p), -np.sin(p)], [0, np.sin(p), np.cos(p)]])
        Ry = np.array([[np.cos(q), 0, np.sin(q)], [0, 1, 0], [-np.sin(q), 0, np.cos(q)]])
        T = np.eye(4)
        T[:3, :3] = R @ Rx @ Ry @ np.diag([1.0, -1.0, -1.0])
        T[:3, 3] = pos
        return T

    def stations(self, t0=START, t1=END, step=2.0):
        t = np.arange(t0, t1, 0.01)
        p = np.stack([np.interp(t, self.drive["t"], self.drive[k]) for k in ("x", "y")], 1)
        s = np.r_[0, np.cumsum(np.linalg.norm(np.diff(p, axis=0), axis=1))]
        marks = np.arange(0, s[-1], step)
        return np.interp(marks, s, t), marks


def chase_fov(aspect):
    vmin = np.degrees(2 * np.arctan(np.tan(np.radians(CHASE["min_hfov"]) / 2) / aspect))
    return max(CHASE["vfov"], vmin) if aspect >= 1 else min(74.0, max(CHASE["vfov"], vmin))


def load_track(path):
    """viewer/tools/chase-track.ts output: times, T_world_cam (OpenCV axes), vfov, size."""
    j = J(path)
    ts = np.array([f["t"] for f in j["frames"]])
    Ts = np.repeat(np.eye(4)[None], len(ts), 0)
    for i, f in enumerate(j["frames"]):
        x, y, z, w = np.array(f["quat"]) / np.linalg.norm(f["quat"])
        R = np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
                      [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
                      [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]])
        Ts[i, :3, :3] = R @ np.diag([1.0, -1.0, -1.0])
        Ts[i, :3, 3] = f["pos"]
    return ts, Ts, float(j["vfov"]), (int(j["width"]), int(j["height"]))


def track_check(scene, ts, Ts, aspect):
    """Largest position (m) and orientation (deg) gap between the replica and the track."""
    dp, da = 0.0, 0.0
    for t, T in zip(ts[::5], Ts[::5]):
        R = scene.chase_T(t, 0.0, aspect)
        dp = max(dp, float(np.linalg.norm(R[:3, 3] - T[:3, 3])))
        for j in range(3):  # axis by axis; arccos of the trace is too coarse near zero
            a, b = R[:3, j], T[:3, j]
            da = max(da, float(np.degrees(np.arctan2(np.linalg.norm(np.cross(a, b)), a @ b))))
    return dp, da


def real_render(scene, e, size):
    """Pinhole crop of a physical exposure; each native-row band uses its own row pose."""
    C = rig[e["c"]]
    K, mx, my, valid, band = design(e["c"], size)
    T = T_world_cam(C, e["t"], C.height / 2, scene.traj)
    pts = scene.local(T[:3, 3])
    rgb, alpha, z = scene.render(T, K, size, e["t"], pts)
    for b in range(BANDS):
        m = band == b
        if m.any():
            Tb = T_world_cam(C, e["t"], C.height * (b + 0.5) / BANDS, scene.traj)
            r, a, d = scene.render(Tb, K, size, e["t"], pts)
            rgb[m], alpha[m], z[m] = r[m], a[m], d[m]
    return T, K, rgb, alpha, z


@lru_cache(None)
def candidates(traj_name):
    """Training photographs of the stitch fit (never held-out) and their centre-row poses."""
    traj = Trajectory(RS / "poses" / traj_name)
    es = [e for e in J(SPLIT) if not e["hold"] and e["yaw"] == 0]
    poses = np.stack([T_world_cam(rig[e["c"]], e["t"], None, traj) for e in es])
    return es, poses


def choose_refs(scene, T, z, K, size, top=3, away_from=None, min_dist=0.0, seed=0):
    """Score training photos by how much of this view's rendered surface they see, from a
    similar direction and distance. Candidates whose camera centre lies within min_dist of
    away_from are skipped. Returns the top entries with their scores."""
    w, h = size
    rng = np.random.default_rng(seed)
    yy, xx = np.nonzero(z > 0.3)
    if len(yy) < 50:
        return []
    pick = rng.choice(len(yy), min(1500, len(yy)), replace=False)
    yy, xx = yy[pick], xx[pick]
    ray = np.stack([(xx + 0.5 - K[0, 2]) / K[0, 0], (yy + 0.5 - K[1, 2]) / K[1, 1], np.ones(len(xx))], 1)
    X = (ray * z[yy, xx, None]) @ T[:3, :3].T + T[:3, 3]
    a = X - T[:3, 3]
    da = np.linalg.norm(a, axis=1)
    a /= da[:, None]
    es, poses = candidates("ego_causal.npz" if scene.live else "ego.npz")
    far = np.ones(len(es), bool) if away_from is None else (
        np.linalg.norm(poses[:, :3, 3] - away_from, axis=1) >= min_dist)
    out = []
    for c in CAMS:
        idx = np.array([i for i, e in enumerate(es) if e["c"] == c and far[i]])
        if not len(idx):
            continue
        Kc, _, _, valid, _ = design(c, size)
        P = poses[idx]
        v = np.einsum("kji,knj->kni", P[:, :3, :3], X[None] - P[:, None, :3, 3])
        zc = v[..., 2]
        u = Kc[0, 0] * v[..., 0] / np.maximum(zc, 1e-3) + Kc[0, 2]
        q = Kc[1, 1] * v[..., 1] / np.maximum(zc, 1e-3) + Kc[1, 2]
        ins = (zc > 0.5) & (u >= 0) & (u < w) & (q >= 0) & (q < h)
        ins &= valid[np.clip(q, 0, h - 1).astype(int), np.clip(u, 0, w - 1).astype(int)]
        b = X[None] - P[:, None, :3, 3]
        db = np.linalg.norm(b, axis=-1)
        cos = np.clip((a[None] * b).sum(-1) / db, 0, 1)
        ratio = np.minimum(da, db) / np.maximum(da, db)
        s = (ins * cos**4 * ratio).mean(1)
        for i in np.argsort(-s)[:top]:
            e = es[idx[i]]
            out.append(dict(c=c, n=e["n"], t=e["t"], score=float(s[i]), seen=float(ins[i].mean()),
                            dist=float(np.linalg.norm(P[i, :3, 3] - T[:3, 3]))))
    return sorted(out, key=lambda r: -r["score"])[:top]


def ensure_ref(folder, r, size):
    p = folder / "ref" / f"{r['c']}_{r['n']:06d}.jpg"
    if not p.exists():
        im, _ = photo(r["c"], r["n"], size)
        p.parent.mkdir(parents=True, exist_ok=True)
        cv.imwrite(str(p), im[:, :, ::-1], [cv.IMWRITE_JPEG_QUALITY, 95])
    return str(p.relative_to(folder))


def write_view(folder, vid, rgb, alpha, z):
    (folder / "render").mkdir(parents=True, exist_ok=True)
    cv.imwrite(str(folder / "render" / f"{vid}.png"), np.clip(rgb + 0.5, 0, 255).astype("B")[:, :, ::-1])
    np.savez_compressed(folder / "render" / f"{vid}.npz", alpha=alpha.astype("f2"), z=z.astype("f2"))


def held_frames(per_cam):
    """Live-map hold set (never fused by live-v4) inside the slice, minus ground-scheduled frames."""
    plan = J(RS / "maps/object-v3/plan.json")
    ground = np.load(RS / "maps/ground-v2/schedule.npz")["ns"]
    out = []
    for c in CAMS:
        used = set(ground[:, ["front", "left_pillar", "right_pillar", "left_repeater", "right_repeater",
                              "back"].index(c)].tolist())
        ee = sorted([dict(c=c, n=e["n"], t=e["t"]) for e in plan[c]
                     if e["hold"] and START <= e["t"] <= END and e["n"] not in used], key=lambda e: e["t"])
        out += [ee[int(i)] for i in np.linspace(0, len(ee) - 1, min(per_cam, len(ee))).round()]
    return out


def make_real(scene_root, out, per_cam=10, workers=6, ref_away=2.0, t_map=None, size=REAL_SIZE):
    folder = Path(out)
    scene = Scene(scene_root, t_map)

    def one(e):
        vid = f"real-{e['c']}-{e['n']:06d}"
        T, K, rgb, alpha, z = real_render(scene, e, size)
        write_view(folder, vid, rgb, alpha, z)
        gt, _ = photo(e["c"], e["n"], size)
        (folder / "gt").mkdir(parents=True, exist_ok=True)
        cv.imwrite(str(folder / "gt" / f"{vid}.png"), gt[:, :, ::-1])
        cv.imwrite(str(folder / "gt" / f"{vid}-mask.png"), photo_mask(e["c"], e["n"], size))
        # A neighbouring exposure is nearly the ground truth, so references must be at least
        # ref_away metres from this camera.
        refs = choose_refs(scene, T, z, K, size, away_from=T[:3, 3], min_dist=ref_away)
        for r in refs:
            r["path"] = ensure_ref(folder, r, size)
        return dict(id=vid, kind="real", c=e["c"], n=e["n"], t=e["t"], t_map=scene.when(e["t"]),
                    T_world_cam=T.tolist(), K=K.tolist(), size=list(size), refs=refs,
                    gt=f"gt/{vid}.png", coverage=float((alpha > 0.5).mean()))

    tic = time.monotonic()
    with ThreadPoolExecutor(workers) as ex:
        views = list(ex.map(one, held_frames(per_cam)))
    savej(folder / "views.json", dict(scene=str(scene.root), size=list(size), ref_away=ref_away, views=views))
    print(len(views), "real views", round(time.monotonic() - tic, 1), "s", flush=True)


def make_chase(scene_root, out, track, t0=START, t1=END, step=2.0, kinds=tuple(KINDS), workers=8,
               t_map=None):
    folder = Path(out)
    scene = Scene(scene_root, t_map)
    tt, TT, vfov, size = load_track(track)
    assert size[0] % 64 == 0 and size[1] % 64 == 0, size
    aspect = size[0] / size[1]
    assert abs(vfov - chase_fov(aspect)) < 1e-3, (vfov, chase_fov(aspect))
    K = K_vfov(vfov, size)
    sel = (tt >= t0 - 0.5) & (tt <= t1 + 0.5)
    dp, da = track_check(scene, tt[sel], TT[sel], aspect)
    print(f"replica vs chase-track: max {dp * 1000:.2f} mm, {da:.4f} deg", flush=True)
    assert dp < 0.01 and da < 0.05, (dp, da)
    ts, marks = scene.stations(t0, t1, step)
    old = {v["id"]: v for v in J(folder / "views.json")["views"]} if (folder / "views.json").exists() else {}
    jobs = []
    for k, (t, m) in enumerate(zip(ts, marks)):
        i = int(np.argmin(np.abs(tt - t)))  # the track frame nearest the 2 m mark
        jobs += [(k, i, float(m), kind) for kind in kinds]

    def one(job):
        k, i, m, kind = job
        t = float(tt[i])
        vid = f"chase-{k:04d}-{kind}"
        if vid in old and (folder / "render" / f"{vid}.png").exists():
            return old[vid]
        T = TT[i] if KINDS[kind] == 0 else scene.chase_T(t, KINDS[kind], aspect)
        rgb, alpha, z = scene.render(T, K, size, t)
        write_view(folder, vid, rgb, alpha, z)
        refs = choose_refs(scene, T, z, K, size)
        for r in refs:
            r["path"] = ensure_ref(folder, r, size)
        return dict(id=vid, kind="chase", station=k, chase=kind, t=t, s=m, orbit=KINDS[kind],
                    pose="chase-track.ts" if KINDS[kind] == 0 else "chasePose replica",
                    t_map=scene.when(t), T_world_cam=T.tolist(), K=K.tolist(), size=list(size),
                    refs=refs, coverage=float((alpha > 0.5).mean()))

    tic = time.monotonic()
    with ThreadPoolExecutor(workers) as ex:
        views = list(ex.map(one, jobs))
    savej(folder / "views.json", dict(scene=str(scene.root), size=list(size), step=step, chase=CHASE,
                                      track=str(track), replica_gap_m_deg=[dp, da], kinds=list(kinds),
                                      views=views))
    print(len(views), "chase views", round(time.monotonic() - tic, 1), "s", flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("set", choices=["real", "chase"])
    ap.add_argument("--scene", default="scene/live-v4")
    ap.add_argument("--out", required=True)
    ap.add_argument("--t-map", type=float, default=None, help="fixed playback time (default: per view)")
    ap.add_argument("--per-cam", type=int, default=10)
    ap.add_argument("--ref-away", type=float, default=2.0)
    ap.add_argument("--t0", type=float, default=START)
    ap.add_argument("--t1", type=float, default=END)
    ap.add_argument("--step", type=float, default=2.0)
    ap.add_argument("--kinds", default=",".join(KINDS))
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--track", default=str(RS / "work/genfill/chase-track-live-v4.json"),
                    help="viewer/tools/chase-track.ts output for the scene (fps 100, 1408x896)")
    a = ap.parse_args()
    if a.set == "real":
        make_real(a.scene, RS / a.out, a.per_cam, a.workers, a.ref_away, a.t_map)
    else:
        make_chase(a.scene, RS / a.out, a.track, a.t0, a.t1, a.step, tuple(a.kinds.split(",")), a.workers,
                   a.t_map)
