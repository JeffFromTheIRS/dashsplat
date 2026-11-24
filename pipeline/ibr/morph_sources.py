"""Task M: full-rate ordered footage and source-attached, two-surface inverse-depth proxies."""

from pathlib import Path
from functools import lru_cache
import json, os
import numpy as np, cv2 as cv
from scipy.spatial.transform import Rotation
from scipy.ndimage import gaussian_filter1d, distance_transform_edt
from pipeline.livemap.geometry import RS, rig, lattice, traj
from pipeline.photo.model import Photometry

W = RS / "work/ibr/task-m"
HOST = Path(os.getenv("IBR_HOST", RS / "scene/stitched-opt-v3"))
SOURCE_START = float(os.getenv("IBR_SOURCE_START", 0))
SOURCE_END = float(os.getenv("IBR_SOURCE_END", 143))
TR = traj("stitched")
PHOTO = Photometry("stitched-ground-v2", RS)
# Forward-looking source lenses; a car without pillar cameras (HW3) has the front alone.
CAMS = [c for c in ("front", "left_pillar", "right_pillar") if c in rig]
GH, GW = 96, 148
OH, OW = 900, 1440
H, L = 180, 288
F = 450 / np.tan(np.deg2rad(25.0))
cv.setNumThreads(1)


def J(p):
    return json.loads(p.read_text())


def savej(p, data):
    tmp = p.with_suffix(".new")
    tmp.write_text(json.dumps(data, separators=(",", ":"), allow_nan=False))
    os.replace(tmp, p)


def smooth(x):
    x = np.clip(x, 0, 1)
    return x * x * (3.0 - 2 * x)


def resize(x, w, h):
    return cv.resize(x, (w, h), interpolation=cv.INTER_LINEAR)


def blur(x, s):
    return cv.GaussianBlur(x, (0, 0), s)


def remap(x, p):
    return cv.remap(
        x,
        np.ascontiguousarray(p, "f"),
        None,
        cv.INTER_LINEAR,
        borderMode=cv.BORDER_REPLICATE,
    )


def coord(w, h):
    y, x = np.mgrid[:h, :w].astype("f")
    return np.stack([x, y], -1)


XY = coord(L, H)
RAY = np.concatenate(
    [(XY + 0.5 - [L / 2, H / 2]) / (F * L / OW), np.ones((H, L, 1))], 2
).astype("f")
GRID = coord(GW, GH)
KEYS = J(RS / "depth/keyframes.json")["frames"]
drive = J(HOST / "drive.json")
ds = np.r_[0, np.cumsum(np.hypot(np.diff(drive["x"]), np.diff(drive["y"])))]
dt = drive["t"]


@lru_cache(None)
def seq(c):
    a = [e for e in J(RS / "data/drive/cams" / f"{c}.json") if SOURCE_START <= e["t"] <= SOURCE_END]
    ts = np.array([e["t"] for e in a])
    n = np.array([e["n"] for e in a])
    C = rig[c]
    T = TR.T_world_ego(C.row_time(ts, C.height * 0.5)) @ C.T_ego_cam
    return ts, n, T, np.interp(ts, dt, ds)


@lru_cache(None)
def kseq(c):
    a = [
        e
        for e in KEYS[c]
        if SOURCE_START <= e["t"] < SOURCE_END
        and (RS / "masks/sky" / c / f"{e['n']:06}.png").exists()
        and (RS / "depth" / c / f"{e['n']:06}.npy").exists()
    ]
    return np.array([e["t"] for e in a]), np.array([e["n"] for e in a])


@lru_cache(240)
def key(c, j):
    ts, ns = kseq(c)
    n = ns[j]
    et = ts[j]
    C = rig[c]
    sz = (GW, GH)
    fn = f"{n:06}"
    load = lambda typ: cv.imread(str(RS / "masks" / typ / c / (fn + ".png")), -1)
    road = resize((load("road") > 0).astype("f"), GW, GH)
    sky = resize((load("sky") > 0).astype("f"), GW, GH)
    dyn = cv.dilate(resize((load("dyn") > 0).astype("f"), GW, GH), np.ones((3, 3), "B"))
    dep = cv.resize(
        np.load(RS / "depth" / c / (fn + ".npy")).astype("f"),
        sz,
        interpolation=cv.INTER_NEAREST,
    )
    ray, valid = lattice(c)
    valid = resize(valid.astype("f"), GW, GH)
    ray = resize(ray, GW, GH)
    T = TR.T_world_ego(C.row_time(et, C.height / 2)) @ C.T_ego_cam
    E = TR.T_world_ego(et)
    normal, height = TR.ground_plane(et)
    normal = E[:3, :3] @ normal
    dz = (T[:3, 3] - E[:3, 3]) @ normal + height
    gInv = np.maximum(
        0,
        -(ray @ (T[:3, :3].T @ normal)) / np.maximum(0.2, ray[:, :, 2]) / max(0.65, dz),
    ).astype("f")
    raw = 1 / np.maximum(3.0, dep)
    ok = (
        (dep > 2)
        & (dep < 450)
        & (sky < 0.6)
        & (road < 0.6)
        & (dyn < 0.2)
        & (valid > 0.9)
    )
    far = ok & (dep > 22)
    ground = road > 0.5
    # Anchor near-object relief to its observed foot when the road is visible immediately below it.
    near = ok & (dep < 26)
    ncc, lab, st, _ = cv.connectedComponentsWithStats(near.astype("B"), 8)
    rel = raw.copy()
    for k in range(1, ncc):
        x, y, w, h, size = st[k]
        if size < 5:
            continue
        sub = lab[y : y + h, x : x + w] == k
        foot = np.argmax(sub[::-1], axis=0)
        yy = y + h - 1 - foot
        xx = np.arange(x, x + w)
        v = sub.any(0) & (yy < GH - 3) & (road[np.minimum(yy + 2, GH - 1), xx] > 0.55)
        d = np.median(raw[y : y + h, x : x + w][sub])
        g = np.median(gInv[yy[v], xx[v]]) if v.sum() >= 2 else d
        if 0.7 * d < g < 1.45 * d:
            rel[y : y + h, x : x + w][sub] = (
                0.84 * rel[y : y + h, x : x + w][sub] + 0.16 * g
            )
    fg = blur(near.astype("f"), 0.72) * smooth((28 - np.maximum(3, dep)) / 5)
    fg *= 1 - dyn
    # Continuous background behind near objects: farther same-exposure samples seed a harmonic-like extension.
    inv = np.where(far, raw, 0).astype("f")
    weight = far.astype("f")
    sc = np.full(dep.shape, 1 / 85, "f")
    for s in [13, 5, 1.4]:
        den = blur(weight, s)
        v = blur(inv, s) / np.maximum(0.002, den)
        a = smooth(den / 0.32)
        sc = sc + (v - sc) * a
    sc = cv.bilateralFilter(sc.astype("f"), 7, 0.010, 3)
    bg = (1 - sky) * ((1 - road) * sc + road * gInv)
    fginv = blur(np.where(near, rel, 0).astype("f"), 1.4) / np.maximum(
        0.02, blur(near.astype("f"), 1.4)
    )
    fginv = np.where(fginv > 0.035, fginv, np.maximum(sc, 1 / 18))
    return T.astype("f"), np.dstack([bg, fginv, fg, dyn, road, sky, valid]).astype("f")


def proj(c, v, w, h):
    C = rig[c]
    rho = np.hypot(v[:, :, 0], v[:, :, 1])
    ang = np.arctan2(rho, v[:, :, 2])
    t2 = ang * ang
    a, b, d, e = C.params[4:]
    f = ang * (1 + t2 * (a + t2 * (b + t2 * (d + t2 * e)))) / np.maximum(1e-6, rho)
    return np.stack(
        [
            (C.params[0] * v[:, :, 0] * f + C.params[2]) * w / C.width - 0.5,
            (C.params[1] * v[:, :, 1] * f + C.params[3]) * h / C.height - 0.5,
        ],
        -1,
    ).astype("f")


@lru_cache(100)
def source(c, i):
    ts, ns, poses, _ = seq(c)
    t = ts[i]
    n = ns[i]
    T = poses[i].astype("f")
    kt, _ = kseq(c)
    j = np.clip(np.searchsorted(kt, t) - 1, 0, len(kt) - 2)
    f = np.clip((t - kt[j]) / (kt[j + 1] - kt[j]), 0, 1)
    ray = resize(lattice(c)[0], GW, GH)
    ray /= np.maximum(0.08, ray[:, :, 2:3])
    fields = []
    for m in [j, j + 1]:
        TK, A = key(c, m)
        dr = ray @ (TK[:3, :3].T @ T[:3, :3]).T
        off = (T[:3, 3] - TK[:3, 3]) @ TK[:3, :3]
        p = proj(c, dr, GW, GH)
        init = remap(A, p)
        b = proj(c, dr + off * init[:, :, 0:1], GW, GH)
        u = proj(c, dr + off * init[:, :, 1:2], GW, GH)
        v = remap(A, b)
        fg = remap(A, u)
        v[:, :, 1:4] = fg[:, :, 1:4]
        v[:, :, 0] = v[:, :, 0] * dr[:, :, 2] / np.maximum(0.5, 1 - off[2] * v[:, :, 0])
        v[:, :, 1] = v[:, :, 1] * dr[:, :, 2] / np.maximum(0.5, 1 - off[2] * v[:, :, 1])
        fields.append(v)
    a = fields[0] * (1 - f) + fields[1] * f
    a[:, :, 3] = np.maximum(
        fields[0][:, :, 3] * (1 - smooth(f - 0.65)),
        fields[1][:, :, 3] * (1 - smooth(0.35 - f)),
    )
    im = cv.imread(
        str(RS / "data/drive/frames" / c / f"{n:06}.jpg"),
        cv.IMREAD_REDUCED_COLOR_2 if rig[c].width >= 2896 else cv.IMREAD_COLOR,
    )
    assert im is not None
    im = np.ascontiguousarray(PHOTO.apply(c, int(n), im[:, :, ::-1])[:, :, ::-1])
    return T, a.astype("f"), im


def track(a, b):
    j = J(W / f"chase-{a}-{b}.json")
    assert (j["width"], j["height"], j["fps"], j["vfov"]) == (OW, OH, 30, 50)
    out = []
    for e in j["frames"]:
        T = np.eye(4)
        T[:3, :3] = Rotation.from_quat(e["quat"]).as_matrix() @ np.diag([1, -1, -1])
        T[:3, 3] = e["pos"]
        out.append((e["t"], T.astype("f")))
    return out


def choose(frames):
    # Project the eye onto every acquisition segment near its route location; choose one continuous clock.
    ts, _, T, S = seq("front")
    xy = T[:, :2, 3]
    dd = np.diff(xy, axis=0)
    length = (dd * dd).sum(1)
    raw = []
    for t, v in frames:
        st = np.interp(t, dt, ds)
        ids = np.where((S[:-1] >= st - 52) & (S[:-1] <= st + 1) & (ts[:-1] <= t))[0]
        k = np.clip(
            ((v[:2, 3] - xy[ids]) * dd[ids]).sum(1) / np.maximum(1e-8, length[ids]),
            0,
            1,
        )
        dist = np.linalg.norm(xy[ids] + k[:, None] * dd[ids] - v[:2, 3], axis=1)
        best = np.argmin(dist + 0.04 * abs(S[ids] - st + 16) + 0.001 * abs(ts[ids] - t))
        j = ids[best]
        raw.append(S[j] * (1 - k[best]) + S[j + 1] * k[best])
    base = np.maximum.accumulate(gaussian_filter1d(raw, 2.0, mode="nearest"))
    return base


@lru_cache(None)
def turn_plan(a, b):
    frames = track(a, b)
    ar = choose(frames)
    by = {}
    for c in CAMS:
        ts, ns, poses, arc = seq(c)
        out = []
        cred = []
        for (t, T), s in zip(frames, ar):
            sel = np.flatnonzero(
                (ts <= t) & (ts >= max(0, t - 35)) & (abs(arc - s) < 40)
            )
            if not len(sel):
                out.append(s)
                cred.append(0.0)
                continue
            dist = np.linalg.norm(poses[sel, :2, 3] - T[:2, 3], axis=1)
            angle = np.rad2deg(np.arccos(np.clip(poses[sel, :3, 2] @ T[:3, 2], -1, 1)))
            j = np.argmin(0.75 * dist + 0.65 * angle)
            out.append(arc[sel[j]])
            cred.append(
                float(
                    (1 - smooth((angle[j] - 14) / 19))
                    * (1 - smooth((dist[j] - 9) / 13))
                )
            )
        ss = np.maximum.accumulate(gaussian_filter1d(out, 4, mode="nearest"))
        by[c] = (ss, gaussian_filter1d(cred, 4, mode="nearest"))
    return by
