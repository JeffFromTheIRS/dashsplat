"""A generated chase clip for the viewer's Generative mode (viewer/README.md).

render  (.venv)  default chase view from viewer/tools/chase-track.ts poses: sky panorama, ground splats,
                 the photographic ground atlas (scene/<s>/ground, sampled with pipeline.groundtex.proof's
                 Atlas, 4 anisotropic taps), object splats depth-tested against the ground, optional tracked
                 movers (objects.json) at their interpolated poses. No ego car. The 1440x900 track is
                 rendered on a 1472x960 canvas with the same focal length (Difix needs multiples of 64)
                 and cropped back. Reference photo per frame with hysteresis.
fix     (genfill env)  Difix-ref every frame, fixed seed; optional feed-forward of the previous repaired
                 frame warped through the render's depth (into the input and/or the output, only where
                 the warp is visible and agrees with the render); the atlas road is put back wherever the
                 atlas is opaque.
flicker          frame-to-frame difference after warping by depth, and after DIS optical flow.
encode           H.264 clip (yuv420p, keyframe every 0.5 s) and generated.json entries.
"""

import argparse
import json
import os
import subprocess
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import cv2 as cv
import numpy as np

RS = Path(os.environ.get("RS", Path.home() / "dev/reality-splat")).resolve()
PAD = (1472, 960)


def J(p):
    return json.loads(Path(p).read_text())


def track(path):
    j = J(path)
    w, h = j["width"], j["height"]
    f = h / 2 / np.tan(np.radians(j["vfov"]) / 2)
    K = np.array([[f, 0, PAD[0] / 2], [0, f, PAD[1] / 2], [0, 0, 1.0]])
    out = []
    for fr in j["frames"]:
        x, y, z, wq = np.array(fr["quat"]) / np.linalg.norm(fr["quat"])
        R = np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - z * wq), 2 * (x * z + y * wq)],
                      [2 * (x * y + z * wq), 1 - 2 * (x * x + z * z), 2 * (y * z - x * wq)],
                      [2 * (x * z - y * wq), 2 * (y * z + x * wq), 1 - 2 * (x * x + y * y)]])
        T = np.eye(4)
        T[:3, :3] = R @ np.diag([1.0, -1.0, -1.0])
        T[:3, 3] = fr["pos"]
        out.append((float(fr["t"]), T))
    crop = ((PAD[0] - w) // 2, (PAD[1] - h) // 2, w, h)
    return out, K, crop, j


def movers(scene_root):
    """Tracked vehicles: per object (track arrays, splats in the object frame with layer flag 1)."""
    from pipeline.road.scene import reader

    p = scene_root / "objects.json"
    if not p.exists():
        return []
    out = []
    for o in J(p)["objects"]:
        a = np.asarray(reader(scene_root / o["file"])).astype("f")
        tr = o["track"]
        yaw = np.unwrap(np.asarray(tr["yaw"], "f8"))
        out.append((o, {k: np.asarray(tr[k], "f8") for k in ("t", "x", "y", "z")}, yaw,
                    np.column_stack([a, np.ones(len(a), "f")])))
    return out


def place(objs, t):
    parts = []
    for o, tr, yaw, a in objs:
        if not (o["t_first"] <= t <= o["t_last"]) or t < tr["t"][0] or t > tr["t"][-1]:
            continue
        x, y, z = (np.interp(t, tr["t"], tr[k]) for k in ("x", "y", "z"))
        th = float(np.interp(t, tr["t"], yaw))
        c, s = np.cos(th), np.sin(th)
        R = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1.0]], "f")
        b = a.copy()
        b[:, :3] = a[:, :3] @ R.T + [x, y, z]
        b[:, 3:6] = a[:, 3:6] @ R.T
        w0, v = np.cos(th / 2), np.sin(th / 2)  # q_yaw = (w0, 0, 0, v), times q (wxyz)
        qw, qx, qy, qz = a[:, 13], a[:, 14], a[:, 15], a[:, 16]
        b[:, 13] = w0 * qw - v * qz
        b[:, 14] = w0 * qx - v * qy
        b[:, 15] = w0 * qy + v * qx
        b[:, 16] = w0 * qz + v * qw
        parts.append(b)
    return np.concatenate(parts).astype("f") if parts else None


def clean(scene, t0, t1, out, workers=8, step=3, max_range=60.0):
    """Hide object splats that the photographs contradict: projected into front/pillar keyframes
    around the stretch, a splat that lands on sky in at least 60 % of the views that see it (3 or
    more), or on a vehicle instance in at least 50 %, is dropped from this render's copy. The
    published tiles are not touched. Cached in out/clean.npz."""
    from pipeline.genfill.views import rig
    from pipeline.common.rig import T_world_cam

    cache = Path(out) / "clean2.npz"
    obj = np.flatnonzero(scene.pts[:, 24] == 1)
    if readable(cache):
        with np.load(cache) as a:
            if len(a["views"]) == len(obj):
                return obj, a["views"], a["sky"], a["dyn"], a["dmin"]
    kf = J(RS / "depth/keyframes.json")["frames"]
    jobs = [(c, e) for c in ("front", "left_pillar", "right_pillar") if c in kf for e in kf[c][::step]
            if t0 - 10 <= e["t"] <= t1 + 10]
    P = scene.pts[obj, :3].astype("f8")
    cell = np.floor(P[:, :2] / 20).astype(int)
    keys = {}
    for i, k in enumerate(map(tuple, cell)):
        keys.setdefault(k, []).append(i)
    keys = {k: np.array(v) for k, v in keys.items()}

    def one(job):
        c, e = job
        C = rig[c]
        n = e["n"]
        sky = cv.imread(str(RS / "masks/sky" / c / f"{n:06d}.png"), cv.IMREAD_UNCHANGED)
        dyn = cv.imread(str(RS / "masks/dyn" / c / f"{n:06d}.png"), cv.IMREAD_UNCHANGED)
        if sky is None or dyn is None:
            return None
        T = T_world_cam(C, e["t"], None, scene.traj)
        o = T[:3, 3]
        r = int(np.ceil(max_range / 20))
        k0 = np.floor(o[:2] / 20).astype(int)
        idx = [keys[(k0[0] + a, k0[1] + b)] for a in range(-r, r + 1) for b in range(-r, r + 1)
               if (k0[0] + a, k0[1] + b) in keys]
        if not idx:
            return None
        idx = np.concatenate(idx)
        v = (P[idx] - o) @ T[:3, :3]
        near = (v[:, 2] > 0.5) & (np.linalg.norm(v, axis=1) < max_range)
        idx, v = idx[near], v[near]
        uv, ok = C.project(v)
        ok &= C.mask_at(uv)
        idx, uv, dist = idx[ok], uv[ok], np.linalg.norm(v[ok], axis=1)
        h, w = sky.shape[:2]
        col = np.clip((uv[:, 0] * w / C.width).astype(int), 0, w - 1)
        row = np.clip((uv[:, 1] * h / C.height).astype(int), 0, h - 1)
        return idx, sky[row, col] != 0, dyn[row, col] != 0, dist

    views = np.zeros(len(obj), "i4")
    dmin = np.full(len(obj), 1e4, "f4")
    skyh = np.zeros(len(obj), "i4")
    dynh = np.zeros(len(obj), "i4")
    with ThreadPoolExecutor(workers) as ex:
        for res in ex.map(one, jobs):
            if res is None:
                continue
            idx, s, d, dist = res
            np.minimum.at(dmin, idx, dist.astype("f4"))
            np.add.at(views, idx, 1)
            np.add.at(skyh, idx[s], 1)
            np.add.at(dynh, idx[d], 1)
    tmp = Path(out) / "clean.tmp.npz"
    np.savez_compressed(tmp, views=views, sky=skyh, dyn=dynh, dmin=dmin)
    tmp.replace(cache)
    return obj, views, skyh, dynh, dmin


def drop_mask(P, views, skyh, dynh, sky_frac=0.6, column_frac=0.3, cell=0.75, ground_z=None):
    """Per splat: sky in >= sky_frac of its views, or a vehicle in >= half. Per upright column
    (splats more than 1 m above the ground, binned in cell x cell metres): when the photographs see
    sky behind at least column_frac of the column's splat views, the whole column goes (a pole that
    is misplaced or smeared would otherwise break into fragments)."""
    seen = views >= 3
    drop = seen & ((skyh >= sky_frac * views) | (dynh >= 0.5 * views))
    if column_frac:
        gz = 0.0 if ground_z is None else ground_z
        up = P[:, 2] - gz > 1.0
        k = np.floor(P[:, :2] / cell).astype("i8")
        key = (k[:, 0] + 2**20) * 2**21 + (k[:, 1] + 2**20)
        u, inv = np.unique(np.where(up, key, -1), return_inverse=True)
        sv = np.bincount(inv, weights=np.where(up & seen, views, 0), minlength=len(u))
        ss = np.bincount(inv, weights=np.where(up & seen, skyh, 0), minlength=len(u))
        n = np.bincount(inv, weights=(up & seen).astype("f8"), minlength=len(u))
        bad = (u != -1) & (n >= 20) & (ss >= column_frac * np.maximum(sv, 1))
        drop |= up & bad[inv]
    return drop


def fade_near(ob, cam, car, T=None, K=None, h=None, inner=7.0, tube=2.2, close=15.0, scale=0.6):
    """What the chase camera sees from a view no photo supports is not drawn, a whole object at a
    time (column 20 of this copy holds a connected-object id, -1 for low splats): an object goes when
    any of its splats is inside 'inner' metres, in the tube from the camera to the car, above the
    camera or in the top third of the frame inside 'close' metres, or seen from nearer than 'scale'
    times the nearest photograph that saw it (column 19). No partial fades: they read as ghosts."""
    X = ob[:, :3].astype("f8")
    d = np.linalg.norm(X - cam, axis=1)
    bad = d < inner
    seg = car - cam
    u = (X - cam) @ seg / (seg @ seg)
    dist = np.linalg.norm(X - cam - np.clip(u, 0, 1)[:, None] * seg, axis=1)
    bad |= (u > 0) & (u < 0.92) & (dist < tube)
    if T is not None:
        v = (X - T[:3, 3]) @ T[:3, :3]
        top = (v[:, 2] > 0.3) & (K[1, 1] * v[:, 1] / np.maximum(v[:, 2], 0.3) + K[1, 2] < h / 3)
        bad |= (d < close) & ((X[:, 2] > cam[2] - 1.0) | top)
    bad |= d < scale * ob[:, 19]
    comp = ob[:, 20].astype("i8")
    gone = np.unique(comp[bad & (comp >= 0)])
    drop = bad | np.isin(comp, gone)
    return ob[~drop]


def components(P, low, voxel=0.5, chunk=8.0):
    """Connected objects among splats (26-connected 0.5 m voxels); -1 where 'low'."""
    from scipy import ndimage

    comp = np.full(len(P), -1, "i8")
    sel = np.flatnonzero(~low)
    if not len(sel):
        return comp
    q = np.floor(P[sel] / voxel).astype("i8")
    q -= q.min(0)
    grid = np.zeros(q.max(0) + 1, bool)
    grid[q[:, 0], q[:, 1], q[:, 2]] = True
    lab, _ = ndimage.label(grid, structure=np.ones((3, 3, 3), bool))
    lab = lab[q[:, 0], q[:, 1], q[:, 2]].astype("i8")
    # a row of touching crowns is one connected piece: split pieces into 8 m cells so only the part
    # near the camera goes
    k = np.floor(P[sel, :2] / chunk).astype("i8") + 2**15
    comp[sel] = np.unique(lab * 2**34 + k[:, 0] * 2**17 + k[:, 1], return_inverse=True)[1]
    return comp


def floating(P, path_xy, path_z, far=25.0, lift=2.5, cell=2.0):
    """Far splats that do not connect to the ground: in 2 m cells more than 'far' metres from the
    driven path whose lowest splat is more than 'lift' metres above the road."""
    from scipy.spatial import cKDTree

    dd, ii = cKDTree(path_xy).query(P[:, :2])
    gz = path_z[ii]
    k = np.floor(P[:, :2] / cell).astype("i8")
    key = (k[:, 0] + 2**20) * 2**21 + (k[:, 1] + 2**20)
    u, inv = np.unique(key, return_inverse=True)
    low = np.full(len(u), np.inf)
    np.minimum.at(low, inv, P[:, 2] - gz)
    return (dd > far) & (low[inv] > lift)


def skylike(rgb, sky_rgb, height, tol=14.0, above=2.0):
    """Bright, sky-coloured splats in crowns (photographic sky bled into the tree colour)."""
    lab = cv.cvtColor(np.clip(rgb, 0, 255).astype("B")[None], cv.COLOR_RGB2LAB)[0].astype("f")
    ref = cv.cvtColor(np.clip(sky_rgb, 0, 255).astype("B")[None, None], cv.COLOR_RGB2LAB)[0, 0].astype("f")
    de = np.linalg.norm((lab - ref) * [100 / 255, 1, 1], axis=1)
    return (de < tol) & (height > above)


def render(scene_name, track_path, out, workers=6, use_movers=False, sky_frac=0.6, column_frac=0.2,
           frames_range=None, refs="all", hand=(), far_bias=1.0, pale=True, float_drop=False):
    from pipeline.genfill import views as V
    import pipeline.groundtex.proof as GP

    out = Path(out)
    (out / "render").mkdir(parents=True, exist_ok=True)
    frames, K, crop, meta = track(track_path)
    scene = V.Scene(scene_name, t_range=(frames[0][0] - 5, frames[-1][0] + 5))
    out.mkdir(parents=True, exist_ok=True)
    obj, views, skyh, dynh, dmin = clean(scene, frames[0][0], frames[-1][0], out)
    scene.pts[obj, 19] = np.where(views > 0, dmin, 0)  # free column here: nearest photo distance (0 = unknown)
    scene.pts[scene.pts[:, 24] == 0, 19:21] = [0, -1]
    P = scene.pts[obj, :3]
    d = scene.drive
    i = np.clip(np.searchsorted(d["t"], (frames[0][0] + frames[-1][0]) / 2), 0, len(d["t"]) - 1)
    gz = np.interp(P[:, 0], [P[:, 0].min(), P[:, 0].max()], [d["z"][i]] * 2)  # flat slice: drive height
    drop = drop_mask(P, views, skyh, dynh, sky_frac, column_frac, ground_z=gz)
    dt_ = np.asarray(d["t"])
    sel = (dt_ > frames[0][0] - 20) & (dt_ < frames[-1][0] + 20)
    pxy = np.stack([np.asarray(d["x"])[sel], np.asarray(d["y"])[sel]], 1)
    pz = np.asarray(d["z"])[sel]
    from scipy.spatial import cKDTree

    near_d, near_i = cKDTree(pxy).query(P[:, :2])
    hz = P[:, 2] - pz[near_i]
    if float_drop:
        drop |= floating(P, pxy, pz)
    if pale and scene.sky is not None:
        # sky bled into crowns: give pale, sky-coloured splats the colour of nearby foliage
        skyc = scene.sky[: scene.sky.shape[0] // 2].reshape(-1, 3).mean(0)
        rgbs = (0.5 + 0.28209479 * scene.pts[obj, 6:9]) * 255
        lab = cv.cvtColor(np.clip(rgbs, 0, 255).astype("B")[None], cv.COLOR_RGB2LAB)[0].astype("f") * [100 / 255, 1, 1]
        tree = (hz > 2.0) & ~drop
        fol = np.median(lab[tree], 0) if tree.any() else lab.mean(0)
        sk = cv.cvtColor(np.clip(skyc, 0, 255).astype("B")[None, None], cv.COLOR_RGB2LAB)[0, 0].astype("f") * [100 / 255, 1, 1]
        # closer to the sky's colour than to the scene's typical foliage, and bright: a washed-out crown
        pal = tree & (np.linalg.norm(lab - sk, axis=1) < np.linalg.norm(lab - fol, axis=1)) & (lab[:, 0] > fol[0] + 8)
        ok = ~pal & ~drop & (hz > 1.0)
        if pal.any() and ok.any():
            dd, ii = cKDTree(P[ok]).query(P[pal], k=16, distance_upper_bound=6.0)
            src = np.flatnonzero(ok)
            dc = scene.pts[obj[src], 6:9]
            got = np.isfinite(dd)
            w = got / np.maximum(got.sum(1, keepdims=True), 1)
            new = (dc[np.where(got, ii, 0)] * w[..., None]).sum(1)
            has = got.any(1)
            scene.pts[obj[np.flatnonzero(pal)[has]], 6:9] = new[has]
            drop[np.flatnonzero(pal)[~has]] = True
    scene.pts[obj, 20] = np.where(near_d < 60, components(P, (hz < 1.0) | (near_d >= 60)), -1)
    for hx, hy, hr in hand:
        drop |= (P[:, 0] - hx) ** 2 + (P[:, 1] - hy) ** 2 < hr * hr
    scene.pts[obj[drop], 9] = -30.0  # invisible to the rasteriser (alpha < 0.002)
    (out / "clean.json").write_text(json.dumps(dict(objects=int(len(obj)), seen=int((views >= 3).sum()),
                                                    sky_frac=sky_frac, column_frac=column_frac, dropped=int(drop.sum()),
                                                    dropped_sky=int(((views >= 3) & (skyh >= sky_frac * views)).sum()),
                                                    dropped_vehicle=int(((views >= 3) & (dynh >= 0.5 * views)).sum()))))
    GP.OUT = scene.root / "ground"
    at = GP.Atlas()
    provdir = scene.root / "ground"
    pcache = {}

    def prov(xy):
        """Ground atlas provenance label at world xy (nearest texel): 0 none, 127 predicted, 255 observed."""
        flat = xy.reshape(-1, 2)
        out = np.zeros(len(flat), "B")
        for k, ix in at.groups(flat):
            t = at.idx.get(k)
            if not t or not t.get("provenance"):
                continue
            if k not in pcache:
                if len(pcache) > 200:
                    pcache.clear()
                pcache[k] = cv.imread(str(provdir / t["provenance"]), cv.IMREAD_GRAYSCALE)
            img = pcache[k]
            uv = (flat[ix] - np.array(k) * GP.M) / t["texelSize"] + t["gutter"]
            i = np.clip(uv[:, 0].astype(int), 0, img.shape[1] - 1)
            j = np.clip(uv[:, 1].astype(int), 0, img.shape[0] - 1)
            out[ix] = img[j, i]
        return out.reshape(xy.shape[:-1])
    objs = movers(scene.root) if use_movers else []
    w, h = PAD
    xx, yy = np.meshgrid(np.arange(w) + 0.5, np.arange(h) + 0.5)
    cam_ray = np.stack([(xx - K[0, 2]) / K[0, 0], (yy - K[1, 2]) / K[1, 1], np.ones_like(xx)], -1)

    todo = range(len(frames)) if frames_range is None else range(*frames_range)
    if frames_range is not None:
        for i in todo:  # a deliberate re-render of these frames
            for ext in ("png", "npz"):
                (out / "render" / f"{i:04d}.{ext}").unlink(missing_ok=True)

    def one(i):
        t, T = frames[i]
        if readable(out / "render" / f"{i:04d}.png", out / "render" / f"{i:04d}.npz"):
            return i
        pts = scene.local(T[:3, 3])
        sky = scene.sky_canvas(T, K, PAD)
        g = np.ascontiguousarray(pts[pts[:, 24] == 0])
        ob = pts[pts[:, 24] == 1]
        mv = place(objs, t) if objs else None
        if mv is not None:
            ob = np.concatenate([ob, mv])
        x, y, zc, _, _ = scene.state(t)
        ob = np.ascontiguousarray(fade_near(ob, T[:3, 3], np.array([x, y, zc + 0.8]), T, K, h))
        G0, a0, z0 = scene.render(T, K, PAD, t, g, canvas=sky)
        O, ao, zo = scene.render(T, K, PAD, t, ob, canvas=np.zeros((h, w, 3), "f"))
        pos = T[:3, 3]
        ray = cam_ray @ T[:3, :3].T
        z = float(at.z(pos[:2][None], pos[2] - 5.5)[0])
        safe = np.minimum(-0.02, ray[:, :, 2])
        dep = np.clip((z - pos[2]) / safe, 0, 160)
        xy = pos[:2] + ray[:, :, :2] * dep[:, :, None]
        for _ in range(3):
            Z = at.z(xy, z)
            dep = np.clip((Z - pos[2]) / safe, 0, 160)
            xy = pos[:2] + ray[:, :, :2] * dep[:, :, None]
        dx, dy = np.gradient(xy, axis=1), np.gradient(xy, axis=0)
        foot = np.maximum(np.linalg.norm(dx, axis=2), np.linalg.norm(dy, axis=2) / 4)
        rgb = np.zeros((h, w, 3), "f")
        alpha = np.zeros((h, w, 1), "f")
        for k in (-0.375, -0.125, 0.125, 0.375):
            r, a = at.rgba(xy + dy * k, foot)
            rgb += r * 0.25  # premultiplied by the Atlas decoder
            alpha += a * 0.25
        ok = (ray[:, :, 2] < -0.02) & (dep < 150)
        aa = np.where(ok, alpha[:, :, 0], 0)
        G = np.where(ok[..., None], rgb + (1 - alpha) * G0, G0)
        zg = np.where(aa > 0.5, dep, np.where(a0 > 0.5, z0, 0))
        behind = (zo > 0) & (zg > 0) & (zo > zg * 1.03 + 0.3)
        ao = np.where(behind, 0, ao)
        O = np.where(behind[..., None], 0, O)
        final = O + (1 - ao[..., None]) * G
        skyw = (1 - ao) * (1 - a0) * (1 - np.where(ok, alpha[:, :, 0], 0))  # how much of the sky canvas shows
        zf = np.where(ao >= 0.5, zo, zg)
        lab = np.where(ok & (aa > 0.02), prov(xy), 0)
        write_png(out / "render" / f"{i:04d}.png", np.clip(final + 0.5, 0, 255).astype("B"))
        tmp = out / "render" / f"{i:04d}.tmp.npz"
        np.savez_compressed(tmp, z=zf.astype("f2"), atlas=(aa * 255).astype("B"),
                            obj=(ao * 255).clip(0, 255).astype("B"), aobs=(lab == 255), apred=(lab == 127),
                            skyw=(skyw * 255 + 0.5).clip(0, 255).astype("B"))
        tmp.replace(out / "render" / f"{i:04d}.npz")
        return i

    tic = time.monotonic()
    with ThreadPoolExecutor(workers) as ex:
        list(ex.map(one, todo))
    print("rendered", len(todo), "frames", round(time.monotonic() - tic, 1), "s", flush=True)
    if refs == "keep" and (out / "frames.json").exists():
        return
    if refs != "all" and (out / "frames.json").exists():
        a0, a1 = map(int, refs.split(":"))
        old = J(out / "frames.json")
        new = references(scene, frames[a0:a1], K, out, offset=a0, far_bias=far_bias)
        for i, r in enumerate(new):
            old["frames"][a0 + i]["ref"] = r
        (out / "frames.json").write_text(json.dumps(old, indent=1))
        return
    refs = references(scene, frames, K, out)
    (out / "frames.json").write_text(json.dumps(dict(
        scene=str(scene.root), track=str(track_path), fps=meta["fps"], vfov=meta["vfov"], size=list(PAD),
        crop=list(crop), K=K.tolist(),
        frames=[dict(n=i, t=t, T_world_cam=T.tolist(), ref=refs[i]) for i, (t, T) in enumerate(frames)]), indent=1))


def candidates(traj):
    """Every depth keyframe of every camera (about 5 Hz per lens) and its centre-row pose."""
    from pipeline.genfill.views import rig, CAMS
    from pipeline.common.rig import T_world_cam

    kf = J(RS / "depth/keyframes.json")["frames"]
    es = [dict(c=c, n=e["n"], t=e["t"]) for c in CAMS for e in kf[c]]
    poses = np.stack([T_world_cam(rig[e["c"]], e["t"], None, traj) for e in es])
    return es, poses


def references(scene, frames, K, out, window=(-14.0, 3.0), switch=1.25, floor=0.12, hold=2.0, offset=0,
               far_bias=1.0):
    """Score photos per frame and hold one: move on only to a later photo, only after 'hold' seconds,
    and only when it scores 'switch' times more (or the held one falls below 'floor' and a later one
    does not). An earlier photo is taken again only if the held one is not visible at all."""
    from pipeline.genfill.views import design

    es, poses = candidates(scene.traj)
    ts = np.array([e["t"] for e in es])
    rng = np.random.default_rng(0)
    cur, refs, last = None, [], -1e9
    for i, (t, T) in enumerate(frames):
        z = np.load(out / "render" / f"{i + offset:04d}.npz")["z"].astype("f")
        yy, xx = np.nonzero(z > 0.3)
        pick = rng.choice(len(yy), min(1500, len(yy)), replace=False)
        yy, xx = yy[pick], xx[pick]
        ray = np.stack([(xx + 0.5 - K[0, 2]) / K[0, 0], (yy + 0.5 - K[1, 2]) / K[1, 1], np.ones(len(xx))], 1)
        X = (ray * z[yy, xx, None]) @ T[:3, :3].T + T[:3, 3]
        a = X - T[:3, 3]
        da = np.linalg.norm(a, axis=1)
        a /= da[:, None]
        idx = np.flatnonzero((ts > t + window[0]) & (ts < t + window[1]))
        sc = np.zeros(len(es))
        for c in sorted({es[j]["c"] for j in idx}):
            jj = np.array([j for j in idx if es[j]["c"] == c])
            Kc, _, _, valid, _ = design(c, PAD)
            P = poses[jj]
            v = np.einsum("kji,knj->kni", P[:, :3, :3], X[None] - P[:, None, :3, 3])
            zc = v[..., 2]
            u = Kc[0, 0] * v[..., 0] / np.maximum(zc, 1e-3) + Kc[0, 2]
            q = Kc[1, 1] * v[..., 1] / np.maximum(zc, 1e-3) + Kc[1, 2]
            ins = (zc > 0.5) & (u >= 0) & (u < PAD[0]) & (q >= 0) & (q < PAD[1])
            ins &= valid[np.clip(q, 0, PAD[1] - 1).astype(int), np.clip(u, 0, PAD[0] - 1).astype(int)]
            b = X[None] - P[:, None, :3, 3]
            db = np.linalg.norm(b, axis=-1)
            cos = np.clip((a[None] * b).sum(-1) / db, 0, 1)
            wf = np.where(da > 30, far_bias, 1.0)  # weight the far field when asked
            sc[jj] = (ins * cos**4 * np.minimum(da, db) / np.maximum(da, db) * wf).sum(1) / wf.sum()
        best = int(np.argmax(sc))
        if cur is None or sc[cur] <= 0:
            cur, last = best, t
        else:
            later = int(np.argmax(np.where(ts >= ts[cur], sc, 0)))
            if (sc[cur] < floor and sc[later] >= floor) or (
                    sc[later] > switch * sc[cur] and t - last >= hold and later != cur):
                cur, last = later, t
        e = es[cur]
        refs.append(dict(c=e["c"], n=e["n"], t=e["t"], score=float(sc[cur]), best=float(sc[best])))
    return refs


def warp(src, Ts, zs, Td, zd, K, far=5000.0):
    """Sample src at the pixels of the destination view through its depth (sky at 'far'); occlusion-tested."""
    h, w = zd.shape
    zd = np.where(zd > 0.3, zd, far)
    zs = np.where(zs > 0.3, zs, far)
    xx, yy = np.meshgrid(np.arange(w) + 0.5, np.arange(h) + 0.5)
    ray = np.stack([(xx - K[0, 2]) / K[0, 0], (yy - K[1, 2]) / K[1, 1], np.ones_like(xx)], -1)
    X = (ray * zd[..., None]) @ Td[:3, :3].T + Td[:3, 3]
    v = (X - Ts[:3, 3]) @ Ts[:3, :3]
    zz = v[..., 2]
    u = (K[0, 0] * v[..., 0] / np.maximum(zz, 1e-3) + K[0, 2] - 0.5).astype("f")
    q = (K[1, 1] * v[..., 1] / np.maximum(zz, 1e-3) + K[1, 2] - 0.5).astype("f")
    ins = (zz > 0.3) & (u >= 0) & (u <= w - 1) & (q >= 0) & (q <= h - 1)
    zsw = cv.remap(zs.astype("f"), u, q, cv.INTER_NEAREST)
    vis = ins & (np.abs(zsw - zz) < 0.05 * zz + 0.15)
    return cv.remap(src, u, q, cv.INTER_CUBIC, borderMode=cv.BORDER_REPLICATE), vis


def rgb(p):
    return cv.imread(str(p))[:, :, ::-1]


def write_png(p, im, gray=False):
    p = Path(p)
    tmp = p.with_name(p.stem + ".tmp.png")
    cv.imwrite(str(tmp), im if gray else im[:, :, ::-1])
    tmp.replace(p)


def readable(*paths):
    """True when every file exists and decodes (a crash can leave truncated files)."""
    try:
        for p in paths:
            p = Path(p)
            if not p.exists():
                return False
            if p.suffix == ".npz":
                with np.load(p) as a:
                    [a[k].shape for k in a.files]
            elif cv.imread(str(p), cv.IMREAD_UNCHANGED) is None:
                return False
        return True
    except Exception:
        return False


def geometry(fr, z, K, drive):
    """Depth with holes closed by the local ground plane (the car's ground height at this time), and
    the masks: below = ray under the horizon; hole = no rendered surface below the horizon."""
    T = np.array(fr["T_world_cam"])
    h, w = z.shape
    xx, yy = np.meshgrid(np.arange(w) + 0.5, np.arange(h) + 0.5)
    ray = np.stack([(xx - K[0, 2]) / K[0, 0], (yy - K[1, 2]) / K[1, 1], np.ones_like(xx)], -1) @ T[:3, :3].T
    below = ray[..., 2] < -0.02
    hole = (z <= 0.3) & below
    zg = float(np.interp(fr["t"], drive["t"], drive["z"]))
    plane = np.where(below, np.clip((zg - T[2, 3]) / np.minimum(ray[..., 2], -0.02), 0.3, 400), 0)
    return T, np.where(hole, plane, z).astype("f"), hole, below, ray


def lowfreq(img, valid):
    """Push-pull fill: mask-weighted Gaussian pyramid, coarse levels fill where fine ones lack support."""
    I = img.astype("f") * valid[..., None]
    W = valid.astype("f")
    pyr = [(I, W)]
    while min(I.shape[:2]) > 12:
        I, W = cv.pyrDown(I), cv.pyrDown(W)
        pyr.append((I, W))
    out = pyr[-1][0] / np.maximum(pyr[-1][1], 1e-6)[..., None]
    for I, W in reversed(pyr[:-1]):
        up = cv.resize(out, (I.shape[1], I.shape[0]), interpolation=cv.INTER_LINEAR)
        a = np.clip(W * 3, 0, 1)[..., None]
        out = a * (I / np.maximum(W, 1e-6)[..., None]) + (1 - a) * up
    return out


def project_ref(ref, ref_valid, Kr, Tr, X):
    """Sample the reference photo at world points X (on the ground plane); returns rgb, ok."""
    v = (X - Tr[:3, 3]) @ Tr[:3, :3]
    zz = v[..., 2]
    u = (Kr[0, 0] * v[..., 0] / np.maximum(zz, 1e-3) + Kr[0, 2] - 0.5).astype("f")
    q = (Kr[1, 1] * v[..., 1] / np.maximum(zz, 1e-3) + Kr[1, 2] - 0.5).astype("f")
    h, w = ref_valid.shape
    ok = (zz > 0.5) & (u >= 0) & (u <= w - 1) & (q >= 0) & (q <= h - 1)
    ok &= cv.remap(ref_valid.astype("B"), u, q, cv.INTER_NEAREST) > 0
    return cv.remap(ref, u, q, cv.INTER_LINEAR).astype("f"), ok


def env_sky(F, E, skyw, t0=10.0, span=30.0, sep=12.0):
    """The environment map put back wherever the repaired frame is sky, with the matte decided on the fixer's
    output. The fixer was given the map as its sky, so its sky is the map plus a slowly varying offset
    (estimated over the render's sky); anything it drew that departs from that by more than t0 levels (a
    pole or wire it completed from the reference photo, foliage in a gap) is structure and kept, ramping to
    fully kept at t0 + span. Near structure, each pixel is split into the fixer's sky and the local structure
    colour (projection on the line between them) and only the sky part is swapped for the map, so no rim of
    the fixer's own sky (a halo) is left at crowns, poles and wires; pure sky becomes exactly the map, so the
    fixer's own sky texture and faint ghost strokes are gone. Solid splats are kept. Returns frame, matte."""
    solid = skyw < 0.02

    def local(img, m, sig):
        w = cv.GaussianBlur(m.astype("f"), (0, 0), sig)
        return cv.GaussianBlur(img * m[..., None], (0, 0), sig) / np.maximum(w, 1e-4)[..., None], w

    sky = skyw > 0.5
    off, ow = local(F - E, sky, 10.0)
    for _ in range(2):  # re-estimate the offset without what the fixer drew
        dist = np.linalg.norm(F - (E + off), axis=-1)
        off, ow = local(F - E, sky & (dist < t0), 10.0)
    P = E + off  # the fixer's sky at each pixel
    dist = np.linalg.norm(F - P, axis=-1)
    st = np.clip((dist - t0) / span, 0, 1)
    Fg, fw = local(F, solid | (st > 0.9), 3.0)
    d = P - Fg
    n2 = (d * d).sum(-1)
    a = np.clip(((F - Fg) * d).sum(-1) / np.maximum(n2, 1e-3), 0, 1)
    a = np.where((n2 > sep * sep) & (fw > 0.05), a, 1 - st)
    a = np.where(solid | (ow < 0.02), 0.0, a)
    h = np.clip((a - 0.9) / 0.1, 0, 1)
    Beff = P + (h * h * (3 - 2 * h))[..., None] * (F - P)
    return F + a[..., None] * (E - Beff), a


def fix(out, tag, timestep=199, ff_in=0.0, ff_out=0.0, n0=0, n1=None, agree_tol=18.0, road=True,
        fill=True, hole_in=1.0, hole_out=0.8, harmonise=0.35, sky_out=0.7, haze_beyond=40.0, pred_gain=0.25, corridor=9.0,
        seed=None, grain=0.0, env=False):
    """Repair frames in order. Holes (no surface below the horizon) start from the previous repaired
    frame carried through the ground plane, else the reference photo projected onto the ground plane,
    else a low-frequency fill; the fixer completes them and its result there is held (hole_out).
    env: the scene's environment map is the sky (env_sky); the fixer does not redraw it.
    Writes <tag>/NNNN.png and <tag>-prov/NNNN.png: 0 observed (atlas road or unchanged render),
    1 repaired, 2 filled (no rendered surface), 3 sky; with env, sky by the map's provenance: 4 observed,
    5 filled, 6 harmonised."""
    import torch
    from pipeline.genfill.difix import Difix
    from pipeline.genfill.views import photo, design, rig
    from pipeline.genfill.fix_views import changed
    from pipeline.common.rig import Trajectory, T_world_cam

    out = Path(out)
    meta = J(out / "frames.json")
    K = np.array(meta["K"])
    drive = J(Path(meta["scene"]) / "drive.json")
    traj = Trajectory(RS / "poses/ego.npz")
    frames = meta["frames"][n0:n1]
    dst, prov = out / tag, out / (tag + "-prov")
    dst.mkdir(parents=True, exist_ok=True)
    prov.mkdir(exist_ok=True)
    model = Difix(True, dtype=torch.float16)
    envm = None
    if env:
        from pipeline.genfill.views import Environment
        envm = Environment(Path(meta["scene"]) / J(Path(meta["scene"]) / "map.json")["environment"])
        raw = out / (tag + "-raw")
        raw.mkdir(exist_ok=True)
    refcache = {}
    prev = None
    stats = []
    tic = time.monotonic()
    x0, y0, cw, ch = meta["crop"]
    if seed and frames and frames[0]["n"] > 0:
        f0 = meta["frames"][frames[0]["n"] - 1]
        z0 = np.load(out / "render" / f"{f0['n']:04d}.npz")["z"].astype("f")
        T0, zw0, *_ = geometry(f0, z0, K, drive)
        prev = (rgb(out / seed / f"{f0['n']:04d}.png"), T0, zw0)
    log = dst / "stats.jsonl"
    done = {}
    if log.exists():
        for line in log.read_text().splitlines():
            try:
                s = json.loads(line)
                done[s["n"]] = s
            except ValueError:
                pass
    for fr in frames:
        n = fr["n"]
        R = rgb(out / "render" / f"{n:04d}.png")
        aux = np.load(out / "render" / f"{n:04d}.npz")
        z, atl = aux["z"].astype("f"), aux["atlas"].astype("f") / 255
        aobs = aux["aobs"] if "aobs" in aux.files else atl > 0.5
        apred = aux["apred"] if "apred" in aux.files else np.zeros_like(aobs)
        obj = aux["obj"].astype("f") / 255
        T, zw, hole, below, ray = geometry(fr, z, K, drive)
        # observed atlas counts as truth only in the driven corridor; off it (car parks, side roads)
        # its paint is often misregistered across photographs, so it is treated like other ground
        X = T[:3, 3] + ray * zw[..., None]
        sel = (np.asarray(drive["t"]) > fr["t"] - 25) & (np.asarray(drive["t"]) < fr["t"] + 25)
        path = np.stack([np.asarray(drive["x"])[sel], np.asarray(drive["y"])[sel]], 1)[::3]
        from scipy.spatial import cKDTree
        sub = X[::4, ::4, :2]
        lat = cKDTree(path).query(sub.reshape(-1, 2))[0].reshape(sub.shape[:2]).astype("f")
        lat = cv.resize(lat, (z.shape[1], z.shape[0]), interpolation=cv.INTER_NEAREST)
        aobs = aobs & (lat < corridor)
        # beyond the ground ring the viewer hazes into the horizon colour: keep the render's haze there
        haze = hole & (zw > haze_beyond)
        hole = hole & ~haze
        sky = (z <= 0.3) & ~hole
        pred = apred & (atl > 0.5) & (obj < 0.5) & ~hole  # ground the atlas itself predicted
        if n in done and readable(dst / f"{n:04d}.png", prov / f"{n:04d}.png"):
            prev = (rgb(dst / f"{n:04d}.png"), T, zw)
            stats.append(done[n])
            continue
        key = (fr["ref"]["c"], fr["ref"]["n"])
        if key not in refcache:
            refcache.clear()
            im, ok = photo(key[0], key[1], tuple(meta["size"]))
            refcache[key] = (im, ok, design(key[0], tuple(meta["size"]))[0],
                             T_world_cam(rig[key[0]], fr["ref"]["t"], None, traj))
        ref, ref_ok, Kr, Tr = refcache[key]
        I = R.astype("f")
        if fill and hole.any():
            base = lowfreq(I, ~hole & (z > 0.3))
            X = T[:3, 3] + ray * zw[..., None]
            pr, pok = project_ref(ref, ref_ok, Kr, Tr, X)
            pok &= zw < 40.0  # far holes: projected photo pixels become fences on the horizon; keep it plain
            I = np.where(hole[..., None], np.where(pok[..., None], pr, base), I)
        M = Mh = Ms = None
        if prev is not None:
            W, vis = warp(prev[0], prev[1], prev[2], T, zw, K)
            Wf = W.astype("f")
            blur = lambda x: cv.GaussianBlur(x.astype("f"), (0, 0), 2.0)
            agree = vis & ~hole & ~pred & ~sky & (np.abs(blur(W) - blur(R)).mean(-1) < agree_tol)
            soft = lambda m: cv.GaussianBlur(cv.erode(m.astype("f"), np.ones((5, 5), "B")), (0, 0), 2.0)[..., None]
            M, Mh, Ms = soft(agree), soft(vis & (hole | pred)), soft(vis & sky)
            # no feed-forward into the fixer's input over the sky: it builds blobs there
            I = I + (ff_in * M + hole_in * Mh) * (Wf - I)
        t0 = time.perf_counter()
        F = model(np.clip(I + 0.5, 0, 255).astype("B"), ref, timestep=timestep, seed=0).astype("f")
        dt = time.perf_counter() - t0
        if M is not None:
            F = F + np.clip(ff_out * M + hole_out * Mh + sky_out * Ms, 0, 1) * (Wf - F)
        if fill and hole.any():
            # pull the filled area's local colour part-way to its observed surroundings (the fixer
            # tends to paint lawns lime); texture is kept, and carried fills converge, not drift
            lf = lambda x: cv.GaussianBlur(x, (0, 0), 12.0)
            F = np.where(hole[..., None], F + harmonise * (lf(base) - lf(F)), F)
            # far holes (beyond the ground ring, filled only when haze_beyond is raised): plain ground
            # colour plus fine grain, so the fixer cannot build railings or roofs along the horizon
            farh = hole & (zw > 40.0)
            if farh.any():
                Fg = base + 0.8 * (F - cv.GaussianBlur(F, (0, 0), 2.5))
                F = np.where(farh[..., None], Fg, F)
        if pred.any():
            # ground the atlas predicted (no photo): keep its colour, admit only part of the fixer's texture
            lf = lambda x: cv.GaussianBlur(x, (0, 0), 12.0)
            Fp = F + 0.5 * (lf(R.astype("f")) - lf(F))
            Fp = R.astype("f") + pred_gain * (Fp - R.astype("f"))
            if grain:
                # quiet texture: the atlas fill's colour plus only the fixer's fine grain (no slabs)
                Fp = R.astype("f") + grain * (F - cv.GaussianBlur(F, (0, 0), 2.5))
            F = np.where(cv.GaussianBlur(pred.astype("f"), (0, 0), 1.5)[..., None] > 0.5, Fp, F)
        F = np.where(haze[..., None], R.astype("f"), F)
        # no road paint outside the observed atlas: remove bright thin strokes on all other ground
        gnd = (obj < 0.5) & below & (z > 0.3) & ~(aobs & (atl > 0.5))
        Fu = np.clip(F, 0, 255).astype("B")
        L = cv.cvtColor(Fu, cv.COLOR_RGB2LAB)[:, :, 0].astype("f")
        strokes = ((L - cv.morphologyEx(L, cv.MORPH_OPEN, np.ones((15, 15), "B"))) > 18) & gnd
        if strokes.any():
            plain = cv.morphologyEx(Fu, cv.MORPH_OPEN, cv.getStructuringElement(cv.MORPH_ELLIPSE, (13, 13))).astype("f")
            wgt = cv.GaussianBlur(cv.dilate(strokes.astype("f"), np.ones((5, 5), "B")), (0, 0), 1.5)[..., None]
            F = F + np.clip(wgt * 1.5, 0, 1) * (plain - F)
        A = (np.clip((atl - 0.5) / 0.48, 0, 1) * aobs)[..., None] if road else 0
        F = F + A * (R.astype("f") - F)
        if envm is not None:
            E, Ep = envm.canvas(T, K, (R.shape[1], R.shape[0]), prov=True)
            sw = aux["skyw"].astype("f") / 255 if "skyw" in aux.files else (z <= 0.3).astype("f")
            write_png(raw / f"{n:04d}.png", np.clip(F + 0.5, 0, 255).astype("B"))  # the fixer's frame, kept
            F, sa = env_sky(F, E, np.where(hole | haze, 0.0, sw))
        Fb = np.clip(F + 0.5, 0, 255).astype("B")
        chg, _ = changed(R, Fb, (z > 0.3).astype("f"))
        cl = np.where(hole | pred, 2, np.where(sky | haze, 3, np.where(chg >= 0.25, 1, 0))).astype("B")
        if envm is not None:
            cl = np.where(sa > 0.5, np.select([Ep == 255, Ep == 170], [4, 6], 5), cl).astype("B")
        if road:
            cl[(atl >= 0.98) & aobs & ~hole] = 0
        write_png(prov / f"{n:04d}.png", cl, gray=True)
        write_png(dst / f"{n:04d}.png", Fb)
        prev = (Fb, T, zw)
        cc = cl[y0:y0 + ch, x0:x0 + cw]
        stats.append(dict(n=n, seconds=dt, observed=float((cc == 0).mean()), repaired=float((cc == 1).mean()),
                          filled=float((cc == 2).mean()), sky=float((cc == 3).mean()),
                          holes=float(hole[y0:y0 + ch, x0:x0 + cw].mean()),
                          atlas_predicted=float(pred[y0:y0 + ch, x0:x0 + cw].mean()),
                          carried_holes=None if Mh is None else float(Mh[hole].mean()) if hole.any() else 0.0))
        with open(log, "a") as fh:
            fh.write(json.dumps(stats[-1]) + "\n")
    (dst / "fix.json").write_text(json.dumps(dict(timestep=timestep, ff_in=ff_in, ff_out=ff_out, road=road,
                                                  fill=fill, hole_in=hole_in, hole_out=hole_out,
                                                  agree_tol=agree_tol, frames=stats), indent=1))
    print(tag, len(frames), "frames", round(time.monotonic() - tic, 1), "s; fixer",
          round(float(np.median([s["seconds"] for s in stats])), 2), "s/frame; filled",
          round(float(np.mean([s["filled"] for s in stats])), 3), "max", round(float(np.max([s["filled"] for s in stats])), 3),
          flush=True)


def flicker(out, tags, n0=0, n1=None, real=False):
    """Mean |previous frame moved onto this one - this frame| in RGB levels, inside the crop:
    by the render's depth (and pose), and by DIS optical flow computed on the sequence itself."""
    out = Path(out)
    meta = J(out / "frames.json")
    K = np.array(meta["K"])
    x0, y0, w, h = meta["crop"]
    frames = meta["frames"][n0:n1]
    drive = J(Path(meta["scene"]) / "drive.json")
    dis = cv.DISOpticalFlow_create(cv.DISOPTICAL_FLOW_PRESET_MEDIUM)
    res = {}

    def seq_flow(imgs):
        e = []
        for a, b in zip(imgs[:-1], imgs[1:]):
            ga, gb = cv.cvtColor(a, cv.COLOR_RGB2GRAY), cv.cvtColor(b, cv.COLOR_RGB2GRAY)
            fl = dis.calc(gb, ga, None)
            yy, xx = np.mgrid[:gb.shape[0], :gb.shape[1]].astype("f")
            wa = cv.remap(a, xx + fl[..., 0], yy + fl[..., 1], cv.INTER_LINEAR)
            e.append(float(np.abs(wa.astype("f") - b.astype("f")).mean()))
        return float(np.mean(e))

    for tag in ["render"] + list(tags):
        d = out / tag
        pv = out / (tag + "-prov")
        imgs, eds, efill = [], [], []
        prev = None
        for fr in frames:
            im = rgb(d / f"{fr['n']:04d}.png")
            z0 = np.load(out / "render" / f"{fr['n']:04d}.npz")["z"].astype("f")
            T, z, _, _, _ = geometry(fr, z0, K, drive)
            if prev is not None:
                W, vis = warp(prev[0], prev[1], prev[2], T, z, K)
                e = np.abs(W.astype("f") - im.astype("f")).mean(-1)[y0:y0 + h, x0:x0 + w]
                vv = vis[y0:y0 + h, x0:x0 + w]
                eds.append(float(e[vv].mean()))
                if pv.exists():
                    fm = (cv.imread(str(pv / f"{fr['n']:04d}.png"), 0)[y0:y0 + h, x0:x0 + w] == 2) & vv
                    if fm.sum() > 500:
                        efill.append(float(e[fm].mean()))
            prev = (im, T, z)
            imgs.append(np.ascontiguousarray(im[y0:y0 + h, x0:x0 + w]))
        res[tag] = dict(depth_warp=float(np.mean(eds)), self_flow=seq_flow(imgs),
                        depth_warp_filled=float(np.mean(efill)) if efill else None)
        print(tag, res[tag], flush=True)
    if real:
        t0, t1 = frames[0]["t"], frames[-1]["t"]
        tab = [r for r in J(RS / "data/drive/cams/front.json") if t0 <= r["t"] <= t1]
        imgs = [cv.resize(rgb(RS / "data/drive/frames/front" / f"{r['n']:06d}.jpg"), (w, int(w * 1876 / 2896)),
                          interpolation=cv.INTER_AREA) for r in tab]
        res["real front video"] = dict(self_flow=seq_flow(imgs), frames=len(imgs))
        print("real", res["real front video"], flush=True)
    (out / f"flicker-{n0}-{n1}.json").write_text(json.dumps(res, indent=1))


def splice(out, tag, new, n0, n1, ramp=15):
    """Put frames n0..n1-1 of 'new' into 'tag', cross-fading over 'ramp' frames at both ends;
    the replaced frames are kept in <tag>-orig/ for a revert."""
    out = Path(out)
    bak = out / (tag + "-orig")
    bak.mkdir(exist_ok=True)
    for n in range(n0, n1):
        for sub in (tag, tag + "-prov"):
            src = out / sub / f"{n:04d}.png"
            b = out / (sub + "-orig")
            b.mkdir(exist_ok=True)
            if not (b / src.name).exists():
                os.link(src, b / src.name)
        k = min(n - n0 + 1, n1 - n) / (ramp + 1)
        w = float(np.clip(k, 0, 1))  # weight of the new frame
        a = rgb(out / (tag + "-orig") / f"{n:04d}.png").astype("f")
        b = rgb(out / new / f"{n:04d}.png").astype("f")
        write_png(out / tag / f"{n:04d}.png", np.clip(a + w * (b - a) + 0.5, 0, 255).astype("B"))
        pv = out / (new + "-prov") / f"{n:04d}.png" if w >= 0.5 else out / (tag + "-prov-orig") / f"{n:04d}.png"
        write_png(out / (tag + "-prov") / f"{n:04d}.png", cv.imread(str(pv), 0), gray=True)
        code = out / (tag + "-codes") / f"{n:04d}.png"
        code.unlink(missing_ok=True)
    print("spliced", n0, n1, flush=True)


def unpublish(scene_dirs, name):
    for sd in scene_dirs:
        gj = RS / sd / "generated.json"
        if gj.exists():
            doc = J(gj)
            doc["clips"] = [c for c in doc["clips"] if c["file"] != f"generated/{name}.mp4"]
            gj.write_text(json.dumps(doc, indent=1))
        (RS / sd / "generated" / f"{name}.mp4").unlink(missing_ok=True)
    print("unpublished", name, flush=True)


def encode(out, tag, scene_dirs, name, crf=18, provenance=True, n0=0, n1=None):
    """H.264 clip into each scene's generated/ and generated.json. With provenance, the frame's mask is
    stacked below the picture (viewer/README.md): 255 observed, 170 repaired, 85 filled (made up: no
    splat surface, ground the atlas predicted, or sky the fixer redrew), 0 none."""
    out = Path(out)
    meta = J(out / "frames.json")
    x0, y0, w, h = meta["crop"]
    fr = meta["frames"][n0:n1]
    tmp = out / f"{name}.mp4"
    rng = ["-start_number", str(fr[0]["n"])]
    args = ["ffmpeg", "-v", "error", "-y", "-framerate", str(meta["fps"])] + rng + ["-i", str(out / tag / "%04d.png")]
    if provenance:
        mdir = out / (tag + "-codes")
        mdir.mkdir(exist_ok=True)
        lut = np.array([255, 170, 85, 85, 255, 85, 170] + [0] * 249, "B")
        for f in fr:
            p = mdir / f"{f['n']:04d}.png"
            if not readable(p):
                write_png(p, lut[cv.imread(str(out / (tag + "-prov") / f"{f['n']:04d}.png"), 0)], gray=True)
        args += ["-framerate", str(meta["fps"])] + rng + ["-i", str(mdir / "%04d.png"), "-filter_complex",
                 f"[0:v]crop={w}:{h}:{x0}:{y0},format=yuv420p[p];[1:v]crop={w}:{h}:{x0}:{y0},format=yuv420p[m];"
                 "[p][m]vstack=inputs=2[v]", "-map", "[v]"]
    else:
        args += ["-vf", f"crop={w}:{h}:{x0}:{y0}"]
    args += ["-frames:v", str(len(fr)), "-c:v", "libx264", "-pix_fmt", "yuv420p", "-g", str(int(meta["fps"] // 2)), "-crf", str(crf),
             "-preset", "slow", "-movflags", "+faststart", "-an", str(tmp)]
    subprocess.run(args, check=True)
    clip = dict(file=f"generated/{name}.mp4", t0=fr[0]["t"], t1=fr[-1]["t"], fps=meta["fps"], pose="chase",
                vfov=meta["vfov"], ego="free", vehicles="overlay")
    if provenance:
        clip["provenance"] = "below"
    for sd in scene_dirs:
        sd = RS / sd
        (sd / "generated").mkdir(exist_ok=True)
        dst = sd / "generated" / f"{name}.mp4"
        dst.unlink(missing_ok=True)
        os.link(tmp, dst)
        gj = sd / "generated.json"
        doc = J(gj) if gj.exists() else {"clips": []}
        doc["clips"] = [c for c in doc["clips"] if c["file"] != clip["file"]] + [clip]
        doc["clips"].sort(key=lambda c: c["t0"])
        gj.write_text(json.dumps(doc, indent=1))
    print("encoded", tmp, tmp.stat().st_size, clip, flush=True)


def sheets(out, tag, name, every_s=2.5):
    """report-clip/: side-by-side H.264 (plain render | repaired, half size), a contact sheet, and the
    three frames with the largest filled share (render | repaired | provenance)."""
    from pipeline.genfill.evaluate import label, save_small

    out = Path(out)
    meta = J(out / "frames.json")
    x0, y0, w, h = meta["crop"]
    rep = RS / "pipeline/genfill/report-clip"
    rep.mkdir(parents=True, exist_ok=True)
    side = rep / f"{name}-side-by-side.mp4"
    vf = (f"[0:v]crop={w}:{h}:{x0}:{y0},scale=720:450[a];[1:v]crop={w}:{h}:{x0}:{y0},scale=720:450[b];"
          "[a][b]hstack=inputs=2[v]")
    for crf in (26, 30, 34):
        subprocess.run(["ffmpeg", "-v", "error", "-y", "-framerate", str(meta["fps"]), "-i", str(out / "render/%04d.png"),
                        "-framerate", str(meta["fps"]), "-i", str(out / tag / "%04d.png"), "-filter_complex", vf,
                        "-map", "[v]", "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", str(crf), "-preset", "slow",
                        "-movflags", "+faststart", "-an", str(side)], check=True)
        if side.stat().st_size < 4.9e6:
            break
    cut = lambda p: rgb(p)[y0:y0 + h, x0:x0 + w]
    small = lambda im: cv.resize(im, (480, 300), interpolation=cv.INTER_AREA)
    fr = meta["frames"]
    step = int(round(every_s * meta["fps"]))
    rows = [np.hstack([label(small(cut(out / "render" / f"{f['n']:04d}.png")), f"render {f['t']:.1f} s"),
                       label(small(cut(out / tag / f"{f['n']:04d}.png")), "repaired")]) for f in fr[::step]]
    pairs = [np.hstack(rows[i:i + 2]) if i + 1 < len(rows) else np.hstack([rows[i], np.zeros_like(rows[i])])
             for i in range(0, len(rows), 2)]
    save_small(rep / f"{name}-contact.jpg", np.vstack(pairs))
    st = J(out / tag / "fix.json")["frames"]
    worst = sorted(st, key=lambda s: -s["filled"])[:3]
    rows = []
    for s in sorted(worst, key=lambda s: s["n"]):
        n = s["n"]
        p = cv.imread(str(out / (tag + "-prov") / f"{n:04d}.png"), 0)[y0:y0 + h, x0:x0 + w]
        f = cut(out / tag / f"{n:04d}.png")
        vis = np.zeros_like(f)
        vis[p == 1] = (230, 170, 60)
        vis[p == 2] = (220, 60, 70)
        vis[p == 3] = (120, 160, 200)
        t = fr[n]["t"]
        rows.append(np.hstack([label(small(cut(out / "render" / f"{n:04d}.png")), f"render {t:.2f} s"),
                               label(small(f), f"repaired, filled {s['filled']:.1%}"),
                               label(small((0.5 * f + 0.5 * vis).astype("B")), "red filled, amber repaired, blue sky")]))
    save_small(rep / f"{name}-worst-fill.jpg", np.vstack(rows))
    print("sheets", side, side.stat().st_size, flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("render")
    p.add_argument("scene")
    p.add_argument("track")
    p.add_argument("out")
    p.add_argument("--movers", action="store_true")
    p.add_argument("--workers", type=int, default=6)
    p.add_argument("--sky-frac", type=float, default=0.6)
    p.add_argument("--column-frac", type=float, default=0.2)
    p.add_argument("--frames", default=None, help="n0:n1 re-render only these frames")
    p.add_argument("--refs", default="all", help="all | keep | n0:n1 (recompute only these)")
    p.add_argument("--hand", default="", help="x,y,r;x,y,r object columns to drop")
    p.add_argument("--far-bias", type=float, default=1.0)
    p.add_argument("--float-drop", action="store_true")
    f = sub.add_parser("fix")
    f.add_argument("out")
    f.add_argument("tag")
    f.add_argument("--timestep", type=int, default=199)
    f.add_argument("--ff-in", type=float, default=0.0)
    f.add_argument("--ff-out", type=float, default=0.0)
    f.add_argument("--n0", type=int, default=0)
    f.add_argument("--n1", type=int, default=None)
    f.add_argument("--no-road", action="store_true")
    f.add_argument("--no-fill", action="store_true")
    f.add_argument("--hole-in", type=float, default=1.0)
    f.add_argument("--hole-out", type=float, default=0.8)
    f.add_argument("--seed", default=None, help="carry from this tag's frame n0-1")
    f.add_argument("--grain", type=float, default=0.0, help="atlas-predicted ground: fixer fine grain only")
    f.add_argument("--haze-beyond", type=float, default=40.0)
    f.add_argument("--env", action="store_true", help="the scene's environment map is the sky")
    k = sub.add_parser("flicker")
    k.add_argument("out")
    k.add_argument("tags", nargs="*")
    k.add_argument("--n0", type=int, default=0)
    k.add_argument("--n1", type=int, default=None)
    k.add_argument("--real", action="store_true")
    e = sub.add_parser("encode")
    e.add_argument("out")
    e.add_argument("tag")
    e.add_argument("name")
    e.add_argument("scenes", nargs="+")
    e.add_argument("--n0", type=int, default=0)
    e.add_argument("--n1", type=int, default=None)
    sp = sub.add_parser("splice")
    sp.add_argument("out")
    sp.add_argument("tag")
    sp.add_argument("new")
    sp.add_argument("n0", type=int)
    sp.add_argument("n1", type=int)
    u = sub.add_parser("unpublish")
    u.add_argument("name")
    u.add_argument("scenes", nargs="+")
    s = sub.add_parser("sheets")
    s.add_argument("out")
    s.add_argument("tag")
    s.add_argument("name")
    a = ap.parse_args()
    if a.cmd == "sheets":
        sheets(RS / a.out, a.tag, a.name)
    elif a.cmd == "render":
        hand = [tuple(map(float, h.split(","))) for h in a.hand.split(";") if h]
        fr = tuple(map(int, a.frames.split(":"))) if a.frames else None
        render(a.scene, RS / a.track, RS / a.out, a.workers, a.movers, a.sky_frac, a.column_frac, fr, a.refs, hand,
               a.far_bias, float_drop=a.float_drop)
    elif a.cmd == "fix":
        fix(RS / a.out, a.tag, a.timestep, a.ff_in, a.ff_out, a.n0, a.n1, road=not a.no_road,
            fill=not a.no_fill, hole_in=a.hole_in, hole_out=a.hole_out, seed=a.seed, grain=a.grain,
            haze_beyond=a.haze_beyond, env=a.env)
    elif a.cmd == "flicker":
        flicker(RS / a.out, a.tags, a.n0, a.n1, a.real)
    elif a.cmd == "splice":
        splice(RS / a.out, a.tag, a.new, a.n0, a.n1)
    elif a.cmd == "unpublish":
        unpublish(a.scenes, a.name)
    else:
        encode(RS / a.out, a.tag, a.scenes, a.name, n0=a.n0, n1=a.n1)
