"""Build independent 50 m tiles; projection, evidence and immutable births are chronological within each tile."""

from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import os, json, time
import cv2 as cv
import numpy as np
from pipeline.common.rig import Trajectory
from pipeline.photo.model import Photometry, basis
import ctypes as ct
import sys, shutil, subprocess
from functools import lru_cache

RS = Path(os.environ.get("RS", ".")).resolve()

W = RS / "maps/ground-work"
FIELDS = "x y z nx ny nz f_dc_0 f_dc_1 f_dc_2 opacity scale_0 scale_1 scale_2 rot_0 rot_1 rot_2 rot_3 t_birth t_death e0 s1 e1 s2 e2".split()
INF = 1.0e9


def binary_header(n):
    return (
        "ply\nformat binary_little_endian 1.0\nelement vertex "
        + str(n)
        + "\n"
        + "".join("property float " + p + "\n" for p in FIELDS)
        + "end_header\n"
    ).encode()


def reader(p):
    with open(p, "rb") as f:
        assert f.readline().strip() == b"ply"
        assert f.readline().strip() == b"format binary_little_endian 1.0"
        n = int(f.readline().split()[-1])
        fields = [f.readline().split()[-1].decode() for _ in FIELDS]
        assert fields == FIELDS and f.readline().strip() == b"end_header"
        off = f.tell()
    assert p.stat().st_size == off + n * 96
    return np.memmap(p, "<f4", "r", off, (n, 24))


def native(work=W):
    lib = ct.CDLL(str(work / "ground.dylib"))
    P = ct.c_void_p
    I = ct.c_int
    F = ct.c_float
    lib.create.argtypes = [I, I, I]
    lib.create.restype = P
    lib.release.argtypes = [P]
    lib.see.argtypes = [P, I, F, P, P, P, I, I, P, I, I, P, P, P]
    lib.packet.argtypes = [P, F, F, I]
    lib.finish.argtypes = [P, ct.c_char_p, ct.c_char_p, P]
    lib.finish.restype = ct.c_long
    return lib


def road_shapes(field, tx, ty):
    # Smooth only supported raised samples, with no extrapolation/internal road deformation.
    if not len(field):
        return []
    xy = (field[:, :2] + 0.5) * 1.5
    sel = (abs(xy[:, 0] - (tx + 0.5) * 50) < 26.9) & (
        abs(xy[:, 1] - (ty + 0.5) * 50) < 26.9
    )
    records = field[sel]
    state = {}
    changes = []
    for x, y, t, z in records[np.argsort(records[:, 2])]:
        key = (x, y)
        prev = state.get(key, 0)
        if abs(z - prev) < 0.028:
            continue
        changes.append((t, x, y, z))
        state[key] = z
    return changes


def build(name, version=1):
    from pipeline.photo.sample import CAMS, NAMES, frame, mask_index, read_mask
    from pipeline.common.rig import frame_table

    start = time.monotonic()
    v2 = version == 2
    work = RS / "maps/ground-v2" if v2 else W
    scene = RS / "scene" / f"road-{name}-v{version}"
    scene.mkdir(exist_ok=True, parents=True)
    (scene / "tiles").mkdir(exist_ok=True)
    scr = work / name
    scr.mkdir(exist_ok=True, parents=True)
    live = name == "live"
    posefile = "poses/" + ("ego_causal.npz" if live else "ego.npz")
    tr = Trajectory(RS / posefile)
    s = dict(np.load(RS / "work/photo/schedule.npz"))
    field = np.load(W / (name + "-height.npy"))
    photo = Photometry("live-v2" if v2 and live else name, RS)
    snap = work / "mask-index.json"
    if snap.exists():
        index = {
            (i, kind): {int(n): p for n, p in paths.items()}
            for i, lens in enumerate(json.loads(snap.read_text()))
            for kind, paths in lens.items()
        }
    else:
        index = mask_index()
        snap.write_text(
            json.dumps(
                [
                    {kind: index[i, kind] for kind in ("dyn", "sky", "road")}
                    for i in range(len(NAMES))
                ]
            )
        )
    tone = None
    if v2:
        tone = np.load(RS / "photo/ground-v2-tone.npz")[name]
        source_ns = s["ns"].copy()
        s["ns"] = s["ns"].copy()
        s["tc"] = s["tc"].copy()
        for i, c in enumerate(CAMS):
            rows0 = frame_table(c)
            ft = {int(v["n"]): float(v["t"]) for v in rows0}
            masked = sorted(
                index[i, "road"].keys()
                & index[i, "sky"].keys()
                & index[i, "dyn"].keys()
            )
            ns0 = np.array(masked, dtype=int)
            mt = np.array([ft[n] for n in masked])
            if not len(mt):
                continue
            j = np.searchsorted(mt, s["tc"][:, i], side="right") - 1
            dt = s["tc"][:, i] - mt[np.maximum(0, j)]
            use = (j >= 0) & (dt <= 0.055)
            s["ns"][use, i] = ns0[j[use]]
            s["tc"][use, i] = mt[j[use]]
        np.savez(work / "schedule.npz", **s, original_ns=source_ns)
        # Publish the exact appearance transform keyed by the physical frames
        # used by this variant, including the common illumination gauge.
        pp = dict(np.load(RS / "photo" / ("live-v2.npz" if live else "stitched.npz")))
        pp.update(gain=tone, ns=s["ns"], ready=s["ready"])
        np.savez(RS / "photo" / f"{name}-ground-v2.npz", **pp)
    lib = native(work)
    t = s["t"]
    mid = tr.T_world_ego(t)
    p = mid[:, :2, 3]
    xyz = {}
    intr = []
    rows = []
    bmask = []
    region = {}
    sh = []
    ux, uy = np.meshgrid((np.arange(65) + 0.5) / 65, (np.arange(49) + 0.5) / 49)
    bb = basis(np.stack((ux, uy), -1).reshape(-1, 2))
    dtype = "d"
    # Relative suspension pitch is more relevant near the car than absolute
    # global attitude. Past innovations set a near-field floor; the full pose
    # uncertainty enters progressively beyond 23 m.
    pose = np.load(RS / posefile)
    rot = np.interp(t, pose["t"], pose["sigma_rot"])
    sp = np.interp(t, pose["t"], pose["susp_pitch"])
    innov = np.abs(np.diff(sp, prepend=sp[0]))
    near = np.array(
        [
            np.clip(
                0.00017 + 0.5 * np.median(innov[max(0, j - 7) : j + 1]),
                0.00017,
                0.00085,
            )
            for j in range(len(t))
        ]
    )
    yaw = np.unwrap(np.arctan2(mid[:, 1, 0], mid[:, 0, 0]))
    ds = s["ds"]
    curvature = np.array(
        [
            abs(yaw[j] - yaw[b]) / max(1.0, ds[j] - ds[b])
            for j in range(len(t))
            for b in [np.searchsorted(ds, max(0, ds[j] - 6))]
        ]
    )
    for i, c in enumerate(CAMS):
        assert c.model == "OPENCV_FISHEYE"
        intr.append(np.ascontiguousarray(np.r_[c.params, c.limit], dtype))
        tt = c.row_time(s["tc"][:, i, None], np.linspace(0, c.height, 17))
        E = tr.T_world_ego(tt)
        cw = E @ c.T_ego_cam
        n, h = tr.ground_plane(tt)
        nw = np.einsum("...ij,...j->...i", E[..., :3, :3], n)
        dw = h - (nw * E[..., :3, 3]).sum(-1)
        rows.append(
            np.ascontiguousarray(
                np.concatenate(
                    (
                        cw[..., :3, :3].swapaxes(-1, -2).reshape(len(t), 17, 9),
                        cw[..., :3, 3],
                        nw,
                        dw[..., None],
                    ),
                    -1,
                ),
                dtype,
            )
        )
        ctm = E[:, 8]
        xyz[i] = np.ascontiguousarray(
            np.stack(
                (
                    ctm[:, 0, 3],
                    ctm[:, 1, 3],
                    np.cos(np.arctan2(ctm[:, 1, 0], ctm[:, 0, 0])),
                    np.sin(np.arctan2(ctm[:, 1, 0], ctm[:, 0, 0])),
                    np.full(len(t), c.T_ego_cam[2, 3]),
                    np.full(len(t), c.rs_readout_s or 0),
                    np.arange(len(t)),
                    s["still"] & np.r_[False, s["still"][:-1]],
                    near,
                    np.clip(rot, 0.0006, 0.004),
                    curvature,
                ),
                1,
            ),
            dtype,
        )
        bmask.append(
            cv.erode(
                cv.resize(
                    c.mask.astype("uint8"),
                    (c.width, c.height)
                    if v2 and i == 0
                    else ((c.width + 1) // 2, (c.height + 1) // 2),
                    interpolation=cv.INTER_NEAREST,
                ),
                np.ones((3, 3), "uint8"),
            )
        )
        sh.append(np.stack([(bb @ v[i]).reshape(49, 65, 3) for v in photo.shade]))
    for k, (x, y) in enumerate(p):
        reach = 30.1 if v2 else 20.1
        head = mid[k, :2, 0]
        head = head / np.linalg.norm(head)
        left = np.array([-head[1], head[0]])
        all_xy = p[k] + np.array(
            [[sx, sy] for sx in [-reach, reach] for sy in [-reach, reach]]
        )
        if v2:
            all_xy = np.vstack(
                (
                    all_xy,
                    *[
                        p[k] + l * head + np.array([-20.5, 20.5])[:, None] * left
                        for l in [-2, 82.5]
                    ],
                )
            )
        lo = np.floor(all_xy.min(0) / 50).astype(int)
        hi = np.floor(all_xy.max(0) / 50).astype(int)
        for tx in range(lo[0], hi[0] + 1):
            for ty in range(lo[1], hi[1] + 1):
                edge = np.hypot(
                    max(tx * 50 - x, 0, x - (tx + 1) * 50),
                    max(ty * 50 - y, 0, y - (ty + 1) * 50),
                )
                dc = np.array([(tx + 0.5) * 50 - x, (ty + 0.5) * 50 - y])
                hwidth = 25 * np.abs(head).sum()
                L = dc @ head
                lat = abs(dc @ left)
                if edge < reach or (
                    v2
                    and L + hwidth >= -2
                    and L - hwidth <= 82.5
                    and lat - hwidth < 20.5
                ):
                    region.setdefault((tx, ty), []).append(k)
    ptr = lambda a: None if a is None else a.ctypes.data

    @lru_cache(maxsize=84)
    def image(i, k):
        im = frame(i, s["ns"][k, i], 1 if v2 and i == 0 else 2)
        shape = (im.shape[1], im.shape[0])
        bad, road = read_mask(index, i, s["ns"][k, i], shape)
        ma = bmask[i] if bad is None else bmask[i] & (1 - bad)
        ma = ma if road is None else ma | (road * 2) | 4
        shade = np.exp(
            np.clip(
                sh[i][photo.epoch[k]] + (tone[k, i] if v2 else photo.gain[k, i]),
                -0.65,
                0.65,
            )
        ).astype("f")
        return im, ma, shade

    def tile(item):
        (tx, ty), keys = item
        g = lib.create(tx, ty, int(live) + 2 * int(v2))
        tag = f"{tx}_{ty}"
        tmp = scr / (tag + ".vertices")
        raw = scr / (tag + ".rgb")
        raised = road_shapes(field, tx, ty)
        height = None
        nextpt = 0
        applied = 0
        geo = np.zeros((220, 220), "f") if raised else None
        stats = np.zeros(8 if v2 else 6, "i4")
        try:
            for k in keys:
                # The map is immutable on a stop; frame masks still update visibility at 4 Hz.
                while nextpt < len(raised) and (not live or raised[nextpt][0] <= t[k]):
                    sec, x, y, z = raised[nextpt]
                    fx = ((x + 0.5) * 1.5 - tx * 50) * 220 / 50
                    fy = ((y + 0.5) * 1.5 - ty * 50) * 220 / 50
                    a = max(0, int(fx) - 7)
                    b = min(220, int(fx) + 9)
                    c0 = max(0, int(fy) - 7)
                    d = min(220, int(fy) + 9)
                    if b > a and d > c0:
                        xx, yy = np.meshgrid(np.arange(a, b), np.arange(c0, d))
                        rr = np.hypot(fx - xx, fy - yy) * 50 / 220
                        geo[c0:d, a:b] = np.maximum(
                            geo[c0:d, a:b],
                            z * np.minimum(1, np.maximum(0, (1.6 - rr) / 0.85)),
                        )
                        applied += 1
                    nextpt += 1
                    height = geo
                far_tile = (
                    np.hypot(
                        max(tx * 50 - p[k, 0], 0, p[k, 0] - (tx + 1) * 50),
                        max(ty * 50 - p[k, 1], 0, p[k, 1] - (ty + 1) * 50),
                    )
                    > 30.1
                )
                for i, c in enumerate(CAMS):
                    if v2 and i and far_tile:
                        continue
                    im, ma, shade = image(i, k)
                    lib.see(
                        g,
                        i,
                        s["tc"][k, i] + c.time_offset_s,
                        ptr(xyz[i][k]),
                        ptr(rows[i][k]),
                        ptr(intr[i]),
                        c.width,
                        c.height,
                        ptr(im),
                        im.shape[1],
                        im.shape[0],
                        ptr(ma),
                        ptr(shade),
                        ptr(height),
                    )
                birth = s["ready"][k] if live else t[k]
                lib.packet(g, t[k], birth if birth <= t[-1] else INF, k)
            n = lib.finish(g, os.fsencode(tmp), os.fsencode(raw), ptr(stats))
        finally:
            lib.release(g)
        if not n:
            return None
        a = np.memmap(tmp, "<f4", "r", shape=(n, 24))
        bbox = np.r_[a[:, :3].min(0), a[:, :3].max(0)].tolist()
        tmin = float(a[:, 17].min())
        f = scene / "tiles" / (tag + ".ply")
        stage = f.with_suffix(".partial")
        with open(stage, "wb") as out, open(tmp, "rb") as inp:
            out.write(binary_header(n))
            shutil.copyfileobj(inp, out, 2**20)
        os.replace(stage, f)
        del a
        tmp.unlink()
        return (
            dict(
                id=tag,
                kind="static",
                layer="ground",
                file="tiles/" + f.name,
                count=n,
                bbox=bbox,
                t_min=tmin,
                t_max=float(t[-1]),
            ),
            stats.tolist(),
            applied,
        )

    items = sorted(region.items(), key=lambda q: q[1][0])
    print(name, "tiles", len(items), "start", flush=True)
    done = []
    with ThreadPoolExecutor(int(os.environ.get("GROUND_THREADS", 12))) as pool:
        for j, res in enumerate(pool.map(tile, items)):
            if res:
                done.append(res)
            if j % 25 == 24:
                print(
                    name,
                    j + 1,
                    "/",
                    len(items),
                    "splats",
                    sum(r[0]["count"] for r in done),
                    flush=True,
                )
    tiles = [r[0] for r in done]
    old = (
        json.loads((scene / "map.json").read_text())["tiles"]
        if (scene / "map.json").exists()
        else []
    )
    content = dict(t_end=float(t[-1]), tiles=tiles, sky=None)
    manifest = scene / "map.json"
    (scene / "map.partial").write_text(json.dumps(content, separators=(",", ":")))
    os.replace(scene / "map.partial", manifest)
    used = {r["file"] for r in tiles}
    for tile0 in old:
        p0 = scene / tile0["file"]
        if tile0["file"] not in used and p0.exists() and p0.parent == scene / "tiles":
            os.replace(p0, scr / (p0.stem + ".previous-ply"))
    shutil.copyfile(RS / "calib/rig.json", scene / "rig.json")
    subprocess.run(
        [
            sys.executable,
            RS / "pipeline/scene/export_drive.py",
            RS,
            scene.name,
            "--poses",
            posefile,
        ],
        check=True,
    )
    ms = dict(
        splats=sum(r["count"] for r in tiles),
        tiles=len(tiles),
        bytes=sum((scene / r["file"]).stat().st_size for r in tiles),
        seconds=time.monotonic() - start,
        levels=np.sum([r[1] for r in done], axis=0).tolist(),
        height_updates=sum(r[2] for r in done),
        mask_hits={
            NAMES[i]: {
                kind: sum(int(n) in index[i, kind] for n in s["ns"][:, i])
                for kind in ("dyn", "sky", "road")
            }
            for i in range(len(NAMES))
        },
        mask_frames={
            NAMES[i]: {z: len(index[i, z]) for z in ("dyn", "sky", "road")}
            for i in range(len(NAMES))
        },
    )
    (scr / "summary.json").write_text(json.dumps(ms, indent=2))
    print(name, ms, flush=True)


if __name__ == "__main__":
    for name in sys.argv[1:] or ("live", "stitched"):
        build(name, int(os.environ.get("GROUND_VERSION", "1")))
