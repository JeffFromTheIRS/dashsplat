"""Gain-normalised direct road patch registration. Collect raw pixels, not pose residuals."""

from pipeline.poses.motion import inputs, knots, RS, W as PW, D, NAMES, NC, OUT
from pipeline.common.rig import load_rig, Trajectory
import numpy as np, cv2 as cv, json, os
from pathlib import Path
from functools import lru_cache
from concurrent.futures import ThreadPoolExecutor

cv.setNumThreads(1)
WORK = RS / "work/road"
PRODUCT = RS / "road"
SEG = [("park", 84.0, 105.0), ("turn", 316.0, 326.0), ("straight", 293.0, 310.0)]
CELL = 0.025
HALF = 12.1


def rot(y):
    return np.array([[np.cos(y), -np.sin(y)], [np.sin(y), np.cos(y)]])


def road_color(im):
    b, g, r = cv.split(im)
    b, g, r = b.astype("f"), g.astype("f"), r.astype("f")
    grass = (g > r * 1.30 + 5) & (g > b * 1.15 + 4)
    verydark = (b + g + r) < 62
    strong = (np.maximum.reduce([b, g, r]) - np.minimum.reduce([b, g, r]) > 70) & ~(
        (r > b * 1.22) & (g > b * 1.15)
    )
    return (~(grass | verydark | strong)).astype("uint8")


def plan_camera(cam):
    axis = np.arange(-HALF, HALF + 0.002, CELL)
    x, y = np.meshgrid(axis, axis)
    P = np.stack((x, y, x * 0), -1)
    T = cam.T_ego_cam
    ray = (P - T[:3, 3]) @ T[:3, :3]
    uv, ok = cam.project(ray)
    ok &= (
        (x * x + y * y < 12**2)
        & (ray[:, :, 2] > 0.05)
        & ~((x > -0.99) & (x < 3.75) & (abs(y) < 0.94))
    )
    iy, ix = np.nonzero(ok)
    sl = np.s_[iy.min() : iy.max() + 1, ix.min() : ix.max() + 1]
    return (
        (uv[sl].astype("f") - 0.5),
        ok[sl].astype("uint8"),
        np.array([axis[ix.min()], axis[iy.min()]]),
    )


def peak(image, template, mask, delta=4):
    res = cv.matchTemplate(image, template, cv.TM_CCOEFF_NORMED)
    res[mask == 0] = -1
    _, hi, _, (x, y) = cv.minMaxLoc(res)
    if hi < 0.78 or x < 1 or y < 1 or x >= res.shape[1] - 1 or y >= res.shape[0] - 1:
        return None
    l, r, u, d = res[y, x - 1], res[y, x + 1], res[y - 1, x], res[y + 1, x]
    dx = np.clip(0.5 * (l - r) / min(l + r - 2 * hi, -1e-6), -0.65, 0.65)
    dy = np.clip(0.5 * (u - d) / min(u + d - 2 * hi, -1e-6), -0.65, 0.65)
    res[max(0, y - delta) : y + delta + 1, max(0, x - delta) : x + delta + 1] = -1
    margin = hi - res.max()
    return (np.array([x + dx, y + dy]), hi, margin) if margin > 0.065 else None


def build():
    s, d, t, B, stop = inputs()
    kt = s["t"]
    ds = s["distance"]
    tc = s["tc"]
    ns = s["ns"]
    sk, _, _ = knots(s, t, stop)
    near = np.zeros(len(kt), bool)
    for _, a, b in SEG:
        near |= (kt >= a - 2) & (kt <= b + 2)
    keep = list(sk)
    last = -1e6
    for k in np.flatnonzero(near):
        if ds[k] - last > 1.05:
            keep.append(k)
            last = ds[k]
    ix = np.unique(keep)
    ix = ix[~s["stop"][ix]]
    C = [load_rig(RS / "calib/rig.json")[n] for n in NAMES]
    cams = [plan_camera(c) for c in C]
    tr = Trajectory(OUT / "ego_causal_v1.npz")
    xy = []
    yaw = []
    for i, c in enumerate(C):
        T = tr.T_world_ego(c.row_time(tc[:, i], c.height / 2))
        xy.append(T[:, :2, 3])
        yaw.append(np.arctan2(T[:, 1, 0], T[:, 0, 0]))
    xy = np.stack(xy, 1)
    yaw = np.stack(yaw, 1)
    WORK.mkdir(exist_ok=True, parents=True)
    cache = WORK / "patch-blocks"
    cache.mkdir(exist_ok=True)
    HP = 13
    rad = 24
    sid = HP + rad
    disk = np.ones((2 * HP + 1, 2 * HP + 1), np.uint8)

    def block(arg):
        b, chunk = arg
        dest = cache / f"{b:04d}.npz"
        if dest.exists():
            return

        @lru_cache(maxsize=100)
        def view(k, ci):
            mp, valid, origin = cams[ci]
            im = cv.imread(str(D / "frames" / NAMES[ci] / f"{ns[k, ci]:06d}.jpg"))
            rgb = cv.remap(im, mp, None, cv.INTER_LINEAR)
            gray = cv.cvtColor(rgb, cv.COLOR_BGR2GRAY)
            m = road_color(rgb) & valid
            m = cv.erode(m, disk)
            m[: HP + 2] = 0
            m[-HP - 2 :] = 0
            m[:, : HP + 2] = 0
            m[:, -HP - 2 :] = 0  # strongest distinct patches across spatial tiles
            cs = []
            h, w = m.shape
            for y in range(0, h, 120):
                for x in range(0, w, 120):
                    vv = m[y : y + 120, x : x + 120]
                    if vv.max() == 0:
                        continue
                    corn = cv.goodFeaturesToTrack(
                        gray[y : y + 120, x : x + 120],
                        2,
                        0.025,
                        17,
                        mask=vv,
                        blockSize=7,
                    )
                    if corn is not None:
                        cs.extend((corn.reshape(-1, 2) + [x, y]).astype(int).tolist())
            return gray, m, np.array(cs).reshape(-1, 2), origin

        def sample_uv(k, c, pp):
            p = cams[c][2] + pp * CELL
            X = np.r_[p, 0]
            R = C[c].T_ego_cam
            return C[c].project((X - R[:3, 3]) @ R[:3, :3])[0]

        rows = []
        for jj in chunk:
            k = ix[jj]
            targets = []
            for off in [0, 1.7, 5.2, 8.4, 12.5, 17.0]:
                qq = (
                    jj
                    if off == 0
                    else np.searchsorted(ds[ix], ds[k] - off, side="right") - 1
                )
                if qq >= 0 and kt[k] - kt[ix[qq]] < 8:
                    targets.append(ix[qq])
            for ci in range(NC):
                gray, mask0, cs, org = view(k, ci)
                if not len(cs):
                    continue
                world = (org + cs * CELL) @ rot(yaw[k, ci]).T + xy[k, ci]
                for j in np.unique(targets):
                    for cj in range(NC):
                        if k == j and ci >= cj:
                            continue
                        rm = rot(yaw[k, ci] - yaw[j, cj])
                        sp = (
                            (world - xy[j, cj]) @ rot(yaw[j, cj]) - cams[cj][2]
                        ) / CELL
                        mp, static, _ = cams[cj]
                        h, w = static.shape
                        good = (
                            (sp[:, 0] >= 5)
                            & (sp[:, 0] < w - 5)
                            & (sp[:, 1] >= 5)
                            & (sp[:, 1] < h - 5)
                        )
                        ids = np.flatnonzero(good)
                        ids = ids[
                            static[sp[ids, 1].astype(int), sp[ids, 0].astype(int)] != 0
                        ]
                        if not len(ids):
                            continue
                        other, om, _, _ = view(j, cj)
                        for a in ids:
                            px, py = cs[a]
                            tpl = gray[py - HP : py + HP + 1, px - HP : px + HP + 1]
                            if tpl.shape != (2 * HP + 1, 2 * HP + 1) or tpl.std() < 5:
                                continue
                            A = np.column_stack((rm, sp[a] - rm @ [sid, sid]))
                            roi = cv.warpAffine(
                                other,
                                A,
                                (2 * sid + 1, 2 * sid + 1),
                                flags=cv.INTER_LINEAR | cv.WARP_INVERSE_MAP,
                            )
                            M = cv.warpAffine(
                                om,
                                A,
                                (2 * sid + 1, 2 * sid + 1),
                                flags=cv.INTER_NEAREST | cv.WARP_INVERSE_MAP,
                            )
                            pk = peak(roi, tpl, M[HP:-HP, HP:-HP])
                            if pk is None:
                                continue
                            pt, score, margin = pk
                            delta = pt - rad
                            sub = cv.getRectSubPix(
                                roi, (2 * HP + 1,) * 2, tuple((pt + HP).astype(float))
                            )
                            rx, ry = 7, 7
                            tmp = cv.getRectSubPix(
                                gray, (2 * (HP + rx) + 1,) * 2, (float(px), float(py))
                            )
                            reverse = peak(
                                tmp, sub, np.ones((2 * rx + 1,) * 2, "uint8"), 3
                            )
                            if (
                                reverse is None
                                or np.linalg.norm(reverse[0] - rx) > 0.95
                            ):
                                continue
                            matched = sp[a] + rm @ delta
                            uv0 = sample_uv(k, ci, cs[a])
                            uv1 = sample_uv(j, cj, matched)
                            hw = np.floor(world[a] / 1.0).astype(np.int64)
                            hold = int((hw[0] * 5519 + hw[1] * 7699) % 5 == 0)
                            rows.append(
                                [
                                    k,
                                    ci,
                                    *uv0,
                                    j,
                                    cj,
                                    *uv1,
                                    score,
                                    margin,
                                    hold,
                                    *world[a],
                                ]
                            )
        np.savez(dest, m=np.array(rows, "f").reshape((-1, 13)))

    chunks = list(enumerate(np.array_split(np.arange(len(ix)), max(1, len(ix) // 24))))
    print("target packets", len(ix), "blocks", len(chunks), flush=True)
    list(ThreadPoolExecutor(12).map(block, chunks))
    m = np.vstack([np.load(p)["m"] for p in sorted(cache.glob("*.npz"))])
    np.save(WORK / "pairs.npy", m)
    print("matched pairs", len(m), "held out", int(m[:, 10].sum()), flush=True)


if __name__ == "__main__":
    build()
