"""A common physical-frame schedule and repeatable overlap radiometry samples."""

from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
import os
import cv2 as cv
import numpy as np
from pipeline.common.hardware import roster
from pipeline.common.rig import load_rig, Trajectory, frame_table
from pipeline.road.geometry import ground_to_pixels

cv.setNumThreads(1)
RS = Path(os.environ.get("RS", ".")).resolve()
DATA = RS / "data/drive"
PHOTO = RS / "photo"
WORK = RS / "work/photo"
# The lenses of the calibrated rig, in the index order of every per-lens array: six on HW4, four on HW3.
NAMES = roster(RS)
NC = len(NAMES)
_rig = load_rig(RS / "calib/rig.json")
CAMS = [_rig[n] for n in NAMES]


def schedule():
    d = np.load(DATA / "timeline.npz")
    t = d["t"]
    v = np.maximum(d["speed"], 0)
    ds = np.r_[0, np.cumsum((v[1:] + v[:-1]) * 0.5 * np.diff(t))]
    keys = [0]
    for i in range(1, len(t)):
        j = keys[-1]
        still = v[i] < 0.2
        if (
            i == len(t) - 1
            or ((v[j] < 0.2) != still)
            or (still and t[i] - t[j] >= 0.25)
            or (
                not still
                and (ds[i] - ds[j] >= max(0.5, v[i] / 9) or t[i] - t[j] >= 0.28)
            )
        ):
            keys.append(i)
    ix = np.array(keys)
    ns = []
    tc = []
    last = np.full(len(ix), -1.0)
    for c in CAMS:
        rr = frame_table(c)
        ft = np.array([z["t"] for z in rr])
        ids = np.clip(np.searchsorted(ft, t[ix]), 1, len(ft) - 1)
        ids -= abs(ft[ids - 1] - t[ix]) < abs(ft[ids] - t[ix])
        tc.append(ft[ids])
        ns.append(np.array([z["n"] for z in rr])[ids])
        last = np.maximum(
            last, ft[ids] + c.time_offset_s + 0.5 * abs(c.rs_readout_s or 0)
        )
    lag = np.load(RS / "work/road/causal-lag.npy")
    mi = np.clip(np.searchsorted(t, last), 0, len(t) - 1)
    ready = np.maximum(
        last,
        np.maximum(
            t[mi] + lag[mi], t[np.maximum(0, mi - 1)] + lag[np.maximum(0, mi - 1)]
        ),
    )
    ready = np.maximum.accumulate(ready)
    return dict(
        t=t[ix],
        ix=ix,
        ns=np.stack(ns, 1),
        tc=np.stack(tc, 1),
        ready=ready,
        still=v[ix] < 0.2,
        ds=ds[ix],
    )


def mask_index():
    return {
        (i, k): {
            int(p.stem): str(p)
            for p in (RS / "masks" / k / n).glob("*.png")
            if p.stem.isdigit()
        }
        for i, n in enumerate(NAMES)
        for k in ("dyn", "sky", "road")
    }


def read_mask(index, i, n, shape):
    bad = None
    road = None
    for kind in ["dyn", "sky", "road"]:
        p = index[i, kind].get(int(n))
        if not p:
            continue
        a = cv.imread(p, cv.IMREAD_UNCHANGED)
        a = (a != 0).astype("uint8")
        if kind == "road":
            road = cv.resize(a, shape, interpolation=cv.INTER_NEAREST)
            continue
        a = cv.dilate(a, np.ones((5, 5), "uint8"))
        a = cv.resize(a, shape, interpolation=cv.INTER_NEAREST)
        bad = a if bad is None else bad | a
    return bad, road


def frame(i, n, reduce=2):
    return cv.imread(
        str(DATA / "frames" / NAMES[i] / f"{n:06d}.jpg"),
        {
            1: cv.IMREAD_COLOR,
            2: cv.IMREAD_REDUCED_COLOR_2,
            4: cv.IMREAD_REDUCED_COLOR_4,
        }[reduce],
    )


def collect(stops_only=False):
    WORK.mkdir(parents=True, exist_ok=True)
    PHOTO.mkdir(parents=True, exist_ok=True)
    s = schedule()
    np.savez(WORK / "schedule.npz", **s)
    tr = {
        n: Trajectory(
            RS / "poses" / ("ego" + ("_causal" if n == "live" else "") + ".npz")
        )
        for n in ["live", "stitched"]
    }
    index = mask_index()
    keys = (
        np.flatnonzero(s["still"] & np.r_[False, s["still"][:-1]])
        if stops_only
        else np.flatnonzero(~s["still"] | np.r_[True, ~s["still"][:-1]])
    )
    print("packets", len(s["t"]), "exposure samples", len(keys), flush=True)

    def block(chunk):
        output = {n: [] for n in tr}
        for k in chunk:
            for i, cam in enumerate(CAMS):
                im = frame(i, s["ns"][k, i], 4)
                w, h = im.shape[1], im.shape[0]
                blur = cv.GaussianBlur(im, (7, 7), 1.4)
                bad, road = read_mask(index, i, s["ns"][k, i], (w, h))
                tf = s["tc"][k, i]
                for name, v in tr.items():
                    T = v.T_world_ego(cam.row_time(tf, cam.height / 2))
                    p = T[:2, 3]
                    lo = np.floor((p - 12) / 0.8).astype(int)
                    hi = np.ceil((p + 12) / 0.8).astype(int)
                    x, y = np.meshgrid(
                        np.arange(lo[0], hi[0] + 1), np.arange(lo[1], hi[1] + 1)
                    )
                    idx = np.stack((x, y), -1).reshape(-1, 2)
                    pos = idx * 0.8
                    xy = (pos - p) @ T[:2, :2]
                    ok = (np.abs(xy[:, 1]) < 5.8) & (np.linalg.norm(xy, axis=1) < 11.7)
                    idx = idx[ok]
                    pos = pos[ok]
                    uv, ok, z = ground_to_pixels(cam, v, tf, pos, 12.0)
                    q = uv / [cam.width / w, cam.height / h] - 0.5
                    ix = np.clip(np.rint(q[:, 0]).astype(int), 0, w - 1)
                    iy = np.clip(np.rint(q[:, 1]).astype(int), 0, h - 1)
                    rgb = blur[iy, ix, ::-1]
                    ok &= (
                        (rgb.min(1) > 22)
                        & (rgb.max(1) < 222)
                        & (rgb.max(1) - rgb.min(1) < 65)
                        & (abs(im[iy, ix, ::-1].astype("i2") - rgb).max(1) < 15)
                    )
                    if bad is not None:
                        ok &= bad[iy, ix] == 0
                    if road is not None:
                        ok &= road[iy, ix] > 0
                    # Deterministic hashing of physical cells (not observations) for validation.
                    output[name].append(
                        np.column_stack(
                            (
                                idx[ok],
                                np.full(ok.sum(), k),
                                np.full(ok.sum(), i),
                                uv[ok] / [cam.width, cam.height],
                                rgb[ok],
                            )
                        ).astype("f")
                    )
        for name, rows in output.items():
            np.save(
                WORK
                / (
                    f"{name}-" + ("stop-" if stops_only else "") + f"{chunk[0]:05d}.npy"
                ),
                np.vstack(rows),
            )

    chunks = np.array_split(keys, max(1, len(keys) // 35))
    list(ThreadPoolExecutor(14).map(block, chunks))
    print("colour samples ready", flush=True)


if __name__ == "__main__":
    collect()
