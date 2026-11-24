"""Track spatially distributed corners on distance/time-selected rig frames, CPU only."""

import os, sys, json
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor
import numpy as np, cv2 as cv
from pipeline.common import hardware
from pipeline.common.rig import load_rig, frame_table

RS = Path(os.environ.get("RS", ".")).resolve()
D = RS / "data/drive"
W = RS / "work/poses"
OUT = RS / "poses"
# The cameras of the calibrated rig, in the index order of every per-camera array: six on HW4, four on HW3.
NAMES = hardware.roster(RS)
NC = len(NAMES)


def schedule():
    W.mkdir(exist_ok=True, parents=True)
    tl = np.load(D / "timeline.npz")
    t = tl["t"]
    v = tl["speed"]
    stop = abs(v) < 0.01
    dist = np.r_[
        0, np.cumsum(np.diff(t) * (np.maximum(v[:-1], 0) + np.maximum(v[1:], 0)) * 0.5)
    ]
    # The car drops 0.2 to 0.7 s of video between one-minute files. The last frame before a long gap gets a
    # packet of its own: otherwise its pose waits for the first exposure after the gap, which is later than
    # the half second the road stage allows the live trajectory.
    gap = np.r_[np.diff(t) > 0.45, False]
    sel = [0]
    for i in range(1, len(t)):
        j = sel[-1]
        change = stop[i] != stop[i - 1]
        border = i + 1 < len(t) and stop[i + 1] != stop[i]
        if (
            change
            or border
            or gap[i]
            or (not stop[i] and (dist[i] - dist[j] >= 0.50 or t[i] - t[j] >= 0.16))
            or i == len(t) - 1
        ):
            sel.append(i)
    ix = np.array(sel)
    T = t[ix]
    ns = []
    cs = []
    cams = load_rig(RS / "calib/rig.json")
    for name in NAMES:
        rr = frame_table(cams[name])
        a = np.array([r["t"] for r in rr])
        idx = np.clip(np.searchsorted(a, T), 1, len(a) - 1)
        idx -= abs(a[idx - 1] - T) < abs(a[idx] - T)
        ns.append(np.array([rr[j]["n"] for j in idx]))
        cs.append(a[idx])
    np.savez(
        W / "schedule.npz",
        ix=ix,
        t=T,
        ns=np.array(ns).T,
        tc=np.array(cs).T,
        stop=stop[ix],
        distance=dist[ix],
    )
    print(
        "selected",
        len(ix),
        "stationary intervals",
        [
            (round(t[a], 2), round(t[b], 2))
            for a, b in zip(
                np.flatnonzero(stop & ~np.r_[False, stop[:-1]]),
                np.flatnonzero(stop & ~np.r_[stop[1:], False]),
            )
        ],
        flush=True,
    )


def detect(img, mask, p, w, h):
    # Per-tile quotas keep roadside, road surface and sparse image quadrants represented.
    nx, ny = 10, 6
    ix = np.clip((p[:, 0] * nx / w).astype(int), 0, nx - 1)
    iy = np.clip((p[:, 1] * ny / h).astype(int), 0, ny - 1)
    cnt = np.bincount(iy * nx + ix, minlength=nx * ny)
    m = mask.copy()
    for x, y in p.astype(int):
        cv.circle(m, (int(x), int(y)), 8, 0, -1)
    new = []
    for tile in np.flatnonzero(cnt < 6):
        y, x = divmod(int(tile), nx)
        a = x * w // nx
        b = (x + 1) * w // nx
        u = y * h // ny
        v = (y + 1) * h // ny
        pts = cv.goodFeaturesToTrack(
            img[u:v, a:b], 9 - int(cnt[tile]), 0.016, 9, mask=m[u:v, a:b], blockSize=5
        )
        if pts is not None:
            new.append(pts.reshape(-1, 2) + [a, u])
    return np.vstack(new).astype("f") if new else np.empty((0, 2), "f")


def camera(name):
    cv.setNumThreads(2)
    cv.setRNGSeed(317)
    s = np.load(W / "schedule.npz")
    ci = NAMES.index(name)
    ns = s["ns"][:, ci]
    t = s["tc"][:, ci]
    cam = load_rig(RS / "calib/rig.json")[name]
    # Track the front lens at up to 1448 px across and the others at up to 960, never above native size.
    w = min(cam.width, 1448 if ci == 0 else 960)
    h = round(cam.height * w / cam.width)
    scale = np.array([cam.width / w, cam.height / h])
    mask = cv.erode(
        cv.resize(
            cam.mask.astype("uint8") * 255, (w, h), interpolation=cv.INTER_NEAREST
        ),
        np.ones((9, 9), np.uint8),
    )
    mode = cv.IMREAD_REDUCED_GRAYSCALE_2 if cam.width >= 2 * w else cv.IMREAD_GRAYSCALE
    px = []
    ids = []
    rows = [0]
    cnt = 0
    prev = None
    points = np.empty((0, 2), "f")
    vv = points.copy()
    tid = np.empty(0, "i")
    for k, n in enumerate(ns):
        im = cv.imread(str(D / "frames" / name / f"{n:06d}.jpg"), mode)
        im = (
            cv.resize(im, (w, h), interpolation=cv.INTER_AREA)
            if im.shape != (h, w)
            else im
        )
        if (
            k
            and len(points)
            and 0 < t[k] - t[k - 1] < 0.53
            and s["distance"][k] - s["distance"][k - 1] < 4.0
        ):
            dt = t[k] - t[k - 1]
            initial = (points + vv * min(dt, 0.18)).astype("f")
            end, st, err = cv.calcOpticalFlowPyrLK(
                prev,
                im,
                points,
                initial,
                winSize=(21, 21),
                maxLevel=3,
                criteria=(3, 20, 0.018),
                flags=cv.OPTFLOW_USE_INITIAL_FLOW,
            )
            back, sta, e = cv.calcOpticalFlowPyrLK(
                im,
                prev,
                end,
                points.copy(),
                winSize=(19, 19),
                maxLevel=2,
                criteria=(3, 15, 0.03),
                flags=cv.OPTFLOW_USE_INITIAL_FLOW,
            )
            valid = (
                (st.ravel() > 0)
                & (sta.ravel() > 0)
                & (np.linalg.norm(back - points, axis=1) < 0.80)
                & (err.ravel() < 26)
                & (end[:, 0] > 4)
                & (end[:, 0] < w - 5)
                & (end[:, 1] > 4)
                & (end[:, 1] < h - 5)
            )
            valid &= (
                mask[
                    np.clip(end[:, 1].astype(int), 0, h - 1),
                    np.clip(end[:, 0].astype(int), 0, w - 1),
                ]
                > 0
            )
            vv = (end[valid] - points[valid]) / dt
            points = end[valid]
            tid = tid[valid]
        else:
            points = np.empty((0, 2), "f")
            vv = points.copy()
            tid = np.empty(0, "i")
        new = detect(im, mask, points, w, h)
        points = np.vstack((points, new))
        tid = np.r_[tid, np.arange(cnt, cnt + len(new), dtype="i")]
        cnt += len(new)
        vv = np.vstack((vv, np.zeros_like(new)))
        ids.append(tid.copy())
        px.append(((points + 0.5) * scale).astype("f"))
        rows.append(rows[-1] + len(tid))
        prev = im
    np.savez(
        W / f"lk-{name}.npz",
        id=np.concatenate(ids),
        uv=np.concatenate(px),
        ptr=np.array(rows),
        total=cnt,
    )
    print(name, len(ns), cnt, rows[-1], flush=True)


if __name__ == "__main__":
    schedule()
    list(ProcessPoolExecutor(NC).map(camera, NAMES))
