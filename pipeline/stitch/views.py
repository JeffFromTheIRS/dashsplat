"""Static split and native-row shutter bands; no held-out RGB in the optimizer."""

import json, os, time
from functools import lru_cache
from pathlib import Path
import cv2 as cv
import numpy as np
from pipeline.common.hardware import reduction, roster
from pipeline.common.rig import load_rig, Trajectory, T_world_cam, undistort_maps
from pipeline.photo.model import Photometry

RS = Path(os.environ.get("RS", ".")).resolve()
WORK = Path(os.getenv("STITCH_WORK", RS / "work/stitch")).resolve()
WORK.mkdir(parents=True, exist_ok=True)
START = float(os.getenv("STITCH_START", 85))
END = float(os.getenv("STITCH_END", 140))
CACHE = WORK / f"views-{START:g}-{END:g}"
# The cameras of the calibrated rig in schedule-column order: six on HW4, four on HW3.
CAMS = roster(RS)
W, H, BANDS = 336, 216, 8
rig = load_rig(RS / "calib/rig.json")
traj = Trajectory(RS / "poses/ego.npz")
photo = Photometry("stitched-ground-v2", RS)
cv.setNumThreads(1)


def J(p):
    return json.loads(p.read_text())


def savej(p, a):
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(a, indent=2))


def designs(c):
    return [-52, 0, 52] if c == "back" else [0]


@lru_cache(None)
def design(c, yaw):
    a = np.deg2rad(yaw)
    R = np.array([[np.cos(a), 0, np.sin(a)], [0, 1, 0], [-np.sin(a), 0, np.cos(a)]])
    f = W / (2 * np.tan(np.deg2rad({"front": 44, "back": 83}.get(c, 80)) / 2))
    K = np.array([[f, 0, W / 2], [0, f, H / 2], [0, 0, 1.0]])
    C = rig[c]
    mx, my, row = undistort_maps(C, K, (W, H), R)
    valid = np.isfinite(row) & C.mask_at(np.stack([mx + 0.5, my + 0.5], -1))
    band = np.where(
        valid,
        np.minimum(BANDS - 1, np.floor(np.nan_to_num(row) * BANDS / C.height)),
        -1,
    ).astype("i1")
    bounds = []
    for i in range(BANDS):
        yy = np.where(band == i)[0]
        bounds.append((int(yy.min()), int(yy.max()) + 1) if len(yy) else (0, 0))
    return K, R, mx, my, band, bounds


def world(c, yaw, t, b):
    C = rig[c]
    row = C.height * (b + 0.5) / BANDS
    T = T_world_cam(C, t, row, traj).copy()
    T[:3, :3] = T[:3, :3] @ design(c, yaw)[1]
    return T


def path(e):
    return CACHE / e["c"] / ("%06d_%d.npz" % (e["n"], e["yaw"]))


def manifest():
    raw = J(RS / "depth/keyframes.json")["frames"]
    ground = np.load(RS / "maps/ground-v2/schedule.npz")["ns"]
    plan = J(RS / "maps/object-v3/plan.json")
    out = []
    for c in CAMS:
        ee = [e for e in raw[c] if START <= e["t_key"] <= END]
        held = {e["n"] for e in ee[7::8]}
        fused = set(ground[:, CAMS.index(c)].tolist())
        fused |= {n for r in plan[c] if r["train"] for n in r["window"]}
        dirs = [RS / "maps/object-v3/stitched/depth" / c, RS / "depth" / c]
        indices = [
            J(d / "index.json") if (d / "index.json").exists() else {} for d in dirs
        ]
        for i, e in enumerate(ee):
            hold = e["n"] in held
            dp = None
            if not hold:
                for d, idx in zip(dirs, indices):
                    rec = idx.get(str(e["n"]), {})
                    fn = d / ("%06d.npy" % e["n"])
                    if (
                        fn.is_file()
                        and rec.get("window")
                        and not (set(rec["window"]) & held)
                    ):
                        dp = str(fn)
                        break
            # Include depth-supervised exposures selected by fusion, plus regular colour views.
            if (
                not hold
                and i % 4 != 1
                and not (dp and any(r["n"] == e["n"] for r in plan[c]))
            ):
                continue
            for yaw in designs(c):
                out.append(
                    dict(
                        c=c,
                        yaw=yaw,
                        n=e["n"],
                        t=e["t"],
                        i=i,
                        hold=hold,
                        clean=e["n"] not in fused,
                        depth=dp,
                    )
                )
    return out


def remap(im, mx, my, C, interp):
    h, w = im.shape[:2]
    return cv.remap(
        im,
        (mx + 0.5) * w / C.width - 0.5,
        (my + 0.5) * h / C.height - 0.5,
        interp,
        borderMode=cv.BORDER_CONSTANT,
    )


def prep_group(group):
    e = group[0]
    c, n = e["c"], e["n"]
    C = rig[c]
    stem = "%06d" % n
    im = cv.imread(
        str(RS / "data/drive/frames" / c / (stem + ".jpg")),
        {1: cv.IMREAD_COLOR, 2: cv.IMREAD_REDUCED_COLOR_2, 4: cv.IMREAD_REDUCED_COLOR_4}[
            reduction(rig[c].width, 724)
        ],
    )
    assert im is not None
    im = photo.apply(c, n, im[:, :, ::-1])
    masks = [
        cv.imread(str(RS / "masks" / k / c / (stem + ".png")), cv.IMREAD_UNCHANGED)
        for k in ("sky", "dyn")
    ]
    assert all(m is not None for m in masks)
    sky, dyn = [(x != 0).astype("B") for x in masks]
    dy = cv.dilate(dyn, np.ones((5, 5), "B"))
    skycore = cv.erode(sky, np.ones((5, 5), "B"))
    skyedge = cv.dilate(sky, np.ones((3, 3), "B"))
    z = np.load(e["depth"]).astype("f") if e["depth"] else None
    for e in group:
        K, R, mx, my, band, _ = design(c, e["yaw"])
        rgb = remap(im, mx, my, C, cv.INTER_LINEAR)
        mask = np.where(
            (band >= 0) & (remap(dy, mx, my, C, 0) == 0),
            np.where(
                remap(skycore, mx, my, C, 0) > 0,
                2,
                np.where(remap(skyedge, mx, my, C, 0) > 0, 0, 1),
            ),
            0,
        ).astype("B")
        depth = (
            remap(z, mx, my, C, 0).astype("float16")
            if z is not None
            else np.zeros(0, "float16")
        )
        p = path(e)
        p.parent.mkdir(parents=True, exist_ok=True)
        np.savez(p, rgb=rgb, mask=mask, z=depth)


if __name__ == "__main__":
    from concurrent.futures import ThreadPoolExecutor
    from itertools import groupby

    tic = time.monotonic()
    # Freeze frame and depth availability for a rerun. Use a new STITCH_WORK for a new input snapshot.
    es = (
        J(CACHE / "manifest.json") if (CACHE / "manifest.json").exists() else manifest()
    )
    groups = [list(g) for _, g in groupby(es, key=lambda e: (e["c"], e["n"]))]
    CACHE.mkdir(parents=True, exist_ok=True)
    with ThreadPoolExecutor(4) as ex:
        list(ex.map(prep_group, groups))
    savej(CACHE / "manifest.json", es)
    print(len(es), len(groups), "views/physical frames", time.monotonic() - tic)
