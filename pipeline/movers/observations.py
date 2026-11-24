"""Native-rate front-to-back LK affine propagation, reset only on real instance masks."""

import cv2, json, numpy as np
from pathlib import Path
from .inventory import W
from pipeline.common.hardware import reduction
from pipeline.common.rig import load_rig, frame_table

cv2.setNumThreads(1)


def read(c, n, grey=False):
    # Decode as small as still covers the 724 px working grid.
    q = reduction(load_rig()[c].width, 724)
    flag = (
        {1: cv2.IMREAD_GRAYSCALE, 2: cv2.IMREAD_REDUCED_GRAYSCALE_2, 4: cv2.IMREAD_REDUCED_GRAYSCALE_4}
        if grey
        else {1: cv2.IMREAD_COLOR, 2: cv2.IMREAD_REDUCED_COLOR_2, 4: cv2.IMREAD_REDUCED_COLOR_4}
    )[q]
    img = cv2.imread(f"data/drive/frames/{c}/{n:06}.jpg", flag)
    return cv2.resize(img, (724, 469)) if img.shape[:2] != (469, 724) else img


def observations(spec):
    extra = spec.get("input") == "extra"
    root = W / "extra" if extra else Path(".")
    data = json.load(open(W / (("extra-" if extra else "") + "inventory.json")))
    rows = []
    cs = load_rig()
    for c, ids in spec["fragments"].items():
        table = frame_table(cs[c])
        src = {
            x[2]: x
            for x in data
            if x[0] == c
            and x[3] in ids
            and spec["window"][0] <= x[1] <= spec["window"][1]
        }
        if not src:
            continue
        indices = {}
        for base in [Path("."), W / "extra"]:
            p = base / "depth" / c / "index.json"
            if p.exists():
                indices.update({int(k): base for k in json.load(open(p))})
        prev = None
        pts = None
        age = 99
        hold = set(sorted(src)[5::9])
        for n in range(min(src), max(src) + 5):
            t = table[n]["t"]
            x = src.get(n)
            if n in hold:
                held_bb = np.array(x[6:10]) + 0.5
                ft = [(held_bb[0] + held_bb[2]) * 0.5, held_bb[3]]
                rt = cs[c].row_time(t, ft[1] * cs[c].height / 469)
                rows.append(
                    dict(
                        c=c,
                        n=n,
                        t=t,
                        rt=rt,
                        release=t + cs[c].time_offset_s + abs(cs[c].rs_readout_s) / 2,
                        id=x[3],
                        key=True,
                        held=True,
                        bb=held_bb.tolist(),
                        foot=ft,
                        depth=0,
                    )
                )
                x = None
            age = 0 if x else age + 1
            if age > 6:
                prev = pts = None
                continue
            im = read(c, n, True)
            dep = 0
            if x:
                mask = (
                    cv2.imread(str(root / "masks/dyn" / c / f"{n:06}.png"), -1) == x[3]
                ).astype("uint8")
                y, u = np.where(mask)
                l, top, r, b = x[6:10]
                bb = np.array([l + 0.5, top + 0.5, r + 0.5, b + 0.5])
                foot = np.array(
                    [
                        np.median(u[y >= np.percentile(y, 97)]) + 0.5,
                        np.percentile(y, 99.7) + 0.5,
                    ]
                )
                points = cv2.goodFeaturesToTrack(im, 80, 0.008, 2, mask=mask)
                pts = points.reshape(-1, 1, 2) if points is not None else None
                mid = x[3]
                if n in indices:
                    z = np.load(indices[n] / "depth" / c / f"{n:06}.npy")[
                        mask.astype(bool)
                    ]
                    z = z[(z > 0) & np.isfinite(z)]
                    dep = float(np.median(z)) if len(z) > 8 else 0
            else:
                if prev is None or pts is None or len(pts) < 4:
                    continue
                nxt, ok, _ = cv2.calcOpticalFlowPyrLK(
                    prev, im, pts, None, winSize=(17, 17), maxLevel=3
                )
                bak, ok2, _ = cv2.calcOpticalFlowPyrLK(
                    im, prev, nxt, None, winSize=(17, 17), maxLevel=3
                )
                keep = (
                    (ok[:, 0] > 0)
                    & (ok2[:, 0] > 0)
                    & (np.linalg.norm((bak - pts)[:, 0], axis=1) < 1.15)
                )
                a, pp = pts[keep, 0], nxt[keep, 0]
                if len(a) < 4:
                    pts = None
                    continue
                # Similarity estimated robustly; use translation and scale, not a rolling box rotation.
                A, good = cv2.estimateAffinePartial2D(
                    a, pp, method=cv2.RANSAC, ransacReprojThreshold=1.3
                )
                if A is None or good.sum() < 4 or abs(A[0, 0] - 1) > 0.22:
                    pts = None
                    continue
                d = np.median((pp - a)[good[:, 0] > 0], axis=0)
                ctr = (bb[:2] + bb[2:]) * 0.5
                bb = np.r_[
                    (bb[:2] - ctr) * A[0, 0] + ctr + d,
                    (bb[2:] - ctr) * A[0, 0] + ctr + d,
                ]
                foot = (foot - ctr) * A[0, 0] + ctr + d
                pts = pp[good[:, 0] > 0, None, :]
                if bb[2] < 5 or bb[0] > 719:
                    pts = None
                    continue
            prev = im
            rt = float(cs[c].row_time(t, foot[1] * cs[c].height / 469))
            release = max(
                t + (cs[c].time_offset_s or 0) + abs(cs[c].rs_readout_s or 0) / 2, t
            )
            rows.append(
                dict(
                    c=c,
                    n=n,
                    t=t,
                    rt=rt,
                    release=release,
                    id=mid,
                    key=bool(x),
                    bb=bb.tolist(),
                    foot=foot.tolist(),
                    depth=dep,
                    depth_root=str(indices[n]) if dep else None,
                )
            )
    return sorted(rows, key=lambda x: x["rt"])
