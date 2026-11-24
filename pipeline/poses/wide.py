"""Independent RootSIFT cross-lens/time links; never gate validation by the close reprojection."""

from .motion import *
from concurrent.futures import ThreadPoolExecutor
from collections import defaultdict

cv.setNumThreads(1)


def run():
    s, d, t, B, st = inputs()
    ix = s["ix"]
    kt = s["t"]
    ds = s["distance"]
    sel = [0]
    for k in range(1, len(ix)):
        j = sel[-1]
        dd = ds[k] - ds[j]
        v = d["speed"][ix[k]]
        if not s["stop"][k] and (
            dd >= np.clip(v * 0.26, 1.05, 4.2) or (dd > 0.55 and kt[k] - kt[j] > 0.85)
        ):
            sel.append(k)
    sel = np.array(sel)
    root = W / "wide"
    root.mkdir(exist_ok=True, parents=True)
    np.save(root / "select.npy", sel)
    cams = list(load_rig(RS / "calib/rig.json")[n] for n in NAMES)
    masks = [
        cv.resize(
            c.mask.astype("uint8") * 255,
            (1200, round(c.height * 1200 / c.width)),
            interpolation=cv.INTER_NEAREST,
        )
        for c in cams
    ]

    def get(i):
        dest = root / f"f{i:05d}.npz"
        if dest.exists():
            return
        q, c = divmod(i, NC)
        n = s["ns"][sel[q], c]
        # Features are taken 1200 px across; an image at least twice that is decoded at half size.
        im = cv.imread(
            str(D / "frames" / NAMES[c] / f"{n:06d}.jpg"),
            cv.IMREAD_REDUCED_GRAYSCALE_2 if cams[c].width >= 2400 else 0,
        )
        h, w = masks[c].shape
        im = cv.resize(im, (w, h), interpolation=cv.INTER_AREA)
        f = cv.SIFT_create(
            nfeatures=1600,
            contrastThreshold=0.016,
            edgeThreshold=12,
            enable_precise_upscale=True,
        )
        kp, dsc = f.detectAndCompute(im, masks[c])
        p = (np.array([x.pt for x in kp]) + 0.5) * [
            cams[c].width / w,
            cams[c].height / h,
        ]
        idx = np.sort(
            np.unique(np.round(2 * p).astype(int), axis=0, return_index=True)[1]
        )
        dsc = dsc[idx]
        dsc = np.rint(
            255 * np.sqrt(dsc / np.maximum(1, dsc.sum(1, keepdims=True)))
        ).astype("uint8")
        np.savez(dest, p=p[idx].astype("f"), d=dsc)

    print("wide frames", len(sel), flush=True)
    list(ThreadPoolExecutor(12).map(get, range(NC * len(sel))))
    print("wide features ready", flush=True)
    pairs = defaultdict(set)

    # Image id = selected keyframe * NC + lens.
    def add(i, j):
        if (
            i != j
            and 0 <= j < NC * len(sel)
            and abs(kt[sel[i // NC]] - kt[sel[j // NC]]) < 12
        ):
            a, b = sorted((i, j))
            pairs[b].add(a)

    for q in range(len(sel)):
        for c in range(NC):
            for off in [1, 2, 4]:
                add(NC * q + c, NC * (q + off) + c)
        # Lens pairs with a shared view among the cameras this car has, and how many keyframes apart.
        for a, b, offs in hardware.cross(NAMES):
            for off in offs:
                add(NC * q + a, NC * (q + off) + b)
    # Descriptors are compact uint8 while idle. Float conversion is local to each match.
    P = []
    F = []
    for i in range(len(sel) * NC):
        with np.load(root / f"f{i:05d}.npz") as f:
            P.append(f["p"])
            F.append(f["d"])

    def group(j):
        dest = root / f"m{j:05d}.npz"
        if dest.exists():
            return
        idx = cv.flann_Index(F[j].astype("f"), dict(algorithm=1, trees=3))
        m = {}
        for i in pairs[j]:
            nn, ds = idx.knnSearch(F[i].astype("f"), 2, params={"checks": 20})
            a = np.flatnonzero(ds[:, 0] < 0.75**2 * ds[:, 1])
            b = nn[a, 0]
            if len(a) < 9:
                continue
            _, ok = cv.findFundamentalMat(
                P[i][a], P[j][b], cv.USAC_MAGSAC, 3, 0.996, 650
            )
            if ok is not None and ok.sum() >= 9:
                ok = ok.ravel().astype(bool)
                m[str(i)] = np.stack((a[ok], b[ok]), -1).astype("uint16")
        np.savez(dest, **m)

    list(ThreadPoolExecutor(14).map(group, pairs))
    print("wide pair matches ready", flush=True)
    # Feature-index union: reject contradictory observations from the same image.
    parent = {}
    views = {}
    edges = []
    for j in pairs:
        with np.load(root / f"m{j:05d}.npz") as f:
            for key in f.files:
                edges.append((int(key), j, f[key]))
    edges.sort(
        key=lambda z: (z[0] % NC != z[1] % NC, abs(z[0] // NC - z[1] // NC), -len(z[2]))
    )

    def rootid(i):
        while i in parent:
            i = parent[i]
        return i

    for i, j, a in edges:
        for u, v in zip(
            (a[:, 0].astype(int) + 2048 * i).tolist(),
            (a[:, 1].astype(int) + 2048 * j).tolist(),
        ):
            x, y = rootid(u), rootid(v)
            if x == y:
                continue
            sx = views.get(x, {i})
            sy = views.get(y, {j})
            if not sx.isdisjoint(sy):
                continue
            if len(sx) < len(sy):
                x, y = y, x
                sx, sy = sy, sx
            parent[y] = x
            sx.update(sy)
            views[x] = sx
            views.pop(y, None)
    good = {
        x: [x]
        for x, vs in views.items()
        if len(vs) >= 4 and len({z % NC for z in vs}) > 1
    }
    for i in parent:
        j = rootid(i)
        if j in good:
            good[j].append(i)
    tr = list(good.values())
    ids = np.repeat(np.arange(len(tr)), [len(v) for v in tr])
    n = np.concatenate([sorted(v) for v in tr])
    im = n // 2048
    p = np.vstack([np.pad(x, ((0, 2048 - len(x)), (0, 0))) for x in P])[n]
    np.savez(
        W / "cross-tracks.npz",
        id=ids.astype("i"),
        k=sel[im // NC].astype("i"),
        cam=(im % NC).astype("uint8"),
        uv=p,
    )
    print("cross-landmarks", len(tr), "obs", len(n), flush=True)


if __name__ == "__main__":
    run()
