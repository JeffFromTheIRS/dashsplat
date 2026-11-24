"""Triangulate high-frequency canopy patches in three independent physical frames; conserve one surface per resolved voxel."""

from .v5structures import *
from scipy.spatial import Delaunay, QhullError


@lru_cache(18)
def image(c, n):
    a = cv.imread(
        str(RS / "data/drive/frames" / c / f"{n:06}.jpg"),
        cv.IMREAD_REDUCED_COLOR_2 if rig[c].width >= 2896 else cv.IMREAD_COLOR,
    )
    return a[:, :, ::-1].copy(), cv.cvtColor(a, cv.COLOR_BGR2GRAY)


def triplet(es, c):
    e = es[1]
    im, g = image(c, e["n"])
    h, w = g.shape
    C = rig[c]
    lab = labels(c, e["n"])
    mask = cv.resize((lab == 8).astype("B"), (w, h), interpolation=0)
    q = cv.goodFeaturesToTrack(g, 1700, 0.005, 4, mask=mask, blockSize=5)
    if q is None or len(q) < 15:
        return []
    p = q.reshape(-1, 2)
    views = []
    valid = np.ones(len(p), bool)
    pix = []
    args = dict(winSize=(19, 19), maxLevel=4, criteria=(3, 18, 0.035))
    for k, ek in enumerate(es):
        if k != 1:
            b, g1 = image(c, ek["n"])
            Q, st, err = cv.calcOpticalFlowPyrLK(g, g1, p, None, **args)
            rev, rv, _ = cv.calcOpticalFlowPyrLK(g1, g, Q, None, **args)
            valid &= (
                (st[:, 0] > 0)
                & (rv[:, 0] > 0)
                & (err[:, 0] < 14)
                & (np.linalg.norm(rev - p, axis=1) < 0.55)
            )
        else:
            Q = p
        S = labels(c, ek["n"])
        xy = (Q + 0.5) * [724 / w, 469 / h] - 0.5
        ix, iy = xy.round().astype(int).T
        valid &= (S[iy.clip(0, 468), ix.clip(0, 723)] != 10) & (
            S[iy.clip(0, 468), ix.clip(0, 723)] != 250
        )
        views.append(rays_at(c, ek, *xy.T))
        pix.append(Q)
    O = np.stack([o for o, u in views])
    U = np.stack([u for o, u in views])
    cross = np.cross(U[0], U[2])
    sn = (cross * cross).sum(1)
    d = (np.cross(O[2] - O[0], U[2]) * cross).sum(1) / np.maximum(1e-10, sn)
    X = O[0] + U[0] * d[:, None]
    R = np.linalg.norm(X - O[1], axis=1)
    err = np.linalg.norm(unit(X - O[1]) - U[1], axis=1) * C.params[0] * w / C.width
    valid &= (sn > 0.00016) & (d > 2) & (d < 65) & (R < 65) & (err < 0.8)
    P = X[valid]
    q = p[valid]
    if len(P) < 10:
        return []
    try:
        tri = Delaunay(q).simplices
    except QhullError:
        return []
    uv = q[tri]
    xyz = P[tri]
    sz = np.linalg.norm(uv - np.roll(uv, 1, axis=1), axis=2).max(1)
    ds = R[valid][tri]
    extent = np.linalg.norm(xyz - np.roll(xyz, 1, axis=1), axis=2).max(1)
    ok = (
        (sz < 20)
        & (sz > 3)
        & (np.ptp(ds, axis=1) < 0.13 + 0.013 * ds.mean(1))
        & (extent < 0.9)
        & (extent > 0.035)
    )
    tri = tri[ok]
    uv = q[tri]
    xyz = P[tri]
    if not len(tri):
        return []
    # The barycentric child is texture-sampled separately in all three photos, never assigned its parent's uniform colour.
    bary = np.array(
        [
            [1 / 3, 1 / 3, 1 / 3],
            [0.68, 0.16, 0.16],
            [0.16, 0.68, 0.16],
            [0.16, 0.16, 0.68],
            [0.45, 0.45, 0.10],
            [0.45, 0.10, 0.45],
            [0.10, 0.45, 0.45],
        ]
    )
    cent = (bary @ xyz).reshape(-1, 3)
    cols = []
    checks = []
    for ek, Q in zip(es, pix):
        x, y = (bary @ Q[valid][tri]).reshape(-1, 2).round().astype(int).T
        im, _ = image(c, ek["n"])
        col = im[y.clip(0, h - 1), x.clip(0, w - 1)]
        cols.append(col)
        S = labels(c, ek["n"])
        checks.append(S[(y * 469 // h).clip(0, 468), (x * 724 // w).clip(0, 723)])
    cols = np.stack(cols).astype("f")
    cl = np.stack(checks)
    good = (
        (np.ptp(cols.mean(2), axis=0) < 24)
        & ((cl == 8).sum(0) >= 2)
        & ~np.isin(cl, [0, 9, 10, 250]).any(0)
    )
    N = unit(np.cross(xyz[:, 1] - xyz[:, 0], xyz[:, 2] - xyz[:, 0]))
    N = np.where(
        (N * (xyz.mean(1) - O[1, valid][tri].mean(1))).sum(1)[:, None] > 0, -N, N
    )
    N = np.repeat(N, len(bary), axis=0)
    good &= np.linalg.norm(N, axis=1) > 0.98
    pitch = (
        np.linalg.norm(xyz[:, 1] - xyz[:, 0], axis=1)
        * np.linalg.norm(xyz[:, 2] - xyz[:, 0], axis=1)
        / 18
    ) ** 0.5
    pitch = np.repeat(pitch, len(bary)).clip(0.016, 0.125)
    uv = (bary @ q[tri]).reshape(-1, 2)
    col = photo.sample(c, e["n"], (uv + 0.5) * [C.width / w, C.height / h], cols[1])
    cent, col, N, pitch = [a[good] for a in [cent, col, N, pitch]]
    B = surfels(
        cent, col, np.c_[pitch * 0.85, pitch * 0.85, pitch * 0.32], 0.94, e["t"]
    )
    tx = unit(np.cross(np.where(abs(N[:, 2:3]) > 0.85, [0.0, 1, 0], [0, 0.0, 1]), N))
    B[:, 3:6] = N
    B[:, 13:17] = Rotation.from_matrix(np.stack([tx, np.cross(N, tx), N], 2)).as_quat()[
        :, [3, 0, 1, 2]
    ]
    return [B]


def run():
    bags = []
    cnt = 0
    vol = Volume()
    for c in [c for c in CAMS if c != "back"]:
        allrows = [e for e in schedule()[c] if not e["hold"]]
        ds = np.array([e["odo"] for e in allrows])
        last = -1e10
        for i, e in enumerate(allrows):
            if e["odo"] - last < 2.4 or not semantic(c, e["n"]):
                continue
            last = e["odo"]
            gap = 1.05 if c == "front" else 0.72
            a, b = np.searchsorted(ds, [ds[i] - gap, ds[i] + gap])
            if a >= i or b <= i or b >= len(ds):
                continue
            bags.extend(triplet([allrows[a], e, allrows[b]], c))
            cnt += 1
        print("canopy", c, cnt, sum(map(len, bags)), flush=True)
    B = np.concatenate(bags)
    g, h, _ = vol.lookup(B[:, :3])
    B = B[(B[:, 2] > g + 0.70) & (B[:, 2] > h + 0.10)]
    idx = np.floor(B[:, :3] / 0.036).astype("i8") + 80000
    key = (idx[:, 0] * 160000 + idx[:, 1]) * 160000 + idx[:, 2]
    o = np.lexsort((B[:, 10], key))
    i = o[np.r_[True, np.diff(key[o]) != 0]]
    B = B[i]
    np.save(W5 / "leaves.npy", B)
    savej(W5 / "leaves.json", dict(triplets=cnt, texels=len(B)))
    print("unique", len(B), flush=True)


def install(src, dest):
    B = np.load(W5 / "leaves.npy")
    near = cKDTree(B[:, :3])
    parts = []
    removed = 0
    for t in J(src / "map.json")["tiles"]:
        if t["layer"] != "objects":
            continue
        a = reader(src / t["file"])
        if not t["id"].startswith(("columns_", "facade_", "office-plane_")):
            d = near.query(a[:, :3], distance_upper_bound=0.21)[0]
            sel = d < np.clip(np.exp(a[:, 10]) * 0.78, 0.035, 0.145)
            removed += sel.sum()
            a = a[~sel]
        parts.append((t, a))
    parts.extend(group(B, "leaf"))
    assemble(dest, src, parts)
    savej(
        W5 / "leaves-installed.json",
        dict(photographic_texels=len(B), old_centres_removed=int(removed)),
    )


if __name__ == "__main__":
    if len(sys.argv) == 3:
        install(Path(sys.argv[1]), Path(sys.argv[2]))
    else:
        run()
