"""Independent near-field road height: masked two-view photometric plane sweep.
Select road behind the tyres, compare band-pass intensities across height hypotheses.
"""

from solve import *
from concurrent.futures import ThreadPoolExecutor
from functools import lru_cache


def measure(s):
    c = CAMS.index("back")
    ix = np.flatnonzero(s.valid)
    cams, cent = s.poses()
    ego = (
        Rot.from_euler("z", s.base[:, 2:3]).as_matrix() @ Rot.from_rotvec(s.dtfr[:, :3]).as_matrix()
    )
    t = np.pad(s.base[:, :2], ((0, 0), (0, 1))) + s.dtfr[:, 3:]
    x, y = np.meshgrid(np.linspace(-8.5, -2.4, 68), np.linspace(-1.2, 1.2, 33))
    z = np.linspace(-0.36, 0.36, 73)
    G = np.stack(np.broadcast_arrays(x[None], y[None], z[:, None, None]), -1).reshape((-1, 3))
    p = s.cam[c]

    @lru_cache(maxsize=12)
    def image(k):
        a = cv.imread(str(DATA / "frames" / CAMS[c] / f"{s.fr[k]['ns'][c]:06d}.jpg"), 0).astype("f")
        g = cv.GaussianBlur(a, (0, 0), 4)
        base = cv.GaussianBlur(a, (0, 0), 0.8)
        return np.clip(
            (base - g) / np.sqrt(np.maximum(3.0, cv.GaussianBlur((base - g) ** 2, (0, 0), 7))),
            -3,
            3,
        )

    def sample(k, X):
        U = X - cent[NC * k + c]
        D = U @ cams[NC * k + c]
        rho = np.linalg.norm(D[:, :2], axis=1)
        th = np.arctan2(rho, D[:, 2])
        t2 = th**2
        uv = D[:, :2] * (th * (1 + t2 * (p[4] + t2 * p[5])) / rho)[:, None] * p[:2] + p[2:4]
        row = (uv[:, 1] / s.sizes[c, 1] - 0.5) * p[15]
        yaw = row * s.base[k, 5]
        co = np.cos(yaw)
        si = np.sin(yaw)
        origin = s.base[k, :2] + s.dtfr[k, 3:5] + (s.dt[k, c] + p[14]) * s.base[k, 3:5]
        a = X[:, :2] - origin - row[:, None] * s.base[k, 3:5]
        U[:, :2] = (
            np.column_stack((co * a[:, 0] + si * a[:, 1], -si * a[:, 0] + co * a[:, 1]))
            + origin
            - cent[NC * k + c, :2]
        )
        D = U @ cams[NC * k + c]
        rho = np.linalg.norm(D[:, :2], axis=1)
        th = np.arctan2(rho, D[:, 2])
        t2 = th**2
        uv = D[:, :2] * (th * (1 + t2 * (p[4] + t2 * p[5])) / rho)[:, None] * p[:2] + p[2:4]
        uv = (uv - 0.5).astype("f").reshape((len(z), -1, 2))
        return cv.remap(image(k), uv, None, cv.INTER_LINEAR)

    def pair(ij):
        a, b = ij
        X = G @ ego[a].T + t[a]
        u, v = sample(a, X), sample(b, X)
        n = (u * v).sum(1) / (np.linalg.norm(u, axis=1) * np.linalg.norm(v, axis=1) + 1e-9)
        j = int(n.argmax())
        return [a, b, float(z[j]), float(n[j]), float(n[j] - np.percentile(n, 25))]

    pairs = [
        (a, b)
        for a, b in zip(ix[:-1], ix[1:])
        if s.fr[a]["seg"] == s.fr[b]["seg"] and np.linalg.norm(t[a] - t[b]) < 3.5
    ]
    results = np.array(list(ThreadPoolExecutor(3).map(pair, pairs)))
    ok = (results[:, 3] > 0.29) & (results[:, 4] > 0.20) & (abs(results[:, 2]) < 0.35)
    vals = results[ok, 2]
    res = dict(
        height=float(np.median(vals)),
        quantiles=np.percentile(vals, [16, 84]).tolist(),
        accepted=int(ok.sum()),
        total=len(ok),
        pairs=results.tolist(),
    )
    json.dump(res, open(s.w / "ground.json", "w"))
    print(
        s.split,
        "road height",
        res["height"],
        res["quantiles"],
        res["accepted"],
        res["total"],
        flush=True,
    )


if __name__ == "__main__":
    for sp in ["all", "A", "B"]:
        measure(Scene(sp).load())
