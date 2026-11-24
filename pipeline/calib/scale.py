"""Blind metric check: calibrate no parameters and read no odometry in the solver.
Two arbitrary starting spacings test whether the rigid rig actually fixes scale.
"""

from solve import *
from align import align


def run(step):
    s = Scene("scale")
    s.w = W / f"scale-{step}"
    s.w.mkdir(exist_ok=True)
    s.cam = np.load(W / "all/result.npz")["cam"].copy()
    ix = np.flatnonzero(s.valid)
    ts = np.array([s.fr[k]["t"] for k in ix])
    s.base[:] = 0
    s.base[ix, 0] = np.arange(len(ix)) * step

    def velocities():
        xyz = s.base[ix, :2] + s.dtfr[ix, 3:5]
        a = s.base[ix, 2] + s.dtfr[ix, 2]
        s.base[ix, 3:5] = np.gradient(xyz, ts, axis=0)
        s.base[ix, 5] = np.gradient(a, ts)

    velocities()
    tr = s.tracks(True)
    s.initialize(tr, 7, 24)
    s.ba(55, 3, 2)
    velocities()
    tr = s.tracks(True)
    s.initialize(tr, 12, 12)
    s.ba(45, 2.5, 2)
    s.prune(4)
    velocities()
    s.ba(35, 1.7, 2)
    s.save()
    xyz = np.pad(s.base[ix, :2], ((0, 0), (0, 1))) + s.dtfr[ix, 3:]
    _, origin, _ = align(Scene().load())
    # a copy, because SciPy 1.17's Rotation.apply refuses the read-only view broadcast_to returns
    xyz += Rot.from_rotvec(s.dtfr[ix, :3]).apply(np.broadcast_to(origin, xyz.shape).copy())
    dist = np.linalg.norm(np.diff(xyz, axis=0), axis=1).sum()
    n = s.obs // 3100 % NC
    cross = len(
        np.unique(
            s.pi[
                np.minimum.reduceat(n, np.r_[0, np.flatnonzero(np.diff(s.pi)) + 1])[s.pi]
                != np.maximum.reduceat(n, np.r_[0, np.flatnonzero(np.diff(s.pi)) + 1])[s.pi]
            ]
        )
    )
    res = dict(
        start_spacing_m=step,
        distance=float(dist),
        start=float(ts[0]),
        end=float(ts[-1]),
        points=len(s.points),
        cross_points=cross,
    )
    print(res, flush=True)
    return res


if __name__ == "__main__":
    runs = [run(s) for s in [0.6, 1.3]]
    tl = np.load(DATA / "timeline.npz")
    sel = (tl["t"] >= runs[0]["start"]) & (tl["t"] <= runs[0]["end"])
    integrate = np.trapezoid if hasattr(np, "trapezoid") else np.trapz  # NumPy 2 renamed it
    telem = float(integrate(tl["speed"][sel], tl["t"][sel]))
    json.dump(dict(visual=runs, telemetry=telem), open(W / "scale-check.json", "w"), indent=2)
