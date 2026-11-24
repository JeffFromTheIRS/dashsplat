"""Score deployed timestamped poses, unused far→near tracks, scale, stops and flags."""

from .offline import Fit, W, OUT, NAMES, np, json, RS
from .motion import inputs, brackets
from pipeline.common.rig import Trajectory, _log, _mv
from collections import defaultdict
from scipy.optimize import minimize_scalar
from scipy.ndimage import gaussian_filter1d
from scipy.spatial.transform import Rotation as Rot
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def exact(s, tr, k, c, uv):
    tt = (
        s.tc[k, c]
        + np.array([v.time_offset_s for v in s.C])[c]
        + (uv[:, 1] / s.sizes[c, 1] - 0.5) * np.array([v.rs_readout_s for v in s.C])[c]
    )
    pose = tr.T_world_ego(tt)
    R = pose[:, :3, :3] @ s.rr[c]
    lever = _mv(pose[:, :3, :3], s.lever[c])
    p = pose[:, :3, 3]
    ray = np.empty((len(k), 3))
    for i in np.unique(c):
        sel = c == i
        ray[sel] = s.C[i].unproject(uv[sel])
    return _mv(R, ray), p + lever, R, p, lever


def solve_ray(ray, C):
    M = np.eye(3) - ray[:, :, None] * ray[:, None, :]
    H = M.sum(0)
    v = np.linalg.eigvalsh(H)
    X = np.linalg.solve(H + 1e-10 * np.eye(3), _mv(M, C).sum(0))
    return X, v[0] / len(ray)


def holdout(s, tr):
    d = np.load(W / "hold.npz")
    num = d["num"]
    k, c, uv = d["k"], d["c"], d["uv"]
    bounds = np.r_[0, num.cumsum()]
    stats = defaultdict(list)
    comparison = defaultdict(list)
    planar = Trajectory(
        W / "no-refined-poses.npz", timeline=RS / "data/drive/timeline.npz"
    )
    live = Trajectory(OUT / "ego_causal.npz")
    ids = defaultdict(set)
    scale = defaultdict(list)
    scale_ids = defaultdict(set)
    kt = s.kt
    speed = s.d["speed"][s.ix]
    yaw = s.B[s.ix, 2]
    yr = np.rad2deg(np.gradient(yaw, kt))
    for a in range(0, len(num), 1800):
        b = min(a + 1800, len(num))
        start = bounds[a]
        sl = slice(start, bounds[b])
        rr, oo, R, P, L = exact(s, tr, k[sl], c[sl], uv[sl])
        alternatives = [exact(s, v, k[sl], c[sl], uv[sl])[:3] for v in (planar, live)]
        for j in range(a, b):
            x, y = bounds[j] - start, bounds[j + 1] - start
            idx = np.arange(x, y)
            cam = c[sl][idx]
            imgs = k[sl][idx]
            pix = uv[sl][idx]
            r = rr[idx]
            O = oo[idx]
            A = R[idx]
            ego = P[idx]
            le = L[idx]
            n = len(r)
            firstt, lastt = kt[imgs[[0, -1]]]
            if len(np.unique(cam)) > 1 and speed[imgs].mean() > 0.8:
                X, ang = solve_ray(r, O)
                dep = ((X - O) * r).sum(1)
                if ang > 0.00012 and dep.min() > 1 and dep.max() < 250:
                    m = np.eye(3) - r[:, :, None] * r[:, None, :]
                    ws = (s.cp[cam, 0] / dep) ** 2
                    ww = m * ws[:, None, None]
                    H = ww.sum(0)
                    xp = np.linalg.solve(H, _mv(ww, ego - ego[0]).sum(0))
                    xl = np.linalg.solve(H, _mv(ww, le).sum(0))
                    p0 = _mv(m, xl - le) * np.sqrt(ws)[:, None]
                    p1 = _mv(m, xp - (ego - ego[0])) * np.sqrt(ws)[:, None]
                    s1 = (p1 * p1).sum()
                    os = -(p0 * p1).sum() / max(s1, 1e-10)
                    res = np.linalg.norm(p0 + os * p1, axis=1)
                    info = s1 / n
                    if info > 2 and np.median(res) < 3 and np.percentile(res, 85) < 6:
                        for key in ["whole", str(int(firstt // 30) * 30)]:
                            scale[key].append((p0, p1))
                            scale_ids[key].add(j)
            if n < 8 or s.ds[imgs[-1]] - s.ds[imgs[0]] < 15:
                continue
            last = np.arange(n - n // 3, n)
            Xn, angle = solve_ray(r[last], O[last])
            rrange = np.linalg.norm(Xn - O, axis=1)
            depth = ((Xn - O) * r).sum(1)
            near = last[rrange[last] < 8]
            far = np.arange(n // 3)
            far = far[rrange[far] > 25]
            if (
                rrange[0] <= 25
                or rrange[-1] >= 8
                or rrange[-1] < 0.6
                or len(far) < 3
                or len(near) < 2
                or angle < 0.000018
                or (depth[last] < 0.5).any()
            ):
                continue
            # Each end must be locally self-consistent. Do not gate near residuals against the FAR point.
            e, _, _, _ = s.project(
                cam[last],
                np.broadcast_to(Xn, (len(last), 3)),
                A[last],
                O[last],
                pix[last],
            )
            if np.median(e) > 2.8 or np.percentile(e, 85) > 5:
                continue
            Xf, ang = solve_ray(r[far], O[far])
            infront = ((Xf - O[far]) * r[far]).sum(1) > 0
            if ang < 0.000025 or not infront.all():
                continue
            e, _, _, _ = s.project(
                cam[far], np.broadcast_to(Xf, (len(far), 3)), A[far], O[far], pix[far]
            )
            if np.median(e) > 2.8 or np.percentile(e, 85) > 5:
                continue
            vals, _, _, _ = s.project(
                cam[near],
                np.broadcast_to(Xf, (len(near), 3)),
                A[near],
                O[near],
                pix[near],
            )
            alt = []
            for ray0, O0, R0 in alternatives:
                rr0, cc0, RR = ray0[idx], O0[idx], R0[idx]
                XX, _ = solve_ray(rr0[far], cc0[far])
                alt.append(
                    s.project(
                        cam[near],
                        np.broadcast_to(XX, (len(near), 3)),
                        RR[near],
                        cc0[near],
                        pix[near],
                    )[0]
                )
            for v0, (m, err) in enumerate(zip(near, vals)):
                time = kt[imgs[m]]
                reg = ["whole"]
                if 82 <= time <= 105:
                    reg += ["car park"]
                if 317.5 <= time <= 325.5:
                    reg += ["90° turn"]
                if 105 < time <= 111.5:
                    reg += ["exit turn"]
                if speed[imgs[m]] >= 17 and abs(yr[imgs[m]]) < 1.25:
                    reg += ["fast straight"]
                for g in reg:
                    key = (g, int(cam[m]))
                    stats[key].append(float(err))
                    ids[key].add(j)
                    comparison[g].append([float(err), alt[0][v0], alt[1][v0]])
    sums = {}
    for g, l in scale.items():
        a, b = map(np.vstack, zip(*l))
        f = lambda v: np.mean(np.log1p(((a + b * v) ** 2).sum(1) / 2.5**2))
        best = minimize_scalar(f, bounds=(0.7, 1.3), method="bounded")
        sums[g] = [float(best.x), len(scale_ids[g])]
    fmt = {
        g: [
            (
                len(ids[g, i]),
                len(stats[g, i]),
                *map(float, np.percentile(stats[g, i], [50, 90])),
            )
            if stats[g, i]
            else (0, 0, None, None)
            for i in range(len(NAMES))
        ]
        for g in ["whole", "car park", "90° turn", "exit turn", "fast straight"]
    }
    return (
        fmt,
        sums,
        {g: np.percentile(v, [50, 90], axis=0).tolist() for g, v in comparison.items()},
    )


def plots(t, off, ca, z, tilt, ct, acc, lat, dp, dr, dg, dc, roi):
    dest = OUT / "debug"
    dest.mkdir(exist_ok=True, parents=True)
    plt.rcParams.update({"font.size": 9})
    fig, ax = plt.subplots(3, 1, figsize=(11, 6.5), sharex=True)
    for w, v, y, c, a in [
        (ax[0], tilt[:, 0], "Pitch (°)", ct[:, 0], acc),
        (ax[1], tilt[:, 1], "Roll (°)", ct[:, 1], lat),
        (ax[2], z, "Height (m)", ca[:, 2, 3], None),
    ]:
        w.plot(t, c, color="#c68e36", lw=0.65, label="causal")
        w.plot(t, v, color="#153d66", lw=0.9, label="offline")
        w.set_ylabel(y)
        w.grid(alpha=0.18)
        w.legend(loc="upper left", ncol=2)
        if a is not None:
            r = w.twinx()
            r.plot(t, a, color="#387b65", alpha=0.35, lw=0.65)
            r.set_ylabel(
                ("−ay: forward" if w is ax[0] else "−ax: left") + " (m/s²)",
                color="#387b65",
            )
            r.set_ylim(-6.5, 6.5)
    for w in ax:
        for a, b in roi:
            w.axvspan(a, b, color="gray", alpha=0.1)
    ax[-1].set_xlabel("Drive time (s) · gray = stationary")
    fig.tight_layout()
    fig.savefig(dest / "attitude-height.png", dpi=130)
    plt.close(fig)
    fig, ax = plt.subplots(2, 1, figsize=(11, 4.5), sharex=True)
    ax[0].plot(t, dr, label="offline − dead reckoning, XY")
    ax[0].plot(t, dg, label="offline − fused GPS, XY")
    ax[0].plot(t, dc, label="causal − offline, XYZ", lw=0.9)
    ax[0].set_ylabel("Distance (m)")
    ax[0].legend(ncol=3)
    ax[1].plot(t, dp, color="#355c83")
    ax[1].set_ylabel("Causal − offline (°)")
    ax[1].set_xlabel("Drive time (s)")
    [a.grid(alpha=0.2) for a in ax]
    fig.tight_layout()
    fig.savefig(dest / "trajectory-differences.png", dpi=130)
    plt.close(fig)


def main():
    s = Fit().load()
    tr = Trajectory(OUT / "ego.npz")
    ca = Trajectory(OUT / "ego_causal.npz")
    t = s.t
    off = tr.poses
    causal = ca.poses
    D = s.d
    station = s.stop
    pi = np.repeat(np.arange(len(s.n)), s.n)
    shift = s.corr[s.kids[0], 3:] + np.r_[s.base[0, :2], 0]
    errs = []
    for j in range(0, len(pi), 60000):
        sl = slice(j, j + 60000)
        _, O, R, _, _ = exact(s, tr, s.k[sl], s.cam[sl], s.uv[sl])
        e, _, _, _ = s.project(s.cam[sl], s.pts[pi[sl]] - shift, R, O, s.uv[sl])
        errs.append(e)
    err = np.concatenate(errs)
    repro = []
    for i in range(len(NAMES)):
        ii = s.cam == i
        e = err[ii]
        repro.append(
            [
                len(set(pi[ii].tolist())),
                len(e),
                float(np.sqrt(np.mean(e**2))),
                float(np.percentile(e, 95)),
            ]
        )
    stats, stereo, paired = holdout(s, tr)
    gps = np.stack((D["gps_x"], D["gps_y"]), -1)
    p = off[:, :3, 3]
    q = causal[:, :3, 3]
    dr = np.linalg.norm(p[:, :2] - s.B[:, :2], axis=1)
    dg = np.linalg.norm(p[:, :2] - gps, axis=1)
    dc = np.linalg.norm(q - p, axis=1)
    ang = np.rad2deg(
        np.linalg.norm(
            _log(off[:, :3, :3].transpose(0, 2, 1) @ causal[:, :3, :3]), axis=1
        )
    )
    eul = Rot.from_matrix(off[:, :3, :3]).as_euler("ZYX", degrees=True)
    ce = Rot.from_matrix(causal[:, :3, :3]).as_euler("ZYX", degrees=True)
    tilt = np.stack((-eul[:, 1], eul[:, 2]), -1)
    ct = np.stack((-ce[:, 1], ce[:, 2]), -1)
    statints = list(
        zip(
            np.flatnonzero(station & ~np.r_[False, station[:-1]]),
            np.flatnonzero(station & ~np.r_[station[1:], False]),
        )
    )
    stationary_ok = all(
        np.array_equal(x[a : b + 1], np.broadcast_to(x[a], x[a : b + 1].shape))
        for x in [off, causal]
        for a, b in statints
    )
    assert stationary_ok
    cor = lambda a, b: float(np.corrcoef(a[~station], b[~station])[0, 1])
    acc = -D["ay"]
    lat = -D["ax"]
    length = lambda x: float(np.linalg.norm(np.diff(x, axis=0), axis=1).sum())
    dist = np.r_[0, np.cumsum(np.linalg.norm(np.diff(s.B[:, :2], axis=0), axis=1))]
    jj = np.searchsorted(dist, np.arange(0, dist[-1], 100))
    gpc = np.linalg.norm(np.diff(gps[jj], axis=0), axis=1)
    drc = np.linalg.norm(np.diff(s.B[jj, :2], axis=0), axis=1)
    opc = np.linalg.norm(np.diff(p[jj, :2], axis=0), axis=1)
    bins = []
    ants = np.load(OUT / "ego.npz")["n_tracks"]
    cnt = np.bincount((s.kt[s.k] // 10).astype(int), minlength=int(t[-1] // 10) + 1)
    mov = []
    for b in range(int(t[-1] // 10) + 1):
        start = b * 10
        sel = (s.kt[s.k] >= start) & (s.kt[s.k] < start + 10)
        ncam = [len(np.unique(pi[sel & (s.cam == i)])) for i in range(len(NAMES))]
        bins.append([start, len(np.unique(pi[sel])), *ncam])
        mo = (t >= start) & (t < start + 10) & ~station
        mov.append(float(np.median(ants[mo])) if mo.any() else 0.0)
    csv = OUT / "debug"
    csv.mkdir(exist_ok=True)
    np.savetxt(
        csv / "tracks-per-10s.csv",
        bins,
        fmt="%d",
        delimiter=",",
        header="start_s,unique_tracks," + ",".join(NAMES),
        comments="",
    )
    quant = lambda x: list(map(float, np.percentile(x, [50, 95, 100])))
    plots(
        t,
        off,
        causal,
        p[:, 2],
        tilt,
        ct,
        acc,
        lat,
        ang,
        dr,
        dg,
        dc,
        [(t[a], t[b]) for a, b in statints],
    )
    jj, frac = brackets(s.kt, t)
    lag = (
        s.tc.max(1)[jj + (frac > 0)]
        + max(c.time_offset_s + c.rs_readout_s / 2 for c in s.C)
        - t
    )
    vh = ~station
    res = dict(
        repro=repro,
        hold=stats,
        paired=paired,
        stereo=stereo,
        windows=bins,
        support=mov,
        length={
            "offline3d": length(p),
            "offlineXY": length(p[:, :2]),
            "causal3d": length(q),
            "wheel": float((np.trapezoid if hasattr(np, "trapezoid") else np.trapz)(D["speed"], t)),
            "gpsXY": length(gps),
        },
        gpswheel=list(map(float, np.percentile(100 * (drc / gpc - 1), [10, 50, 90]))),
        gpsoff=list(map(float, np.percentile(100 * (opc / gpc - 1), [10, 50, 90]))),
        dead_delta=quant(dr),
        gps_delta=quant(dg),
        causal_delta=quant(dc),
        causal_angle=quant(ang),
        causal_moving=quant(dc[vh]),
        causal_ang_moving=quant(ang[vh]),
        max_lag=float(max(lag[vh])),
        pitch=quant(abs(tilt[:, 0])),
        roll=quant(abs(tilt[:, 1])),
        height=[float(p[:, 2].min()), float(p[:, 2].max())],
        pitch_forward_accel=cor(tilt[:, 0], acc),
        roll_left_accel=cor(tilt[:, 1], lat),
        pitch_dv=cor(tilt[:, 0], gaussian_filter1d(np.gradient(D["speed"], t), 5)),
        roll_vyaw=cor(tilt[:, 1], D["speed"] * np.gradient(s.B[:, 2], t)),
        stop=[(float(t[a]), float(t[b])) for a, b in statints],
        n_master=len(t),
        n_vision=len(s.kt),
        n_knots=len(s.corr),
        n_landmarks=len(s.pts),
        cross_inliers=int(
            np.sum(
                np.minimum.reduceat(s.cam, np.r_[0, s.n.cumsum()[:-1]])
                != np.maximum.reduceat(s.cam, np.r_[0, s.n.cumsum()[:-1]])
            )
        ),
    )
    json.dump(res, open(W / "metrics.json", "w"), indent=2)
    print("validated", res["repro"], res["causal_delta"], res["hold"], flush=True)


if __name__ == "__main__":
    main()
