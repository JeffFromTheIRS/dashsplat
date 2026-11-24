"""Frozen-rig metric holdout. Only the delivered camera parameters/masks cross drives."""

import hashlib
import os
import shutil
import subprocess
import sys
import tempfile
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import cv2 as cv
import numpy as np
from scipy.spatial.transform import Rotation as Rot

from raw import write
from stages import HOME, binlink, read

sys.path.insert(0, str(HOME / "pipeline/calib"))
from deliver import bundle, transfers
from features import CAMS, NC, W, extract, match
from ground import measure
from solve import B, Scene


class SparseEvidence(ValueError):
    pass


def samples():
    tl = np.load("data/drive/timeline.npz")
    rows = [read(f"data/drive/cams/{c}.json") for c in CAMS]
    times = [np.array([e["t"] for e in rr]) for rr in rows]
    dt = tl["t"]
    xy = np.c_[tl["x"], tl["y"]]
    ds = np.r_[0, np.cumsum(np.linalg.norm(np.diff(xy, axis=0), axis=1))]
    # Pick straight, forward-moving support by odometer, not a benchmark timestamp.
    cand = []
    for lo, hi in read("inventory.json")["moving"]:
        for a in range(int(np.searchsorted(dt, lo)), int(np.searchsorted(dt, hi)), 9):
            b = np.searchsorted(ds, ds[a] + 16)
            if b < len(dt) and dt[b] <= hi and np.min(tl["speed"][a : b + 1]) > 0.5:
                cand.append((np.ptp(tl["yaw"][a : b + 1]), a, b))
    if not cand:
        raise SparseEvidence(
            "not enough forward motion for a 16 m independent rig scale check"
        )
    _, a, b = min(cand)
    holds = [(a, b)]
    # Reserve panels using telemetry alone, before computing visual residuals.
    for _ in range(2):
        possible = [
            (yaw, i, j)
            for yaw, i, j in cand
            if yaw <= np.deg2rad(6)
            and all(dt[j] < dt[l] - 0.4 or dt[i] > dt[h] + 0.4 for l, h in holds)
        ]
        if not possible:
            break
        _, i, j = max(
            possible,
            key=lambda r: (min(abs(ds[r[1]] - ds[l]) for l, _ in holds), -r[0]),
        )
        holds.append((i, j))
    train = []
    for lo, hi in read("inventory.json")["moving"]:
        for i in range(np.searchsorted(dt, lo), np.searchsorted(dt, hi), 12):
            j = np.searchsorted(ds, ds[i] + 13)
            if (
                j >= len(dt)
                or dt[j] > hi
                or any(dt[j] >= dt[l] - 0.4 and dt[i] <= dt[h] + 0.4 for l, h in holds)
            ):
                continue
            train.append((float(np.ptp(tl["yaw"][i : j + 1])), i, j))
    selected = []
    for _, i, j in sorted(train, reverse=True):
        if all(j < l or i > h for l, h in selected):
            selected.append((i, j))
        if len(selected) == 5:
            break

    def make(i, j, seg):
        out = []
        for k in np.unique(np.searchsorted(ds, np.linspace(ds[i], ds[j], 18))):
            t = dt[k]
            ids = [int(np.argmin(abs(tt - t))) for tt in times]
            delta = [float(tt[u] - t) for tt, u in zip(times, ids)]
            if max(map(abs, delta)) > 0.021:
                continue
            out.append(
                dict(
                    t=float(t),
                    xy=xy[k].tolist(),
                    yaw=float(tl["yaw"][k]),
                    seg=seg,
                    ns=[rr[u]["n"] for rr, u in zip(rows, ids)],
                    dt=delta,
                )
            )
        return out

    hh = [make(*hold, 5) for hold in holds]
    if any(len(panel) < 10 for panel in hh):
        raise SparseEvidence(
            "not enough shutters synchronized across all cameras to verify the rig"
        )
    return hh, [r for s, (i, j) in enumerate(sorted(selected)) for r in make(i, j, s)]


def prepare(fr):
    # Feature caches are indexed by position in frames.json. Never reuse with new frames.
    for d in ("features", "matches", "matches-split"):
        if (W / d).exists():
            shutil.rmtree(W / d)
    (W / "features").mkdir(parents=True)
    write(W / "frames.json", fr)
    with ThreadPoolExecutor(4) as pool:
        list(
            pool.map(
                extract,
                [(k, c, n) for k, f in enumerate(fr) for c, n in enumerate(f["ns"])],
            )
        )
    match(fr)


def frozen(spec, step):
    # Same blind BA sequence as calib/scale.py, expressed directly in delivered ego
    # coordinates. That script's final align(Scene().load()) would load the old drive.
    s = Scene("scale")
    s.w = W / f"blind-{step}"
    s.w.mkdir(exist_ok=True)
    ix = np.flatnonzero(s.valid)
    ts = np.array([s.fr[k]["t"] for k in ix])
    s.base[:] = 0
    s.base[ix, 0] = step * np.arange(len(ix))
    for c, n in enumerate(CAMS):
        q = spec[n]
        T = np.asarray(q["T_ego_cam"])
        s.cam[c] = np.r_[
            q["params"],
            Rot.from_matrix(B[c].T @ T[:3, :3]).as_rotvec(),
            T[:3, 3],
            q["time_offset_s"],
            q["rs_readout_s"],
        ]
    fixed = s.cam.copy()

    def velocities():
        pp = s.base[ix, :2] + s.dtfr[ix, 3:5]
        s.base[ix, 3:5] = np.gradient(pp, ts, axis=0)
        s.base[ix, 5] = np.gradient(s.base[ix, 2] + s.dtfr[ix, 2], ts)

    velocities()
    s.initialize(s.tracks(True), 7, 24)
    s.ba(55, 3, 2)
    velocities()
    s.initialize(s.tracks(True), 12, 12)
    s.ba(45, 2.5, 2)
    s.prune(4)
    velocities()
    s.ba(35, 1.7, 2)
    if not np.allclose(fixed, s.cam, rtol=1e-8, atol=1e-8):
        raise RuntimeError("metric check changed fixed camera parameters")
    x = np.pad(s.base[ix, :2], ((0, 0), (0, 1))) + s.dtfr[ix, 3:]
    dist = np.linalg.norm(np.diff(x, axis=0), axis=1).sum()
    tar, _, uv = transfers(s)
    err = np.linalg.norm(s.pixels[tar] - uv, axis=1)
    per = {}
    target_rows = {}
    spatial = {}
    for i, n in enumerate(CAMS):
        sel = tar // 3100 % NC == i
        # One residual per target image feature even if several other lenses
        # independently triangulated it. Windows have disjoint target shutters.
        unique, ix, multiplicity = np.unique(
            tar[sel], return_inverse=True, return_counts=True
        )
        es = err[sel][np.argsort(ix, kind="stable")]
        target_rows[n] = [
            (int(s.fr[k // 3100 // NC]["ns"][i]), int(k % 3100)) for k in unique
        ]
        if len(unique):
            uvp = s.pixels[unique]
            size = np.array([spec[n]["width"], spec[n]["height"]])
            frac = uvp / size
            spatial[n] = dict(
                native_size=size.tolist(),
                uv_p10_p50_p90=np.round(
                    np.percentile(uvp, [10, 50, 90], axis=0), 1
                ).tolist(),
                left_middle_right=np.bincount(
                    np.clip((frac[:, 0] * 3).astype(int), 0, 2), minlength=3
                ).tolist(),
            )
        per[n] = (
            [float(np.median(v)) for v in np.split(es, np.cumsum(multiplicity)[:-1])]
            if len(unique)
            else []
        )
    tl = np.load("data/drive/timeline.npz")
    sel = (tl["t"] >= ts[0]) & (tl["t"] <= ts[-1])
    integrate = np.trapezoid if hasattr(np, "trapezoid") else np.trapz
    telem = integrate(tl["speed"][sel], tl["t"][sel])
    return dict(
        start_spacing_m=step,
        interval_s=[float(ts[0]), float(ts[-1])],
        distance_m=float(dist),
        telemetry_m=float(telem),
        transfer=per,
        target_rows=target_rows,
        spatial=spatial,
        speed_mean=float(np.mean(tl["speed"][sel])),
        yaw_rate_deg_s_p50=float(
            np.median(np.rad2deg(np.gradient(np.unwrap(tl["yaw"]), tl["t"]))[sel])
        ),
    )


def check(spec, panels, matched=None):
    # Equal lens votes, not feature-count votes: the wide rear lens can contribute
    # hundreds of times more correspondences than the front. Normalize to 1448 px.
    combined = [{c: [] for c in CAMS} for _ in range(2)]
    checks = []
    warnings = []
    for panel in panels:
        prepare(panel)
        runs = []
        for i, step in enumerate((0.6, 1.3)):
            try:
                r = frozen(spec, step)
            except (ValueError, IndexError, np.linalg.LinAlgError, cv.error) as e:
                warnings.append(
                    f"{panel[0]['t']:.2f}s, spacing {step}: insufficient visual support ({e})"
                )
                continue
            ids = r.pop("target_rows")
            for c, es in r["transfer"].items():
                combined[i][c].extend(es)
                med = float(np.median(es)) if es else None
                if matched is not None:
                    matched[i][c].update(zip(ids[c], es))
                r["transfer"][c] = dict(count=len(es), median_px=med)
                if med is not None and med > (5 if c == "front" else 3):
                    warnings.append(
                        f"{c} {r['interval_s'][0]:.2f}–{r['interval_s'][1]:.2f}s: {med:.2f} native px on {len(es)} targets (spacing {step} m)"
                    )
            if abs(r["distance_m"] / r["telemetry_m"] - 1) > 0.05:
                warnings.append(
                    f"{r['interval_s'][0]:.2f}s: visual scale differs from telemetry by over 5%"
                )
            runs.append(r)
        checks.append(runs)
    pools = []
    counts = {}
    lens = {}
    support = {}
    for bucket in combined:
        pools.append(
            {
                c: dict(count=len(es), median_px=float(np.median(es)) if es else None)
                for c, es in bucket.items()
            }
        )
    for c in CAMS:
        counts[c] = min(p[c]["count"] for p in pools)
        support[c] = sum(
            len(rs) == 2 and all(r["transfer"][c]["count"] >= 5 for r in rs)
            for rs in checks
        )
        if counts[c] >= 10 and support[c] >= 2:
            lens[c] = (
                float(np.median([p[c]["median_px"] for p in pools]))
                * 1448
                / spec[c]["width"]
            )
        else:
            warnings.append(
                f"{c}: sparse visual support, {counts[c]} targets, {support[c]} supported panels; not a mismatch verdict"
            )
    return dict(
        panels=checks,
        pooled_transfers=pools,
        counts=counts,
        supported_panels=support,
        lens_px=lens,
        # The pooled verdict needs four supported lenses, or all but one on a car with fewer cameras.
        pooled_px=float(np.median(list(lens.values())))
        if len(lens) >= min(4, NC - 1)
        else None,
        warnings=warnings,
    )


BASELINE = Path(__file__).with_name("rig-baseline.json")
OWN_BASELINE = HOME / "calib/rig-baseline.json"


def limits_file():
    """The limits belong to one rig, so `baseline` writes them beside it. The file in this folder is that of
    the car this was built on, read when the installation has none of its own."""
    return OWN_BASELINE if OWN_BASELINE.is_file() else BASELINE


def rig_hash():
    return hashlib.sha256((HOME / "calib/rig.json").read_bytes()).hexdigest()


def baseline():
    from drive import heavy
    from transactions import atomic_json, identity, lock

    # Three stretches of the calibrated car's drive, disjoint from the rig solve and its scale test. The
    # defaults are those of the drive this was built on; pass your own as `baseline T0 T1 T0 T1 ...`.
    times = [float(x) for x in sys.argv[2:]]
    if len(times) % 2:
        raise ValueError("baseline windows are pairs of start and end seconds")
    windows = list(zip(times[::2], times[1::2])) or [(115, 145), (161, 191), (329, 359)]
    rows = []
    with lock(
        HOME / "work/locks/reconstruction.lock",
        True,
        identity(),
        drive="stored-rig-reference",
        stage="baseline",
    ):
        for a, b in windows:
            while heavy():
                time.sleep(10)
            with tempfile.TemporaryDirectory(
                dir=HOME / "work", prefix="rig-reference-"
            ) as td:
                root = Path(td)
                (root / "data").mkdir()
                (root / "data/drive").symlink_to(HOME / "data/drive")
                (root / "calib").mkdir()
                (root / "calib/masks").symlink_to(HOME / "calib/masks")
                write(root / "inventory.json", dict(moving=[[a, b]]))
                subprocess.run(
                    [sys.executable, __file__, "reference-panel"],
                    cwd=root,
                    env=dict(os.environ, RS=str(root), PYTHONPATH=str(HOME)),
                    check=True,
                )
                m = read(root / "measure.json")
                rows.append(
                    dict(
                        window=[a, b],
                        pooled_px=m["pooled_px"],
                        lens_px=m["lens_px"],
                        counts=m["counts"],
                        supported_panels=m["supported_panels"],
                        panel_ranges=[r[0]["interval_s"] for r in m["panels"] if r],
                    )
                )
                print("baseline", rows[-1], flush=True)

    def limit(xs):
        x = np.asarray(xs)
        return round(
            float(
                max(
                    3 * x.max(), x.max() + 6 * 1.4826 * np.median(abs(x - np.median(x)))
                )
            ),
            4,
        )

    if any(r["pooled_px"] is None for r in rows):
        raise ValueError("reference missing gross-pooled evidence")
    per = {
        c: limit([r["lens_px"][c] for r in rows if c in r["lens_px"]])
        for c in CAMS
        if sum(c in r["lens_px"] for r in rows) >= 2
    }
    atomic_json(
        OWN_BASELINE,
        dict(
            rig_sha256=rig_hash(),
            reference="drive of the calibrated car, native exposures and telemetry only",
            normalization="1448 px image width; median per lens over disjoint shutters, median across two starts, median across lenses",
            rule="max(3 * worst reference, worst reference + 6 * scaled MAD)",
            windows=rows,
            limit_px=limit([r["pooled_px"] for r in rows]),
            lens_limits=per,
        ),
    )


def gross(r, base, spec):
    why = []
    if r["pooled_px"] is not None and r["pooled_px"] > base["limit_px"]:
        why.append(f"pooled {r['pooled_px']:.2f} > {base['limit_px']:.2f} px")
    for c, v in r["lens_px"].items():
        lim = base["lens_limits"].get(c)
        if lim is None or v <= lim:
            continue
        # One bad panel is never a hardware veto. Confirm the pooled lens change
        # independently on at least two panels, at both starting spacings.
        n = sum(
            len(rs) == 2
            and all(
                x["transfer"][c]["count"] >= 5
                and x["transfer"][c]["median_px"] * 1448 / spec[c]["width"] > lim
                for x in rs
            )
            for rs in r["panels"]
        )
        if n >= 2:
            why.append(
                f"{c} pooled {v:.2f} > {lim:.2f} px, independently confirmed on {n} panels"
            )
    return why


def paired(a, b, spec):
    results = []
    for i in range(2):
        scores = {}
        for c in CAMS:
            x, y = a[i][c], b[i][c]
            common = sorted(x.keys() & y.keys())
            count = len(common)
            if count < 50 or count < 0.8 * len(x):
                return False, results
            fac = 1448 / spec[c]["width"]
            scores[c] = [float(np.median([p[k] for k in common])) * fac for p in (x, y)]
        results.append(scores)
    passed = all(
        np.median([p[1] for p in s.values()])
        < 0.95 * np.median([p[0] for p in s.values()])
        and all(y <= 1.1 * x for x, y in s.values())
        for s in results
    )
    return bool(passed), results


def candidate(old, train):
    prepare(sorted(train, key=lambda r: r["t"]))
    s = Scene()
    s.run()
    measure(s)
    fitted = bundle(s)
    new = {}
    for i, c in enumerate(CAMS):
        T = np.eye(4)
        T[:3, :3] = fitted["R"][i]
        T[:3, 3] = fitted["pos"][i]
        p = s.cam[i]
        ref = np.array(old[c]["T_ego_cam"])
        if (
            np.linalg.norm(T[:3, 3] - ref[:3, 3]) > 0.25
            or Rot.from_matrix(ref[:3, :3].T @ T[:3, :3]).magnitude() > np.deg2rad(5)
            or max(abs(p[:2] / old[c]["params"][:2] - 1)) > 0.1
        ):
            raise ValueError(f"{c}: implausible refit displacement or focal length")
        new[c] = dict(
            old[c],
            T_ego_cam=T.tolist(),
            params=p[:8].tolist(),
            time_offset_s=float(p[14]),
            rs_readout_s=float(p[15]),
        )
    return new


def target_map():
    return [{c: {} for c in CAMS} for _ in range(2)]


def run():
    mode = os.environ.get("RUN_RIG", "verify")
    saved = read(HOME / "calib/rig.json")
    old = saved["cameras"]
    meta = read("data/drive/meta.json")
    if set(meta["cams"]) != set(old) or any(
        meta["sizes"][c] != [q["width"], q["height"]] for c, q in old.items()
    ):
        raise ValueError("incompatible camera roster or native resolution")
    W.mkdir(parents=True, exist_ok=True)
    shutil.copytree(HOME / "calib/masks", "calib/masks", dirs_exist_ok=True)
    report = dict(mode=mode, selected="stored", warnings=[])
    file = W / "verification.json"
    if mode != "stored":
        base = read(limits_file())
        if base["rig_sha256"] != rig_hash():
            raise ValueError(
                "empirical baseline belongs to a different stored bundle; rebuild baseline"
            )
        binlink(HOME / "work/calib/build/rig_ba", W / "build/rig_ba")
        held = target_map()
        try:
            hh, train = samples()
        except SparseEvidence as e:
            hh = []
            train = []
            report["warnings"].append(str(e))
        r = check(old, hh, held if mode == "refit" else None)
        report.update(measurements=r, baseline=base)
        report["warnings"] += r["warnings"]
        bad = gross(r, base, old)
        if bad:
            report.update(selected=None, gross_mismatch=bad)
            write(file, report)
            raise ValueError("incompatible rig: " + "; ".join(bad))
        if mode == "refit":
            duration = sum(b - a for a, b in read("inventory.json")["moving"])
            turns = sum(
                np.ptp(np.unwrap([r["yaw"] for r in train if r["seg"] == i]))
                >= np.deg2rad(12)
                for i in {r["seg"] for r in train}
            )
            small = min(r["counts"].values())
            report["refit_evidence"] = dict(
                moving_seconds=duration,
                disjoint_training_turns=int(turns),
                min_held_targets_per_lens=small,
            )
            if duration < 120 or turns < 2 or small < 50:
                report["warnings"].append(
                    "refit skipped: needs 120 moving seconds, two disjoint training turns of at least 12°, and 50 held targets per lens"
                )
            else:
                try:
                    fit = candidate(old, train)
                    fresh = target_map()
                    q = check(fit, hh, fresh)
                    ok, scores = paired(held, fresh, old)
                    report.update(refit_measurements=q, paired_scores=scores)
                    if ok and not gross(q, base, fit):
                        saved = dict(saved, cameras=fit)
                        report["selected"] = "refit"
                    else:
                        report["warnings"].append(
                            "refit did not improve paired held-out evidence; retained stored rig"
                        )
                except (IndexError, ValueError, np.linalg.LinAlgError, cv.error) as e:
                    report["warnings"].append(
                        f"refit rejected: {e}; retained stored rig"
                    )
    write(file, report)
    write("calib/rig.json", saved)
    print("selected", report["selected"], "warnings", report["warnings"], flush=True)


if __name__ == "__main__":
    if sys.argv[1] == "baseline":
        baseline()
    elif sys.argv[1] == "reference-panel":
        W.mkdir(parents=True, exist_ok=True)
        binlink(HOME / "work/calib/build/rig_ba", W / "build/rig_ba")
        panels, _ = samples()
        write("measure.json", check(read(HOME / "calib/rig.json")["cameras"], panels))
