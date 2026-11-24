"""Causal joint exposure/shading: periodic prefix fits + packet overlap updates.
A prefix fit re-anchors the common RGB gauge that pairwise online updates cannot observe.
No stitched arrays, inspection timestamps or target colours are inputs."""

import json
import numpy as np
from .fit import observations, solve
from .sample import WORK, PHOTO, CAMS, NAMES, NC
from .model import basis


def run(until=None):
    s = dict(np.load(WORK / "schedule.npz"))
    t = s["t"]
    M, i, j = observations("live")
    k = M[:, 2].astype(int)
    c = M[:, 3].astype(int)
    hold = (M[i, 0].astype(int) * 31 + M[i, 1].astype(int) * 97) % 8 == 0
    n = len(t) if until is None else np.searchsorted(t, until)
    t = t[:n]
    raw = np.log(M[:, 6:9])
    bb = basis(M[:, 4:6])
    order = np.argsort(k[i[~hold]], kind="stable")
    pi = i[~hold][order]
    pj = j[~hold][order]
    ends = np.searchsorted(k[pi], np.arange(n + 1))
    sh = [np.zeros((NC, 33, 3))]
    epoch = np.zeros(n, int)
    gain = np.zeros((n, NC, 3))
    prior = np.zeros((NC, 3))
    prevfit = -10.0
    last = 0
    for m in range(1, n):
        dt = t[m] - t[m - 1]
        K = ends[m]
        period = 1.25 if s["ds"][m] < 160 else 3.0
        prev = gain[m - 1].copy()
        if (
            K >= 150
            and t[m] - prevfit > period
            and (not s["still"][m] or t[m] - prevfit > 12)
        ):
            # Include old captures to preserve gauge, but do not let identical stopped images dominate shading.
            sel = np.flatnonzero(~s["still"][k[pi[:K]]] | ((k[pi[:K]] % 8) == 0))
            sel = sel[:: max(1, len(sel) // 38000)]
            a, b = pi[sel], pj[sel]
            sub = np.unique(np.r_[a, b])
            gg, v = solve(
                M[sub],
                np.searchsorted(sub, a),
                np.searchsorted(sub, b),
                t,
                m - 1,
                local_prior=12 if s["ds"][m] > 35 else 28,
            )
            prior = gg[m - 1]
            prev = prior.copy()
            sh.append(v)
            last = len(sh) - 1
            prevfit = t[m]
        epoch[m] = last
        v = sh[last]
        a = pi[ends[m] : ends[m + 1]]
        b = pj[ends[m] : ends[m + 1]]
        G = prev.copy()
        if len(a):
            past = k[b] < m
            old = np.einsum("ni,nij->nj", bb[b], np.array(sh)[epoch[k[b]], c[b]])
            rhs = (
                raw[b]
                + old
                - raw[a]
                - np.einsum("ni,nij->nj", bb[a], v[c[a]])
                + gain[k[b], c[b]] * past[:, None]
            )
            A = np.eye(NC)[c[a]] - np.eye(NC)[c[b]] * (~past)[:, None]
            H = np.eye(NC) * 10 + np.ones((NC, NC)) * 3
            rhs0 = 2 * prior + 8 * prev + 3 * prior.sum(0)
            for _ in range(2):
                wt = 1 / np.sqrt(1 + (abs(A @ G - rhs).mean(1) / 0.065) ** 2)
                G = np.linalg.solve(
                    H + A.T @ (wt[:, None] * A), rhs0 + A.T @ (wt[:, None] * rhs)
                )
        gain[m] = np.clip(G, -0.50, 0.50)
    sh = np.array(sh)
    out = dict(
        gain=gain.astype("f"),
        shade=sh.astype("f"),
        epoch=epoch,
        ns=s["ns"][:n],
        t=t,
        ready=s["ready"][:n],
        names=NAMES,
        sizes=[[c.width, c.height] for c in CAMS],
    )
    if until is not None:
        return out
    np.savez(PHOTO / "live-v2.npz", **out)
    a = i[hold][::2]
    b = j[hold][::2]
    sub = np.unique(np.r_[a, b])
    z = M[sub, 6:9] * np.exp(
        np.clip(
            gain[k[sub], c[sub]]
            + np.einsum("ni,nij->nj", bb[sub], sh[epoch[k[sub]], c[sub]]),
            -0.65,
            0.65,
        )
    )
    e = abs(z[np.searchsorted(sub, a)] - z[np.searchsorted(sub, b)]).mean(1)
    cross = c[a] != c[b]
    print(
        "prefix profiles",
        len(sh),
        "heldout cross",
        np.percentile(e[cross], [50, 90]),
        flush=True,
    )
    (PHOTO / "live-v2-metrics.json").write_text(
        json.dumps(
            dict(profiles=len(sh), cross=np.percentile(e[cross], [50, 90]).tolist())
        )
    )


def tone():
    """Use the same physical, camera-balanced illumination gauge in both scenes.
    The reference is a causal running median of observed road patches, not a
    white/grey target. Equal camera votes avoid cold narrow-FOV front dominance.
    Corrections are low frequency; texture, paint and local shadows survive."""
    s = dict(np.load(WORK / "schedule.npz"))
    M, _, _ = observations("live")
    k = M[:, 2].astype(int)
    c = M[:, 3].astype(int)
    o = np.argsort(k, kind="stable")
    bounds = np.searchsorted(k[o], np.arange(len(s["t"]) + 1))
    bb = basis(M[:, 4:6])
    raw = np.log(M[:, 6:9])
    gauge = np.zeros((len(s["t"]), 3))
    correct = {}
    for name in ("live-v2", "stitched"):
        p = np.load(PHOTO / (name + ".npz"))
        g = p["gain"].copy()
        sh = p["shade"]
        ep = p["epoch"]
        z = raw + g[k, c] + np.einsum("ni,nij->nj", bb, sh[ep[k], c])
        anchor = None
        corr = np.zeros((NC, 3))
        old = -2.0
        for j, t in enumerate(s["t"]):
            ix = o[bounds[j] : bounds[j + 1]]
            med = {}
            uc = {}
            for lens in range(NC):
                sub = ix[c[ix] == lens]
                if len(sub) < 4:
                    continue
                # Reject highlights and strongly coloured verge when setting a
                # camera's broad road illuminant; vote on JPEG log-code levels.
                qq = raw[sub]
                u = (
                    (qq[:, 1] > np.log(42))
                    & (qq[:, 1] < np.log(184))
                    & (np.ptp(qq, axis=1) < 0.65)
                )
                sub = sub[u]
                if len(sub) < 4:
                    continue
                med[lens] = np.median(z[sub], axis=0)
                uc[lens] = np.median(raw[sub], axis=0)
            if len(uc) >= 3:
                target = np.median(list(uc.values()), axis=0)
                dt = min(1.0, max(0.05, t - old))
                lam = 1 - np.exp(-dt / 1.3)
                anchor = target if anchor is None else anchor + lam * (target - anchor)
                for lens, v in med.items():
                    delta = anchor - v
                    lum = delta[1]
                    chrom = delta - lum
                    adjust = np.clip(0.88 * chrom, -0.24, 0.24) + np.clip(
                        0.78 * lum, -0.16, 0.16
                    )
                    corr[lens] += (1 - np.exp(-dt / 0.5)) * (adjust - corr[lens])
                old = t
            if j:
                g[j] += corr
            if anchor is not None:
                gauge[j] = anchor
        correct[name] = g.astype("f")
    np.savez(
        PHOTO / "ground-v2-tone.npz",
        live=correct["live-v2"],
        stitched=correct["stitched"],
        gauge=gauge,
        t=s["t"],
    )
    print("ground tone", len(gauge), flush=True)


if __name__ == "__main__":
    run()
    tone()
