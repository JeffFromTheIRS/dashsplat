import json
import numpy as np
from .sample import CAMS, NAMES, NC, WORK, PHOTO
from .model import basis
from scipy.sparse import coo_matrix, diags
from scipy.sparse.linalg import spsolve


def observations(name):
    M = np.vstack([np.load(p) for p in sorted(WORK.glob(name + "-*.npy"))])
    M = M[np.lexsort((M[:, 3], M[:, 2], M[:, 1], M[:, 0]))]
    bounds = np.r_[
        0, np.flatnonzero((np.diff(M[:, :2], axis=0) != 0).any(1)) + 1, len(M)
    ]
    I = []
    J = []
    for a, b in zip(bounds[:-1], bounds[1:]):
        if b - a < 2:
            continue
        m = np.arange(a + 1, b)
        I.extend([m, m[1::3]])
        J.extend([m - 1, np.full(len(m[1::3]), a)])
    return M, np.concatenate(I), np.concatenate(J)


def solve(M, i, j, t, end, local_prior=6):
    knots = np.unique(np.r_[t[0], np.arange(t[0], t[end] + 0.001, 0.65), t[end]])
    nn = len(knots)
    ng = NC * nn  # one gain per knot and lens, then D shading terms per lens
    D = 33
    L = ng + NC * D
    k = M[:, 2].astype(int)
    c = M[:, 3].astype(int)
    tm = t[np.minimum(k, end)]
    # A window cut can already have enough same-packet matches for its first prefix fit.
    b = np.clip(np.searchsorted(knots, tm, "right") - 1, 0, max(0, nn - 2))
    upper = np.minimum(b + 1, nn - 1)
    u = (tm - knots[b]) / np.maximum(knots[upper] - knots[b], 1e-8)
    idx = np.column_stack(
        (NC * b + c, NC * upper + c, ng + D * c[:, None] + np.arange(D))
    )
    val = np.column_stack((1 - u, u, basis(M[:, 4:6])))
    N = len(i)
    row = np.arange(N)[:, None] + np.zeros((1, 2 * (2 + D)), int)
    A = coo_matrix(
        (
            np.column_stack((val[i], -val[j])).ravel(),
            (row.ravel(), np.column_stack((idx[i], idx[j])).ravel()),
        ),
        shape=(N, L),
    ).tocsr()
    A.eliminate_zeros()
    y = np.log(M[j, 6:9] / M[i, 6:9])
    rr = np.arange(ng - NC)
    P = coo_matrix(
        (
            np.tile([-4.0, 4.0], len(rr)),
            (np.repeat(rr, 2), np.stack((rr, rr + NC), 1).ravel()),
        ),
        shape=(len(rr), L),
    ).tocsr()
    prior = np.r_[
        np.full(ng, 0.68), np.tile([5, 5, 7, 7, 8, 12, 12, 13] + [local_prior] * 25, NC)
    ]
    prior[0] = 1e4
    normal = P.T @ P + diags(prior**2)
    w = np.ones(N)
    X = np.zeros((L, 3))
    for _ in range(4):
        H = A.T @ A.multiply(w[:, None]) + normal
        X = spsolve(H.tocsc(), A.T @ (w[:, None] * y))
        e = np.median(abs(A @ X - y), axis=1)
        w = 1 / np.sqrt(1 + (e / 0.063) ** 2)
    G = X[:ng].reshape(nn, NC, 3)
    gain = np.stack([np.interp(t, knots, v) for v in G.reshape(nn, NC * 3).T], 1).reshape(
        -1, NC, 3
    )
    return gain, X[ng:].reshape(NC, D, 3)


def run(name):
    s = dict(np.load(WORK / "schedule.npz"))
    t = s["t"]
    M, i, j = observations(name)
    cell = M[i, :2].astype(int)
    hold = (cell[:, 0] * 31 + cell[:, 1] * 97) % 8 == 0
    k = M[:, 2].astype(int)
    c = M[:, 3].astype(int)
    train = np.flatnonzero(~hold)
    rng = np.random.default_rng(311)
    train = np.sort(rng.choice(train, min(340000, len(train)), replace=False))
    epoch = np.zeros(len(t), int)
    sh = []
    if name == "stitched":
        gain, V = solve(M, i[train], j[train], t, len(t) - 1)
        sh = [V]
    else:
        sh = [np.zeros((NC, 33, 3))]
        base = [
            np.zeros((NC, 3))
        ]  # Prefix fits exclude the current packet. Freeze the fixed sensor calibration after the exit.
        # Calibrate shading only from moving viewpoints. A seventy-second stop
        # otherwise contributes thousands of copies of the same shadow.
        Mm = M[~s["still"][k]]
        bounds = np.r_[
            0, np.flatnonzero((np.diff(Mm[:, :2], axis=0) != 0).any(1)) + 1, len(Mm)
        ]
        I = []
        J = []
        for aa, bb0 in zip(bounds[:-1], bounds[1:]):
            idx = np.arange(aa + 1, bb0)
            I.extend([idx, idx[1::3]])
            J.extend([idx - 1, np.full(len(idx[1::3]), aa)])
        im, jm = np.concatenate(I), np.concatenate(J)
        km = Mm[:, 2].astype(int)
        hm = (Mm[im, 0].astype(int) * 31 + Mm[im, 1].astype(int) * 97) % 8 == 0
        for sec in [5.6, 88, 96, 105]:
            end = np.searchsorted(t, sec) - 1
            tr = np.flatnonzero((~hm) & (km[im] <= end))
            tr = tr[:: max(1, len(tr) // 100000)]
            if len(tr) < 100:
                continue
            gg, V = solve(Mm, im[tr], jm[tr], t, end, local_prior=65)
            epoch[end + 1 :] = len(sh)
            sh.append(V)
            base.append(np.median(gg[: end + 1], axis=0))
        gain = np.zeros((len(t), NC, 3))
        bb = basis(M[:, 4:6])
        raw = np.log(M[:, 6:9])
        o = np.argsort(k[i[~hold]], kind="stable")
        pi = i[~hold][o]
        pj = j[~hold][o]
        begin = np.searchsorted(k[pi], np.arange(len(t) + 1))
        for m in range(len(t)):
            prev = gain[max(0, m - 1)]
            v = sh[epoch[m]]
            G = prev.copy()
            a = pi[begin[m] : begin[m + 1]]
            b = pj[begin[m] : begin[m + 1]]
            prior = base[epoch[m]]
            dt = t[m] - t[max(0, m - 1)]
            if len(a):
                old = raw[b] + np.einsum(
                    "ni,nij->nj", bb[b], np.stack(sh)[epoch[k[b]], c[b]]
                )
                y = old - raw[a] - np.einsum("ni,nij->nj", bb[a], v[c[a]])
                past = k[b] < m
                A = np.eye(NC)[c[a]] - np.eye(NC)[c[b]] * (~past)[:, None]
                rhs = y + gain[k[b], c[b]] * past[:, None]
                reg = 5.0
                smooth = 8.0
                H0 = np.eye(NC) * (reg + smooth) + np.ones((NC, NC)) * 12.0
                rr0 = reg * prior + smooth * prev + np.ones((NC, 1)) * 12 * prior.sum(0)
                for _ in range(3):
                    err = np.mean(abs(A @ G - rhs), axis=1)
                    w = 1 / np.sqrt(1 + (err / 0.055) ** 2)
                    G = np.linalg.solve(
                        H0 + A.T @ (A * w[:, None]), rr0 + A.T @ (rhs * w[:, None])
                    )
            gain[m] = np.clip(G, -0.45, 0.45) if m else np.zeros_like(G)
    sh = np.stack(sh)
    np.savez(
        PHOTO / (name + ".npz"),
        gain=gain.astype("f"),
        shade=sh.astype("f"),
        epoch=epoch,
        ns=s["ns"],
        t=t,
        ready=s["ready"],
        names=NAMES,
        sizes=[[c.width, c.height] for c in CAMS],
    )
    ix = np.flatnonzero(hold)[::2]
    a, b = i[ix], j[ix]
    sub = np.unique(np.r_[a, b])
    raw = M[sub, 6:9]
    corrected = raw * np.exp(
        np.clip(
            gain[k[sub], c[sub]]
            + np.einsum("ni,nij->nj", basis(M[sub, 4:6]), sh[epoch[k[sub]], c[sub]]),
            -0.65,
            0.65,
        )
    )
    aa, bb = np.searchsorted(sub, a), np.searchsorted(sub, b)
    e0 = abs(raw[aa] - raw[bb]).mean(1)
    e1 = abs(corrected[aa] - corrected[bb]).mean(1)
    out = {}
    for key, mask in [("all", a >= 0), ("same", c[a] == c[b]), ("cross", c[a] != c[b])]:
        out[key] = [
            int(mask.sum()),
            np.percentile(e0[mask], [50, 90]).tolist(),
            np.percentile(e1[mask], [50, 90]).tolist(),
        ]
    json.dump(out, open(PHOTO / (name + "-metrics.json"), "w"), indent=2)
    print(name, out, flush=True)


if __name__ == "__main__":
    for n in ["live", "stitched"]:
        run(n)
