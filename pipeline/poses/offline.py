"""Whole-drive sparse metric bundle, with held-out tracks reserved before pose fitting."""

from .motion import *
from pipeline.common.rig import _exp, _mv, _T, _log, _skew
import subprocess
from scipy.ndimage import gaussian_filter1d
from collections import defaultdict


class Fit:
    def __init__(self):
        self.s, self.d, self.t, self.B, self.stop = inputs()
        s = self.s
        self.ix = s["ix"]
        self.tc = s["tc"]
        self.kt = s["t"]
        self.ds = s["distance"]
        self.sk, self.kids, self.first = knots(s, self.t, self.stop)
        self.tk = self.kt[self.sk]
        self.base = self.B[self.ix[self.sk]]
        self.unique = self.base[self.first]
        self.C = list(load_rig(RS / "calib/rig.json")[n] for n in NAMES)
        self.sizes = np.array([[c.width, c.height] for c in self.C])
        self.kidx = np.clip(np.searchsorted(self.tk, self.kt), 0, len(self.tk) - 1)
        self.rr = np.stack([c.T_ego_cam[:3, :3] for c in self.C])
        self.lever = np.stack([c.T_ego_cam[:3, 3] for c in self.C])
        self.cp = np.stack(
            [
                np.r_[c.params, c.T_ego_cam[:3, :3].ravel(), c.T_ego_cam[:3, 3]]
                for c in self.C
            ]
        )
        if not all(c.model == "OPENCV_FISHEYE" for c in self.C):
            raise ValueError("Ceres factor supports the fitted fisheye rig only")
        self.corr = np.load(W / "causal-knots.npz")["c"][self.sk[self.first]].copy()

    def pools(self):
        train = [[], [], [], []]
        hold = [[], [], [], []]
        # One pool of single-lens tracks per camera, then (ci == NC) the cross-lens tracks.
        for ci in range(NC + 1):
            if ci < NC:
                x = np.load(W / f"lk-{NAMES[ci]}.npz")
                id = x["id"]
                k = np.repeat(np.arange(len(self.ix), dtype="i"), np.diff(x["ptr"]))
                c = np.full(len(id), ci, "uint8")
                uv = x["uv"]
                order = np.argsort(id, kind="stable")
                id = id[order]
                k = k[order]
                c = c[order]
                uv = uv[order]
            else:
                x = np.load(W / "cross-tracks.npz")
                id = x["id"]
                k = x["k"]
                c = x["cam"]
                uv = x["uv"]
                order = np.lexsort((k, id))
                id, k, c, uv = [v[order] for v in [id, k, c, uv]]
            st = np.r_[0, np.flatnonzero(np.diff(id)) + 1]
            end = np.r_[st[1:], len(id)]
            length = end - st
            dd = self.ds[k[end - 1]] - self.ds[k[st]]
            tid = id[st]
            is_hold = ((tid.astype("uint64") * 9176 + ci * 37 + 17) % 13) == 0
            enough = (length >= 4) & (dd > 1.0)
            cell = (
                (self.kidx[k[st]] * NC + c[st]) * 4
                + np.minimum(3, (uv[st, 0] / self.sizes[c[st], 0] * 4).astype(int))
            ) * 3 + np.minimum(2, (uv[st, 1] / self.sizes[c[st], 1] * 3).astype(int))
            rng = np.random.default_rng(127 + ci)
            prio = np.log1p(np.minimum(length, 50)) + 0.9 * rng.random(len(st))
            cand = np.flatnonzero(enough & ~is_hold)
            cand = cand[np.argsort(-prio[cand])]
            ii = np.unique(cell[cand], return_index=True)[1]
            keep = cand[ii]
            if ci == NC:  # each cross-lens point independently informs metric baselines
                rest = np.delete(cand, ii)
                keep = np.r_[keep, rest[np.unique(cell[rest], return_index=True)[1]]]
            for idx, out, maxn in [
                (np.sort(keep), train, 16),
                (np.flatnonzero(is_hold & enough & ((dd > 12) | (ci == NC))), hold, 60),
            ]:
                num = np.minimum(length[idx], maxn)
                rep = np.repeat(np.arange(len(idx)), num)
                xindex = np.arange(len(rep)) - np.repeat(
                    np.r_[0, num.cumsum()[:-1]], num
                )
                obs = st[idx[rep]] + np.rint(
                    xindex * (length[idx[rep]] - 1) / np.maximum(1, num[rep] - 1)
                ).astype(int)
                out[0].append(num)
                out[1].append(k[obs])
                out[2].append(c[obs])
                out[3].append(uv[obs])
            print("pool", ci, len(keep), "held-out", len(hold[0][-1]), flush=True)
        for filename, out in [("pool", train), ("hold", hold)]:
            np.savez(
                W / (filename + ".npz"),
                num=np.concatenate(out[0]),
                k=np.concatenate(out[1]),
                c=np.concatenate(out[2]),
                uv=np.vstack(out[3]),
            )

    def raydata(self, k, c, uv):
        tt = (
            self.tc[k, c]
            + np.array([a.time_offset_s for a in self.C])[c]
            + (uv[:, 1] / self.sizes[c, 1] - 0.5)
            * np.array([a.rs_readout_s for a in self.C])[c]
        )
        bas = sample(self.t, self.B, tt)
        j, alpha = brackets(self.tk, tt)
        fr = (
            self.corr[self.kids[j]] * (1 - alpha[:, None])
            + self.corr[self.kids[j + 1]] * alpha[:, None]
        )
        pose = assemble(bas, fr)
        wcR = pose[:, :3, :3] @ self.rr[c]
        cent = pose[:, :3, 3] + _mv(pose[:, :3, :3], self.lever[c])
        raw = np.zeros((len(uv), 3))
        for ci in np.unique(c):
            idx = c == ci
            raw[idx] = self.C[ci].unproject(uv[idx])
        return _mv(wcR, raw), cent, wcR, bas, self.kids[j], self.kids[j + 1], alpha

    def project(self, c, X, R, O, uv):
        local = _mv(R.transpose(0, 2, 1), X - O)
        dst = np.linalg.norm(local, axis=1)
        pred = np.empty_like(uv, dtype=float)
        valid = np.zeros(len(c), bool)
        for i in np.unique(c):
            ii = c == i
            pred[ii], valid[ii] = self.C[i].project(local[ii])
        return np.linalg.norm(pred - uv, axis=1), dst, valid, local[:, 2]

    def prepare(self):
        d = np.load(W / "pool.npz")
        num = d["num"]
        k = d["k"]
        c = d["c"]
        uv = d["uv"]
        bounds = np.r_[0, np.cumsum(num)]
        accepted = []
        pts = []
        for a in range(0, len(num), 2500):
            end = min(len(num), a + 2500)
            sl = slice(bounds[a], bounds[end])
            n = num[a:end]
            st = np.r_[0, n.cumsum()[:-1]]
            ri, O, R, *_ = self.raydata(k[sl], c[sl], uv[sl])
            M = np.eye(3) - ri[:, :, None] * ri[:, None, :]
            H = np.add.reduceat(M, st)
            X = np.linalg.solve(
                H + 1e-9 * np.eye(3), np.add.reduceat(_mv(M, O), st)[:, :, None]
            )[:, :, 0]
            e, di, valid, z = self.project(c[sl], np.repeat(X, n, axis=0), R, O, uv[sl])
            mu = np.add.reduceat(np.minimum(e, 48), st) / n
            rad = np.maximum.reduceat(di, st)
            g = (
                (mu < 11)
                & (np.minimum.reduceat(z, st) > 0.15)
                & (rad < 320)
                & (np.linalg.eigvalsh(H)[:, 0] / n > 1.4e-5)
            )
            accepted.append(g)
            pts.append(X[g])
        keep = np.concatenate(accepted)
        oo = np.repeat(keep, num)
        self.n = num[keep]
        self.k = k[oo]
        self.cam = c[oo]
        self.uv = uv[oo]
        self.pts = np.vstack(pts)
        print("init", len(self.n), len(self.k), flush=True)

    def errors(self):
        ids = np.repeat(np.arange(len(self.n)), self.n)
        ee = []
        for i in range(0, len(ids), 60000):
            sl = slice(i, i + 60000)
            r, O, R, *_ = self.raydata(self.k[sl], self.cam[sl], self.uv[sl])
            e, _, _, z = self.project(
                self.cam[sl], self.pts[ids[sl]], R, O, self.uv[sl]
            )
            ee.append(np.where(z > 0.05, e, 1e3))
        return np.concatenate(ee)

    def prune(self, lim):
        e = self.errors()
        st = np.r_[0, self.n.cumsum()[:-1]]
        good = e < lim
        ns = np.add.reduceat(good.astype(int), st)
        keep = (
            (ns >= 4)
            & (ns / self.n > 0.75)
            & (np.add.reduceat(np.minimum(e, 20), st) / self.n < 4.0)
        )
        obs = good & np.repeat(keep, self.n)
        self.n = ns[keep]
        self.pts = self.pts[keep]
        self.k = self.k[obs]
        self.cam = self.cam[obs]
        self.uv = self.uv[obs]
        print("static retained", len(self.n), len(self.k), flush=True)

    def ba(self, it, loss):
        st = np.r_[0, self.n.cumsum()[:-1]]
        ptids = np.repeat(np.arange(len(st)), self.n)
        tx = self.tk
        ids = self.kids
        B = self.base
        gg = np.stack((self.d["gps_x"], self.d["gps_y"]), -1) - self.B[:, :2]
        gps = gaussian_filter1d(gg, 24, axis=0)[self.ix[self.sk[self.first]]]
        prior = np.column_stack((gps, np.tile([0.034, 2.2, 0.70], (len(gps), 1))))
        prior[0, :2] = 0
        prior[0, 3:] = 0.01
        edges = []
        smooth = []
        for j in range(len(tx) - 1):
            aa, bb = ids[j : j + 2]
            dx, dy = B[j + 1, :2] - B[j, :2]
            dt = tx[j + 1] - tx[j]
            if aa != bb:
                edges.append(
                    [aa, bb, dx, dy, (B[j, 2] + B[j + 1, 2]) / 2, np.hypot(dx, dy), dt]
                )
            if (
                j
                and ids[j - 1] != aa
                and aa != bb
                and max(dt, tx[j] - tx[j - 1]) < 0.55
            ):
                smooth.append([ids[j - 1], aa, bb, tx[j] - tx[j - 1], dt])
        with open(W / "problem.bin", "wb") as f:
            np.array(
                [
                    len(self.corr),
                    len(self.n),
                    len(self.k),
                    len(edges),
                    len(smooth),
                    it,
                    14,
                ],
                dtype="i",
            ).tofile(f)
            # bundle.cpp (and road/poses.py, which patches this file in place) reads six camera slots. A car
            # with fewer cameras leaves the last ones zero; no observation refers to them.
            if NC > 6:
                raise ValueError("the trajectory solver's problem format holds at most six cameras")
            cp = np.zeros((6, self.cp.shape[1]))
            cp[:NC] = self.cp
            for x in [[loss], cp, self.corr, prior, self.pts, edges, smooth]:
                np.ascontiguousarray(x, dtype="d").tofile(f)
            for j in range(0, len(self.k), 100000):
                sl = slice(j, j + 100000)
                k, c, uv = self.k[sl], self.cam[sl], self.uv[sl]
                tt = (
                    self.tc[k, c]
                    + self.cp[c, 0] * 0
                    + np.array([v.time_offset_s for v in self.C])[c]
                    + (uv[:, 1] / self.sizes[c, 1] - 0.5)
                    * np.array([v.rs_readout_s for v in self.C])[c]
                )
                bas = sample(self.t, self.B, tt)
                idx, al = brackets(tx, tt)
                np.ascontiguousarray(
                    np.column_stack(
                        (ids[idx], ids[idx + 1], ptids[sl], c, al, bas, uv)
                    ),
                    dtype="d",
                ).tofile(f)
        subprocess.run(
            [W / "build/trajectory_ba", W / "problem.bin", W / "solution.bin"],
            check=True,
        )
        x = np.fromfile(W / "solution.bin")
        self.corr = x[: self.corr.size].reshape(self.corr.shape)
        self.pts = x[self.corr.size :].reshape((-1, 3))
        print(
            "RMS", *[round(np.sqrt(np.mean(e**2)), 2) for e in self.bycam()], flush=True
        )

    def bycam(self):
        e = self.errors()
        return [e[self.cam == i] for i in range(NC)]

    def save(self):
        e = self.errors()
        np.savez(
            W / "offline-fit.npz",
            corr=self.corr,
            pts=self.pts,
            n=self.n,
            k=self.k,
            cam=self.cam,
            uv=self.uv,
            error=e,
        )
        c = corr_interp(self.tk, self.corr[self.kids], self.t)
        p = assemble(self.B, c)
        ids = np.repeat(np.arange(len(self.n)), self.n)
        nt = np.zeros(len(self.sk))
        er = np.full(len(self.sk), 3.0)
        ki = self.kidx[self.k]
        sort = np.argsort(ki)
        lo = np.searchsorted(ki[sort], np.arange(len(self.sk) + 1))
        for j in range(len(nt)):
            ii = sort[lo[max(j - 1, 0)] : lo[min(j + 2, len(lo) - 1)]]
            nt[j] = len(np.unique(ids[ii]))
            er[j] = np.median(e[ii]) if len(ii) else 7.0
        good = np.interp(self.t, self.tk, nt)
        good[self.stop] = 0
        noise = np.interp(self.t, self.tk, er)
        sp = 0.035 + (1 + noise) / np.sqrt(np.maximum(1, good))
        sr = np.deg2rad(0.045 + 1.3 * (1 + noise) / np.sqrt(np.maximum(1, good)))
        export("ego", self.t, p, sp, sr, good.astype("i"), self.stop)
        print("offline saved", flush=True)

    def load(self):
        a = np.load(W / "offline-fit.npz")
        self.corr, self.pts, self.n, self.k, self.cam, self.uv = [
            a[x] for x in ["corr", "pts", "n", "k", "cam", "uv"]
        ]
        return self


if __name__ == "__main__":
    s = Fit()
    s.pools()
    s.prepare()
    s.ba(32, 3.2)
    s.prune(5.0)
    s.ba(24, 2.0)
    s.save()
