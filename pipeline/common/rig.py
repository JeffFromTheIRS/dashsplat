"""Metric camera geometry for Reality Splat. Only NumPy and the standard library.

All UVs, including returned source_row, are COLMAP pixel centres (top-left
0.5,0.5). Only remap_x/y are zero-centred OpenCV array coordinates. R in
undistort_maps maps VIRTUAL camera vectors INTO the physical camera. Use
Trajectory('poses/ego_causal.npz') explicitly for live mapping.
"""

from pathlib import Path
from functools import lru_cache
import json, struct, zlib, re
import numpy as np


def _mask_png(path):
    # Byte grayscale noninterlaced PNG, including all five PNG row filters.
    data = Path(path).read_bytes()
    off = 8
    payload = bytearray()
    if data[:8] != bytes.fromhex("89504e470d0a1a0a"):
        raise ValueError(f"Not PNG: {path}")
    while off < len(data):
        n, tag = struct.unpack_from(">I4s", data, off)
        chunk = data[off + 8 : off + 8 + n]
        off += n + 12
        if tag == b"IHDR":
            w, h, bits, color, comp, filt, inter = struct.unpack(">IIBBBBB", chunk)
            if (bits, color, comp, filt, inter) != (8, 0, 0, 0, 0):
                raise ValueError("Expected 8-bit grayscale mask PNG")
        if tag == b"IDAT":
            payload.extend(chunk)
    raw = np.frombuffer(zlib.decompress(payload), np.uint8).reshape(h, w + 1)
    out = np.empty((h, w), np.uint8)
    zero = np.zeros(w, np.uint8)
    for i, row in enumerate(raw):
        f = row[0]
        a = row[1:]
        prev = out[i - 1] if i else zero
        if f == 0:
            out[i] = a
        elif f == 1:
            out[i] = a.cumsum(dtype=np.uint8)
        elif f == 2:
            out[i] = a + prev
        elif f in (3, 4):
            l = ul = 0
            for j, (value, above) in enumerate(zip(a.tolist(), prev.tolist())):
                if f == 3:
                    q = (l + above) // 2
                else:
                    p = l + above - ul
                    pl, pu, pul = abs(p - l), abs(p - above), abs(p - ul)
                    q = l if pl <= pu and pl <= pul else above if pu <= pul else ul
                l = (value + q) & 255
                ul = above
                out[i, j] = l
        else:
            raise ValueError("Invalid PNG filter")
    return out != 0


def _skew(w):
    x, y, z = np.moveaxis(np.asarray(w), -1, 0)
    o = np.zeros_like(x)
    return np.stack((o, -z, y, z, o, -x, -y, x, o), -1).reshape(x.shape + (3, 3))


def _exp(w):
    a = np.linalg.norm(w, axis=-1)
    K = _skew(w)
    return (
        np.eye(3)
        + np.sinc(a / np.pi)[..., None, None] * K
        + (0.5 * np.sinc(a / (2 * np.pi)) ** 2)[..., None, None] * (K @ K)
    )


def _log(R):
    v = (
        np.stack(
            (
                R[..., 2, 1] - R[..., 1, 2],
                R[..., 0, 2] - R[..., 2, 0],
                R[..., 1, 0] - R[..., 0, 1],
            ),
            -1,
        )
        * 0.5
    )
    sn = np.linalg.norm(v, axis=-1)
    cs = (np.trace(R, axis1=-2, axis2=-1) - 1) / 2
    theta = np.arctan2(sn, cs)
    w = v * np.where(sn > 1e-10, theta / np.maximum(sn, 1e-30), 1)[..., None]
    near = (cs < 0) & (sn < 1e-7)
    if np.any(near):
        val, vec = np.linalg.eigh((R[near] + R[near].swapaxes(-1, -2)) / 2)
        axis = vec[..., :, -1]
        axis *= np.where((axis * v[near]).sum(-1) < 0, -1, 1)[..., None]
        w[near] = axis * theta[near, None]
    return w


def _J(w):
    a = np.linalg.norm(w, axis=-1)
    K = _skew(w)
    b = np.where(
        a < 1e-3,
        1 / 6 - a * a / 120,
        (1 - np.sinc(a / np.pi)) / np.maximum(a * a, 1e-30),
    )
    return (
        np.eye(3)
        + (0.5 * np.sinc(a / (2 * np.pi)) ** 2)[..., None, None] * K
        + b[..., None, None] * (K @ K)
    )


def _mv(M, v):
    return np.einsum("...ij,...j->...i", M, v)


def _T(R, p):
    t = np.broadcast_to(
        np.eye(4), np.broadcast_shapes(R.shape[:-2], p.shape[:-1]) + (4, 4)
    ).copy()
    t[..., :3, :3] = R
    t[..., :3, 3] = p
    return t


class Camera:
    def __init__(self, name, spec, folder):
        self.name = name
        self.width = spec["width"]
        self.height = spec["height"]
        self.model = spec["model"]
        self.params = np.array(spec["params"], float)
        self.T_ego_cam = np.array(spec["T_ego_cam"], float)
        self.time_offset_s = spec["time_offset_s"]
        self.rs_readout_s = spec["rs_readout_s"]
        self.mask = _mask_png(folder / spec["mask"])
        self.root = folder.parent
        if self.mask.shape != (self.height, self.width):
            raise ValueError("Mask dimensions do not match camera")
        if self.model not in ("OPENCV", "OPENCV_FISHEYE"):
            raise ValueError(self.model)
        if self.model == "OPENCV_FISHEYE":
            k = self.params[4:]
            roots = np.polynomial.polynomial.polyroots(
                [1, 3 * k[0], 5 * k[1], 7 * k[2], 9 * k[3]]
            )
            real = [np.sqrt(z.real) for z in roots if abs(z.imag) < 1e-9 and z.real > 0]
            self.limit = min([np.pi] + real)
            self.rmax = self._radial(self.limit)

    def _radial(self, t):
        a, b, c, d = self.params[4:]
        u = t * t
        return t * (1 + u * (a + u * (b + u * (c + u * d))))

    def mask_at(self, uv):
        uv = np.asarray(uv, float)
        x, y = np.moveaxis(uv, -1, 0)
        ok = (x >= 0) & (y >= 0) & (x < self.width) & (y < self.height)
        return (
            ok
            & self.mask[np.where(ok, y, 0).astype(int), np.where(ok, x, 0).astype(int)]
        )

    def _brown(self, xy, jac=False):
        x, y = np.moveaxis(xy, -1, 0)
        k1, k2, p1, p2 = self.params[4:]
        r = x * x + y * y
        g = 1 + r * (k1 + r * k2)
        v = np.stack(
            (
                x * g + 2 * p1 * x * y + p2 * (r + 2 * x * x),
                y * g + p1 * (r + 2 * y * y) + 2 * p2 * x * y,
            ),
            -1,
        )
        if not jac:
            return v
        q = 2 * (k1 + 2 * k2 * r)
        cross = q * x * y + 2 * (p1 * x + p2 * y)
        return v, np.stack(
            (
                g + q * x * x + 2 * p1 * y + 6 * p2 * x,
                cross,
                cross,
                g + q * y * y + 6 * p1 * y + 2 * p2 * x,
            ),
            -1,
        ).reshape(x.shape + (2, 2))

    def project(self, X_cam):
        X = np.asarray(X_cam, float)
        z = X[..., 2]
        rho = np.linalg.norm(X[..., :2], axis=-1)
        if self.model == "OPENCV_FISHEYE":
            t = np.arctan2(rho, z)
            q = self._radial(t) / np.maximum(rho, 1e-300)
            xy = X[..., :2] * q[..., None]
            geom = (t < self.limit) & ((rho > 0) | (z > 0))
        else:
            xy = self._brown(X[..., :2] / np.where(z > 0, z, 1)[..., None])
            geom = z > 0
        uv = xy * self.params[:2] + self.params[2:4]
        return uv, geom & np.all(np.isfinite(X), axis=-1) & self.mask_at(uv)

    def unproject(self, uv):
        xy = (np.asarray(uv, float) - self.params[2:4]) / self.params[:2]
        if self.model == "OPENCV_FISHEYE":
            r = np.linalg.norm(xy, axis=-1)
            t = np.minimum(r, self.limit * 0.99)
            lo = np.zeros_like(r)
            hi = np.full_like(r, self.limit)
            a, b, c, d = self.params[4:]
            for _ in range(15):
                e = self._radial(t) - r
                lo = np.where(e < 0, t, lo)
                hi = np.where(e > 0, t, hi)
                u = t * t
                der = 1 + u * (3 * a + u * (5 * b + u * (7 * c + 9 * d * u)))
                new = t - e / np.maximum(der, 1e-14)
                t = np.where((new < lo) | (new > hi), (lo + hi) / 2, new)
            ray = np.concatenate(
                (
                    xy * (np.sin(t) / np.maximum(r, 1e-300))[..., None],
                    np.cos(t)[..., None],
                ),
                -1,
            )
            return np.where((r <= self.rmax)[..., None], ray, np.nan)
        q = xy.copy()
        for _ in range(12):
            pq, J = self._brown(q, True)
            e = pq - xy
            det = J[..., 0, 0] * J[..., 1, 1] - J[..., 0, 1] ** 2
            d = np.stack(
                (
                    J[..., 1, 1] * e[..., 0] - J[..., 0, 1] * e[..., 1],
                    J[..., 0, 0] * e[..., 1] - J[..., 0, 1] * e[..., 0],
                ),
                -1,
            )
            q -= d / np.where(abs(det) > 1e-15, det, np.nan)[..., None]
        ray = np.concatenate((q, np.ones(q.shape[:-1] + (1,))), -1)
        return ray / np.linalg.norm(ray, axis=-1, keepdims=True)

    def row_time(self, t_frame, v):
        return (
            np.asarray(t_frame)
            + self.time_offset_s
            + (np.asarray(v) / self.height - 0.5) * (self.rs_readout_s or 0.0)
        )


@lru_cache(maxsize=8)
def _load(path, mtime):
    return {
        n: Camera(n, c, path.parent)
        for n, c in json.loads(path.read_text())["cameras"].items()
    }


def load_rig(path="calib/rig.json"):
    """Return a dict keyed by camera name."""
    p = Path(path).resolve()
    return _load(p, p.stat().st_mtime_ns)


class Trajectory:
    """Piecewise constant body twist: T0 exp(alpha * log(inv(T0) T1)).
    velocity(t) returns (linear origin velocity, angular velocity), both in ENU
    metres/s and radians/s. Outside the sampled interval poses clamp, velocity=0.
    """

    def __init__(self, path="poses/ego.npz", timeline=None):
        p = Path(path).resolve()
        tl = Path(timeline) if timeline else p.parent.parent / "data/drive/timeline.npz"
        self.source = p if p.exists() else tl
        with np.load(self.source) as d:
            self.t = d["t"]
            self.road = {
                key: d[key]
                for key in ("road_pitch", "road_roll", "susp_heave")
                if key in d
            }
            self.poses = (
                d["T_world_ego"]
                if p.exists()
                else _T(
                    _exp(np.stack((d["yaw"] * 0, d["yaw"] * 0, d["yaw"]), -1)),
                    np.stack((d["x"], d["y"], d["x"] * 0), -1),
                )
            )
        self.dt = np.diff(self.t)
        if len(self.dt) == 0 or np.any(self.dt <= 0):
            raise ValueError("Trajectory requires strictly increasing times")
        self.R = self.poses[:, :3, :3]
        self.p = self.poses[:, :3, 3]
        rt = self.R[:-1].transpose(0, 2, 1)
        self.w = _log(rt @ self.R[1:])
        self.u = np.linalg.solve(
            _J(self.w), _mv(rt, np.diff(self.p, axis=0))[..., None]
        )[..., 0]

    def _at(self, t):
        t = np.asarray(t, float)
        i = np.clip(np.searchsorted(self.t, t, "right") - 1, 0, len(self.dt) - 1)
        a = np.clip((t - self.t[i]) / self.dt[i], 0, 1)
        return i, a

    def T_world_ego(self, t):
        i, a = self._at(t)
        w = self.w[i] * a[..., None]
        return _T(
            self.R[i] @ _exp(w),
            self.p[i] + _mv(self.R[i], _mv(_J(w), self.u[i] * a[..., None])),
        )

    def ground_plane(self, t):
        """Return (unit upward normal in ego, signed normal height of ego origin).
        Road points satisfy normal @ X_ego + height = 0. V2 road pitch/roll
        are tangent angles in the heading-aligned world frame, pitch nose-up,
        roll left-side-up. Legacy/fallback files use the ENU plane z=0.
        """
        T = self.T_world_ego(t)
        R = T[..., :3, :3]
        if not self.road:
            return R[..., 2, :], T[..., 2, 3]
        p, r, h = [
            np.interp(t, self.t, self.road[key])
            for key in ("road_pitch", "road_roll", "susp_heave")
        ]
        yaw = np.arctan2(R[..., 1, 0], R[..., 0, 0])
        nx = -np.sin(p) * np.cos(r)
        ny = -np.sin(r)
        nw = np.stack(
            (
                np.cos(yaw) * nx - np.sin(yaw) * ny,
                np.sin(yaw) * nx + np.cos(yaw) * ny,
                np.cos(p) * np.cos(r),
            ),
            -1,
        )
        return _mv(R.swapaxes(-1, -2), nw), h

    def velocity(self, t):
        i, a = self._at(t)
        R = self.R[i] @ _exp(self.w[i] * a[..., None])
        dt = self.dt[i][..., None]
        v = _mv(R, self.u[i]) / dt
        w = _mv(R, self.w[i]) / dt
        valid = ((np.asarray(t) >= self.t[0]) & (np.asarray(t) <= self.t[-1]))[
            ..., None
        ]
        return np.where(valid, v, 0), np.where(valid, w, 0)

    def linear_velocity(self, t):
        return self.velocity(t)[0]

    def angular_velocity(self, t):
        return self.velocity(t)[1]


@lru_cache(maxsize=4)
def _trajectory(p, mtime):
    return Trajectory(p)


def ground_plane(t, trajectory=None):
    """Local tangent road plane (normal_ego, signed height in metres)."""
    if trajectory is None:
        p = Path("poses/ego.npz").resolve()
        trajectory = _trajectory(p, p.stat().st_mtime_ns if p.exists() else -1)
    return trajectory.ground_plane(t)


def frame_table(cam):
    camera = load_rig()[cam] if isinstance(cam, str) else cam
    return json.loads(
        (camera.root / "data/drive/cams" / f"{camera.name}.json").read_text()
    )


def T_world_cam(cam, t_frame, v=None, trajectory=None):
    cam = load_rig()[cam] if isinstance(cam, str) else cam
    if trajectory is None:
        p = cam.root / "poses/ego.npz"
        trajectory = _trajectory(p, p.stat().st_mtime_ns if p.exists() else -1)
    return (
        trajectory.T_world_ego(
            cam.row_time(t_frame, cam.height / 2 if v is None else v)
        )
        @ cam.T_ego_cam
    )


def undistort_maps(cam, K_new, size, R=None):
    """(width,height), K_new in COLMAP centres. Returns float32 (mx,my,source_row);
    invalid samples have mx=my=-1 and source_row=NaN. R maps virtual to physical."""
    cam = load_rig()[cam] if isinstance(cam, str) else cam
    w, h = size
    y, x = np.mgrid[:h, :w]
    r = np.stack((x + 0.5, y + 0.5, np.ones_like(x)), -1) @ np.linalg.inv(K_new).T
    if R is not None:
        r = r @ np.asarray(R).T
    uv, ok = cam.project(r)
    mx, my = np.moveaxis(np.where(ok[..., None], uv - 0.5, -1).astype("f"), -1, 0)
    return mx, my, np.where(ok, uv[..., 1], np.nan).astype("f")


def _selfcheck():
    cams = load_rig()
    root = next(iter(cams.values())).root
    for n, c in cams.items():
        worst = 0.0
        count = 0
        for j in range(0, c.height, 32):
            y, x = np.nonzero(c.mask[j : j + 32])
            uv = np.stack((x + 0.5, y + j + 0.5), -1)
            p, valid = c.project(c.unproject(uv))
            assert valid.all()
            worst = max(worst, np.linalg.norm(p - uv, axis=-1).max(initial=0))
            count += len(uv)
        assert worst < 0.01
        print(f"{n}: {count} valid centres, max round-trip {worst:.7g} px")
    # Read the original final Ceres problem (actual world landmarks and their
    # exposure rows); recover the one rigid gauge transform used at delivery.
    p = root / "work/calib/all"
    f = open(p / "ba.txt")
    nc, nf, np_, no, it, loss, mode, threads, ne = map(float, next(f).split())
    nc, nf, np_, no, ne = map(int, (nc, nf, np_, no, ne))
    meta = np.loadtxt(f, max_rows=nc)
    fs = np.loadtxt(f, max_rows=nf)
    np.loadtxt(f, max_rows=ne)
    np.loadtxt(f, max_rows=np_)
    obs = np.loadtxt(f, max_rows=no)
    f.close()
    res = np.load(p / "result.npz")
    par = res["cam"]
    corr = res["frame"]
    points = res["points"]
    # The solver's camera order: these names, less any this rig does not have (HW3 has no pillar cameras).
    names = [
        n
        for n in (
            "front",
            "back",
            "left_pillar",
            "right_pillar",
            "left_repeater",
            "right_repeater",
        )
        if n in cams
    ]
    raw = _T(meta[:, 16:25].reshape(nc, 3, 3) @ _exp(par[:, 8:11]), par[:, 11:14])
    gauge = raw[0] @ np.linalg.inv(cams["front"].T_ego_cam)
    report = (root / "calib/report.md").read_text()
    for i, n in enumerate(names):
        c = cams[n]
        z = obs[obs[:, 1] == i]
        k = z[:, 0].astype(int)
        uv = z[:, 4:6]
        tau = c.row_time(z[:, 3], uv[:, 1])
        base = fs[k, :6]
        yaw = base[:, 2] + base[:, 5] * tau
        R = _exp(np.stack((yaw * 0, yaw * 0, yaw), -1)) @ _exp(corr[k, :3])
        p0 = corr[k, 3:].copy()
        p0[:, :2] += base[:, :2] + base[:, 3:5] * tau[:, None]
        wc = _T(R, p0) @ gauge @ c.T_ego_cam
        pred, _ = c.project(
            _mv(
                wc[:, :3, :3].transpose(0, 2, 1),
                points[z[:, 2].astype(int)] - wc[:, :3, 3],
            )
        )
        rms = np.sqrt(np.mean(np.sum((pred - uv) ** 2, axis=1)))
        line = next(l for l in report.splitlines() if l.startswith("|" + n + "|"))
        target = float(line.split("|")[6].split("/")[0])
        assert abs(rms - target) < 0.00055, (n, rms, target)
        print(
            f"{n}: {len(k)} real BA observations, RMS {rms:.5f} px vs report {target:.3f}"
        )


if __name__ == "__main__":
    _selfcheck()
