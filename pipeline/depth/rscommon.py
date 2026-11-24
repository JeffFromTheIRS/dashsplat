"""Shared helpers for the depth and mask stages: paths, frames, rig access, virtual pinhole views,
depth utilities.

Camera maths (projection, unprojection, undistortion maps, trajectory) comes from the shared module
pipeline/common/rig.py. Camera parameters come from $RS/calib/rig.json (COLMAP OPENCV_FISHEYE; pixel
centre of the top-left pixel at (0.5, 0.5)). Ego poses come from $RS/poses/ego.npz when present,
otherwise from the planar telemetry trajectory (common.rig.Trajectory falls back by itself).
"""
import json
import os
import sys
import types
from functools import lru_cache
from pathlib import Path

import numpy as np

RS = Path(os.environ.get("RS", "~/dev/reality-splat")).expanduser()
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common import rig as crig  # noqa: E402
from common.hardware import roster  # noqa: E402

# What "--cams all" means: the cameras of the calibrated rig, six on HW4 and four on HW3.
CAMS = roster(RS)

def setup_env(threads=8):
    """Process-wide settings: thread caps (shared CPU), MPS fallback, duplicate-OpenMP workaround."""
    os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
    os.environ.setdefault("OMP_NUM_THREADS", str(threads))
    os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    try:
        import cv2
        cv2.setNumThreads(threads)
    except Exception:
        pass
    try:
        import torch
        torch.set_num_threads(threads)
    except Exception:
        pass


def stub_optional_modules(names=("moviepy", "moviepy.editor", "open3d", "pillow_heif")):
    """Some model repos import heavy viewer/export packages at module load; stub the ones we never call."""
    for name in names:
        if name in sys.modules:
            continue
        try:
            __import__(name)
        except Exception:
            sys.modules[name] = types.ModuleType(name)


# ----------------------------------------------------------------------------------------------
# Frames and time


@lru_cache(maxsize=None)
def frame_table(cam):
    """(n, t) arrays for a camera, cached as npz under work/depth/cache."""
    cache = RS / "work" / "depth" / "cache" / f"frames_{cam}.npz"
    src = RS / "data" / "drive" / "cams" / f"{cam}.json"
    if cache.exists() and cache.stat().st_mtime >= src.stat().st_mtime:
        z = np.load(cache)
        return z["n"], z["t"]
    rows = json.loads(src.read_text())
    n = np.array([r["n"] for r in rows], dtype=np.int64)
    t = np.array([r["t"] for r in rows], dtype=np.float64)
    cache.parent.mkdir(parents=True, exist_ok=True)
    np.savez(cache, n=n, t=t)
    return n, t


def frame_time(cam, n):
    ns, ts = frame_table(cam)
    if n < len(ns) and ns[n] == n:
        return float(ts[n])
    return float(ts[np.searchsorted(ns, n)])


def frame_at_time(cam, t):
    ns, ts = frame_table(cam)
    i = int(np.clip(np.searchsorted(ts, t), 1, len(ts) - 1))
    if abs(ts[i - 1] - t) < abs(ts[i] - t):
        i -= 1
    return int(ns[i])


def frame_path(cam, n):
    return RS / "data" / "drive" / "frames" / cam / f"{n:06d}.jpg"


def load_image(cam, n, size=None):
    """RGB uint8 image, optionally resized to size=(w, h) with area filtering."""
    import cv2
    im = cv2.imread(str(frame_path(cam, n)), cv2.IMREAD_COLOR)
    if im is None:
        raise FileNotFoundError(frame_path(cam, n))
    im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
    if size is not None and (im.shape[1], im.shape[0]) != tuple(size):
        im = cv2.resize(im, tuple(size), interpolation=cv2.INTER_AREA)
    return im


# ----------------------------------------------------------------------------------------------
# Shared-machine manners and atomic writes


def memory_free_pct():
    """macOS system-wide free memory percentage from memory_pressure (None if unavailable)."""
    import re
    import subprocess
    try:
        out = subprocess.run(["memory_pressure"], capture_output=True, text=True, timeout=30).stdout
        m = re.search(r"free percentage:\s*(\d+)%", out)
        return int(m.group(1)) if m else None
    except Exception:
        return None


def wait_for_memory(min_free=20, poll=30, what="", log=print):
    """Block while free memory is below min_free percent (another job may need it), then return."""
    import time
    waited = 0
    while True:
        f = memory_free_pct()
        if f is None or f >= min_free:
            if waited:
                log(f"memory free {f}% after waiting {waited}s {what}", flush=True)
            return f
        if waited % 300 == 0:
            log(f"memory free {f}% < {min_free}%: waiting before {what or 'continuing'}", flush=True)
        time.sleep(poll)
        waited += poll


def atomic_write_bytes(path, data):
    path = Path(path)
    tmp = path.with_name(path.name + ".tmp")
    with open(tmp, "wb") as f:
        f.write(data)
    os.replace(tmp, path)


def atomic_write_text(path, text):
    atomic_write_bytes(path, text.encode())


def atomic_save_npy(path, arr):
    import io
    buf = io.BytesIO()
    np.save(buf, arr)
    atomic_write_bytes(path, buf.getvalue())


def atomic_imwrite(path, img):
    import cv2
    ok, enc = cv2.imencode(Path(path).suffix, img)
    if not ok:
        raise IOError(f"encode failed: {path}")
    atomic_write_bytes(path, enc.tobytes())


def time_filter(frames, t0=None, t1=None):
    """Keyframe entries whose key time lies in [t0, t1] (either bound may be None)."""
    def key(e):
        return e.get("t_key", e["t"])
    return [e for e in frames if (t0 is None or key(e) >= t0) and (t1 is None or key(e) <= t1)]


# ----------------------------------------------------------------------------------------------
# Rig, trajectory and virtual pinhole views


def load_rig():
    """dict name -> common.rig.Camera (native resolution)."""
    return crig.load_rig(RS / "calib" / "rig.json")


def rig_available():
    return (RS / "calib" / "rig.json").exists()


def trajectory(path=None):
    """common.rig.Trajectory; poses/ego.npz if it exists, else planar telemetry."""
    p = Path(path) if path else RS / "poses" / "ego.npz"
    return crig.Trajectory(p, timeline=RS / "data" / "drive" / "timeline.npz")


def traj_name(traj):
    return str(Path(str(traj.source)).relative_to(RS)) if str(traj.source).startswith(str(RS)) else str(traj.source)


def T_world_cam(cam, n, traj, v=None):
    """Camera-to-world for frame n of cam at native image row v (default: middle row)."""
    return crig.T_world_cam(cam, frame_time(cam.name, n), v, traj)


def cam_K(cam, size=None):
    p = cam.params
    K = np.array([[p[0], 0, p[2]], [0, p[1], p[3]], [0, 0, 1.0]])
    if size is not None:
        K[0] *= size[0] / cam.width
        K[1] *= size[1] / cam.height
    return K


def level_rotation(yaw_deg, pitch_deg):
    """Ego-from-virtual rotation of a level (roll-free) camera looking at yaw/pitch in the ego frame."""
    y, p = np.deg2rad(yaw_deg), np.deg2rad(pitch_deg)
    fwd = np.array([np.cos(y) * np.cos(p), np.sin(y) * np.cos(p), np.sin(p)])
    right = np.cross(fwd, [0, 0, 1.0])
    right /= np.linalg.norm(right)
    down = np.cross(fwd, right)
    return np.stack([right, down, fwd], 1)


class VirtualView:
    """A rectified pinhole view of a physical camera.

    K is in COLMAP pixel centres for the virtual image of size (w, h); R maps virtual camera vectors
    into the physical camera (as in common.rig.undistort_maps)."""

    def __init__(self, cam, size, K, R=None, name=None):
        import cv2
        self.cam, self.size, self.K = cam, tuple(size), np.asarray(K, np.float64)
        self.R = np.eye(3) if R is None else np.asarray(R, np.float64)
        self.name = name or cam.name
        mx, my, _ = crig.undistort_maps(cam, self.K, self.size, self.R)
        self.valid = mx >= 0
        # pre-shrink factor so the remap samples a source about 1-1.4x the virtual resolution
        f_src = cam.params[0]
        self.pre = float(min(1.0, 1.4 * self.K[0, 0] / f_src))
        s = self.pre
        self.mx = np.where(self.valid, (mx + 0.5) * s - 0.5, -1).astype(np.float32)
        self.my = np.where(self.valid, (my + 0.5) * s - 0.5, -1).astype(np.float32)
        self.pre_size = (round(cam.width * s), round(cam.height * s))
        w, h = self.size
        jj, ii = np.meshgrid(np.arange(w) + 0.5, np.arange(h) + 0.5)
        rays = np.stack([(jj - self.K[0, 2]) / self.K[0, 0], (ii - self.K[1, 2]) / self.K[1, 1],
                         np.ones_like(jj)], -1)
        self.ray_norm = np.linalg.norm(rays, axis=-1).astype(np.float32)  # range = z * ray_norm
        self._back = {}
        self._cv2 = cv2

    def rectify(self, img_native):
        cv2 = self._cv2
        src = img_native
        if self.pre < 1.0:
            src = cv2.resize(img_native, self.pre_size, interpolation=cv2.INTER_AREA)
        out = cv2.remap(src, self.mx, self.my, cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
        out[~self.valid] = 0
        return out

    def back_maps(self, out_size):
        """For the original camera grid of out_size: sampling coords into this view, cosine to the
        view axis, ray z in the physical camera, and validity (calib mask, in view, in model range)."""
        if out_size in self._back:
            return self._back[out_size]
        cam = self.cam
        W, H = out_size
        sx, sy = cam.width / W, cam.height / H
        jj, ii = np.meshgrid((np.arange(W) + 0.5) * sx, (np.arange(H) + 0.5) * sy)
        uv = np.stack([jj, ii], -1)
        r = cam.unproject(uv)
        ok = np.all(np.isfinite(r), -1) & cam.mask_at(uv)
        r = np.nan_to_num(r)
        rv = r @ self.R  # physical -> virtual: R^T r, written as row-vector product
        z = rv[..., 2]
        zz = np.where(z > 1e-6, z, 1e-6)
        u = self.K[0, 0] * rv[..., 0] / zz + self.K[0, 2]
        v = self.K[1, 1] * rv[..., 1] / zz + self.K[1, 2]
        w, h = self.size
        inside = (z > 1e-6) & (u >= 0) & (u < w) & (v >= 0) & (v < h)
        cos_axis = np.where(inside, z / np.linalg.norm(rv, axis=-1).clip(1e-9), -1)
        m = dict(u=(u - 0.5).astype(np.float32), v=(v - 0.5).astype(np.float32), inside=inside & ok,
                 cos_axis=cos_axis, ray_z=r[..., 2].astype(np.float32), mask=ok)
        self._back[out_size] = m
        return m


def views_to_original(views, ranges, out_size, min_ray_z=np.cos(np.deg2rad(88)), interp="nearest"):
    """Merge per-view range maps (range along the ray, metres) back onto the original camera grid.
    Each output pixel takes the view whose axis is closest to its ray. Returns (z_depth, range)."""
    import cv2
    W, H = out_size
    best = np.full((H, W), -2.0)
    rng = np.zeros((H, W), np.float32)
    ray_z = None
    flag = cv2.INTER_NEAREST if interp == "nearest" else cv2.INTER_LINEAR
    for vw, r in zip(views, ranges):
        m = vw.back_maps(out_size)
        ray_z = m["ray_z"]
        s = cv2.remap(r.astype(np.float32), m["u"], m["v"], flag, borderMode=cv2.BORDER_CONSTANT,
                      borderValue=0)
        take = m["inside"] & (m["cos_axis"] > best) & (s > 0)
        rng[take] = s[take]
        best[take] = m["cos_axis"][take]
    valid = (rng > 0) & (ray_z > min_ray_z)
    z = np.where(valid, rng * ray_z, 0).astype(np.float32)
    return z, np.where(valid, rng, 0).astype(np.float32)


def default_views(cam, work_w):
    """Virtual pinhole views used for a camera at model input width work_w (multiple of 14).

    Narrow and wide lenses get one view with the lens's own principal point and a focal length that
    keeps the horizontal extent of the calibration mask; the 180-degree rear fisheye gets three views
    yawed -60/0/+60 degrees about the camera's vertical axis."""
    W, H = cam.width, cam.height
    if cam.name == "back":
        # level views (no roll, 20 degrees down) yawed about the ego vertical around the lens's heading
        w = work_w
        h = int(round(w * 0.75 / 14) * 14)
        f = (w / 2) / np.tan(np.deg2rad(45))
        K = np.array([[f, 0, w / 2], [0, f, h / 2], [0, 0, 1.0]])
        R_ec = cam.T_ego_cam[:3, :3]
        axis = R_ec[:, 2]
        yaw0 = np.degrees(np.arctan2(axis[1], axis[0]))
        out = []
        for dy, n in [(55, "l"), (0, "c"), (-55, "r")]:
            out.append(VirtualView(cam, (w, h), K, R_ec.T @ level_rotation(yaw0 + dy, -20.0), f"back_{n}"))
        return out
    h = int(round(work_w * H / W / 14) * 14)
    w = work_w
    # widest horizontal half-angle inside the calibration mask on the principal row
    row = cam.mask[int(cam.params[3])]
    xs = np.nonzero(row)[0]
    p = cam.params
    rays = cam.unproject(np.array([[xs.min() + 0.5, p[3]], [xs.max() + 0.5, p[3]]]))
    half = np.arctan2(np.abs(rays[:, 0]), rays[:, 2])
    s = w / W
    cx, cy = p[2] * s, p[3] * h / H
    f = min(cx / np.tan(half[0]), (w - cx) / np.tan(half[1]))
    K = np.array([[f, 0, cx], [0, f, cy], [0, 0, 1.0]])
    return [VirtualView(cam, (w, h), K, None, cam.name)]


# ----------------------------------------------------------------------------------------------
# Depth utilities


def inv(T):
    Ti = np.eye(4)
    Ti[:3, :3] = T[:3, :3].T
    Ti[:3, 3] = -T[:3, :3].T @ T[:3, 3]
    return Ti


def resample_depth(depth, K_src, K_dst, size_dst, mode="nearest"):
    """Resample a pinhole depth map between two intrinsics of the same camera (crop/scale only)."""
    import cv2
    w, h = size_dst
    jj, ii = np.meshgrid(np.arange(w) + 0.5, np.arange(h) + 0.5)
    us = (jj - K_dst[0, 2]) / K_dst[0, 0] * K_src[0, 0] + K_src[0, 2] - 0.5
    vs = (ii - K_dst[1, 2]) / K_dst[1, 1] * K_src[1, 1] + K_src[1, 2] - 0.5
    interp = cv2.INTER_NEAREST if mode == "nearest" else cv2.INTER_LINEAR
    return cv2.remap(depth.astype(np.float32), us.astype(np.float32), vs.astype(np.float32), interp,
                     borderMode=cv2.BORDER_CONSTANT, borderValue=0)


def edge_mask(depth, rel=0.06):
    """Depth discontinuities and the flying pixels around them.

    A pixel is flagged when log-depth varies by more than rel within its 3x3 window AND the second
    difference (curvature) near it is large, so steep but smooth surfaces (road at grazing angles)
    are kept while occlusion boundaries are removed."""
    import cv2
    d = depth.astype(np.float32)
    ld = np.log(np.where(d > 0, d, 1e-3).clip(1e-3))
    k = np.ones((3, 3), np.uint8)
    jump = (cv2.dilate(ld, k) - cv2.erode(ld, k)) > rel
    d2 = np.zeros_like(ld)
    d2[:, 1:-1] = np.abs(ld[:, 2:] - 2 * ld[:, 1:-1] + ld[:, :-2])
    d2[1:-1, :] = np.maximum(d2[1:-1, :], np.abs(ld[2:, :] - 2 * ld[1:-1, :] + ld[:-2, :]))
    curv = cv2.dilate((d2 > 0.5 * rel).astype(np.uint8), k) > 0
    return jump & curv


def colorize_depth(depth, vmin=1.0, vmax=80.0):
    """Inverse-depth turbo colouring, black where invalid. Returns RGB uint8."""
    import cv2
    d = depth.astype(np.float32)
    inv_d = np.where(d > 0, 1.0 / np.maximum(d, 1e-3), 0)
    x = np.clip((inv_d - 1 / vmax) / (1 / vmin - 1 / vmax), 0, 1)
    c = cv2.applyColorMap((x * 255).astype(np.uint8), cv2.COLORMAP_TURBO)[..., ::-1]
    c[d <= 0] = 0
    return np.ascontiguousarray(c)
