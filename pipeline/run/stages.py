"""Drive-local adapters. Executed from a private RS-shaped workspace."""

import json
import os
import shlex
import shutil
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

from raw import ingest, inventory, write

HOME = Path(os.environ["RS_INSTALL"])
ROOT = Path.cwd()
PY = sys.executable


def read(p):
    return json.loads(Path(p).read_text())


def command(*args, **kwargs):
    args = list(map(str, args))
    print("$", shlex.join(args), flush=True)
    subprocess.run(args, check=True, **kwargs)


def mod(name, *args, **kwargs):
    command(PY, "-u", "-m", "pipeline." + name, *args, **kwargs)


def cache(scene):
    command("bun", HOME / "viewer-cache-tool/tools/build-tile-cache.ts", scene)


def binlink(src, dst):
    dst = Path(dst)
    dst.parent.mkdir(parents=True, exist_ok=True)
    if not src.is_file():
        raise FileNotFoundError(src)
    if not dst.exists():
        dst.symlink_to(src)


def main():
    stage = sys.argv[1]
    if stage == "inventory":
        inventory(sys.argv[2], json.loads(sys.argv[3]), HOME / "calib/rig.json")
    elif stage == "ingest":
        ingest()
    else:
        reconstruct(stage)


def keyframes(production=False):
    import numpy as np

    field = read("inventory.json")
    tl = np.load("data/drive/timeline.npz")
    first = float(tl["t"][0])
    for a, b in field["moving"]:
        a = max(a, first + 0.65)  # two released causal knots before first object packet
        if a >= b:
            continue
        command(
            PY,
            "pipeline/depth/keyframes.py",
            "--t0",
            a,
            "--t1",
            b,
            "--step",
            0.5 if production else 1.75,
            "--stopped-hz",
            0,
        )
    from pipeline.common.hardware import roster

    doc = read("depth/keyframes.json")
    if any(not rows for rows in doc["frames"].values()) or set(doc["frames"]) != set(roster()):
        raise ValueError("missing keyframe support for one or more cameras")
    if not production:
        write("work/run/preview-keys.json", doc)


def masks():
    command(PY, "pipeline/masks/run_masks.py", "--cams", "all")
    for c, rows in read("depth/keyframes.json")["frames"].items():
        for e in rows:
            for kind, ext in [
                ("road", "png"),
                ("sky", "png"),
                ("dyn", "png"),
                ("dyn", "json"),
            ]:
                p = Path(f"masks/{kind}/{c}/{e['n']:06d}.{ext}")
                if not p.is_file() or not p.stat().st_size:
                    raise FileNotFoundError(p)


def tracking():
    # Use the production classifier binary, never the installation's classifications.
    exe = HOME / "work/movers/photo-full-native/semantics/classify"
    if exe.is_file():
        out = Path("work/movers/photo-full-native/semantics/classify")
        out.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(exe, out)
    mod("movers.packets", "--all")
    for k in ["handoff", "footprint", "motion"]:
        mod("movers." + k)
    # The Vision CLI emits one [] for a blank line. The upstream semantics
    # producer always adds a newline, even when there are zero confirmed IDs.
    # Represent genuine absence of candidates as an empty score table instead.
    import pickle

    path = Path("work/movers/photo-full-native")
    with (path / "identities.pickle").open("rb") as f:
        candidates = pickle.load(f)
    if any(h["confirm"] is not None for h in candidates):
        mod("movers.semantics")
    else:
        with (path / "semantics.pickle").open("wb") as f:
            pickle.dump({}, f)
    mod("movers.filter")


def road():
    import numpy as np
    from pipeline.common.hardware import roster
    from pipeline.common.rig import Trajectory, load_rig
    from pipeline.road import pairs
    from pipeline.road.geometry import ground_to_pixels, pixels_to_ground

    inv = read("inventory.json")
    pairs.SEG = [(str(i), a, b) for i, (a, b) in enumerate(inv["moving"])]
    pairs.build()
    binlink(HOME / "work/road/build/trajectory_ba", "work/road/build/trajectory_ba")
    mod("road.fit")
    mod("road.poses")
    tl = np.load("data/drive/timeline.npz")
    lag = np.load("work/road/causal-lag.npy")
    if not np.isfinite(lag).all() or lag.min() < 0 or lag.max() > 0.5:
        raise ValueError("invalid road pose release lag")
    sc = np.load("work/poses/schedule.npz")
    m = np.load("work/road/pairs.npy", mmap_mode="r")
    for stem in ("ego", "ego_causal"):
        p = Path("work/road") / (stem + ".npz")
        tr = Trajectory(p)
        R = tr.poses[:, :3, :3]
        if (
            not np.array_equal(tr.t, tl["t"])
            or not np.isfinite(tr.poses).all()
            or np.max(abs(np.linalg.det(R) - 1)) > 1e-8
        ):
            raise ValueError("invalid SE(3) road poses: " + stem)
        per = {}
        for i, c in enumerate(roster()):
            cam = load_rig()[c]
            vals = m[(m[:, 1] == i) & (m[:, 10] == 1)]
            es = []
            for row in vals[:: max(1, len(vals) // 100)][:100]:
                t = sc["tc"][int(row[0]), i]
                uv = row[2:4].astype(float)
                G, ok = pixels_to_ground(cam, tr, t, uv)
                vv, good, _ = ground_to_pixels(cam, tr, t, G[:2])
                if ok and good:
                    es.append(np.linalg.norm(vv - uv))
            if not es or max(es) >= 0.15:
                raise ValueError(
                    f"{stem} {c}: insufficient or inconsistent fresh road transfer, max {max(es, default=None)} px"
                )
            per[c] = float(max(es))
        shutil.copy2(p, Path("poses") / (stem + ".npz"))
        print(stem, per, flush=True)
    mod("road.field")


def bake_movers():
    # source_tiles only needs instance runs for splats sourced from dynamic pixels.
    # The preview contains exclusively road discs, no tagged object-layer splats.
    src = Path("scene/live-v5")
    shutil.copytree("scene/road-live-v2", src, dirs_exist_ok=True)
    cache(src)
    write(
        "maps/live-v5/slice-instances.json",
        {
            "frames": [],
            "reason": "ground-only preview; no instance-tagged static object splats",
        },
    )
    mod("movers.online_scene", "publish")
    mod("movers.skin")
    dst = Path("scene/live-v5-movers-v4")
    shutil.copytree(dst / "objects", "work/run/objects", dirs_exist_ok=True)
    shutil.copy2(dst / "objects.json", "work/run/objects.json")


def publish(src, objs, atlas, tier):
    from drive import sig
    from transactions import content_signature

    # Immutable private release, local cache build, then one atomic host pointer swap.
    name = os.environ["RUN_NAME"]
    base = Path(os.environ["RUN_ROOT"])
    parent = Path(os.environ["RUN_RELEASES"])
    parent.mkdir(exist_ok=True)
    out = parent / f"{tier}-{time.time_ns()}"
    shutil.copytree(
        src,
        out,
        ignore=shutil.ignore_patterns("rsb", "ground", "objects", "objects.json"),
    )
    shutil.copytree(atlas, out / "ground")
    shutil.copytree(objs / "objects", out / "objects")
    shutil.copy2(objs / "objects.json", out / "objects.json")
    cache(out)
    mp = read(out / "map.json")
    gr = read(out / "ground/ground.json")
    ix = read(out / "rsb/index.json")
    if not mp["tiles"] or not gr["tiles"] or not ix["ground"] or not ix["objects"]:
        raise ValueError("incomplete cache or empty scene")
    for t in mp["tiles"]:
        v = ix["tiles"][t["id"]]
        if (
            v["count"] != t["count"]
            or (out / "rsb" / v["file"]).stat().st_size != v["bytes"]
        ):
            raise ValueError("stale cached tile " + t["id"])
    for t in gr["tiles"]:
        if (out / "ground" / t["texture"]).stat().st_size != t["bytes"]:
            raise ValueError("stale atlas texture")
    for o in read(out / "objects.json")["objects"]:
        for k in ["file", "predicted_file"]:
            if not (out / o[k]).is_file():
                raise FileNotFoundError(out / o[k])
    dest = HOME / "scene" / name
    temp = dest.with_name("." + name + "-next")
    if (dest.exists() or dest.is_symlink()) and (
        not dest.is_symlink() or dest.resolve().parent != parent
    ):
        raise ValueError(f"refusing to replace unrelated viewer scene {dest}")
    previous = os.readlink(dest) if dest.is_symlink() else None
    temp.unlink(missing_ok=True)
    temp.symlink_to(out)
    temp.replace(dest)
    host = os.getenv("RS_VIEWER_URL", "http://127.0.0.1:8090").rstrip("/")
    t = read(base / "inventory.json")["moving"][0][0] + 0.7
    url = f"{host}/?scene={name}&mode=stitched&t={t:.2f}"
    try:

        def get(path):
            with urllib.request.urlopen(host + path, timeout=15) as h:
                return h.read()

        if not any(
            x["name"] == name and x["hasMap"]
            for x in json.loads(get("/scenes/index.json"))["scenes"]
        ):
            raise ValueError("scene not listed by viewer")
        if b"<html" not in get("/").lower():
            raise ValueError("viewer not serving HTML")
        for f in [
            "drive.json",
            "map.json",
            "rsb/index.json",
            "objects.json",
            "ground/ground.json",
            "ground/" + gr["tiles"][0]["texture"],
            "rsb/" + ix["tiles"][mp["tiles"][0]["id"]]["file"],
        ]:
            if get("/scenes/" + name + "/" + f) != (out / f).read_bytes():
                raise ValueError("served bytes differ: " + f)
    except BaseException:
        if previous:
            temp.symlink_to(previous)
            temp.replace(dest)
        else:
            dest.unlink()
        raise
    write(
        base
        / "work/run"
        / ("published.json" if tier == "preview" else "published-full.json"),
        dict(
            url=url,
            name=name,
            directory=str(out),
            signature=sig(out),
            content_sha256=content_signature(out),
            tier=tier,
        ),
    )


def reconstruct(stage):
    if stage == "calibration":
        from calibration import run

        run()
    elif stage == "poses":
        binlink(
            HOME / "work/poses/build/trajectory_ba", "work/poses/build/trajectory_ba"
        )
        for s in ("tracks", "wide", "causal", "offline"):
            mod("poses." + s)
        for s in ("ego", "ego_causal"):
            shutil.copy2(f"poses/{s}.npz", f"poses/{s}_v1.npz")
    elif stage == "road":
        road()
    elif stage == "keyframes":
        keyframes()
    elif stage == "masks":
        masks()
    elif stage == "photo":
        import numpy as np
        from pipeline.photo import fit, live_v2, sample

        sample.collect()
        sc = np.load("work/photo/schedule.npz")
        if np.any(sc["still"][1:] & sc["still"][:-1]):
            sample.collect(True)
        fit.run("stitched")
        live_v2.run()
        live_v2.tone()
    elif stage == "ground":
        p = Path("maps/ground-v2/ground.dylib")
        p.parent.mkdir(exist_ok=True, parents=True)
        command(
            "c++",
            "-std=c++17",
            "-O3",
            *(["-shared", "-fPIC"] if sys.platform.startswith("linux") else ["-dynamiclib"]),
            "pipeline/road/ground_native.cpp",
            "-o",
            p,
        )
        mod("road.scene", "live", "stitched")
    elif stage == "atlas":
        inv = read("inventory.json")
        env = dict(
            os.environ,
            STITCH_START=str(inv["window"][0]),
            STITCH_END=str(inv["window"][1]),
        )
        command(
            PY, "-c", "from pipeline.stitch.v2data import schedule; schedule()", env=env
        )
        for a, b in inv["moving"]:
            mod("groundtex.atlas", a, b, "--workers", 3)
        # The upstream verifier checks the real bytes, mesh and alpha before
        # serializing statistics. A lens with zero registered observations has
        # no median. Represent only those undefined report fields as JSON null.
        import math
        from pipeline.groundtex import verify as atlas_verify

        output = atlas_verify.dump

        def report(path, data):
            for stats in data["sources"].values():
                if stats["registered"] == 0:
                    for name in (
                        "before_cm",
                        "after_cm",
                        "median_initial_ncc",
                        "median_aligned_ncc",
                    ):
                        if name in stats and math.isnan(stats[name]):
                            stats[name] = None
            output(path, data)

        atlas_verify.dump = report
        try:
            atlas_verify.verify()
        finally:
            atlas_verify.dump = output
        v = read("work/run/atlas-report/verified.json")
        if not v["tiles"] or v["surface_square_metres"]["opaque"] <= 0:
            raise ValueError("atlas has no observed opaque ground")
    elif stage == "movers":
        tracking()
        bake_movers()
    elif stage == "publish":
        publish(
            Path("scene/road-live-v2"),
            Path("work/run"),
            Path("work/run/atlas"),
            "preview",
        )
    else:
        from full import run

        run(stage)


if __name__ == "__main__":
    main()
