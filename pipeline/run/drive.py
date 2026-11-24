#!/usr/bin/env python3
"""Run every stage on one drive: a folder of Tesla dashcam clips to a published viewer scene.

The installation root (environments, models, calibration, work and scene folders) is RS_INSTALL, by default
this checkout. RS_VIEWER_URL is the viewer host that serves <root>/scene (default http://127.0.0.1:8090; on
Linux a private loopback host is started when it is unset)."""

import argparse
import contextlib
import datetime
import hashlib
import json
import os
import re
import shlex
import signal
import subprocess
import sys
import time
from pathlib import Path
from zipfile import BadZipFile

from transactions import (
    atomic_json as dump,
)
from transactions import (
    begin,
    content_signature,
    identity,
    initialize,
    lock,
    promote,
)

HERE = Path(__file__).resolve().parent
HOME = Path(os.getenv("RS_INSTALL", HERE.parents[1])).resolve()
PREVIEW = [
    "inventory",
    "ingest",
    "calibration",
    "poses",
    "road",
    "keyframes",
    "masks",
    "photo",
    "ground",
    "atlas",
    "movers",
    "publish",
]
FULL = [
    "production",
    "removal",
    "teacher",
    "optimize",
    "attach",
    "clip",
    "publish-full",
]
ART = {
    "inventory": ["inventory.json"],
    "ingest": [
        "data/drive/meta.json",
        "data/drive/timeline.npz",
        "data/drive/cams",
        "data/drive/frames",
    ],
    "calibration": ["calib/rig.json", "calib/masks", "work/calib/verification.json"],
    "poses": [
        "poses/ego_v1.npz",
        "poses/ego_causal_v1.npz",
        "work/poses/offline-fit.npz",
        "work/poses/schedule.npz",
        "work/poses/hold.npz",
    ],
    "road": [
        "poses/ego.npz",
        "poses/ego_causal.npz",
        "work/road/causal-lag.npy",
        "maps/ground-work/live-height.npy",
        "maps/ground-work/stitched-height.npy",
    ],
    "keyframes": ["work/run/preview-keys.json", "depth/keyframes.json"],
    "masks": ["masks"],
    "photo": [
        "photo/stitched.npz",
        "photo/live-v2.npz",
        "photo/ground-v2-tone.npz",
        "work/photo/schedule.npz",
    ],
    "ground": [
        "scene/road-live-v2",
        "scene/road-stitched-v2",
        "photo/stitched-ground-v2.npz",
    ],
    "atlas": ["work/run/atlas", "work/run/atlas-report/verified.json"],
    "movers": ["work/run/objects", "work/run/objects.json"],
    "publish": ["work/run/published.json"],
    "production": ["full/depth", "full/masks"],
    "removal": ["full/maps/movers-v1/remove-index.json", "full/maps/movers-v1/remove"],
    "teacher": [
        "full/work/run/teacher.json",
        "full/work/run/depth-measurements.json",
        "full/work/run/static-masks",
        "full/scene/live-v4/sky.jpg",
    ],
    "optimize": [
        "full/scene/optimized/map.json",
        "full/scene/optimized/tiles",
        "full/scene/optimized/drive.json",
        "full/scene/optimized/rig.json",
        "full/scene/optimized/sky.jpg",
    ],
    "attach": [
        "full/scene/optimized/objects.json",
        "full/scene/optimized/objects",
        "full/scene/optimized/ground",
    ],
    "clip": ["full/work/run/clip.json"],
    "publish-full": ["work/run/published-full.json"],
}


def read(p):
    return json.loads(Path(p).read_text())


def stamp():
    return datetime.datetime.now(datetime.timezone.utc).isoformat(
        timespec="milliseconds"
    )


def digest(x):
    return hashlib.sha256(json.dumps(x, sort_keys=True).encode()).hexdigest()


def sig(p):
    if not p.exists():
        raise FileNotFoundError(p)
    files = sorted(q for q in p.rglob("*") if q.is_file()) if p.is_dir() else [p]
    h = hashlib.sha256()
    size = 0
    for q in files:
        st = q.stat()
        size += st.st_size
        if not st.st_size:
            raise ValueError(f"empty output: {q}")
        h.update(
            str(
                (str(q.relative_to(p)) if q != p else ".", st.st_size, st.st_mtime_ns)
            ).encode()
        )
        if q.suffix == ".json" and st.st_size < 2000000:
            h.update(q.read_bytes())
    return dict(files=len(files), bytes=size, sha256=h.hexdigest())


def outputs(root, k):
    paths = list(ART[k])
    if k == "teacher":
        for entry in read(root / paths[0]):
            p = Path("full") / entry["path"]
            paths.extend(
                [
                    str(p.parent),
                    str(p.parent.parent / "ground-selection.json"),
                    str(p.parent.parent / "v2/exposures.json"),
                ]
            )
    if k == "clip" and read(root / paths[0]).get("clips"):
        paths += [
            "full/scene/optimized/generated.json",
            "full/scene/optimized/generated",
        ]
    return {x: sig(root / x) for x in paths}


# Version of the driver-to-producer invocation contract. Its initial value
# preserves receipts created before controller-only changes had their own hash.
# Bump for changed producer environment/arguments; the actual controller digest
# is recorded independently as runner_version on every invocation.
DRIVER_CONTRACT = "355118065116748adc05735f8e3696ed9848392970a6d4448f9a4842c4fc6c82"


def sources(stage, legacy=None):
    names = {
        "inventory": ["ingest"],
        "ingest": ["ingest"],
        "calibration": ["common", "calib"],
        "poses": ["common", "poses"],
        "road": ["road", "poses"],
        "keyframes": ["depth"],
        "masks": ["masks", "depth"],
        "photo": ["photo"],
        "ground": ["road", "photo", "scene"],
        "atlas": ["groundtex"],
        "movers": ["movers"],
        "production": ["depth", "masks"],
        "removal": ["movers"],
        "teacher": ["ibr", "stitch", "livemap"],
        "optimize": ["stitch"],
        "attach": ["movers"],
        "clip": ["genfill", "groundtex"],
        "publish-full": ["scene"],
    }.get(stage, [])
    p = HOME / "pipeline"
    files = [
        HERE / ("raw.py" if stage in ("ingest", "inventory") else "stages.py"),
    ]
    if legacy or stage not in ("inventory", "ingest"):
        files.append(HERE / "drive.py")
    if stage == "calibration":
        files.append(HERE / "calibration.py")
    if stage in FULL:
        files.append(HERE / "full.py")
    if stage in ("teacher", "clip"):
        files.append(HOME / "tools/run-chase/chase-track.js")
    for name in names:
        files.extend(
            x for x in (p / name).glob("*") if x.suffix in (".py", ".cpp", ".swift")
        )
    if stage.startswith("publish"):
        files.extend((HOME / "viewer-cache-tool").rglob("*.ts"))
    return digest(
        [
            (
                str(x),
                (legacy or DRIVER_CONTRACT)
                if x == HERE / "drive.py"
                else hashlib.sha256(x.read_bytes()).hexdigest(),
            )
            for x in sorted(files)
            if x.is_file()
        ]
    )


def heavy(group=None):
    text = subprocess.check_output(
        ["ps", "-ww", "-Ao", "pid=,pcpu=,command="], text=True
    )
    found = []
    for line in text.splitlines():
        cols = line.strip().split(None, 2)
        if len(cols) < 3:
            continue
        pid, pct, cmd = cols
        exe = cmd.split(None, 1)[0]
        if int(pid) == os.getpid() or "python" not in exe.lower():
            continue
        if group is not None:
            try:
                if os.getpgid(int(pid)) == group:
                    continue
            except ProcessLookupError:
                continue
        if float(pct) >= 85 or re.search(
            r"(pipeline[/.](?:masks|depth|stitch|ibr|groundtex|genfill|genvideo)[/.]|run_depth.py|run_masks.py)",
            cmd,
        ):
            found.append(dict(pid=int(pid), command=cmd[:220]))
    return found


def wait_worker(child, priority=False):
    if not priority:
        return child.wait()
    paused = False
    while child.poll() is None:
        # Repair jobs may arrive after our stage started. Yield only our own
        # process group; never signal another engineer's processes.
        busy = any(
            "genfill" in j["command"] or "genvideo" in j["command"]
            for j in heavy(child.pid)
        )
        if busy != paused:
            try:
                os.killpg(child.pid, signal.SIGSTOP if busy else signal.SIGCONT)
            except ProcessLookupError:
                break
            paused = busy
            print(
                "yielding GPU to clip repair" if busy else "resuming GPU worker",
                flush=True,
            )
        try:
            return child.wait(timeout=3)
        except subprocess.TimeoutExpired:
            pass
    return child.wait()


def publication_ok(root, stage):
    if not stage.startswith("publish"):
        return True
    r = read(root / ART[stage][0])
    dest = HOME / "scene" / r["name"]
    release = Path(r["directory"])
    if not dest.is_symlink() or content_signature(release) != r.get("content_sha256"):
        return False
    if dest.resolve() == release:
        return True
    # A later, verified full release legitimately supersedes the preview pointer.
    full = root / "work/run/published-full.json"
    progressive = release.parent.parent / "progress-public/work/run/published-full.json"
    if progressive.is_file():
        full = progressive
    return (
        stage == "publish"
        and full.is_file()
        and dest.resolve() == Path(read(full)["directory"])
        and content_signature(dest.resolve()) == read(full).get("content_sha256")
    )


# Stage-private scratch contains producers' own existence-only caches. Retire it
# together with outputs before replay; the previous immutable public release survives.
PRIVATE = {
    "inventory": ["inventory.json"],
    "ingest": ["data"],
    "calibration": ["calib", "work/calib"],
    "poses": ["work/poses", "poses"],
    "road": ["work/road", "maps/ground-work", "poses/ego.npz", "poses/ego_causal.npz"],
    "keyframes": ["depth", "work/depth", "work/run/preview-keys.json"],
    "masks": ["masks"],
    "photo": ["work/photo", "photo"],
    "ground": ["scene/road-live-v2", "scene/road-stitched-v2", "maps/ground-v2"],
    "atlas": [
        "work/run/atlas",
        "work/run/atlas-state",
        "work/run/atlas-report",
        "work/stitch-v2",
    ],
    "movers": [
        "work/movers",
        "scene/live-v5",
        "scene/live-v5-movers-v4",
        "maps/live-v5",
        "work/run/objects",
        "work/run/objects.json",
    ],
    "production": ["full"],
    "removal": ["full/work/movers", "full/maps/movers-v1"],
    "teacher": [
        "full/work/run",
        "full/work/ibr",
        "full/maps/object-v3",
        "full/scene/live-v4",
    ],
    "optimize": ["full/scene/optimized"],
    "attach": [
        "full/scene/live-v5",
        "full/scene/live-v5-movers-v4",
        "full/work/run/objects",
        "full/work/run/objects.json",
        "full/scene/optimized/objects",
        "full/scene/optimized/objects.json",
        "full/scene/optimized/ground",
    ],
    "clip": [
        "full/work/run/chase",
        "full/scene/optimized/generated",
        "full/scene/optimized/generated.json",
    ],
}


def exclusions(root, k):
    paths = list(PRIVATE.get(k, []))
    if k == "optimize":
        for chunk in (root / "full/work/run").glob("chunk-*"):
            paths.extend(
                str((chunk / d).relative_to(root))
                for d in ("fit", "v4", "v2/dense-init", "v2/targets", "v2/train.json")
            )
    return paths


def validate(root, k):
    signs = outputs(root, k)
    if k == "inventory":
        inv = read(root / "inventory.json")
        # Every clip group carries the same cameras (six on HW4, four on HW3), front among them.
        if (
            not inv["clips"]
            or not inv["windows"]
            or "front" not in inv["counts"]
            or any(set(g["paths"]) != set(inv["counts"]) for g in inv["clips"])
        ):
            raise ValueError("invalid inventory")
    if k == "ingest":
        import numpy as np
        from PIL import Image

        d = root / "data/drive"
        inv = read(root / "inventory.json")
        meta = read(d / "meta.json")
        with np.load(d / "timeline.npz") as tl:
            t = tl["t"]
            if (
                len(t) < 2
                or not np.all(np.diff(t) > 0)
                or any(len(tl[x]) != len(t) for x in ("speed", "x", "y", "yaw"))
            ):
                raise ValueError("invalid master timeline")
        if set(meta["cams"]) != set(inv["counts"]):
            raise ValueError("camera roster mismatch")
        for c in meta["cams"]:
            rows = read(d / "cams" / f"{c}.json")
            ns = [r["n"] for r in rows]
            frames = d / "frames" / c
            if (
                len(ns) < 2
                or not np.all(np.diff(ns) > 0)
                or not np.all(np.diff([r["t"] for r in rows]) > 0)
            ):
                raise ValueError("invalid camera index: " + c)
            for n in ns:
                file = frames / f"{n:06d}.jpg"
                with Image.open(file) as im, file.open("rb") as f:
                    f.seek(-2, 2)
                    if (
                        im.format != "JPEG"
                        or list(im.size) != meta["sizes"][c]
                        or f.read() != b"\xff\xd9"
                    ):
                        raise ValueError("invalid JPEG: " + str(file))
    return signs, {x: content_signature(root / x) for x in signs}


# On-disk v1 receipts from the deployed pre-reboot runner. Only adopt the two
# raw stages, and only after reconstructing the exact OLD key, validating the
# real indexes/JPEGs and recording content hashes. No blanket cache exemption.
LEGACY_RAW_RUNNER = "01e4dbf63c927b1dc6e2d51356ee542f4b8e2c9d954ebb0b017b3cc565f88285"


def run(a):
    if not re.fullmatch(r"[a-zA-Z0-9][a-zA-Z0-9_-]{0,63}", a.name):
        raise ValueError("invalid drive name")
    base = HOME / "work/drives" / a.name
    p = base / "manifest.json"
    if a.action == "status":
        m = read(p)
        print(f"{a.name}: {m['workspace']}")
        for k, v in m["stages"].items():
            print(
                k,
                v["status"],
                f"{v.get('duration_s', 0):.1f}s",
                v.get("detail", ""),
                v.get("log", ""),
            )
        if m.get("progress"):
            print("progress", json.dumps(m["progress"]))
            for label, unit in m.get("units", {}).items():
                if unit["status"] != "done":
                    print(
                        label,
                        unit["status"],
                        unit.get("detail", ""),
                        unit.get("log", ""),
                    )
        if m.get("viewer"):
            print(m["viewer"])
        return
    src = Path(a.footage).expanduser().resolve(strict=True)
    params = dict(footage=str(src), window=a.window, name=a.name)
    raw = digest(
        [
            (str(q.relative_to(src)), q.stat().st_size, q.stat().st_mtime_ns)
            for q in sorted(src.rglob("*.mp4"))
        ]
    )
    clock_path = src / "window-clock.json"
    if clock_path.is_file():
        raw = digest([raw, content_signature(clock_path)])
    base.mkdir(parents=True, exist_ok=True)
    from clips import selection

    clip_select = selection(a.clip)
    stages = PREVIEW + (FULL if a.quality == "full" else [])
    py = HOME / ".venv/bin/python"
    if not py.exists():
        raise ValueError(f"Studio installation not found: {py}")
    for k in [a.start, a.only]:
        if k and k not in stages:
            raise ValueError(f"{k} not in {a.quality}: {', '.join(stages)}")
    who = identity()
    with lock(base / "runner.lock", False, who, drive=a.name) as runner_lease:
        m = read(p) if p.exists() else {}
        fingerprint = digest([params, raw])
        old = m.get("fingerprint")
        if old and old != fingerprint:
            # Separate generations; a previous publication stays live until replacement is verified.
            dump(base / ("manifest-" + old[:12] + ".json"), m)
            m = {}
        root = base / fingerprint[:16]
        initialize(root)
        m.update(
            workspace=str(root),
            fingerprint=fingerprint,
            parameters=params,
            quality=a.quality,
            rig_mode=a.rig,
            owner=who,
            runner_version=digest(
                [
                    hashlib.sha256(f.read_bytes()).hexdigest()
                    for f in (HERE / "drive.py", HERE / "transactions.py")
                ]
            ),
        )
        rec = m.setdefault("stages", {})
        dump(p, m)
        for n in ("pipeline", ".venv", "envs", "third_party"):
            if not (root / n).exists():
                (root / n).symlink_to(HOME / n, target_is_directory=True)
        env = dict(
            os.environ,
            RS=str(root),
            RS_INSTALL=str(HOME),
            RUN_NAME=a.name,
            RUN_RIG=a.rig,
            RUN_RELEASES=str(base / "releases"),
            RUN_ROOT=str(root),
            PYTHONPATH=str(root),
            PYTHONDONTWRITEBYTECODE="1",
            PYTHONUNBUFFERED="1",
            HF_HUB_OFFLINE="1",
            TRANSFORMERS_OFFLINE="1",
            VECLIB_MAXIMUM_THREADS="1",
            OPENBLAS_NUM_THREADS="1",
            OMP_NUM_THREADS="1",
            MOVERS_RUN="photo-full-native",
            GT_REPORT="work/run/atlas-report",
            GT_WORK="work/run/atlas-state",
            GT_OUT="work/run/atlas",
            GROUND_VERSION="2",
            GROUND_THREADS="4",
        )
        upstream = raw
        legacy_upstream = raw
        forced = False
        for i, k in enumerate(PREVIEW):
            if a.start == k:
                forced = True
            code = sources(k)
            inputs = {"preceding_outputs": upstream, "raw_inventory": raw}
            if k in ("inventory", "calibration"):
                inputs["installed_rig"] = sig(HOME / "calib/rig.json")
            if k == "calibration":
                inputs["installed_masks"] = sig(HOME / "calib/masks")
                inputs["rig_mode"] = a.rig
                if a.rig != "stored":
                    inputs["rig_ba"] = sig(HOME / "work/calib/build/rig_ba")
                    # the limits of this installation's own rig when it has them (calibration.limits_file)
                    own = HOME / "calib/rig-baseline.json"
                    inputs["gross_baseline_sha256"] = hashlib.sha256(
                        (own if own.is_file() else HERE / "rig-baseline.json").read_bytes()
                    ).hexdigest()
            key = digest([k, inputs, params, code])
            prev = rec.get(k, {})
            valid = False
            old_key = prev.get("key")
            old_signs = prev.get("signatures")
            try:
                if (
                    prev.get("status") == "prepared"
                    and prev.get("key") == key
                    and not forced
                ):
                    prepared = Path(prev["generation"])
                    signs, hashes = validate(prepared, k)
                    if (signs, hashes) == (
                        prev["signatures"],
                        prev["content_sha256"],
                    ) and publication_ok(prepared, k):
                        promote(root, prepared)
                        prev.update(status="done", recovered=stamp())
                        dump(p, m)
                        if k.startswith("publish"):
                            m["viewer"] = read(root / ART[k][0])["url"]
                            dump(p, m)
                        valid = True
                elif prev.get("status") == "done" and not forced:
                    adopt = False
                    if "content_sha256" not in prev and k in ("inventory", "ingest"):
                        old_inputs = dict(inputs, preceding_outputs=legacy_upstream)
                        old_code = sources(k, legacy=LEGACY_RAW_RUNNER)
                        adopt = prev.get("key") == digest(
                            [k, old_inputs, params, old_code]
                        )
                    if adopt or prev.get("key") == key:
                        signs, hashes = validate(root, k)
                        valid = (
                            signs == prev["signatures"]
                            and (adopt or prev.get("content_sha256") == hashes)
                            and publication_ok(root, k)
                        )
                        if valid and adopt:
                            prev.update(
                                key=key,
                                inputs=inputs,
                                code_version=code,
                                legacy_key=old_key,
                                content_sha256=hashes,
                                adopted=stamp(),
                            )
                            dump(p, m)
            except (OSError, ValueError, KeyError, EOFError, BadZipFile):
                valid = False
            if old_signs:
                legacy_upstream = digest([prev.get("legacy_key", old_key), old_signs])
            if valid and not forced:
                print(k, "cached", flush=True)
                upstream = digest([key, prev["signatures"]])
                if a.only == k:
                    break
                continue
            if (a.only and k != a.only) or (a.start and not forced):
                raise ValueError(
                    f"{k}: prerequisite missing or stale; resume without --from/--only"
                )
            log = base / "logs" / f"{time.time_ns()}-{k}.log"
            log.parent.mkdir(exist_ok=True)
            cmd = [str(py), str(HERE / "stages.py"), k]
            if k == "inventory":
                cmd += [str(src), json.dumps(a.window)]
            r = dict(
                status="waiting",
                key=key,
                parameters=params,
                code_version=code,
                inputs=inputs,
                outputs=ART[k],
                command=cmd,
                log=str(log),
                queued=stamp(),
                owner=who,
            )
            if prev.get("status") in ("waiting", "running", "prepared"):
                r["interrupted_attempt"] = {
                    f: prev.get(f)
                    for f in ("status", "owner", "generation", "log", "queued")
                }
            log.write_text(shlex.join(cmd) + "\n")
            rec[k] = r
            dump(p, m)
            for later in (PREVIEW + FULL)[i + 1 :]:
                if later in rec:
                    rec[later]["status"] = "stale"
            start = None
            try:
                # inventory and sparse sequential clip decoding use no GPU. All other
                # expensive stages (including CPU Ceres) share the installation lock.
                lease = (
                    lock(
                        HOME / "work/locks/reconstruction.lock",
                        True,
                        who,
                        drive=a.name,
                        stage=k,
                    )
                    if k
                    not in (
                        "inventory",
                        "ingest",
                        "keyframes",
                        "publish",
                        "publish-full",
                    )
                    else contextlib.nullcontext()
                )
                with lease as heavy_lease:
                    if k not in (
                        "inventory",
                        "ingest",
                        "keyframes",
                        "publish",
                        "publish-full",
                    ):
                        while busy := heavy():
                            r["detail"] = "waiting for " + ", ".join(
                                str(j["pid"]) for j in busy
                            )
                            if busy != r.get("blockers"):
                                with log.open("a") as f:
                                    print(stamp(), r["detail"], file=f)
                            r["blockers"] = busy
                            dump(p, m)
                            time.sleep(20)
                    start = time.monotonic()
                    work = begin(root, k, exclusions(root, k))
                    r.update(
                        status="running", start=stamp(), detail="", generation=str(work)
                    )
                    dump(p, m)
                    print(k, shlex.join(cmd), "log", log, flush=True)
                    with log.open("a") as f:
                        f.flush()
                        child = subprocess.Popen(
                            cmd,
                            cwd=work,
                            env=dict(
                                env,
                                RS=str(work),
                                RUN_ROOT=str(work),
                                PYTHONPATH=str(work),
                            ),
                            pass_fds=tuple(
                                f.fileno()
                                for f in (runner_lease, heavy_lease)
                                if f is not None
                            ),
                            stdout=f,
                            stderr=subprocess.STDOUT,
                            start_new_session=True,
                        )
                        try:
                            rc = wait_worker(
                                child, k in ("masks", "ground", "atlas", "movers")
                            )
                        except BaseException:
                            os.killpg(child.pid, signal.SIGCONT)
                            os.killpg(child.pid, signal.SIGTERM)
                            child.wait()
                            raise
                        if rc:
                            raise RuntimeError(
                                f"exit {rc}: "
                                + log.read_text(errors="replace")
                                .strip()
                                .splitlines()[-1]
                            )
                    signatures, hashes = validate(work, k)
                    if not publication_ok(work, k):
                        raise ValueError("published scene or cache is stale")
                    r.pop("blockers", None)
                    r.update(
                        status="prepared",
                        signatures=signatures,
                        content_sha256=hashes,
                        finish=stamp(),
                        duration_s=round(time.monotonic() - start, 3),
                    )
                    dump(
                        p, m
                    )  # crash on either side of pointer rename: revalidate this exact generation
                    promote(root, work)
                    r.update(status="done")
                    upstream = digest([key, signatures])
                    if k.startswith("publish"):
                        m["viewer"] = read(root / ART[k][0])["url"]
                        print(m["viewer"], flush=True)
            except BaseException as e:
                r.update(status="failed", detail=str(e))
                print(f"{k} FAILED: {shlex.join(cmd)}; {e}; log {log}", file=sys.stderr)
                raise
            finally:
                if k == "calibration" and r.get("generation"):
                    receipt = Path(r["generation"]) / "work/calib/verification.json"
                    if receipt.is_file():
                        measured = read(receipt)
                        r["rig"] = measured
                        r["warnings"] = measured["warnings"]
                r["commands_run"] = [
                    line[2:]
                    for line in log.read_text(errors="replace").splitlines()
                    if line.startswith("$ ")
                ]
                r.update(
                    finish=stamp(),
                    duration_s=round(time.monotonic() - start, 3) if start else 0,
                )
                dump(p, m)
            if a.only == k:
                break
        if a.quality == "full" and a.only not in PREVIEW:
            from progressive import advance

            advance(base, root, env, m, p, runner_lease, a, clip_select)
        if m.get("viewer"):
            print(m["viewer"])


@contextlib.contextmanager
def viewer_host(args):
    # A standalone Linux worker verifies the same served bytes without exposing a network port.
    if not sys.platform.startswith('linux') or args.action != 'run' or os.getenv('RS_VIEWER_URL'):
        yield
        return
    import socket, urllib.request
    with socket.socket() as s:
        s.bind(('127.0.0.1',0));port=s.getsockname()[1]
    url=f'http://127.0.0.1:{port}'
    host=subprocess.Popen(['bun',HOME/'deploy/serve.ts',HOME/'viewer/dist',HOME/'scene',str(port),'127.0.0.1'])
    try:
        for _ in range(100):
            if host.poll() is not None:raise RuntimeError('private viewer exited')
            try:
                with urllib.request.urlopen(url+'/scenes/index.json',timeout=1): break
            except OSError:time.sleep(.1)
        else:raise RuntimeError('private viewer did not start')
        os.environ['RS_VIEWER_URL']=url;os.environ.setdefault('HF_HOME',str(HOME/'third_party/hf'));os.environ.setdefault('TORCH_HOME',str(HOME/'third_party/torch'))
        yield
    finally:
        host.terminate();host.wait(timeout=10)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    sub = ap.add_subparsers(dest="action", required=True)
    r = sub.add_parser("run")
    r.add_argument("footage")
    r.add_argument("--name", required=True)
    r.add_argument(
        "--rig",
        choices=["stored", "verify", "refit"],
        default="verify",
        help="hardware compatibility gate; explicit refit requires supported motion and improvement on reserved images",
    )
    r.add_argument("--quality", choices=["preview", "full"], default="preview")
    g = r.add_mutually_exclusive_group()
    g.add_argument("--from", dest="start")
    g.add_argument("--only")
    r.add_argument("--window", type=float, nargs=2)
    r.add_argument(
        "--clip",
        nargs="+",
        default=["all"],
        metavar="COVERAGE",
        help="full-tier generated coverage: all (default), none, or t0 t1",
    )
    s = sub.add_parser("status")
    s.add_argument("--name", required=True)
    signal.signal(signal.SIGTERM, lambda *_: sys.exit(143))
    try:
        args = ap.parse_args()
        with viewer_host(args):
            run(args)
    except (OSError, RuntimeError, ValueError, KeyError) as ex:
        sys.exit(str(ex))
