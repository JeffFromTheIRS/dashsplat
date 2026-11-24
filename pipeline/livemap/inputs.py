"""Private producer roots. Keep the shared depth/masks and their manifests untouched."""

from pathlib import Path
import os, json, sys, subprocess, shutil
import numpy as np

RS = Path(os.environ.get("RS", Path.home() / "dev/reality-splat")).resolve()
W = RS / "maps/object-v3"
W.mkdir(parents=True, exist_ok=True)
from pipeline.common.hardware import roster

# NAMES is the rig's cameras in index order (six on HW4, four on HW3); CAMS the same in processing order,
# front first and back last.
NAMES = roster(RS)
CAMS = [c for c in NAMES if c != "back"] + [c for c in NAMES if c == "back"]


def make():
    meta = json.loads((RS / "depth/keyframes.json").read_text())
    tl = np.load(RS / "data/drive/timeline.npz")
    odo = np.r_[0, np.cumsum(np.hypot(np.diff(tl["x"]), np.diff(tl["y"])))]
    plan = {}
    for c in CAMS:
        ft = json.loads((RS / "data/drive/cams" / f"{c}.json").read_text())
        ids = np.array([v["n"] for v in ft])
        od = np.interp([v["t"] for v in ft], tl["t"], odo)
        sel = []
        for e in meta["frames"][c]:
            if e["t_key"] < 80 or e["t_key"] > 140:
                continue
            if sel and (
                e["t_key"] - sel[-1]["t_key"] < 0.28 or e["odo"] - sel[-1]["odo"] < 0.56
            ):
                continue
            sel.append(e)
        hold = {e["n"] for j, e in enumerate(sel) if j % 10 == 5 and e["t_key"] >= 85}
        rows = []
        for j, e in enumerate(sel):
            i = np.searchsorted(ids, e["n"])
            window = [int(ids[i])]
            o = od[i]
            for q in range(2 if c == "back" else 4):
                i = np.searchsorted(od[:i], o - 1, side="right") - 1
                if i < 0:
                    break
                window.append(int(ids[i]))
                o = od[i]
            h = e["n"] in hold
            rows.append(
                dict(
                    **e,
                    hold=h,
                    window=window,
                    train=not h and not bool(set(window) & hold),
                    ordinal=j,
                )
            )
        plan[c] = rows
    (W / "plan.json").write_text(json.dumps(plan))
    return plan


def prepare():
    plan = make()
    meta = json.loads((RS / "depth/keyframes.json").read_text())
    print(
        "original files",
        {c: len(list((RS / "depth" / c).glob("*.npy"))) for c in CAMS},
        flush=True,
    )
    for mode in ("live", "stitched"):
        base = W / mode
        (base / "depth").mkdir(parents=True, exist_ok=True)
        for n in ["data", "calib", "poses", "masks", "third_party"]:
            if not (base / n).exists():
                (base / n).symlink_to(RS / n)
        target = {k: [r for r in v if r["train"] or r["hold"]] for k, v in plan.items()}
        (base / "depth/keyframes.json").write_text(
            json.dumps({"frames": target, "selections": meta["selections"]})
        )
        if mode == "stitched" and meta["depth"]["trajectory"] == "poses/ego.npz":
            for c, rows in target.items():
                dst = base / "depth" / c
                dst.mkdir(exist_ok=True)
                idxpath = RS / "depth" / c / "index.json"
                idx = json.loads(idxpath.read_text()) if idxpath.exists() else {}
                private = dst / "index.json"
                items = json.loads(private.read_text()) if private.exists() else {}
                for e in rows:
                    key = str(e["n"])
                    src = RS / "depth" / c / f"{e['n']:06d}.npy"
                    out = dst / src.name
                    v = idx.get(key, {})
                    if (
                        not out.exists()
                        and src.is_file()
                        and v.get("window") == e["window"]
                        and v.get("model") == "da3-large-1.1"
                        and v.get("scale_applied") == 1
                    ):
                        os.link(src, out)
                        items[key] = v
                private.write_text(json.dumps(items))
    print(
        "scheduled",
        {
            c: (len(v), sum(x["train"] for x in v), sum(x["hold"] for x in v))
            for c, v in plan.items()
        },
        flush=True,
    )


def produce():
    prepare()
    plan = json.loads((W / "plan.json").read_text())
    script = RS / "pipeline/depth/run_depth.py"
    for step in ["audit", "front", "sides", "back"]:
        cams = (
            CAMS
            if step == "audit"
            else (
                ["front"]
                if step == "front"
                else [c for c in CAMS if c not in ("front", "back")]
                if step == "sides"
                else ["back"]
            )
        )
        for mode in ["stitched", "live"]:
            base = W / mode
            path = base / "depth/keyframes.json"
            meta = json.loads(path.read_text())
            meta["frames"] = {
                c: [
                    r
                    for i, r in enumerate(v)
                    if (r["train"] or r["hold"])
                    and (step != "audit" or i % 10 in [0, 5])
                ]
                for c, v in plan.items()
            }
            path.write_text(json.dumps(meta))
            pose = "ego_causal.npz" if mode == "live" else "ego.npz"
            print(step, mode, flush=True)
            subprocess.run(
                [
                    sys.executable,
                    str(script),
                    "--cams",
                    ",".join(cams),
                    "--trajectory",
                    str(base / "poses" / pose),
                    "--scale-factor",
                    "1.0",
                    "--window",
                    "5",
                    "--window-multiview",
                    "3",
                ],
                env={**os.environ, "RS": str(base)},
                check=True,
            )


if __name__ == "__main__":
    produce()
