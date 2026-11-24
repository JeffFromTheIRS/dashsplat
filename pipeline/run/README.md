# dashsplat drive runner

## On your machine

The rest of this README is the record of the runner as built and measured on the author's machines: "the Studio" is
his Mac Studio (M5 Ultra), "basebook" his working Mac, `ferry` his file-copy tool and `10.0.50.51` the Studio's LAN
address. To run it elsewhere:

- The installation root is `RS_INSTALL`, by default this checkout. Environments (`.venv`, `envs/`), weights
  (`third_party/`), footage (`data/`), the calibrated rig (`calib/rig.json`, `calib/masks/`), per-drive work
  (`work/`) and published scenes (`scene/`) live under it; the root `.gitignore` keeps them out of git.
- Run from the root: `.venv/bin/python pipeline/run/drive.py run /path/to/clips --name my-drive` (add
  `--quality full` for the full tier).
- The camera list is the installed rig's: HW4's six cameras or HW3's four (front, back, two repeaters). The
  footage must have exactly those cameras at the rig's sizes, or the inventory stage stops. Everything below
  that says "six" describes the HW4 car this was measured on; on HW3 the stages up to photometry that need no
  image model have been run on one 75-second window, the others not
  ([calibrating another car](../calib/README.md#another-car-including-hw3)).
- The runner checks each publication over HTTP at `RS_VIEWER_URL`, default `http://127.0.0.1:8090`. On macOS start
  that host first: `cd viewer && bun install && bun run build && cd ..`, `mkdir -p scene`, then
  `bun deploy/serve.ts viewer/dist scene 8090 127.0.0.1`. On Linux the runner starts a private loopback host itself.
- The runner does not build its prerequisites: the calibrated rig (not in this repository yet), the Ceres solvers and
  C++ helpers, the chase-camera bundle (`tools/run-chase/`) and the cache-tool link (`viewer-cache-tool`).
  `pipeline/cuda/setup-drive.sh` builds them on Linux; on a Mac follow the same steps with Homebrew
  (`brew install ceres-solver cmake`) in place of apt. The rig check's thresholds (`rig-baseline.json`) were
  measured on the author's car; `calibration.py baseline` writes those of your own rig to
  `calib/rig-baseline.json`, which is read in preference.
- `install.sh` copies the runner to a second machine with `ferry`; you do not need it.

## Stage inventory (before implementation)

Paths below are relative to the selected drive workspace, except **installation** (Studio's existing RS directory). A path in a script is not evidence that its contents can be shared across drives.

| Stage / entry point | Inputs → outputs | Hard-coded paths, time or other dependency |
|---|---|---|
| Ingest: ingest/ingest.py | six HW4 MP4 streams, embedded SEI → data/drive camera frames, index and telemetry | expects data/clips/<clip>/<clip>-<cam>.mp4; preserve native frame IDs |
| Motion selection | drive-local speed and timestamps → stopped spans and overlapping fit windows | old drive has long stops; do not reuse its intervals |
| Rig: calib/features, solve, scale, deliver; common/rig.py | fresh image correspondences + car motion + installed six-camera calibration → verified rig or new solve | features.select uses fixed intervals; scale.run calls align(Scene().load()); rig selfcheck replays the original solver problem, not new footage |
| Pose: poses/{motion,tracks,offline,causal} | rig + indexed frames + telemetry → smoothed and causal trajectories | uses whole-drive sampling and native Ceres executables |
| Ground: road/{pairs,fit,field,scene,deliver} | trajectory + fresh road correspondences → road surfaces and ground splats | causal-lag.npy required downstream; deliver contains t[1000] and 293–326 s checks |
| Appearance: photo/{sample,fit,live_v2} | road, lag, images → fitted offline and live exposure models | outputs under work/photo; stitch exposure JSON required separately |
| Atlas: groundtex/atlas.py + verify.py | drive-local exposures.json, road-stitched-v2 and terrain/photometry → tiled atlas | GT_WORK/GT_OUT; native start/end support; release-specific v5/v6 audit/hand-selected states are not generic |
| Instances: depth/keyframes + masks/run_masks.py | fresh window's camera rows → window keyframes + per-camera semantic/instance masks | production environments and model weights are installation resources; preview movers also need masks |
| Movers: movers/{packets,handoff,footprint,motion,semantics,filter,landmarks,removal,online_scene,skin} | fresh masks, rays and poses → tracks, exclusion/removal and attached dynamic objects | MOVERS_RUN=photo-full-native; removal uses original intervals; online_scene.publish expects live-v5 cache and slice-instances.json |
| Depth: depth/run_depth.py, produce.py | window keyframes + imagery → calibrated metric depth | production audit uses original drive; avoid importing its scale correction as fresh evidence |
| Teacher: ibr/teacher.py and morph_sources | fresh poses, depth and world → teacher views | four fixed chase shots in 85–140 s; morph_sources caps source times at 143 s and expects stitched-opt-v3/drive.json |
| Optimizer/stitch: stitch/{v2data,v4teacher,v5world,v5release} | masks/removal, teacher observations, calibrated depth, prior world → optimized tiled scene | release expects existing prior optimized world + hand-reviewed structures; overlapping object IDs cannot simply be concatenated |
| Chase clip (optional): genfill/{clip,views} | viewer-correct chase track and rendered views → encoded clip | uses work/stitch/views-85-140/manifest.json and maps/object-v3/raster.dylib; use no paid services |
| Publish: deploy/serve.ts + installed viewer cache tool | scene/<name>/drive.json and packed assets → rebuilt cache and LAN viewer | stable host serves the installation's scene directory, not a private workspace; check HTTP after publication |

## Run on the Studio

From the installation directory, where the raw footage and fitted HW4 camera bundle live:

```sh
cd ~/dev/reality-splat
.venv/bin/python pipeline/run/drive.py run data/clips --name hw4-drive --quality full
```

Use `--quality preview` (the default) to stop after publication. Use `--window 224 254` to restrict the decode and reconstruction to seconds on the **raw front SEI clock**, including a short shutter guard. A single folder may contain nested minute files or flat files; pass one continuous recording from one car. Before first use on a Studio without the runner, run `pipeline/run/install.sh` on basebook from this checkout. It copies only runner files, the drive-local path/time adapters listed below, and the existing viewer chase camera bundled with Bun. No pip, npm, or system installs.

Requirements: complete six-file minute groups of the calibrated HW4 dimensions, decodable H.264 with Tesla SEI, forward motion and shared visibility between lenses; existing Studio .venv and model caches, Ceres solvers, compiler, Bun/cache tool, Apple Vision classifier, and the already-running viewer host. Full adds the existing stitch packages, DA3, native teacher rasterizer, and for the default `--clip all`, the local genfill environment, ffmpeg and model weights. The compatibility check measures 16 m moving holdouts, when available. Thin visual support produces a warning, not a mismatch. This is a camera-hardware compatibility check, not vehicle identification: it cannot distinguish identical rigs by VIN.

### Inspect or resume

```sh
.venv/bin/python pipeline/run/drive.py status --name my-drive
.venv/bin/python pipeline/run/drive.py run /path/to/clips --name my-drive --quality full
.venv/bin/python pipeline/run/drive.py run /path/to/clips --name my-drive --from masks
.venv/bin/python pipeline/run/drive.py run /path/to/clips --name my-drive --only calibration
```

Repeat the **same source path, name and window** to resume. `--only` checks prerequisites and runs just that stage if stale; `--from` forces that stage and the rest of the selected tier. Valid prerequisite outputs are reused across tiers. A changed source file (path, size, mtime) or time window gets a new private workspace. Source changes, missing outputs and changed output signatures invalidate dependent stages; an invalid stage starts a private APFS generation without its stale intermediates. Each producer writes there first. The runner checks its output contract, hashes the output bytes, durably records `prepared`, then renames a temporary symlink over the active workspace pointer. A crash before `prepared` leaves only private partial files; a crash at the rename is recovered by revalidating the prepared generation. Prior generations, uncommitted attempts and releases use disk space. Failed stages can be replayed without manually picking frames or rebuilding caches.

Output: `$RS/work/drives/<name>/manifest.json`, per-attempt logs in `logs/` and per-generation artifacts. The manifest records options, stage command, logged nested commands, timestamps, duration excluding queue time, input and output signatures, and source-code digest. `waiting` includes the external Python PIDs. Non-light stages take an OS lock in `$RS/work/locks/reconstruction.lock` and wait without starting their model. Its JSON owner records PID, process start, kernel boot time, drive and stage. The stage child inherits the lock while it runs. When an owner dies or the Studio reboots, the kernel releases the lock; the next holder records the stale owner and reuses the same lock file. Do not delete it. After a reboot, rerun the same command. Raw inventory and single-thread decoding can run while GPU work is queued. Separate releases are assembled and cached away from the host directory. Only after validating the scene is the host's `scene/<name>` symlink swapped; the runner checks the index, viewer HTML, manifests, an atlas texture and a binary tile over HTTP. On failure it restores the previous pointer. The link is printed on successful publication. `RS_VIEWER_URL` overrides the viewer host (default `http://127.0.0.1:8090`; the author set it to the Studio's LAN host); `RS_INSTALL` overrides the installation path (default: the checkout).

### What's automatic, and which result it builds

- **Motion and calibration:** the runner chooses 24 s moving fit windows with 3 s overlap. `--rig stored` checks all six native camera names and dimensions, then copies the installed bundle. The default, `--rig verify`, also measures up to three disjoint, telemetry-selected 16 m panels with two blind BA starts. Each target is withheld from its other-camera triangulation. We pool distinct target residuals and give each adequately supported lens one vote, normalized to a 1448 px image width. A single soft-limit exceedance (5 native px front, 3 elsewhere) is reported in the manifest. The gross pooled test needs ≥4 supported lenses; a single-lens gross verdict needs a pooled exceedance and replication on two panels at both starts. No scale or single-panel pixel limit independently rejects reuse. Thresholds come from `pipeline/run/rig-baseline.json`: fresh matches on three disjoint, non-calibration stretches of the calibrated car (115–145, 161–191 and 329–359 s), not previous reconstructions. Use `--rig refit` only for a deliberate recalibration. It requires ≥120 moving seconds, two disjoint ≥12° training turns and ≥50 held-out targets per lens. On matching held-out targets (≥50/lens and ≥80% coverage), the candidate must lower the lens-balanced median by ≥5% at both blind starts and worsen no lens by >10%. Sparse or unimproved fits keep the stored rig. Missing cameras, wrong dimensions or supported gross mismatch stop publication. Measurements, chosen rig and warnings are recorded in the calibration stage of the manifest.
- **Preview:** road-relative poses and causal lag are fitted on the selected drive extent. Stopped exposures are thinned by the pose/photo schedules and omitted from mask and atlas schedules. Existing v2 ground splats plus the **base photographic atlas** are built; this is not a v5/v6 ground release. Automatic photo-full mover discovery verifies masks, projected footprints, visual motion and independent category evidence. An empty object list is allowed after real discovery. The compatibility source for the mover exporter is a fresh ground-only scene; it correctly has no instance-tagged upright tiles. No manually labelled tracker or reviewed hindsight vehicle is imported.
- **Full, in time order:** each window builds its own 0.5 m production masks/depth, removal ledger and audited depths, then pose-correct chase teachers. The generic v5 extension recipe puts optimizer capacity into standing objects, uses native photographic witnesses and foliage opacity regularization, and proposes shafts and generic facade planes automatically. Planes need both road clearance and displaced-camera depth agreement. Six-lens visibility is checked before release. No named office planes, manually protected first-slice shafts/arms or reviewed generated training views are imported. A short initial crawl can borrow moving depth witnesses from the immediately following departure; parked frames are excluded and borrowed support is not counted as extra fitted coverage. The whole-drive ground/atlas and preview movers remain available beyond the completed windows. Published standing tile overlaps use nearest-route ownership.
- **Chase coverage:** `--clip all` is the full-tier default, `--clip none` makes geometry only, and `--clip 224 254` intersects those clock bounds with moving spans. Each window publishes geometry before its ≤4 s videos. Adjacent chunks share an exact endpoint; the next repair seeds from the previous RGB, depth and camera pose, and carries the provenance mask too. A stop resets the seed. The documented repair settings are timestep 199 and input/output feed-forward 0.5. The installed renderer does near-camera fading and rejects photographically contradicted splats. No moving model is drawn into the video: metadata specifies viewer overlays. Opaque source-photo atlas texels are labelled as observed; only the driven corridor receives the fixer's road protection. Each chunk has a private, producer/version-keyed frame cache, an encoded-frame-count check and a completion receipt. The whole-scene cache is rebuilt and the hosted bytes checked after **each** geometry window and clip. Changing a producer need not redo a verified earlier window.
- **GPU priority:** the runner waits before compute. If a generated-clip repair begins during our GPU work, it pauses its own worker process group and resumes when the repair clears. It never signals the repair engineer's job. Rerun the same command after a reboot. `status --name hw4-drive` reports coverage on the drive clock, including published full and clip seconds separately. There is no interactive frame selection or approval step.

### Whole-drive proof (started 02:50 EST)

The 361.44 s recording contains **249.53 moving seconds**, five moving spans and **14 overlapping fit windows**. Full clip selection schedules **70** chunks, with no overlapped time counted twice. Whole-drive JPEGs and per-window depth are recomputed: the previous runner receipt covers only a 30 s subset, and the original installation's processed drive has no matching full raw-input/output content receipt. Installed models and calibration are shared. All releases are private until their cache is validated and the one host pointer moves.

**Vehicle inspection, earlier 224–254 s proof:** all 36 tracks have class and finite boxes; none had a paint name, but 35 had the viewer's ≥50 opaque side samples at door height for automatic paint. At exactly 234.0 and 234.5 s no confirmed track was active. At 235.0 s the old static layer contained 114 centres inside the active car's box. Native per-camera removal mattes therefore did **not** guarantee an empty 3-D body. The new merger removes standing centres inside a tracked, oriented body **at the centre's photographic source time**, above 0.15 m. It also emits measured colour names where the scan supports one and keeps scan-paint fallback otherwise. This does not remove unrelated walls throughout a car's swept path or manufacture a vehicle when there is no track. Garbled red sign faces are a known reconstruction/appearance gap; there is no sign-specific patch or manual paint in this runner.

**New outside-runner edits:** `pipeline/stitch/v5world.py` selects the window's work directory, metadata and audited per-lens range masks; `pipeline/stitch/v5fit.py` gates the old 85 s generated-supervision branch on the historical benchmark; `pipeline/stitch/v5release.py` gates protected structures and named office inputs the same way. Historical invocations without these arguments keep their original paths. No changes to genfill, groundtex or viewer.

**Measured whole-drive preview:** published at 04:50 EST. Per minute of moving footage (seconds of measured stage wall time): inventory 0.4, ingest 131.2, calibration 8.5, poses 53.2, road 102.1, keyframes 0.5, masks 229.6, photometry 21.7, ground 66.7, atlas 94.7, movers 103.5, publish/cache 3.9. Total **13.6 min per moving minute** (56.6 min measured stages, about 2 h elapsed including waits). All 1,217 photographic atlas tiles and 121 automatic preview tracks were published before the first full window.

**First full window (0–4.8606 s):** first native v5 fit and 0–4 / 4–4.8606 s videos published by 05:07 EST. The 4 s join's seed depth, repaired RGB and provenance mask are byte-identical; both hosted MP4s matched their private receipts. The right repeater supplied no admitted range band in this window: its metric geometry was excluded, while physical photos were retained. Five other lenses supplied audited bands. The automatic v5 fit used 898 native and 702 teacher updates. Window release audit: 13 standing centres removed inside witnessed source-time boxes, 90 named paints, 117 measured scan colours across 122 attached/preview tracks. Raw clip generation measured **85.7 minutes of compute per generated minute** on those first two chunks; extrapolating short windows is noisy.

### 06:30 EST owner checkpoint

[Open hw4-drive at 98 s, inside a generated interval](http://10.0.50.51:8090/?scene=hw4-drive&mode=stitched&t=98). Whole preview: **249.53 / 249.53 moving s** over the 361.44 s timeline. Published v5 geometry: **49.86 s**, 3/14 windows (0–4.86 and 81.94–126.94). Published video: **32.86 s**, 9/70 chunks (0–4.86 and 81.94–109.94). The 109.94–113.94 s chunk was running at cutoff. All seven adjacent clip joins, including the 105.94 s fit-window boundary, matched RGB/depth/provenance; the discontinuity across the parked interval intentionally did not carry a seed. Hosted generated index, RSB flags and latest MP4 were independently checked. The viewer link stays the same as more data is published.

**Full rates so far**, stage minutes per newly covered moving minute: production masks/depth **38.98**, removal **2.27**, teachers **4.00**, v5 fit **4.63**, mover attachment **0.75**, merge **0.58**, window and chunk publication combined **0.73**. Generated render/repair/encode: **89.10 min per generated minute**. With 199.67 s of full geometry and 216.67 s of video outstanding, these observed rates project **8 h 15 min** more, approximately **14:45 EST** without further queues; scene complexity and competing GPU activity can move that estimate. Current release: 126 tracks, 93 measured paint names, 121 scan-paint candidates, 17 source-time standing centres removed. There is no guarantee of a clean model for untracked cars. The whole-drive process remains detached on Studio.

### Historical 30 s proof and measured rates

The installation contained one raw recording. The proof uses its raw MP4s under new name `hw4-raw-224-254`, never its processed `data/drive`. First measured run: **6 minute groups, 6,530 decoded, SEI-indexed images verified present across all six cameras**, approximately 294 m including guards. Candidate rig holdout: 18 shutters at 225.505–227.671 s, outside the calibration report's old scale-check interval. These are candidate observations, not a calibration pass. The selected optimizer windows are 224–248 s and 245–254 s. The installed runner repeated inventory in **1.37 s** and ingest in **75.22 s** after its code changed. An explicit `--only ingest` restart then printed `inventory cached` and `ingest cached`; both stage receipts, timestamps, logs and signatures remained unchanged. After the Studio lost power at 21:19 EST, the unattended full-tier command restarted and both raw stages were cache hits again. Their original timestamps and durations remained intact while the updated runner read and hashed the actual indexes, timeline and JPEGs and checked JPEG dimensions and end markers. The original precision gate stopped the proof at calibration. It was replaced with the hardware-compatibility check described above.

A second, isolated Studio run was killed after writing rear frame 007992. That JPEG remained private; the committed workspace still contained only inventory. Restart reclaimed the dead PID and validated and promoted all 6,530 images in **68.799 s**. Replaying a `prepared` receipt after restoring the old pointer recovered the completed generation without decoding again. Flipping a JPEG byte while preserving its size and modification time made the runner reject the ingest prerequisite before calibration. The live drive and its cache were not modified by these checks.

| Stage | Wall time for this 30 s window | Per moving minute / interpretation |
|---|---:|---|
| Raw inventory | 1.37 s | Scans the full 361 s source; 2.74 s per selected minute here |
| Single-thread window ingest | 75.22 s | 150.4 s per selected minute; includes partial minute files |
| Hardware compatibility (default verify) | 33.74 s | Stored rig; no refit. Prior verification 35.81 s. |
| Poses / road / keyframes | 19.19 / 28.95 / 0.78 s | 97.8 s combined per selected minute |
| Preview masks / photometry / ground splats | 92.86 / 9.19 / 30.41 s | 264.9 s combined per selected minute |
| Atlas / automatic movers / publication and cache | 33.19 / 98.50 / 2.23 s | 267.8 s combined per selected minute |
| **Verified preview total** | **425.61 s (7 min 06 s)** | **14 min 11 s per selected minute** on this 30 s run, excluding queues and earlier failed attempts |
| Full production depth + masks | 908.53 s | 2,306 camera keyframes verified |
| Mover removal / teacher views | 183.17 / 96.55 s | 665 accepted teacher views across two overlapping windows |
| Optimize and merge / attach movers | 75.17 / 49.40 s | One scene with 161 unique tile IDs |
| Generated chase / full cache and publication | 408.47 / 5.51 s | Hosted H.264 and release bytes verified |
| **Additional full-tier time** | **1,726.78 s (28 min 47 s)** | 57 min 34 s per selected minute |
| **Preview + full** | **2,152.39 s (35 min 52 s)** | 71 min 45 s per selected minute; this is the sum of successful stage times, not elapsed time including queue waits |

Queue time, algorithms' elapsed time, and media duration are distinct. The README does not promise a preview at driving speed. No human action is needed to release the OS lock when existing jobs exit. An incompatible rig, inadequate visual/depth evidence, missing model or a failing producer is reported at the first affected stage; inspect its actual command and log before resuming.

### Edits outside `pipeline/run/`

Paths below are source files in this checkout. All five preserve their installed defaults when the runner is not invoking them.

| File | Argument plumbing |
|---|---|
| `pipeline/livemap/audit.py` | Accept drive-local audit start/end bounds. |
| `pipeline/stitch/v2data.py` | Accept drive-local mask root, audited depth measurements file and ground tile selection. |
| `pipeline/ibr/morph_sources.py` | Accept fresh host scene and source-time bounds. |
| `pipeline/ibr/teacher.py` | Accept teacher output root, chase window and preceding source bound. |
| `pipeline/movers/removal.py` | Accept this drive's removal intervals. |

### Superseded precision gate: hw4-raw-224-254

Inventory and ingest were cache hits on both calibration attempts. The first attempt had only five front target transfers per start; its refit failed during field-of-view calculation. The revised check reserved three disjoint holdouts from telemetry before running the solver: 225.505–227.671, 250.475–253.863 and 241.226–242.142 s. All scale estimates were within 1.7% of telemetry, and the two starting spacings agreed within 0.03% on each panel. The pooled front result passed (162 distinct targets at either start; medians 4.43 and 4.56 px), but the first panel exceeded the 5 px bound on 28 targets (7.19 and 7.23 px). The gate rejected reuse. The fallback trained on disjoint shutters; its left-repeater focal estimate was 881.7 px, 15.26% below the delivered 1040.5 px. Its field-of-view calculation did not bracket the masked sensor edge. This rejects that fit; it does not establish that the physical car has a different rig. The owner confirmed that these clips are from the calibrated vehicle. These measurements motivated the soft warnings and empirical gross-mismatch gate above; short-window refits are disabled.

### Empirical compatibility result and front-overlap follow-up

Native targets pooled by lens at 1448 px width gave 1.416 px (115–145 s), 1.785 px (161–191 s), and 1.331 px (329–359 s). The gross threshold is **5.3565 px**; the proof window scored **1.5252 px** and retained the stored rig. The stage manifest includes per-lens counts, percentiles and panel motion; it carries the warning without marking the stage failed. With insufficient reference coverage for a lens-specific threshold, that lens can participate in the overall pooled test but has no lens-only veto.

At 225.505–227.671 s, the follow-up reproduced the prior 7.2 native-pixel front holdout: 6.99/7.08 px across the two blind starts, with 35 target features in each. **All 35 were in the left image third**, concentrated around native (264, 1038); x p10–p90 was 88–408 on a 2896 px image. Mean speed was 7.43 m/s and median yaw rate +0.009°/s. The other front panel with support had a 3.81–4.02 px median and covered both image sides. This is localized cross-camera overlap evidence, not a measured timing shift or drive-wide pose fault. If downstream cross-lens residuals reproduce it, examine front-left matches and row timing there; no timing adjustment was applied.

### Published preview

The preview was published and verified before full-tier work started. The first publication served 2,890,052 live ground splats across 18 tiles, 114 photographic atlas tiles and 28 automatic car tracks. All 1,719 shared mesh vertices agreed (maximum mismatch 0 m); opaque texture coverage was 6,650 m². The hosted map, cache, mover list, atlas and sampled texture and binary tile were checked over HTTP before full production. On this publication run the pooled rig score was 1.454 px, below the 5.357 px reference limit. A lens with no successful atlas registrations has null registration medians and an explicit zero count; this does not waive any texture, opacity or mesh check.

### Published full scene and operator handoff

[Open the full viewer scene](http://10.0.50.51:8090/?scene=hw4-raw-224-254&mode=stitched&t=224.70) · [Generated chase MP4 (233–237 s)](http://10.0.50.51:8090/scenes/hw4-raw-224-254/generated/chase.mp4). The stable preview URL now selects the verified full release. Both fit windows (224–248 s and 245–254 s) are merged: 18 ground tiles plus 143 static-object tiles, 1,467,516 ground splats plus 1,439,423 static-object splats, 36 automatic car tracks, and 114 atlas tiles. The six-camera production verifier reported no missing or unreadable depth files and no missing masks. The hosted full MP4 and five scene manifests matched the immutable release by SHA-256. ffprobe measured 121 H.264 frames at 30 fps; measured coverage does not assert that all detected tracks correspond to different physical vehicles.

Final resume check: all 17 stages through mover attachment were cache hits, including preview publication after full replaced the pointer. Publication validates released content hashes rather than modification times. The clip producer changed during the original clip run; a future full invocation will correctly invalidate the clip and its dependent publication. The already served clip and full release were separately revalidated without replacing either. No manual frame selection, masks, refit, mover labeling, cache rebuild, scene switching or publication was required. A Studio power loss requires restarting the same command; GPU contention only requires waiting. No torch/numpy or system package changes were needed.

The four-second proof's **408.47 s** for render/fix/encode corresponds to **102.1 compute minutes per generated minute**, not 13.6 minutes per driving minute once every moving second is generated. The earlier non-clip full stages consumed **43.9 minutes per driving minute**, before v5 or repeated per-window setup. A first budget for 249.53 s moving is roughly 7.1 h of clip work plus 3 h of full reconstruction and the preview; queue waits and v5 costs come on top. Use measured whole-drive receipts as they arrive.
