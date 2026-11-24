# Rig calibration

From a car's dashcam frames and telemetry, estimate a metric rig, rectify each lens, and validate against disjoint frames and a visual-only drive segment. Results and limitations for the author's drive (a Model 3 with HW4's six cameras) are in [REPORT.md](REPORT.md).

On anomastudio, run from the Reality Splat root:

```sh
# Once: C++17 toolchain, Ceres sparse CPU backend and CMake.
brew install ceres-solver cmake
# Uses the project's existing Python 3.12, NumPy, SciPy and OpenCV.
bash pipeline/calib/run.sh
```

The default root is `$HOME/dev/reality-splat`; override with `RS`, and the interpreter (`$RS/.venv/bin/python`) with `PY`. Inputs: `data/drive/{frames,cams,meta.json,timeline.npz}`. Intermediates and compiled solver: `work/calib/`. Features/pairs are cached for this drive and selection; use a fresh work root for different frames or mask geometry. The final products are `calib/rig.json`, one byte mask per camera, `calib/report.md`, and `calib/debug/`. No neural weights or GPU are used. BA defaults to 14 threads; override with `BA_THREADS`.

1. `features.py` samples moving windows on the shared clock, masks bodywork and builds RootSIFT tracks across cameras and time. Split-only pairs restore sequential coverage when every other keyframe is removed.
2. `solve.py` initializes metric points and runs `ba.cpp`, maximizing image consistency while allowing small 6-DoF corrections to the telemetry trajectory. The same solver fits two independent image subsets.
3. `ground.py` and `align.py` estimate road height from a photometric sweep, vehicle heading from translation, lateral centre from the side mounts, and axle location from turns subject to the rear overhang bound.
4. `scale.py` freezes the rig, discards telemetry poses and speeds, and reconstructs an unseen departure twice from arbitrary initial spacings. Telemetry is only the comparison standard, not a scale input to those solves.
5. `deliver.py` writes the PLAN schema, native-resolution fit and uncertainty tables, masked rectifications, and cross-lens overlays with the target lens excluded from triangulation.

## Another car, including HW3

The cameras come from the footage: HW4 records six, HW3 four (front, back and the two repeaters, 1280x960). The solver starts each lens and mount from the values in [../common/hardware.py](../common/hardware.py). **Two cars have been calibrated this way: the author's HW4 Model 3 on one drive with hand-picked windows, and one HW3 car on two drives with the automatic windows below.** Read `calib/report.md` critically before using a rig from any other car.

On six minutes of the HW3 car's footage the report gave 0.57 to 0.90 px RMS per lens, 3.8 mm of lateral residual in the rear-axle fit, halves that agree within 2.8 cm and 0.09 degrees, and a held-out 15 m measured from images alone within 0.14 % of the odometer. Starting the solver from focal lengths 10 to 16 % off moved no mount by more than 6 mm. A second drive of the same car six weeks later, in rain, calibrated on its own, agreed with the first: every mount within 5 cm (4 cm of it a height shift common to all four cameras, the rest at most 3.2 cm), every angle within 0.3 degrees, every focal length within 0.5 %. The first drive's rig, frozen, also reproduced distance on 21 stretches of the second drive driven at 5 to 19 m/s: 1.2 % short with 1.1 % of scatter and no trend over speed, which is the test of the cameras' timing against each other.

Two things had to be fixed in ingest first, and it now does both by itself: HW3's back camera has a frame counter of its own (up to a frame a minute fast or slow, and 30 instead of 36 fps at a standstill), and HW3's telemetry heading is about 0.6 s old. Either one left in shows in the results. With the late heading, the first drive gave 152 mm of lateral residual and mounts up to 4.5 cm away. With the back camera timed from its counter on a drive without a stop, the second drive gave mounts 6 to 10 cm off along the car and a frozen-rig distance error that grew with speed.

1. Ingest the drive without the runner (the runner needs a rig to exist): put each minute in its own folder, `data/clips/<clip>/<clip>-<camera>.mp4`, and run `python pipeline/ingest/ingest.py data` from the root.
2. Draw the body masks. `python pipeline/calib/features.py --median` writes `calib/debug/median-<camera>.jpg`, the temporal median of moving frames, in which the road blurs away and the car's own bodywork stays sharp. Outline the bodywork (and any dark lens border) in `calib/mask-polygons.json`:

   ```json
   {"space": [600, 450],
    "left_repeater": {"drop": [[[0, 0], [150, 0], [60, 450], [0, 450]]]},
    "back": {"keep": [[[5, 5], [595, 5], [595, 330], [5, 330]]]}}
   ```

   `drop` polygons are removed from the image; `keep` polygons, when given, are the only area kept. `space` is the size the coordinates were drawn in (the size of the median sheets), default `[1, 1]` for image fractions. Without this file, footage that is not HW4 loses only its three-pixel border, and the car's body ends up in the solve and in every later stage.
3. Choose the keyframe windows with `CALIB_SEGS`. Unset, HW4 footage uses the windows picked by hand for the author's drive, and anything else uses `auto`: from telemetry, the straightest 16 m as the held-out scale check (driven at 13 m/s or less when the drive has such a stretch: faster, the lenses share too few landmarks) and up to five 60 m stretches with the most turning as training. Set `CALIB_SEGS=auto` for your own HW4 car, or pass your own as JSON, `[[start_s, end_s, spacing_m], ...]`, scale check last. The drive needs turns: they are what locate the mounts.
4. `RS=$PWD bash pipeline/calib/run.sh`, then read `calib/report.md` and look at `calib/debug/`.
5. The runner's default `--rig verify` compares each drive with thresholds measured on the rig it was built for, so with a new rig either run with `--rig stored`, or rebuild them from three stretches of the calibrated drive that the solve did not use: `RS_INSTALL=$PWD PYTHONPATH=$PWD .venv/bin/python pipeline/run/calibration.py baseline T0 T1 T0 T1 T0 T1` from the root. That writes `calib/rig-baseline.json`, beside the rig it belongs to, which the runner reads in preference to the author's [../run/rig-baseline.json](../run/rig-baseline.json). Without pillar cameras the check says nothing about the front lens: within a 16 m stretch no other lens sees what it sees.

What differs without pillar cameras: no lens overlaps the front one at the same instant, so the front is tied to the rest by the car's motion, plus whatever features survive from a front frame to a repeater frame 5 to 20 keyframes later. On the HW3 car that was 700 to 1200 landmarks against 20000 to 35000 for the other lenses, and the front's cross-camera error was correspondingly the largest (median 2.6 to 3.1 px). The report's cross-camera table shows 0 transfers for a lens that shares no landmark with another.

The rear-axle fit keeps the back lens within 0.977 m of the axle, the rear overhang of a Model 3 ([align.py](align.py)). On a longer car, change that bound.

## On Windows

Ingest and calibration run natively on Windows (the HW3 car above was calibrated that way); the runner, `calibration.py baseline` included, needs macOS or Linux (it uses POSIX file locks and symlinks), though its trajectory, road, keyframe and photometry stages have run on Windows with each stage's code called directly.

1. A conda environment from conda-forge: `conda create -p <env> --override-channels -c conda-forge python=3.12 numpy=1.26.4 scipy opencv=4.10.0 av cmake ninja "ceres-solver=*=cpu*"`. Left to itself conda picks the CUDA build of Ceres, which does not configure without a matching CUDA toolkit.
2. Visual Studio Build Tools with the C++ workload, with `vcvars64.bat` loaded into the shell you run from (the "x64 Native Tools" prompt does that).
3. Put `<env>`, `<env>\Library\bin` and `<env>\Scripts` on `PATH`, set `CMAKE_GENERATOR=Ninja`, `CMAKE_PREFIX_PATH=<env>\Library`, `PY=<env>\python.exe` and `RS`, and run `pipeline/calib/run.sh` with Git Bash. Ingest is `%PY% pipeline\ingest\ingest.py %RS%\data`.
