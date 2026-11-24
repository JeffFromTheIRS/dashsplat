#!/usr/bin/env bash
set -euo pipefail
RS="${RS:-$HOME/dev/reality-splat}"
export RS OPENBLAS_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1
cd "$RS"
P="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"; PY="${PY:-$RS/.venv/bin/python}"; BUILD="$RS/work/calib/build"
cmake -S "$P" -B "$BUILD" -DCMAKE_BUILD_TYPE=Release
cmake --build "$BUILD" -j4
"$PY" "$P/features.py"
for split in all A B; do "$PY" "$P/solve.py" "$split"; done
"$PY" "$P/ground.py"
"$PY" "$P/scale.py"
"$PY" "$P/deliver.py"
