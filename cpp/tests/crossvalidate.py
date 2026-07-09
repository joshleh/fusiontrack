"""Cross-validate the C++ tracking core against the Python reference.

Runs an identical predict/update sequence through both the C++ `EkfTracker`
(compiled via pybind11) and the Python `EKFTracker` (filterpy) and asserts the
posterior state agrees to a tight tolerance. This is the guarantee that the
C++ port is a faithful reimplementation, not an independent approximation.

Usage:
    cmake -S cpp -B cpp/build -DFUSIONTRACK_PYBIND=ON \
        -Dpybind11_DIR=$(python -c "import pybind11; print(pybind11.get_cmake_dir())")
    cmake --build cpp/build
    python cpp/tests/crossvalidate.py
"""
from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "cpp" / "build"))

import fusiontrack_cpp as cpp  # noqa: E402
from src.ekf import EKFTracker  # noqa: E402

TOL = 1e-9


def main() -> int:
    rng = np.random.default_rng(0)
    x0 = np.array([120.0, 30.0, 4.0, -1.5], dtype=np.float64)

    cpp_ekf = cpp.EkfTracker(list(x0), 1.0)
    py_ekf = EKFTracker(x0.copy(), dt=1.0)

    max_diff = 0.0
    for _ in range(60):
        cpp_ekf.predict()
        py_ekf.predict()

        # Alternate a camera (x, y) update and a polar radar update, using the
        # same simulated measurements for both implementations.
        py_state = py_ekf.get_state()
        cam = py_state[:2] + rng.normal(0.0, 2.0, size=2)
        cpp_ekf.update_camera(list(cam))
        py_ekf.update_camera(cam.copy())

        px, py = py_ekf.get_state()[:2]
        rng_m = math.hypot(px, py) + rng.normal(0.0, 3.0)
        az = math.atan2(py, px) + rng.normal(0.0, math.radians(0.5))
        z_polar = np.array([rng_m, az], dtype=np.float64)
        cpp_ekf.update_radar_polar(list(z_polar))
        py_ekf.update_radar_polar(z_polar.copy())

        diff = float(np.max(np.abs(np.array(cpp_ekf.state()) - py_ekf.get_state())))
        max_diff = max(max_diff, diff)

    print(f"max |C++ - Python| state difference over 60 frames: {max_diff:.3e}")
    if max_diff <= TOL:
        print("PASS: C++ core matches the Python reference.")
        return 0
    print(f"FAIL: difference exceeds tolerance {TOL:.0e}")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
