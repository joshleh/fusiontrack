"""Adapter exposing the compiled C++ tracking core behind the Python API.

The C++ core (``cpp/``) is built into a pybind11 module ``fusiontrack_cpp``.
This module wraps it so ``CppKFTracker`` / ``CppEKFTracker`` are drop-in
replacements for :class:`ekf.KFTracker` / :class:`ekf.EKFTracker` in
``src/fusion.py``: same constructor keywords, same method names, same return
types (NumPy arrays and :class:`ekf.UncertaintyEllipse2D`).

Build the module first (from the repo root)::

    cmake -S cpp -B cpp/build -DFUSIONTRACK_PYBIND=ON \\
        -Dpybind11_DIR=$(python -c "import pybind11; print(pybind11.get_cmake_dir())")
    cmake --build cpp/build -j

Then ``run_fusion_demo(backend="cpp")`` (or ``python -m src.fusion --backend cpp``)
drives the identical simulation through the C++ filters.
"""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Optional

import numpy as np
from numpy.typing import NDArray

from . import ekf

_REPO_ROOT = Path(__file__).resolve().parents[1]
_BUILD_DIR = _REPO_ROOT / "cpp" / "build"


class CppBackendUnavailable(RuntimeError):
    """Raised when the compiled ``fusiontrack_cpp`` module cannot be imported."""


def _import_module():
    """Import ``fusiontrack_cpp``, adding the CMake build dir to the path if needed."""
    try:
        import fusiontrack_cpp  # type: ignore
        return fusiontrack_cpp
    except ImportError:
        pass
    if _BUILD_DIR.is_dir():
        sys.path.insert(0, str(_BUILD_DIR))
        try:
            import fusiontrack_cpp  # type: ignore
            return fusiontrack_cpp
        except ImportError:
            pass
    raise CppBackendUnavailable(
        "Could not import 'fusiontrack_cpp'. Build it with:\n"
        "  cmake -S cpp -B cpp/build -DFUSIONTRACK_PYBIND=ON "
        "-Dpybind11_DIR=$(python -c \"import pybind11; print(pybind11.get_cmake_dir())\")\n"
        "  cmake --build cpp/build -j"
    )


def is_available() -> bool:
    """Return True if the compiled C++ module can be imported."""
    try:
        _import_module()
        return True
    except CppBackendUnavailable:
        return False


def _flat_r(r: Optional[NDArray[np.float64]]):
    """Flatten a 2x2 covariance to a row-major length-4 list, or None."""
    if r is None:
        return None
    arr = np.asarray(r, dtype=np.float64).reshape(2, 2)
    return [float(arr[0, 0]), float(arr[0, 1]), float(arr[1, 0]), float(arr[1, 1])]


def _ellipse(tup) -> ekf.UncertaintyEllipse2D:
    cx, cy, width, height, angle_deg = tup
    return ekf.UncertaintyEllipse2D(
        center=(float(cx), float(cy)),
        width=float(width),
        height=float(height),
        angle_deg=float(angle_deg),
    )


class CppKFTracker:
    """C++-backed linear Kalman filter, matching :class:`ekf.KFTracker`."""

    def __init__(
        self,
        initial_state: NDArray[np.float64],
        *,
        dt: float = ekf.DEFAULT_DT_S,
        r_camera: Optional[NDArray[np.float64]] = None,
        r_radar: Optional[NDArray[np.float64]] = None,
    ) -> None:
        mod = _import_module()
        self._t = mod.KfTracker([float(v) for v in np.asarray(initial_state).reshape(4)], float(dt))
        self._r_camera = _flat_r(r_camera)
        self._r_radar = _flat_r(r_radar)

    def predict(self) -> None:
        self._t.predict()

    def update_camera(
        self, z_xy: NDArray[np.float64], r_override: Optional[NDArray[np.float64]] = None
    ) -> None:
        r = _flat_r(r_override) if r_override is not None else self._r_camera
        self._t.update_camera([float(z_xy[0]), float(z_xy[1])], r)

    def update_radar(
        self, z_xy: NDArray[np.float64], r_override: Optional[NDArray[np.float64]] = None
    ) -> None:
        r = _flat_r(r_override) if r_override is not None else self._r_radar
        self._t.update_radar([float(z_xy[0]), float(z_xy[1])], r)

    def get_state(self) -> NDArray[np.float64]:
        return np.asarray(self._t.state(), dtype=np.float64)

    def get_covariance(self) -> NDArray[np.float64]:
        return np.asarray(self._t.covariance(), dtype=np.float64).reshape(4, 4)

    def get_position_covariance_2d(self) -> NDArray[np.float64]:
        return self.get_covariance()[0:2, 0:2]

    def get_uncertainty_ellipse(self) -> ekf.UncertaintyEllipse2D:
        return _ellipse(self._t.uncertainty_ellipse())


class CppEKFTracker:
    """C++-backed Extended Kalman filter, matching :class:`ekf.EKFTracker`."""

    def __init__(
        self,
        initial_state: NDArray[np.float64],
        *,
        dt: float = ekf.DEFAULT_DT_S,
        r_camera: Optional[NDArray[np.float64]] = None,
        r_radar_polar: Optional[NDArray[np.float64]] = None,
    ) -> None:
        mod = _import_module()
        self._t = mod.EkfTracker([float(v) for v in np.asarray(initial_state).reshape(4)], float(dt))
        self._r_camera = _flat_r(r_camera)
        self._r_radar_polar = _flat_r(r_radar_polar)

    def predict(self) -> None:
        self._t.predict()

    def update_camera(
        self, z_xy: NDArray[np.float64], r_override: Optional[NDArray[np.float64]] = None
    ) -> None:
        r = _flat_r(r_override) if r_override is not None else self._r_camera
        self._t.update_camera([float(z_xy[0]), float(z_xy[1])], r)

    def update_radar_polar(
        self, z_polar: NDArray[np.float64], r_override: Optional[NDArray[np.float64]] = None
    ) -> None:
        r = _flat_r(r_override) if r_override is not None else self._r_radar_polar
        self._t.update_radar_polar([float(z_polar[0]), float(z_polar[1])], r)

    def compute_innovation_polar(self, z_polar: NDArray[np.float64]):
        y, s = self._t.compute_innovation_polar([float(z_polar[0]), float(z_polar[1])])
        return (
            np.asarray(y, dtype=np.float64).reshape(2, 1),
            np.asarray(s, dtype=np.float64).reshape(2, 2),
        )

    def get_state(self) -> NDArray[np.float64]:
        return np.asarray(self._t.state(), dtype=np.float64)

    def get_covariance(self) -> NDArray[np.float64]:
        return np.asarray(self._t.covariance(), dtype=np.float64).reshape(4, 4)

    def get_position_covariance_2d(self) -> NDArray[np.float64]:
        return self.get_covariance()[0:2, 0:2]

    def get_uncertainty_ellipse(self) -> ekf.UncertaintyEllipse2D:
        return _ellipse(self._t.uncertainty_ellipse())
