# FusionTrack C++ core

A dependency-free C++17 port of the single-target tracking core in
[`src/ekf.py`](../src/ekf.py): a linear Kalman filter (`KfTracker`) and a true
Extended Kalman filter with native polar radar (`EkfTracker`). This is the layer
that would run in a real-time deployment; the Python side stays for simulation,
plotting, and the multi-object tracker.

## Why a separate C++ implementation

- **Deterministic, allocation-free.** Every matrix is fixed-size
  (`std::array` storage in [`linalg.hpp`](include/fusiontrack/linalg.hpp)); the
  filter never touches the heap, which is what a real-time autonomy payload
  needs. The only inverse the Kalman update requires is the 2x2 innovation
  covariance `S`, so it is done in closed form.
- **Joseph-form covariance update** `P = (I - KH) P (I - KH)^T + K R K^T`, which
  preserves symmetric positive-definiteness better than the naive `(I - KH) P`
  and matches the Python `filterpy` reference exactly.
- **Bit-for-bit faithful to the reference.** The cross-validation script drives
  identical measurements through both implementations and asserts agreement.
  Current result: **max state difference 5.7e-14 over 60 frames** (machine
  precision).

## Layout

| Path                              | Contents                                                        |
| --------------------------------- | --------------------------------------------------------------- |
| `include/fusiontrack/linalg.hpp`  | Fixed-size matrix ops, 2x2 inverse, symmetric-2x2 eigensolver   |
| `include/fusiontrack/ekf.hpp`     | Tuning constants, `KfTracker`, `EkfTracker` interfaces          |
| `src/ekf.cpp`                     | F/Q, `h(x)`, analytic Jacobian, angle-wrapping residual, update |
| `tests/test_ekf.cpp`              | 10 unit tests (predict, Jacobian vs finite-diff, wrap, gating)  |
| `tests/crossvalidate.py`          | C++ vs Python `filterpy` equivalence check                      |
| `bindings/pymodule.cpp`           | Optional pybind11 module `fusiontrack_cpp`                       |

## Build and test

```bash
cmake -S . -B build
cmake --build build -j
ctest --test-dir build --output-on-failure
```

### Optional: Python bindings + cross-validation

```bash
pip install pybind11
cmake -S . -B build -DFUSIONTRACK_PYBIND=ON \
  -Dpybind11_DIR=$(python -c "import pybind11; print(pybind11.get_cmake_dir())")
cmake --build build -j
python tests/crossvalidate.py   # run from repo root: python cpp/tests/crossvalidate.py
```

## Design notes

The `# INTERVIEW CRITICAL` reasoning from the Python core carries over
verbatim: the radar Jacobian is evaluated at the *predicted* state (linearization
error grows for large steps / sharp turns → a UKF avoids this), the azimuth
residual must be wrapped to `(-pi, pi]`, and the polar `R = diag(sigma_r^2,
sigma_theta^2)` is physically derived rather than hand-tuned. See
[`docs/ekf_explainer.md`](../docs/ekf_explainer.md) for the full matrix
cheat-sheet.
