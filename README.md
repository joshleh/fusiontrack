# FusionTrack

Multi-sensor UAV tracking in 2D by fusing **simulated radar** (range, azimuth with miss and false-alarm behavior) and **simulated camera** (noisy bounding-box centers in pixels) through both a **linear KF** (Cartesian radar) and a **true Extended Kalman Filter** (native polar radar with analytic Jacobian). This repository is a portfolio-quality **math + simulation** slice: no API server, no Docker, no experiment tracking yet.

## What to run

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

End-to-end fusion with two Matplotlib figures (trajectory + error, with occlusion shading):

```bash
cd /path/to/fusiontrack
python -m src.fusion
```

Unit tests (14 cases: KFTracker predict/update/PSD/ellipse + EKFTracker polar update, angle wrapping, covariance calibration):

```bash
pytest tests/ -q
```

Pedagogical notebook (step-by-step prints and occlusion discussion):

```bash
jupyter notebook notebooks/01_fusion_demo.ipynb
```

## Layout

| Path | Role |
|------|------|
| `src/ekf.py` | `KFTracker` (linear KF) and `EKFTracker` (polar EKF) — predict, update, covariance, ellipse |
| `src/radar_sim.py` | Polar returns with Gaussian noise, misses, false alarms |
| `src/camera_sim.py` | Pixel centers with noise, stochastic misses, occlusion window |
| `src/fusion.py` | Synthetic trajectory, four parallel trackers (EKF fused / KF fused / camera-only / radar-only) |
| `src/utils.py` | Pixel ↔ world scaling, polar ↔ Cartesian |
| `docs/ekf_explainer.md` | Matrix cheat-sheet (placeholders for your interview notes) |

## Two trackers: KFTracker vs EKFTracker

`src/ekf.py` now provides **both**:

| Class | Measurement model | Radar update | R matrix |
|-------|------------------|--------------|----------|
| `KFTracker` | Linear $H \mathbf{x}$ for both sensors | Polar → Cartesian first, diagonal world $R$ | ~2 m 1-σ isotropic (tunable knob) |
| `EKFTracker` | Linear $H$ for camera; nonlinear $h(\mathbf{x}) = [r, \theta]$ for radar | **Native polar**, analytic Jacobian, angle-normalizing residual | $\text{diag}(\sigma_r^2, \sigma_\theta^2)$ — physically derived |

The EKF radar update avoids the linearization error that the Cartesian approximation introduces at large azimuth offsets or long range.  Camera measurements stay linear (they are already in world Cartesian after the pixel → metre mapping in `utils`).

**Interview notes (marked `# INTERVIEW CRITICAL` in code)**
- Jacobian is evaluated at the **predicted** state → linearization error grows for large prediction steps or sharp turns → a UKF sigma-point approach avoids this.
- Angle wrapping in the innovation ($y_\theta \in (-\pi, \pi]$) is mandatory whenever a bearing appears; identical issue in SLAM, GPS/compass fusion, quaternion EKF.
- Sequential two-update-per-frame (camera then radar) is *not* equivalent to a single joint update unless measurements are conditionally independent given the state — acceptable for tutorial; production uses gating + MHT.
- `tr(P)` shrinks after a good update, but it is not a geometric “area” — use NEES or the full ellipse for consistency checking.
