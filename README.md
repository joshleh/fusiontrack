# FusionTrack

Multi-sensor UAV tracking in 2D by fusing **simulated radar** (range, azimuth with miss and false-alarm behavior) and **simulated camera** (noisy bounding-box centers in pixels) through a **constant-velocity Kalman filter** with separate **camera** and **radar** measurement noise matrices. This repository is a portfolio-quality **math + simulation** slice: no API server, no Docker, no experiment tracking yet.

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

Unit tests (seven cases covering `predict`, L2 after update, trace reduction, repeated updates, PSD covariance, ellipse):

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
| `src/ekf.py` | `KFTracker` (linear KF) — `predict`, `update_camera`, `update_radar`, covariance / ellipse |
| `src/radar_sim.py` | Polar returns with Gaussian noise, misses, false alarms |
| `src/camera_sim.py` | Pixel centers with noise, stochastic misses, occlusion window |
| `src/fusion.py` | Synthetic trajectory, three parallel filters (fused / camera-only / radar-only), plots |
| `src/utils.py` | Pixel ↔ world scaling, polar ↔ Cartesian |
| `docs/ekf_explainer.md` | Matrix cheat-sheet (placeholders for your interview notes) |

## Naming: linear KF, not an EKF (yet)

The **class** is `KFTracker` and uses `filterpy.kalman.KalmanFilter` — **linear** constant-velocity dynamics and linear position measurements in a shared world $(x, y)$ frame. The filename `ekf.py` is kept as a short handle; a true **Extended** Kalman update in native polar $h(x)$ (or a UKF) is a natural follow-on so the colloquial “EKF fusion” in READMEs matches the implementation.

Sensors: camera $R$ comes from **pixel** noise (weaker, ~4 m 1-σ in world for the default sim); radar $R$ is set **tighter** (~2 m 1-σ) so fusion has a defensible *sensor weighting* story. See `src/fusion.RADAR_MEASUREMENT_VAR_M2` and comments there.

The **polar** simulator still draws noise in $(r, \theta)$; mapping to world with a diagonal $R$ is the usual interview trade-off: see `# INTERVIEW CRITICAL` in `src/ekf.py` and `src/utils.py`.
