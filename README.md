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

Unit tests:

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
| `src/ekf.py` | `EKFTracker` — `predict`, `update_camera`, `update_radar`, covariance / ellipse |
| `src/radar_sim.py` | Polar returns with Gaussian noise, misses, false alarms |
| `src/camera_sim.py` | Pixel centers with noise, stochastic misses, occlusion window |
| `src/fusion.py` | Synthetic trajectory, three parallel filters (fused / camera-only / radar-only), plots |
| `src/utils.py` | Pixel ↔ world scaling, polar ↔ Cartesian |
| `docs/ekf_explainer.md` | Matrix cheat-sheet (placeholders for your interview notes) |

## Note on “EKF” vs linear Kalman

The process and measurement models in **Cartesian world space** are **linear** in this scaffold: both sensors eventually provide a **position** in $(x, y)$ with diagonal (or block) $R$. True **polar** radar with linearization or a full **EKF/UKF** in the native $(r, \theta)$ measurement space is the natural next step for production accuracy; see `# INTERVIEW CRITICAL` comments in `src/ekf.py` and `src/utils.py`.
