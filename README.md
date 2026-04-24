# FusionTrack

[![Site preview](site_preview.png)](https://joshleh.github.io/fusiontrack)

Multi-sensor UAV tracking in 2D — single-object EKF fusion and **multi-object tracking** with Hungarian data association and Mahalanobis gating. Fuses **simulated radar** (range, azimuth, misses, false alarms) and **simulated camera** (noisy pixels, structured occlusion) through a linear KF and a true Extended Kalman Filter with native polar measurements.

## What to run

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

Single-object EKF fusion (trajectory + error figures, occlusion shading):

```bash
cd /path/to/fusiontrack
python -m src.fusion
```

Multi-object tracker demo — prints metrics, shows static plot, and saves `mot_demo.gif`:

```bash
python -m src.mot
```

Unit tests (25 cases — KFTracker, EKFTracker, TrackerManager lifecycle/gating):

```bash
pytest tests/ -q
```

Pedagogical notebook (step-by-step prints, inline plots, cov-trace panel):

```bash
jupyter notebook notebooks/01_fusion_demo.ipynb
```

## Layout

| Path | Role |
|------|------|
| `src/ekf.py` | `KFTracker` (linear KF) and `EKFTracker` (polar EKF) — predict, update, covariance, ellipse, `compute_innovation_polar` for gating |
| `src/radar_sim.py` | Polar returns with Gaussian noise, misses, false alarms |
| `src/camera_sim.py` | Pixel centers with noise, stochastic misses, occlusion window |
| `src/fusion.py` | Single-object demo: 4 parallel trackers (EKF fused / KF fused / camera-only / radar-only) |
| `src/mot.py` | **Multi-Object Tracker** — `TrackerManager`, `Track`, `TrackState`; GNN + Mahalanobis gating; `compute_mot_metrics`; demo + plot + GIF animation |
| `src/mot_sim.py` | 3-target crossing scenario + Poisson clutter generation |
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

## Multi-Object Tracking (MOT)

`src/mot.py` implements a **Global Nearest-Neighbour (GNN)** tracker on top of the EKFTracker:

| Component | Implementation | Interview depth |
|-----------|---------------|-----------------|
| Gate | Mahalanobis distance² in polar measurement space: $d^2 = y^T S^{-1} y$, $S = H P H^T + R_{\text{polar}}$; chi-square 99% threshold (2 DOF ≈ 9.21) | Too tight → valid targets lost; too loose → clutter absorbed |
| Assignment | `scipy.optimize.linear_sum_assignment` (Hungarian / Jonker-Volgenant, O(n³)) | JPDA maintains soft weights; MHT maintains a hypothesis tree |
| Birth | Unmatched measurement → TENTATIVE track with zero velocity, large $P_v$ | One detection never confirms — clutter blips die in ≤3 frames |
| Death | CONFIRMED track with ≥3 consecutive misses → DELETED | Tuning max_misses trades ID switches for ghost tracks |
| Scenario | 3 straight CV targets that converge within ~9 m of each other at frame 49 (crossing stress-test) + Poisson(0.5) clutter per frame | At the crossing: two tracks are inside each other's chi-square gate; Hungarian finds the globally optimal one-shot assignment |

**What GNN cannot handle well:** closely spaced crossing targets in high clutter (JPDA), track fragmentation after long occlusion (MHT), multi-sensor joint measurement origin (probabilistic FISST). These are the natural follow-ons to this implementation.

## Results

All numbers from `run_mot_demo(rng=default_rng(42))`, 100 frames, Poisson(0.5) clutter/frame.

### Multi-object tracker (3 crossing targets)

| Metric | Value |
|--------|-------|
| **ID switches through crossing (frame 49)** | **0** |
| Confirmed tracks matching GT trajectories | 3 (track IDs 2, 4, 10; lifetimes 82–100 frames) |
| Per-target position RMSE | 2.51 m / 2.56 m / 2.62 m |
| Mean RMSE (GT-matched tracks, lifetime ≥ 50 frames) | **2.57 m** |
| Clutter tracks born and pruned | several (lifetime ≤ 15 frames each, by tentative-age rule) |

The zero ID-switch result reflects the crossing-scenario design: Targets 1 & 2 pass within ~9 m at frame 49 (within each other's chi-square gate), but Target 3 stays ~40 m clear. The Hungarian globally optimal assignment distinguishes the two close targets without a swap. To observe ID switches, increase `clutter_rate` or halve the gate threshold — both degrade GNN before needing JPDA.

The RMSE of ~2.5 m is consistent with the physical noise: $\sigma_r = 3$ m, $\sigma_\theta = 0.5°$, giving a cross-range error of $r\,\sigma_\theta \approx 2.3$ m at the typical slant range of ~270 m for these trajectories.

The demo animation (`mot_demo.gif`, generated by `python -m src.mot`) shows track birth, ellipse convergence, the crossing at frame 49, and clutter pruning in real time.

### Single-object EKF vs. linear KF (100-frame camera+radar scenario, occlusion frames 40–60)

The EKF's polar $R = \text{diag}(\sigma_r^2, \sigma_\theta^2)$ is physically derived; the KF's Cartesian $R$ is a hand-tuned approximation. Both perform comparably on straight-line motion at short range — the EKF advantage grows at long range or high-azimuth offsets where cross-range error scales with $r\,\sigma_\theta$.
