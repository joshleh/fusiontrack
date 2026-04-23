# EKF + MOT: matrix cheat-sheet and interview notes

Reference for every design decision in FusionTrack. Read alongside the source code — each section maps to a `# INTERVIEW CRITICAL` comment.

---

## 1. State vector — why four numbers?

$$x = [x,\; y,\; v_x,\; v_y]^T \in \mathbb{R}^4$$

Tracking only $[x, y]$ forces velocity estimation by finite-differencing consecutive positions, which amplifies measurement noise and lags one frame behind. Carrying velocity in the state lets the prediction step propagate the estimate forward with a *consistent* joint distribution over position and velocity — so the predicted position already encodes the current speed. This also makes the covariance meaningful: off-diagonal blocks $P_{xv}$ capture the correlation between position uncertainty and velocity uncertainty, which a two-element state cannot represent at all.

---

## 2. Motion model — matrix $F$ (constant-velocity)

For time step $dt$ the discrete constant-velocity $F$ is:

$$F = \begin{bmatrix}1 & 0 & dt & 0 \\ 0 & 1 & 0 & dt \\ 0 & 0 & 1 & 0 \\ 0 & 0 & 0 & 1\end{bmatrix}$$

**What it assumes:** zero acceleration between frames — "the target keeps flying straight." For a UAV in level cruise this is excellent. For a banking turn or climb, the true position diverges from the CV prediction *quadratically* in $dt$, and the filter compensates through $Q$ (see §3).

**What it misses:** coordinated turns, wind gusts, speed changes. A constant-turn model (CT) or interacting multiple models (IMM) are the next rung up when maneuver statistics are known.

---

## 3. Process noise $Q$ — the maneuver budget

The standard "discretized white acceleration" model (Bar-Shalom §6.3):

$$G = \begin{bmatrix}dt^2/2 & 0 \\ 0 & dt^2/2 \\ dt & 0 \\ 0 & dt\end{bmatrix}, \qquad Q = \sigma_a^2 \; G G^T$$

$\sigma_a$ is the root-mean-square acceleration the CV model cannot predict (m/s²). In this simulator $\sigma_a = 0.25$ m/s².

**Physical meaning:** raising $\sigma_a$ inflates the *predicted* covariance $P^-$ before each measurement, which increases the Kalman gain $K = P^- H^T S^{-1}$, so the filter trusts fresh measurements *more* and the prior prediction *less*. The price is faster reaction to clutter and higher steady-state position noise. Lowering $\sigma_a$ makes the filter "stiff": it ignores noisy blips but also takes longer to pull back after a real maneuver.

**Interview phrasing:** "$Q$ encodes my belief about how much the world can surprise me between frames. Bigger $Q$ = more surprised = more gain."

---

## 4. Measurement matrix $H$ — position only

$$H = \begin{bmatrix}1 & 0 & 0 & 0 \\ 0 & 1 & 0 & 0\end{bmatrix}$$

Both sensors deliver only position $(x, y)$ in world meters — neither gives a direct velocity measurement. So $H$ is the same for both sensors on the linear KF path. The velocity is estimated indirectly through the temporal dynamics of consecutive position measurements and the cross-correlation terms in $P$.

A Doppler radar would add a third measurement row observing $\frac{\partial r}{\partial t} = \frac{x v_x + y v_y}{r}$ — nonlinear in state, requiring the full EKF Jacobian.

---

## 5. Camera noise → world $R$

The simulator adds independent Gaussian pixel noise $\sigma_u = \sigma_v = 8\,\text{px}$. The orthographic map $u_m = u_{px} \cdot m/\text{px}$ converts this to world:

$$R_{\text{cam}} = \text{diag}\!\left[(\sigma_u \cdot m)^2,\; (\sigma_v \cdot m)^2\right] = \text{diag}[16,\; 16]\;\text{m}^2$$

so camera is ~4 m 1-σ per axis.

**In a real system** $R_{\text{world}}$ is altitude- and gimbal-dependent. The full derivation uses the projective Jacobian $\frac{\partial(x,y)}{\partial(u,v)}$ evaluated at the current altitude and camera orientation. This linear approximation holds only for near-nadir imagery.

---

## 6. Radar: Cartesian approximation (`KFTracker`)

Radar noise is physically in $(r, \theta)$: range noise $\sigma_r = 3$ m, azimuth noise $\sigma_\theta = 0.5°$. After converting to world Cartesian:

$$x_{\text{meas}} = (r + \varepsilon_r)\cos(\theta + \varepsilon_\theta) \approx r\cos\theta + \varepsilon_r\cos\theta - r\sin\theta\,\varepsilon_\theta$$

The cross-range component $r\sin\theta\,\varepsilon_\theta$ grows **linearly with range**. At $r = 300$ m and $\sigma_\theta = 0.5°$:

$$\sigma_{\text{cross}} = r\,\sigma_\theta = 300 \times 0.00873 \approx 2.6\;\text{m}$$

`KFTracker` uses a fixed isotropic $R_{\text{cart}} = \text{diag}[4, 4]\;\text{m}^2$ (≈ 2 m 1-σ). This is a known modeling lie: the error ellipse is actually elongated in the range direction at close range and in cross-range at long range. It is defensible when the target is never far from the sensor boresight and the approximation error is small relative to $P$.

---

## 7. Radar: native polar EKF (`EKFTracker`) — the correct path

**Forward model (nonlinear):**
$$h(x) = \begin{bmatrix}\sqrt{x_0^2 + x_1^2} \\ \operatorname{atan2}(x_1, x_0)\end{bmatrix}$$

**2×4 Jacobian** evaluated at the predicted state $x^-$:

$$H_{\text{jac}} = \begin{bmatrix}\dfrac{x_0}{r} & \dfrac{x_1}{r} & 0 & 0 \\[6pt] -\dfrac{x_1}{r^2} & \dfrac{x_0}{r^2} & 0 & 0\end{bmatrix}, \qquad r = \sqrt{x_0^2 + x_1^2}$$

**Polar $R$** (physical units, no approximation):
$$R_{\text{polar}} = \text{diag}[\sigma_r^2,\; \sigma_\theta^2] = \text{diag}[9.0,\; 7.6\times10^{-5}]$$

The innovation covariance $S = H_{\text{jac}} P^- H_{\text{jac}}^T + R_{\text{polar}}$ is a 2×2 matrix in polar space that correctly captures the range-dependent anisotropy via the Jacobian — no free parameters, just physics.

**Why this is better:** the Cartesian approximation smears anisotropic polar noise into an isotropic world blob and requires tuning $R_{\text{cart}}$ by hand. The EKF $R_{\text{polar}}$ is derived directly from the sensor spec.

**Linearization error:** the Jacobian is evaluated at $x^-$ (the predicted state), not the true state. The linearization error grows as the prediction step gets large or the trajectory curves sharply — exactly the regime where a UKF (unscented transform) is preferred.

### Angle wrapping (mandatory)

Innovation: $y = z - h(x^-)$ includes an azimuth difference. Without wrapping, a target at azimuth $+\pi - \epsilon$ measured at $-\pi + \epsilon$ produces $y_\theta \approx 2\pi$ — a Kalman gain-scaled position shift of ~600 m. The fix:

```python
y[1] = (y[1] + π) % (2π) - π   # normalize to (-π, π]
```

**The same bug appears in:** GPS/compass fusion (heading near ±180°), bearing-only SLAM, quaternion attitude EKF (slerp vs additive, different fix). If you see a tracker that occasionally shoots off to the horizon, this is usually why.

---

## 8. Covariance $P$ — geometry and meaning

The $2\times2$ position block of $P$ after an update:

$$P_{xy} = \begin{bmatrix}\sigma_x^2 & \sigma_{xy} \\ \sigma_{xy} & \sigma_y^2\end{bmatrix}$$

Its eigenvectors point along the principal axes of the uncertainty ellipse; the eigenvalues give the squared semi-axes. The 95% ellipse has semi-axes $a_i = \sqrt{\lambda_i \cdot \chi^2_{2,0.95}}$ where $\chi^2_{2,0.95} = 5.991$.

**tr(P$_{xy}$)** = $\sigma_x^2 + \sigma_y^2$ — a scalar "total spread" used in the fusion plots. It is *not* the area of the ellipse (area $= \pi a_1 a_2 = \pi \sqrt{\det P_{xy}} \cdot \chi^2$ scale). Two filters can have equal trace but very different ellipse shapes (one elongated vs. circular) — the full matrix matters.

**NEES (Normalized Estimation Error Squared):** the proper consistency check. $\text{NEES} = (x - \hat x)^T P^{-1} (x - \hat x)$ should be chi-square distributed with $d$ DOF for a consistent filter. A NEES that trends upward means $P$ is *optimistic* (filter thinks it's better than it is). This repository does not compute NEES — it's the natural next diagnostic step.

---

## 9. Innovation gating — chi-square filter

**Innovation:** $y = z - H x^-$  
**Innovation covariance:** $S = H P^- H^T + R$  
**Mahalanobis distance²:** $d^2 = y^T S^{-1} y$  
**Gate threshold:** $d^2 < \chi^2_{2,0.99} = 9.21$ (99% confidence, 2 DOF)

**What the gate does:** a measurement outside the gate is *not* ingested. Without gating, a single false-alarm 200 m away with Mahalanobis distance 400 would still update the track (just with a small gain), slowly biasing the state and inflating the residual. With gating, the filter never sees it.

**Tuning the gate:**
- Too tight (e.g., 95% → 5.99): valid targets at long prediction intervals gated out, tracks go coast-only and drift.
- Too loose (e.g., 99.99%): clutter passes the gate and pollutes the track update.
- In practice, tune against a NEES vs. gate-miss-rate curve on representative data.

---

## 10. Multi-object data association

### GNN (this implementation)

1. Build cost matrix: $C_{ij} = d^2(\text{track}_i, \text{meas}_j)$ if within gate else $\infty$.
2. Hungarian algorithm: globally optimal one-to-one assignment, $O(n^3)$.
3. Commit: each track updates on exactly one measurement (or none if all gated).

**Failure mode:** at the crossing (frame 49), Tracks 1 and 2 are within each other's gate. GNN commits to the globally optimal assignment — correct if clutter is low. With even moderate clutter, a false alarm inside both tracks' gates can steal the optimal assignment and cause an ID switch.

### JPDA (Joint Probabilistic Data Association)

Computes the probability that each measurement originated from each track, then updates every track with the *expected-value* measurement (weighted sum over all hypotheses). No hard commitment. Cost: exponential in the number of tracks × measurements per frame without approximation; practical implementations cap the number of feasible events.

**When to use:** targets within each other's gate + moderate clutter. For UAV surveillance in clear weather, GNN is sufficient; for maritime or ground clutter, JPDA is standard.

### MHT (Multiple Hypothesis Tracking)

Maintains a tree of all feasible measurement-to-track assignment histories, pruning by likelihood and N-scan depth. Handles long occlusions and uncertain births correctly. The complexity is exponential before pruning; Reid's 1979 paper and Blackman's "Multiple Target Tracking with Radar Applications" (1986) are the canonical references.

**When to use:** safety-critical applications, long-range sensors, highly cluttered environments. Production UAV defense trackers (like what Anduril builds) use variants of MHT.

---

## 11. Interview checklist

- **Write $F$ on a whiteboard:** for $dt=1$ s, $x \leftarrow x + v_x$, $y \leftarrow y + v_y$, velocities constant. Predict updates $P \leftarrow F P F^T + Q$.
- **What does raising $\sigma_a$ do?** Inflates $Q$ → inflates $P^-$ → increases $K$ → more trust in fresh measurements vs. prior.
- **Why is Cartesian radar $R$ a "lie"?** Cross-range error is $r\,\sigma_\theta$ — grows with range. A fixed isotropic $R$ is wrong at long range.
- **When does angle wrapping bite you?** Any bearing innovation near ±180°. Same pattern in SLAM, GPS/compass, quaternion EKF.
- **$\text{tr}(P)$ vs. NEES?** $\text{tr}(P)$ is the filter's own self-assessment. NEES uses ground truth to check whether $P$ is honest.
- **Why GNN and not greedy?** Greedy (assign each measurement to its nearest track in sequence) is order-dependent and can create unnecessary ID switches. Hungarian is order-independent and minimizes total cost.
- **Why GNN and not JPDA?** At low clutter density and well-separated targets, GNN is optimal in expectation. JPDA is necessary when targets are within each other's gate in significant clutter — you pay $O(2^n)$ to avoid ID switches.
- **What is a track's "tentative" phase for?** Clutter rejection. A false alarm fires at most one unmatched measurement; a real target fires consistently. Requiring 2+ hits before confirming eliminates most point clutter without multi-hypothesis tracking.

---

## 12. References

- Bar-Shalom, Rong Li, Kirubarajan — *Estimation with Applications to Tracking and Navigation* (2001, Wiley-IEEE) — the definitive KF/EKF/IMM reference; chapters 4–6 cover everything above.
- Thrun, Burgard, Fox — *Probabilistic Robotics* (2005, MIT Press) — chapter 3 for EKF derivation; chapter 6 for SLAM bearing-range update.
- Blackman & Popoli — *Design and Analysis of Modern Tracking Systems* (1999, Artech House) — Hungarian, JPDA, MHT in tracking context.
- Reid — "An Algorithm for Tracking Multiple Targets" (1979, IEEE TAC) — original MHT paper.
- `scipy.optimize.linear_sum_assignment` — implements the Jonker-Volgenant algorithm ($O(n^3)$, faster than classical Hungarian in practice).
- `scipy.stats.chi2.ppf(0.99, df=2)` → 9.210 — the gate threshold used here.
