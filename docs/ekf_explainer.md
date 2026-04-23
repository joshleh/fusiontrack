# Linear Kalman (CV) + fusion: matrix cheat‑sheet (fill in your interview voice)

> **Placeholder doc**: complete each [TODO: …] in your own words before sharing externally.

## State $x$ — why four numbers?

* **Vector**: $x = [x, y, v_x, v_y]^T$ (world frame, 2D).
* [TODO: 2–3 sentences: why you separate position and velocity; what breaks if you only track $x, y$ with differencing.]

## Motion model — matrix $F$

* [TODO: write the discrete $F$ you use for time step $dt$ and name the *constant-velocity* assumption.]
* [TODO: one sentence: when a UAV yaws and accelerates, what does this model miss, and what knob compensates?]

## Process noise $Q$

* [TODO: what physical process is approximated as white random acceleration, and what does $\sigma_a$ stand for?]
* [TODO: in one line: *increasing* $Q$ means the filter tends to *trust the … more / less* because … (finish).]

## Measurement $z$ and $H$

* [TODO: why $H$ only picks out position for both sensors in this build.]
* [TODO: why camera noise is in pixels first, and how the Jacobian / linear mapping in `utils` turns that into $R$ in meters.]

## Camera vs radar: two $R$ matrices

* [TODO: numerically compare expected axis errors: ~8–15 px vs ~3–5 m after your conversion — is one sensor always “best”? in what region of the map?]
* [TODO: **INTERVIEW CRITICAL**: *polar to Cartesian* after independent Gaussian noise in $(r, \theta)$ is not actually Gaussian in $(x, y)$ — when is a linear KF in $(x, y)$ still defensible, and when do you use an EKF/UKF in the polar update?]

## Covariance $P$

* [TODO: interpret an eigenvector of the $2 \times 2$ position block as “direction of most uncertainty” with a one-sentence picture.]
* [TODO: what $\mathrm{tr}(P_{xy})$ measures in your fusion plots, and what it *doesn’t* measure.]

## Innovation (optional section)

* [TODO: define $y = z - H \hat x^-$, name $S = H P^- H^T + R$, and when Mahalanobis gating is used in tracking interviews.]

## References to cite out loud (books / papers, not blog posts if you can avoid it)

* [TODO: e.g. Bar-Shalom, Rong Li, Kirubarajan — *Estimation with Applications to Tracking and Navigation*; Thrun/Burgard/Fox — *Probabilistic Robotics* — your pick.]
