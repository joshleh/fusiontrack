// FusionTrack tracking core (C++).
//
// A faithful C++ port of the Python reference in src/ekf.py: single-target 2D
// constant-velocity tracking with two flavours of Kalman filter.
//
//   * KfTracker  - linear KF. Both sensors observe position (x, y). Radar is
//                  assumed pre-converted from polar, so R is a diagonal world
//                  covariance (a modeling approximation).
//
//   * EkfTracker - Extended KF. Radar measurements stay in native polar
//                  (range, azimuth) and pass through the nonlinear h(x), its
//                  analytic 2x4 Jacobian, and an angle-normalizing residual.
//                  Camera stays linear (already in world Cartesian).
//
// State vector: x = [px, py, vx, vy]^T (metres, metres/second).
// Both trackers use the identical constant-velocity process model so they are
// directly comparable, exactly as in the Python demo.
//
// The covariance update uses the Joseph (symmetric) form
//   P = (I - K H) P (I - K H)^T + K R K^T
// matching filterpy and preserving symmetric positive-definiteness better than
// the naive (I - K H) P form.
#pragma once

#include "fusiontrack/linalg.hpp"

namespace ft {

// --- Kinematic and noise tuning (mirrors src/ekf.py; no magic numbers) -----
inline constexpr double kDefaultDtS = 1.0;
// RMS acceleration (m/s^2) the constant-velocity model cannot predict; sizes Q.
inline constexpr double kSigmaAccelMS2 = 0.25;
inline constexpr double kInitPosVarM2 = 25.0;
inline constexpr double kInitVelVarM2S2 = 4.0;
// Camera: ~8 px 1-sigma -> ~4 m 1-sigma -> 16 m^2 per axis.
inline constexpr double kRCameraDefaultM2 = 16.0;
// Radar (linear KF, Cartesian): ~2 m 1-sigma per axis.
inline constexpr double kRRadarDefaultM2 = 2.0 * 2.0;
// Radar (EKF, native polar): physical range/azimuth noise.
inline constexpr double kRadarRangeStdM = 3.0;
inline constexpr double kRadarAzimuthStdRad = 0.5 * 3.14159265358979323846 / 180.0;
// 95% chi-square, 2 DOF, for ellipse axis scaling.
inline constexpr double kChi2_95_2D = 5.991;

using State = Vec<4>;
using Meas = Vec<2>;
using StateCov = Mat<4, 4>;
using MeasCov = Mat<2, 2>;

// Constant-velocity discrete transition F for state [px, py, vx, vy].
Mat<4, 4> build_f_cv(double dt);

// Discretized white-acceleration process noise Q = sigma_a^2 * G G^T.
Mat<4, 4> build_q_cv(double dt, double sigma_a);

// Linear position measurement matrix H (observe px, py).
Mat<2, 4> h_position();

// Nonlinear radar forward model h(x) = [range, azimuth].
Meas h_radar_polar(const State& x);

// Analytic 2x4 Jacobian of h_radar_polar, evaluated at x.
Mat<2, 4> h_radar_polar_jacobian(const State& x);

// Innovation z - h(x) with the azimuth component wrapped to (-pi, pi].
// Mandatory whenever a bearing appears in the residual.
Meas radar_polar_residual(const Meas& z, const Meas& hx);

// Position-marginal NEES: e^T P^{-1} e for the 2D error e = x_true - x_est.
// Chi-square(2)-distributed (expected value 2) for a consistent filter; a value
// persistently above 2 means the reported covariance P is optimistic.
double nees_2d(const Meas& error, const MeasCov& P);

// 95% confidence ellipse of the position marginal.
struct UncertaintyEllipse2D {
    double cx, cy;      // centre
    double width;       // full major-axis length (diameter)
    double height;      // full minor-axis length (diameter)
    double angle_deg;   // orientation of the major axis
};

// --- Linear Kalman filter tracker -----------------------------------------
class KfTracker {
public:
    explicit KfTracker(const State& initial_state, double dt = kDefaultDtS);

    void predict();
    void update_camera(const Meas& z_xy);
    void update_camera(const Meas& z_xy, const MeasCov& r);
    void update_radar(const Meas& z_xy);
    void update_radar(const Meas& z_xy, const MeasCov& r);

    const State& state() const { return x_; }
    const StateCov& covariance() const { return P_; }
    Mat<2, 2> position_covariance_2d() const;
    UncertaintyEllipse2D uncertainty_ellipse() const;

private:
    void update_linear(const Meas& z, const MeasCov& r);

    State x_;
    StateCov P_;
    Mat<4, 4> F_;
    Mat<4, 4> Q_;
    Mat<2, 4> H_;
    MeasCov R_camera_;
    MeasCov R_radar_;
};

// --- Extended Kalman filter tracker ---------------------------------------
class EkfTracker {
public:
    explicit EkfTracker(const State& initial_state, double dt = kDefaultDtS);

    void predict();
    void update_camera(const Meas& z_xy);
    void update_camera(const Meas& z_xy, const MeasCov& r);
    // Native polar radar update: z = [range_m, azimuth_rad]. Do NOT pre-convert.
    void update_radar_polar(const Meas& z_polar);
    void update_radar_polar(const Meas& z_polar, const MeasCov& r);

    // Predicted innovation (y) and 2x2 innovation covariance (S) for a polar
    // measurement, used by multi-object trackers for Mahalanobis gating. Call
    // after predict() but before update_radar_polar().
    struct Innovation {
        Meas y;
        MeasCov S;
    };
    Innovation compute_innovation_polar(const Meas& z_polar) const;

    const State& state() const { return x_; }
    const StateCov& covariance() const { return P_; }
    Mat<2, 2> position_covariance_2d() const;
    UncertaintyEllipse2D uncertainty_ellipse() const;

private:
    State x_;
    StateCov P_;
    Mat<4, 4> F_;
    Mat<4, 4> Q_;
    MeasCov R_camera_;
    MeasCov R_radar_polar_;
};

}  // namespace ft
