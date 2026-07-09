#include "fusiontrack/ekf.hpp"

#include <algorithm>
#include <cmath>

namespace ft {

namespace {
constexpr double kPi = 3.14159265358979323846;
constexpr double kTwoPi = 2.0 * kPi;

// Common Joseph-form covariance/state update shared by the linear KF and the
// linearized EKF step. H is the (Jacobian) measurement matrix, y the already
// wrapped innovation, R the measurement noise in the measurement's own frame.
template <typename SetXP>
void kalman_update(State& x, StateCov& P, const Mat<2, 4>& H, const Meas& y,
                   const MeasCov& R, SetXP&& commit) {
    const Mat<4, 2> PHt = P * transpose(H);
    const MeasCov S = H * PHt + R;
    const Mat<4, 2> K = PHt * inverse2x2(S);

    const State x_new = x + K * y;
    const Mat<4, 4> IKH = eye<4>() - K * H;
    const StateCov P_new = IKH * P * transpose(IKH) + K * R * transpose(K);
    commit(x_new, P_new);
}

UncertaintyEllipse2D ellipse_from(const State& x, const StateCov& P) {
    Mat<2, 2> p2;
    p2(0, 0) = P(0, 0);
    p2(0, 1) = P(0, 1);
    p2(1, 0) = P(1, 0);
    p2(1, 1) = P(1, 1);

    const SymEig2 e = sym_eig_2x2(p2);
    const double l0 = std::max(e.eigenvalues[0], 1e-9);
    const double l1 = std::max(e.eigenvalues[1], 1e-9);
    const double semi_major = std::sqrt(l0 * kChi2_95_2D);
    const double semi_minor = std::sqrt(l1 * kChi2_95_2D);
    const double angle_deg =
        std::atan2(e.major_axis[1], e.major_axis[0]) * 180.0 / kPi;
    return UncertaintyEllipse2D{x(0, 0),          x(1, 0),
                                2.0 * semi_major,  2.0 * semi_minor,
                                angle_deg};
}

StateCov init_p() {
    StateCov P;
    P(0, 0) = kInitPosVarM2;
    P(1, 1) = kInitPosVarM2;
    P(2, 2) = kInitVelVarM2S2;
    P(3, 3) = kInitVelVarM2S2;
    return P;
}

MeasCov scaled_identity(double v) {
    MeasCov m;
    m(0, 0) = v;
    m(1, 1) = v;
    return m;
}
}  // namespace

Mat<4, 4> build_f_cv(double dt) {
    Mat<4, 4> F = eye<4>();
    F(0, 2) = dt;
    F(1, 3) = dt;
    return F;
}

Mat<4, 4> build_q_cv(double dt, double sigma_a) {
    // G maps a per-axis white acceleration into [pos; vel] increments.
    Mat<4, 2> G;
    G(0, 0) = 0.5 * dt * dt;
    G(1, 1) = 0.5 * dt * dt;
    G(2, 0) = dt;
    G(3, 1) = dt;
    const Mat<2, 2> q = scaled_identity(sigma_a * sigma_a);
    return G * q * transpose(G);
}

Mat<2, 4> h_position() {
    Mat<2, 4> H;
    H(0, 0) = 1.0;
    H(1, 1) = 1.0;
    return H;
}

Meas h_radar_polar(const State& x) {
    const double px = x(0, 0);
    const double py = x(1, 0);
    const double r = std::max(std::hypot(px, py), 1e-6);  // guard the origin
    Meas z;
    z(0, 0) = r;
    z(1, 0) = std::atan2(py, px);
    return z;
}

Mat<2, 4> h_radar_polar_jacobian(const State& x) {
    const double px = x(0, 0);
    const double py = x(1, 0);
    const double r2 = std::max(px * px + py * py, 1e-12);
    const double r = std::sqrt(r2);
    Mat<2, 4> H;
    H(0, 0) = px / r;
    H(0, 1) = py / r;
    H(1, 0) = -py / r2;
    H(1, 1) = px / r2;
    return H;
}

Meas radar_polar_residual(const Meas& z, const Meas& hx) {
    Meas y = z - hx;
    // Wrap azimuth residual into (-pi, pi]; without this a target crossing the
    // +/-pi seam produces a ~2*pi innovation that flings the state ~hundreds of
    // metres. Same bug pattern as GPS/compass fusion and bearing-only SLAM.
    y(1, 0) = std::fmod(y(1, 0) + kPi, kTwoPi);
    if (y(1, 0) < 0.0) y(1, 0) += kTwoPi;
    y(1, 0) -= kPi;
    return y;
}

// --- KfTracker -------------------------------------------------------------
KfTracker::KfTracker(const State& initial_state, double dt)
    : x_(initial_state),
      P_(init_p()),
      F_(build_f_cv(dt)),
      Q_(build_q_cv(dt, kSigmaAccelMS2)),
      H_(h_position()),
      R_camera_(scaled_identity(kRCameraDefaultM2)),
      R_radar_(scaled_identity(kRRadarDefaultM2)) {}

void KfTracker::predict() {
    x_ = F_ * x_;
    P_ = F_ * P_ * transpose(F_) + Q_;
}

void KfTracker::update_linear(const Meas& z, const MeasCov& r) {
    const Meas y = z - H_ * x_;
    kalman_update(x_, P_, H_, y, r,
                  [this](const State& x, const StateCov& P) {
                      x_ = x;
                      P_ = P;
                  });
}

void KfTracker::update_camera(const Meas& z_xy) { update_linear(z_xy, R_camera_); }
void KfTracker::update_camera(const Meas& z_xy, const MeasCov& r) { update_linear(z_xy, r); }
void KfTracker::update_radar(const Meas& z_xy) { update_linear(z_xy, R_radar_); }
void KfTracker::update_radar(const Meas& z_xy, const MeasCov& r) { update_linear(z_xy, r); }

Mat<2, 2> KfTracker::position_covariance_2d() const {
    Mat<2, 2> p2;
    p2(0, 0) = P_(0, 0);
    p2(0, 1) = P_(0, 1);
    p2(1, 0) = P_(1, 0);
    p2(1, 1) = P_(1, 1);
    return p2;
}

UncertaintyEllipse2D KfTracker::uncertainty_ellipse() const { return ellipse_from(x_, P_); }

// --- EkfTracker ------------------------------------------------------------
EkfTracker::EkfTracker(const State& initial_state, double dt)
    : x_(initial_state),
      P_(init_p()),
      F_(build_f_cv(dt)),
      Q_(build_q_cv(dt, kSigmaAccelMS2)),
      R_camera_(scaled_identity(kRCameraDefaultM2)) {
    R_radar_polar_(0, 0) = kRadarRangeStdM * kRadarRangeStdM;
    R_radar_polar_(1, 1) = kRadarAzimuthStdRad * kRadarAzimuthStdRad;
}

void EkfTracker::predict() {
    x_ = F_ * x_;
    P_ = F_ * P_ * transpose(F_) + Q_;
}

void EkfTracker::update_camera(const Meas& z_xy) { update_camera(z_xy, R_camera_); }

void EkfTracker::update_camera(const Meas& z_xy, const MeasCov& r) {
    const Mat<2, 4> H = h_position();  // camera is linear
    const Meas y = z_xy - H * x_;
    kalman_update(x_, P_, H, y, r,
                  [this](const State& x, const StateCov& P) {
                      x_ = x;
                      P_ = P;
                  });
}

void EkfTracker::update_radar_polar(const Meas& z_polar) {
    update_radar_polar(z_polar, R_radar_polar_);
}

void EkfTracker::update_radar_polar(const Meas& z_polar, const MeasCov& r) {
    const Mat<2, 4> H = h_radar_polar_jacobian(x_);
    const Meas hx = h_radar_polar(x_);
    const Meas y = radar_polar_residual(z_polar, hx);
    kalman_update(x_, P_, H, y, r,
                  [this](const State& x, const StateCov& P) {
                      x_ = x;
                      P_ = P;
                  });
}

EkfTracker::Innovation EkfTracker::compute_innovation_polar(const Meas& z_polar) const {
    const Meas hx = h_radar_polar(x_);
    const Mat<2, 4> H = h_radar_polar_jacobian(x_);
    const MeasCov S = H * P_ * transpose(H) + R_radar_polar_;
    const Meas y = radar_polar_residual(z_polar, hx);
    return Innovation{y, S};
}

Mat<2, 2> EkfTracker::position_covariance_2d() const {
    Mat<2, 2> p2;
    p2(0, 0) = P_(0, 0);
    p2(0, 1) = P_(0, 1);
    p2(1, 0) = P_(1, 0);
    p2(1, 1) = P_(1, 1);
    return p2;
}

UncertaintyEllipse2D EkfTracker::uncertainty_ellipse() const { return ellipse_from(x_, P_); }

}  // namespace ft
