// Unit tests for the FusionTrack C++ core.
//
// Dependency-free harness (no GoogleTest/Catch2 needed): each CHECK records a
// failure and the process exits non-zero if any check fails, which is all
// ctest needs to report pass/fail.
#include <cmath>
#include <cstdio>
#include <string>

#include "fusiontrack/ekf.hpp"

namespace {
int g_failures = 0;
int g_checks = 0;

void report(bool ok, const std::string& what, const char* file, int line) {
    ++g_checks;
    if (!ok) {
        ++g_failures;
        std::printf("  FAIL [%s:%d] %s\n", file, line, what.c_str());
    }
}

#define CHECK(cond) report((cond), #cond, __FILE__, __LINE__)
#define CHECK_NEAR(a, b, tol)                                                  \
    report(std::fabs((a) - (b)) <= (tol),                                      \
           std::string(#a " ~= " #b " (got ") + std::to_string(a) + " vs " +  \
               std::to_string(b) + ")",                                        \
           __FILE__, __LINE__)

double trace2(const ft::Mat<2, 2>& m) { return m(0, 0) + m(1, 1); }
}  // namespace

// F is constant-velocity: predict advances position by velocity * dt.
static void test_predict_advances_position() {
    ft::State x0;
    x0(0, 0) = 0.0;
    x0(1, 0) = 0.0;
    x0(2, 0) = 3.0;   // vx
    x0(3, 0) = -2.0;  // vy
    ft::KfTracker kf(x0, 1.0);
    kf.predict();
    CHECK_NEAR(kf.state()(0, 0), 3.0, 1e-12);
    CHECK_NEAR(kf.state()(1, 0), -2.0, 1e-12);
    CHECK_NEAR(kf.state()(2, 0), 3.0, 1e-12);
    CHECK_NEAR(kf.state()(3, 0), -2.0, 1e-12);
}

// Q from build_q_cv must be symmetric with the known CV block structure.
static void test_q_structure() {
    const auto Q = ft::build_q_cv(1.0, ft::kSigmaAccelMS2);
    const double s2 = ft::kSigmaAccelMS2 * ft::kSigmaAccelMS2;
    CHECK_NEAR(Q(0, 0), 0.25 * s2, 1e-15);  // (dt^2/2)^2 = 1/4 at dt=1
    CHECK_NEAR(Q(2, 2), s2, 1e-15);         // dt^2 = 1 at dt=1
    CHECK_NEAR(Q(0, 2), 0.5 * s2, 1e-15);   // (dt^2/2)*dt = 1/2
    CHECK_NEAR(Q(0, 2), Q(2, 0), 1e-15);    // symmetry
    CHECK_NEAR(Q(1, 3), Q(3, 1), 1e-15);
}

// Radar forward model: 3-4-5 triangle gives range 5, azimuth atan2(4,3).
static void test_h_radar_polar() {
    ft::State x;
    x(0, 0) = 3.0;
    x(1, 0) = 4.0;
    const ft::Meas z = ft::h_radar_polar(x);
    CHECK_NEAR(z(0, 0), 5.0, 1e-12);
    CHECK_NEAR(z(1, 0), std::atan2(4.0, 3.0), 1e-12);
}

// Analytic Jacobian must match a central finite difference.
static void test_jacobian_matches_finite_difference() {
    ft::State x;
    x(0, 0) = 40.0;
    x(1, 0) = 25.0;
    x(2, 0) = 1.0;
    x(3, 0) = -1.0;
    const auto J = ft::h_radar_polar_jacobian(x);
    const double h = 1e-6;
    for (std::size_t col = 0; col < 4; ++col) {
        ft::State xp = x, xm = x;
        xp(col, 0) += h;
        xm(col, 0) -= h;
        const ft::Meas fp = ft::h_radar_polar(xp);
        const ft::Meas fm = ft::h_radar_polar(xm);
        const double dr = (fp(0, 0) - fm(0, 0)) / (2 * h);
        const double dtheta = (fp(1, 0) - fm(1, 0)) / (2 * h);
        CHECK_NEAR(J(0, col), dr, 1e-5);
        CHECK_NEAR(J(1, col), dtheta, 1e-5);
    }
}

// Azimuth residual must wrap across the +/-pi seam instead of returning ~2pi.
static void test_angle_wrapping() {
    const double pi = 3.14159265358979323846;
    ft::Meas z, hx;
    z(0, 0) = 100.0;
    z(1, 0) = -pi + 0.1;  // just past -pi
    hx(0, 0) = 100.0;
    hx(1, 0) = pi - 0.1;  // just under +pi
    const ft::Meas y = ft::radar_polar_residual(z, hx);
    // True angular difference is +0.2 rad, not (-pi+0.1)-(pi-0.1) = -2pi+0.2.
    CHECK_NEAR(y(1, 0), 0.2, 1e-9);
    CHECK(std::fabs(y(1, 0)) <= pi);
}

// A camera update must not increase the position covariance trace.
static void test_camera_update_shrinks_covariance() {
    ft::State x0;
    x0(0, 0) = 10.0;
    x0(1, 0) = 10.0;
    ft::KfTracker kf(x0, 1.0);
    kf.predict();
    const double before = trace2(kf.position_covariance_2d());
    ft::Meas z;
    z(0, 0) = 10.5;
    z(1, 0) = 9.5;
    kf.update_camera(z);
    const double after = trace2(kf.position_covariance_2d());
    CHECK(after < before);
}

// A polar radar update must pull the estimate toward the measurement and
// shrink covariance.
static void test_ekf_radar_update() {
    ft::State x0;
    x0(0, 0) = 200.0;
    x0(1, 0) = 0.0;
    x0(2, 0) = 0.0;
    x0(3, 0) = 0.0;
    ft::EkfTracker ekf(x0, 1.0);
    ekf.predict();
    const double before = trace2(ekf.position_covariance_2d());
    // Measure a slightly longer range at the same bearing.
    ft::Meas z;
    z(0, 0) = 210.0;  // range
    z(1, 0) = 0.0;    // azimuth
    ekf.update_radar_polar(z);
    const double after = trace2(ekf.position_covariance_2d());
    CHECK(after < before);
    CHECK(ekf.state()(0, 0) > 200.0);   // moved toward the 210 m measurement
    CHECK(ekf.state()(0, 0) < 210.0);   // but not all the way (finite gain)
}

// compute_innovation_polar returns a symmetric S and a wrapped innovation, and
// must not mutate the tracker (it is a const query).
static void test_innovation_gating_query() {
    ft::State x0;
    x0(0, 0) = 150.0;
    x0(1, 0) = 20.0;
    ft::EkfTracker ekf(x0, 1.0);
    ekf.predict();
    const ft::State before = ekf.state();
    ft::Meas z = ft::h_radar_polar(ekf.state());
    z(0, 0) += 5.0;
    const auto inn = ekf.compute_innovation_polar(z);
    CHECK_NEAR(inn.S(0, 1), inn.S(1, 0), 1e-12);  // symmetric
    CHECK(inn.S(0, 0) > 0.0 && inn.S(1, 1) > 0.0);
    CHECK_NEAR(inn.y(0, 0), 5.0, 1e-9);
    CHECK_NEAR(ekf.state()(0, 0), before(0, 0), 1e-12);  // unchanged
}

// 2x2 inverse sanity: A * inv(A) == I.
static void test_inverse2x2() {
    ft::Mat<2, 2> a;
    a(0, 0) = 4.0;
    a(0, 1) = 3.0;
    a(1, 0) = 6.0;
    a(1, 1) = 3.0;
    const auto prod = a * ft::inverse2x2(a);
    CHECK_NEAR(prod(0, 0), 1.0, 1e-12);
    CHECK_NEAR(prod(0, 1), 0.0, 1e-12);
    CHECK_NEAR(prod(1, 0), 0.0, 1e-12);
    CHECK_NEAR(prod(1, 1), 1.0, 1e-12);
}

// Covariance stays symmetric after a full predict/update cycle (Joseph form).
static void test_covariance_symmetric_after_update() {
    ft::State x0;
    x0(0, 0) = 30.0;
    x0(1, 0) = 40.0;
    ft::EkfTracker ekf(x0, 1.0);
    ekf.predict();
    ft::Meas z = ft::h_radar_polar(ekf.state());
    z(0, 0) += 2.0;
    ekf.update_radar_polar(z);
    const auto& P = ekf.covariance();
    for (std::size_t i = 0; i < 4; ++i)
        for (std::size_t j = 0; j < 4; ++j)
            CHECK_NEAR(P(i, j), P(j, i), 1e-9);
}

int main() {
    std::printf("FusionTrack C++ core tests\n");
    test_predict_advances_position();
    test_q_structure();
    test_h_radar_polar();
    test_jacobian_matches_finite_difference();
    test_angle_wrapping();
    test_camera_update_shrinks_covariance();
    test_ekf_radar_update();
    test_innovation_gating_query();
    test_inverse2x2();
    test_covariance_symmetric_after_update();

    std::printf("\n%d checks, %d failures\n", g_checks, g_failures);
    return g_failures == 0 ? 0 : 1;
}
