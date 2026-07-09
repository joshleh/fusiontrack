// Fixed-size, heap-free linear algebra for the FusionTrack tracking core.
//
// The state is 4-dimensional and every measurement is 2-dimensional, so all
// matrix shapes are known at compile time. Using std::array storage keeps the
// filter allocation-free and deterministic, which is the property a real-time
// tracking pipeline (e.g. on an embedded autonomy payload) actually cares
// about. The only matrix inverse the Kalman update ever needs is the 2x2
// innovation covariance S, so a closed-form 2x2 inverse is all we implement.
#pragma once

#include <array>
#include <cmath>
#include <cstddef>
#include <stdexcept>

namespace ft {

// Row-major, zero-initialized fixed-size matrix.
template <std::size_t R, std::size_t C>
struct Mat {
    std::array<double, R * C> d{};

    double& operator()(std::size_t i, std::size_t j) { return d[i * C + j]; }
    double operator()(std::size_t i, std::size_t j) const { return d[i * C + j]; }

    static constexpr std::size_t rows = R;
    static constexpr std::size_t cols = C;
};

// Column vector alias.
template <std::size_t N>
using Vec = Mat<N, 1>;

template <std::size_t N>
inline Mat<N, N> eye() {
    Mat<N, N> m;
    for (std::size_t i = 0; i < N; ++i) m(i, i) = 1.0;
    return m;
}

// C = A * B
template <std::size_t A, std::size_t B, std::size_t D>
inline Mat<A, D> operator*(const Mat<A, B>& x, const Mat<B, D>& y) {
    Mat<A, D> out;
    for (std::size_t i = 0; i < A; ++i) {
        for (std::size_t k = 0; k < B; ++k) {
            const double xik = x(i, k);
            for (std::size_t j = 0; j < D; ++j) {
                out(i, j) += xik * y(k, j);
            }
        }
    }
    return out;
}

template <std::size_t R, std::size_t C>
inline Mat<R, C> operator+(const Mat<R, C>& a, const Mat<R, C>& b) {
    Mat<R, C> o;
    for (std::size_t i = 0; i < R * C; ++i) o.d[i] = a.d[i] + b.d[i];
    return o;
}

template <std::size_t R, std::size_t C>
inline Mat<R, C> operator-(const Mat<R, C>& a, const Mat<R, C>& b) {
    Mat<R, C> o;
    for (std::size_t i = 0; i < R * C; ++i) o.d[i] = a.d[i] - b.d[i];
    return o;
}

template <std::size_t R, std::size_t C>
inline Mat<R, C> operator*(double s, const Mat<R, C>& a) {
    Mat<R, C> o;
    for (std::size_t i = 0; i < R * C; ++i) o.d[i] = s * a.d[i];
    return o;
}

template <std::size_t R, std::size_t C>
inline Mat<C, R> transpose(const Mat<R, C>& a) {
    Mat<C, R> o;
    for (std::size_t i = 0; i < R; ++i)
        for (std::size_t j = 0; j < C; ++j) o(j, i) = a(i, j);
    return o;
}

// Closed-form 2x2 inverse. The innovation covariance S is symmetric positive
// definite for a well-posed filter, so a near-zero determinant means the model
// has become numerically degenerate and we fail loudly rather than propagate
// NaNs into the state.
inline Mat<2, 2> inverse2x2(const Mat<2, 2>& m) {
    const double det = m(0, 0) * m(1, 1) - m(0, 1) * m(1, 0);
    if (std::abs(det) < 1e-300) {
        throw std::runtime_error("inverse2x2: singular innovation covariance");
    }
    const double inv_det = 1.0 / det;
    Mat<2, 2> o;
    o(0, 0) = m(1, 1) * inv_det;
    o(0, 1) = -m(0, 1) * inv_det;
    o(1, 0) = -m(1, 0) * inv_det;
    o(1, 1) = m(0, 0) * inv_det;
    return o;
}

// Eigen-decomposition of a symmetric 2x2 matrix, closed form. Returns the two
// eigenvalues (lambda[0] >= lambda[1]) and the eigenvector of the larger
// eigenvalue. Used to draw the 95% position-uncertainty ellipse.
struct SymEig2 {
    std::array<double, 2> eigenvalues;   // descending
    std::array<double, 2> major_axis;    // unit eigenvector of eigenvalues[0]
};

inline SymEig2 sym_eig_2x2(const Mat<2, 2>& p) {
    const double a = p(0, 0);
    const double b = 0.5 * (p(0, 1) + p(1, 0));  // symmetrize defensively
    const double c = p(1, 1);
    const double tr = a + c;
    const double diff = std::sqrt((a - c) * (a - c) + 4.0 * b * b);
    const double l0 = 0.5 * (tr + diff);
    const double l1 = 0.5 * (tr - diff);

    // Eigenvector for l0: (b, l0 - a) unless b ~ 0 (already axis-aligned).
    double vx, vy;
    if (std::abs(b) > 1e-12) {
        vx = b;
        vy = l0 - a;
    } else {
        vx = (a >= c) ? 1.0 : 0.0;
        vy = (a >= c) ? 0.0 : 1.0;
    }
    const double norm = std::hypot(vx, vy);
    if (norm > 0.0) {
        vx /= norm;
        vy /= norm;
    }
    return SymEig2{{l0, l1}, {vx, vy}};
}

}  // namespace ft
