// Optional pybind11 bindings for the FusionTrack C++ core.
//
// Built only when -DFUSIONTRACK_PYBIND=ON. Exposes the C++ KfTracker/EkfTracker
// with the same surface the Python demo (src/fusion.py) needs -- state, full
// 4x4 covariance, the 95% uncertainty ellipse, optional per-update R overrides,
// and the polar innovation used for gating -- so the exact C++ filter math can
// back the Python simulation and be cross-validated against the reference.
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <array>
#include <optional>

#include "fusiontrack/ekf.hpp"

namespace py = pybind11;

namespace {
ft::State to_state(const std::array<double, 4>& v) {
    ft::State x;
    for (std::size_t i = 0; i < 4; ++i) x(i, 0) = v[i];
    return x;
}

ft::Meas to_meas(const std::array<double, 2>& v) {
    ft::Meas m;
    m(0, 0) = v[0];
    m(1, 0) = v[1];
    return m;
}

// Row-major 2x2 [r00, r01, r10, r11] -> MeasCov.
ft::MeasCov to_meascov(const std::array<double, 4>& r) {
    ft::MeasCov m;
    m(0, 0) = r[0];
    m(0, 1) = r[1];
    m(1, 0) = r[2];
    m(1, 1) = r[3];
    return m;
}

std::array<double, 4> from_state(const ft::State& x) {
    return {x(0, 0), x(1, 0), x(2, 0), x(3, 0)};
}

// Row-major flatten of the full 4x4 covariance.
std::array<double, 16> from_cov(const ft::StateCov& p) {
    std::array<double, 16> out{};
    for (std::size_t i = 0; i < 4; ++i)
        for (std::size_t j = 0; j < 4; ++j) out[i * 4 + j] = p(i, j);
    return out;
}

py::tuple ellipse_tuple(const ft::UncertaintyEllipse2D& e) {
    return py::make_tuple(e.cx, e.cy, e.width, e.height, e.angle_deg);
}

using OptR = std::optional<std::array<double, 4>>;
}  // namespace

PYBIND11_MODULE(fusiontrack_cpp, m) {
    m.doc() = "FusionTrack C++ tracking core (KF + EKF)";

    py::class_<ft::KfTracker>(m, "KfTracker")
        .def(py::init([](const std::array<double, 4>& x0, double dt) {
                 return ft::KfTracker(to_state(x0), dt);
             }),
             py::arg("initial_state"), py::arg("dt") = ft::kDefaultDtS)
        .def("predict", &ft::KfTracker::predict)
        .def(
            "update_camera",
            [](ft::KfTracker& t, const std::array<double, 2>& z, OptR r) {
                if (r) {
                    t.update_camera(to_meas(z), to_meascov(*r));
                } else {
                    t.update_camera(to_meas(z));
                }
            },
            py::arg("z"), py::arg("r") = py::none())
        .def(
            "update_radar",
            [](ft::KfTracker& t, const std::array<double, 2>& z, OptR r) {
                if (r) {
                    t.update_radar(to_meas(z), to_meascov(*r));
                } else {
                    t.update_radar(to_meas(z));
                }
            },
            py::arg("z"), py::arg("r") = py::none())
        .def("state", [](const ft::KfTracker& t) { return from_state(t.state()); })
        .def("covariance", [](const ft::KfTracker& t) { return from_cov(t.covariance()); })
        .def("uncertainty_ellipse",
             [](const ft::KfTracker& t) { return ellipse_tuple(t.uncertainty_ellipse()); });

    py::class_<ft::EkfTracker>(m, "EkfTracker")
        .def(py::init([](const std::array<double, 4>& x0, double dt) {
                 return ft::EkfTracker(to_state(x0), dt);
             }),
             py::arg("initial_state"), py::arg("dt") = ft::kDefaultDtS)
        .def("predict", &ft::EkfTracker::predict)
        .def(
            "update_camera",
            [](ft::EkfTracker& t, const std::array<double, 2>& z, OptR r) {
                if (r) {
                    t.update_camera(to_meas(z), to_meascov(*r));
                } else {
                    t.update_camera(to_meas(z));
                }
            },
            py::arg("z"), py::arg("r") = py::none())
        .def(
            "update_radar_polar",
            [](ft::EkfTracker& t, const std::array<double, 2>& z, OptR r) {
                if (r) {
                    t.update_radar_polar(to_meas(z), to_meascov(*r));
                } else {
                    t.update_radar_polar(to_meas(z));
                }
            },
            py::arg("z"), py::arg("r") = py::none())
        .def("compute_innovation_polar",
             [](const ft::EkfTracker& t, const std::array<double, 2>& z) {
                 const auto inn = t.compute_innovation_polar(to_meas(z));
                 return py::make_tuple(
                     std::array<double, 2>{inn.y(0, 0), inn.y(1, 0)},
                     std::array<double, 4>{inn.S(0, 0), inn.S(0, 1), inn.S(1, 0),
                                           inn.S(1, 1)});
             })
        .def("state", [](const ft::EkfTracker& t) { return from_state(t.state()); })
        .def("covariance", [](const ft::EkfTracker& t) { return from_cov(t.covariance()); })
        .def("uncertainty_ellipse",
             [](const ft::EkfTracker& t) { return ellipse_tuple(t.uncertainty_ellipse()); });
}
