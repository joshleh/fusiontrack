// Optional pybind11 bindings for the FusionTrack C++ core.
//
// Built only when -DFUSIONTRACK_PYBIND=ON. Exposes the C++ KfTracker/EkfTracker
// so the Python demo (and the cross-validation test) can drive the exact same
// filter math that a C++ deployment would run.
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <array>

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

std::array<double, 4> from_state(const ft::State& x) {
    return {x(0, 0), x(1, 0), x(2, 0), x(3, 0)};
}

std::array<double, 4> flat_pos_cov(const ft::Mat<2, 2>& p) {
    return {p(0, 0), p(0, 1), p(1, 0), p(1, 1)};
}
}  // namespace

PYBIND11_MODULE(fusiontrack_cpp, m) {
    m.doc() = "FusionTrack C++ tracking core (KF + EKF)";

    py::class_<ft::KfTracker>(m, "KfTracker")
        .def(py::init([](const std::array<double, 4>& x0, double dt) {
                 return ft::KfTracker(to_state(x0), dt);
             }),
             py::arg("initial_state"), py::arg("dt") = ft::kDefaultDtS)
        .def("predict", &ft::KfTracker::predict)
        .def("update_camera",
             [](ft::KfTracker& t, const std::array<double, 2>& z) {
                 t.update_camera(to_meas(z));
             })
        .def("update_radar",
             [](ft::KfTracker& t, const std::array<double, 2>& z) {
                 t.update_radar(to_meas(z));
             })
        .def("state", [](const ft::KfTracker& t) { return from_state(t.state()); })
        .def("position_covariance_2d",
             [](const ft::KfTracker& t) { return flat_pos_cov(t.position_covariance_2d()); });

    py::class_<ft::EkfTracker>(m, "EkfTracker")
        .def(py::init([](const std::array<double, 4>& x0, double dt) {
                 return ft::EkfTracker(to_state(x0), dt);
             }),
             py::arg("initial_state"), py::arg("dt") = ft::kDefaultDtS)
        .def("predict", &ft::EkfTracker::predict)
        .def("update_camera",
             [](ft::EkfTracker& t, const std::array<double, 2>& z) {
                 t.update_camera(to_meas(z));
             })
        .def("update_radar_polar",
             [](ft::EkfTracker& t, const std::array<double, 2>& z) {
                 t.update_radar_polar(to_meas(z));
             })
        .def("compute_innovation_polar",
             [](const ft::EkfTracker& t, const std::array<double, 2>& z) {
                 const auto inn = t.compute_innovation_polar(to_meas(z));
                 return py::make_tuple(
                     std::array<double, 2>{inn.y(0, 0), inn.y(1, 0)},
                     std::array<double, 4>{inn.S(0, 0), inn.S(0, 1), inn.S(1, 0),
                                           inn.S(1, 1)});
             })
        .def("state", [](const ft::EkfTracker& t) { return from_state(t.state()); })
        .def("position_covariance_2d",
             [](const ft::EkfTracker& t) { return flat_pos_cov(t.position_covariance_2d()); });
}
