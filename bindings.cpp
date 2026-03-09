#include <pybind11/pybind11.h>
#include <pybind11/eigen.h>
#include <pybind11/stl.h>
#include <vector>
#include "src/fluid.h"
#include "src/geometry.h"
#include "src/link.h"
#include "src/joint.h"
#include "src/kinematics.h"
#include "src/dynamics.h"
#include "src/simulation.h"

namespace py = pybind11;

PYBIND11_MODULE(fish_sim, m) {
    m.doc() = "fish simulator"; // optional module docstring

    py::register_exception<std::runtime_error>(m, "RuntimeError");

    py::class_<Fluid>(m, "Fluid")
        .def(py::init<double, const Eigen::Vector2d &>())
        .def_readwrite("rho", &Fluid::rho)
        .def_readwrite("flowV", &Fluid::flowV);

    py::class_<Geometry>(m, "Geometry")
        .def(py::init<double>())
        .def("calc_R", py::overload_cast<const Eigen::VectorXd&>(&Geometry::calc_R, py::const_))
        .def("calc_r", py::overload_cast<const Eigen::VectorXd&>(&Geometry::calc_r, py::const_));
    
    py::class_<Link>(m, "Link")
        .def(py::init<int, double, const Eigen::VectorXd&, const Fluid&, const Geometry&>())
        .def_readonly("length", &Link::length)
        .def_readonly("lc", &Link::lc)
        .def_readonly("Izz", &Link::Izz)
        .def_readonly("m_total", &Link::mTotal);
    
    py::class_<Joint>(m, "Joint")
        .def(py::init<int, const Eigen::VectorXd &, const Eigen::VectorXd &, double, double, double>())
        .def_readwrite("k", &Joint::k)
        .def_readwrite("mu", &Joint::mu)
        .def_readwrite("amp", &Joint::amp)
        .def_readwrite("freq", &Joint::freq)
        .def_readwrite("phase", &Joint::phase);
    
    py::class_<HydroDyn>(m, "HydroDyn")
        .def(py::init<int, const Eigen::VectorXd &, const Eigen::VectorXd &, double, const Eigen::Vector2d &>());

    m.def("get_q0", &getQ0, "Get initial state");
    
    py::class_<FishSimulator>(m, "FishSimulator")
        .def(py::init<int, double, double, const Eigen::VectorXd &, Link &, Joint &, HydroDyn &>())
        .def("run", &FishSimulator::run);

    py::class_<SimulationResults>(m, "SimulationResults")
        .def(py::init<>())
        .def_readonly("t_traj", &SimulationResults::t_traj)
        .def_readonly("q_traj", &SimulationResults::q_traj)
        .def_readonly("Ft_rec", &SimulationResults::Ft_rec)
        .def_readonly("M_rec", &SimulationResults::M_rec);
}
