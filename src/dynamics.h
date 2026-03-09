#ifndef DYNAMICS_H
#define DYNAMICS_H

#include <vector>
#include <cmath>
#include <tuple>
#include "geometry.h"
#include "link.h"
#include "joint.h"
#define EIGEN_USE_BLAS 1
#define EIGEN_USE_LAPACKE 1
#include "Eigen/Dense"
#include "Eigen/Sparse"
#include "math_utils.h"
#include <stdexcept>


// compute fluid drag on body i
std::pair<Eigen::Vector2d, double> calcBodyDrag(int i, const Link& links, const HydroDyn& hydrodyn, double theta, const Eigen::Vector2d& vV);

/* compute body penducle thrust
sx - local height at penducle
x_next - distance to link N-1 COM
vV - velocity of link N-1 in robot frame
*/
Eigen::Vector2d calcThrust(double sx, double x_next, const Eigen::Vector2d& vV, double theta, double omega, double rho);

/* compute lift and drag force on caudal fin
theta - orientation of caudal fin (the last link)
*/
std::pair<Eigen::Vector3d, Eigen::Vector3d> calcFoilLiftDrag(double theta, double rho, const Eigen::Vector2d& vV, double foil_area, const HydroDyn& hydrodyn);

/* assemble dynamics matrix
*/
std::pair<Eigen::SparseMatrix<double>, Eigen::VectorXd> assembleChainDynamicMatrix(
    double t, const Link& links, const Joint& joints,
    const Eigen::VectorXd& thetas, const Eigen::VectorXd& omegas,
    const Eigen::MatrixXd& Fs, const Eigen::VectorXd& Ts
);

/* dynamic equations (ODE)
q (6N) = [x * N; y * N; vx * N; vy * N; theta * N; omega * N]
Return: dq, Fs (forces on each link), headT (hydrodynamic torque on the head), F_in (interactive forces between two adjacent links)
*/
std::tuple<Eigen::VectorXd, Eigen::Vector2d, double, Eigen::VectorXd> dyn(
    double t,
    const Eigen::VectorXd& q,
    const Link& links,
    const Joint& joints,
    const HydroDyn& hydrodyn 
);

#endif