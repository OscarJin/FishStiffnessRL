#ifndef KINEMATICS_H
#define KINEMATICS_H

#define EIGEN_USE_BLAS 1
#define EIGEN_USE_LAPACKE 1
#include "Eigen/Dense"
#include "link.h"

Eigen::VectorXd getQ0(const Eigen::VectorXd& q_init, const Link& links);

#endif