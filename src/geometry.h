#ifndef GEOMETRY_H
#define GEOMETRY_H

#include <cmath>
#define EIGEN_USE_BLAS 1
#define EIGEN_USE_LAPACKE 1
#include "Eigen/Dense"

class Geometry {
public:
    Geometry(double L);

    Eigen::VectorXd calc_R(const Eigen::VectorXd& ls) const;
    Eigen::VectorXd calc_r(const Eigen::VectorXd& ls) const;

    double calc_R(double l) const;
    double calc_r(double l) const;

private:
    double L;
    double R1, R2, R3, R4;
    double r1, r2, r3, r4, r5;  
};

#endif