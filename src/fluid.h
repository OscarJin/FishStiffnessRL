#ifndef FLUID_H
#define FLUID_H

#define EIGEN_USE_BLAS 1
#define EIGEN_USE_LAPACKE 1
#include "Eigen/Dense"
#include <stdexcept>
#include <cmath>

struct Fluid {
    double rho;             // fluid density
    Eigen::Vector2d flowV;  // flow velocity

    Fluid(double density, const Eigen::Vector2d& velocity)
        : rho(density), flowV(velocity) {}
};

class HydroDyn {
    // hydrodynamic parameters
public:
    Eigen::VectorXd Cf;
    Eigen::VectorXd Cd;
    double Cl_fin;                   // Cl = Cl_fin_coeff * alpha
    Eigen::Vector2d Cd_fin;          // [a; b] Cd = a * alpha^2 + b

    HydroDyn(int num_links, const Eigen::VectorXd& friction_coeffs, const Eigen::VectorXd& drag_coeffs, double Cl_fin_coeff, const Eigen::Vector2d& Cd_fin_coeff)
        : Cf(friction_coeffs), Cd(drag_coeffs), Cl_fin(Cl_fin_coeff), Cd_fin(Cd_fin_coeff) {
        if (Cf.size() != num_links - 1 || Cd.size() != num_links - 1) {
            throw std::runtime_error("Error: Cf or Cd size mismatch!");
        }
    }

    Eigen::Vector2d computeCaudalFinLiftDrag(double angle_of_attack) const {
        double Cl = Cl_fin * std::abs(angle_of_attack);
        double Cd = Cd_fin(0) * std::pow(angle_of_attack, 2) + Cd_fin(1);
        return Eigen::Vector2d(Cl, Cd);
    }

};

#endif
