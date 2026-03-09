#ifndef LINK_H
#define LINK_H

#include "fluid.h"
#include "geometry.h"
#define EIGEN_USE_BLAS 1
#define EIGEN_USE_LAPACKE 1
#include "Eigen/Dense"
#include <vector>

class Link {
public:
    int N;                      // number of links
    double L;                   // total length
    Eigen::VectorXd length;     // length of each link
    Eigen::MatrixXd mass;       // mass of each link
    double mTotal;              // total mass (not include added mass)
    Eigen::VectorXd lc;         // center of mass (to previous joint)
    Eigen::VectorXd Izz;        // moment of inertia w.r.t. COM
    Eigen::VectorXd xNext;      // COM to next joint
    std::vector<bool> foil;     // foil presence
    double foil_area;           // wetted area of foil

    const Fluid& fluid;
    const Geometry& geometry;

    /* for integrals */
    const double _step_size;        
    std::vector<Eigen::VectorXd> l_seg;
    std::vector<Eigen::VectorXd> R_seg;
    std::vector<Eigen::VectorXd> r_seg;
    std::vector<Eigen::VectorXd> p_seg;

    Link(int num_links, double l_tot, const Eigen::VectorXd& lengths, const Fluid& fluid, const Geometry& geometry);

private:
    Eigen::VectorXd calcBodyMass() const;
    Eigen::MatrixXd calcAddedMass() const;
    Eigen::VectorXd calcLc() const;
    Eigen::VectorXd calcBodyIzz() const;
    Eigen::VectorXd calcAddedIzz() const;
    double calcFoilArea() const;
};

#endif