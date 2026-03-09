#ifndef SIMULATION_H
#define SIMULATION_H

#include <vector>
#include <tuple>
#include <cmath>
#define EIGEN_USE_BLAS 1
#define EIGEN_USE_LAPACKE 1
#include "Eigen/Dense"
#include "dynamics.h"
#include <stdexcept>
#include <chrono>

struct SimulationResults {
    std::vector<double> t_traj;
    std::vector<Eigen::VectorXd> q_traj;
    std::vector<double> Ft_rec;     // record thrust force
    std::vector<double> M_rec;      // record motor torque
};

class FishSimulator {
public:
    FishSimulator(int sim_freq, double t_start, double t_end, const Eigen::VectorXd& q0,
                const Link& links, const Joint& joints, const HydroDyn& hydrodyn);
    
    SimulationResults run(double timeout = 30.);

private:
    int sim_freq;
    double dt;
    double t_start, t_end;
    Eigen::VectorXd q0;
    int N;
    const Geometry& geometry;
    const Link& links;
    const Joint& joints;
    const Fluid& fluid;
    const HydroDyn& hydrodyn;

    // Dormand-Prince 45 ODE solver
    Eigen::VectorXd dopri45_step(double t, const Eigen::VectorXd& q);
};


#endif