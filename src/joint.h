#ifndef JOINT_H
#define JOINT_H

#define EIGEN_USE_BLAS 1
#define EIGEN_USE_LAPACKE 1
#include "Eigen/Dense"
#include <stdexcept>

struct Joint {
public:
    int nJoint;            // Number of joints = N-1
    Eigen::VectorXd k;      // Stiffness
    Eigen::VectorXd mu;     // Damping
    // driving input: amp * sin(2*pi*freq + phase)
    double amp;             // Driving amplitude
    double freq;            // Driving frequency
    double phase;           // Driving phase

    Joint(int link_num, const Eigen::VectorXd& stiffness, const Eigen::VectorXd& damping, double amplitude, double frequency, double phase)
        : nJoint(link_num - 1), k(stiffness), mu(damping), amp(amplitude), freq(frequency), phase(phase) {
        bool err = (k.size() != nJoint) || (mu.size() != nJoint);
        if (err) {
            throw std::runtime_error("Error: Joint parameters size mismatch!");
        }
    }
};

#endif