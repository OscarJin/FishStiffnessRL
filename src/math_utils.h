#ifndef MATH_UTILS_H
#define MATH_UTILS_H

#define EIGEN_USE_BLAS 1
#define EIGEN_USE_LAPACKE 1
#include "Eigen/Dense"

namespace math_utils {

    inline double trapz(const Eigen::VectorXd& values, double step_size) {
        double result = 0.0;
        int n = values.size();

        if (n < 2) {
            return result;
        }

        return (values.head(n - 1) + values.tail(n - 1)).sum() * (step_size / 2.0);
    }
}

#endif