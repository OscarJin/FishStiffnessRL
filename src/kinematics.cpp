#include "kinematics.h"

Eigen::VectorXd getQ0(const Eigen::VectorXd &q_init, const Link &links)
{   
    int _N = links.N;
    double x = q_init(0);
    double y = q_init(1);
    double vx = q_init(2);
    double vy = q_init(3);
    Eigen::VectorXd thetas = q_init.segment(4, _N);
    Eigen::VectorXd omegas = q_init.segment(_N + 4, _N);

    Eigen::VectorXd q0 = Eigen::VectorXd::Zero(6 * _N);

    for (int i = 0; i < _N; i++) {
        // Transform to link center
        x += links.lc(i) * std::cos(thetas(i));
        y += links.lc(i) * std::sin(thetas(i));
        vx -= omegas(i) * links.lc(i) * std::sin(thetas(i));
        vy += omegas(i) * links.lc(i) * std::cos(thetas(i));

        q0(i) = x;
        q0(_N + i) = y;
        q0(2 * _N + i) = vx;
        q0(3 * _N + i) = vy;

        // Transform to next joint
        x += links.xNext(i) * std::cos(thetas(i));
        y += links.xNext(i) * std::sin(thetas(i));
        vx -= omegas(i) * links.xNext(i) * std::sin(thetas(i));
        vy += omegas(i) * links.xNext(i) * std::cos(thetas(i));
    }

    q0.segment(4 * _N, _N) = thetas;
    q0.segment(5 * _N, _N) = omegas;

    return q0;
}