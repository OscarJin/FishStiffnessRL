#include "simulation.h"

FishSimulator::FishSimulator(int sim_freq, double t_start, double t_end, const Eigen::VectorXd& q0,
                            const Link& links, const Joint& joints, const HydroDyn& hydrodyn)
    : sim_freq(sim_freq), t_start(t_start), t_end(t_end), q0(q0), N(links.N), 
        geometry(links.geometry), links(links), joints(joints), fluid(links.fluid), hydrodyn(hydrodyn) {
    dt = 1.0 / (double)sim_freq;
}

SimulationResults FishSimulator::run(double timeout)
{
    SimulationResults results;
    int record_size = static_cast<int>((t_end - t_start) * sim_freq);

    double t_now = t_start;
    Eigen::VectorXd q_now = q0;

    // logging
    results.t_traj.reserve(record_size);
    results.q_traj.reserve(record_size);
    results.Ft_rec.reserve(record_size);
    results.M_rec.reserve(record_size);

    // timer
    auto startTime = std::chrono::steady_clock::now();

    while (t_now < t_end) {
        // check timeout
        auto curTime = std::chrono::steady_clock::now();
        double elapsed_seconds = std::chrono::duration<double>(curTime - startTime).count();
        if (elapsed_seconds > timeout) {
            throw std::runtime_error("Simulation timed out!");
        }

        q_now.noalias() = dopri45_step(t_now, q_now);
        
        // record traj
        t_now += dt;
        results.t_traj.push_back(t_now);
        results.q_traj.push_back(q_now);

        Eigen::VectorXd dq, Fi;
        Eigen::Vector2d Ft;
        double headT;
        std::tie(dq, Ft, headT, Fi) = dyn(t_now, q_now, links, joints, hydrodyn);
        double theta1 = q_now(4 * N);
        double cos_theta1 = std::cos(theta1);
        double sin_theta1 = std::sin(theta1);
        // calculate thrust force
        double Ftx = -Ft(0) * cos_theta1 - Ft(1) * sin_theta1;
        results.Ft_rec.push_back(Ftx);

        // calculate thrust force
        double ddth1 = dq(5 * N);
        double M_in = headT - links.Izz(0) * ddth1 - links.xNext(0) * sin_theta1 * Fi(0) + links.xNext(0) * cos_theta1 * Fi(N - 1);
        results.M_rec.push_back(M_in);
    }

    return results;
}

Eigen::VectorXd FishSimulator::dopri45_step(double t, const Eigen::VectorXd &q)
{
    const double tol = 1e-6;    // error tolerance
    double h = std::pow(tol, 0.2) / 4.0;    // initial step
    double te = t + dt;
    double t_now = t;
    Eigen::VectorXd q_now = q;

    // Dormand-Prince coeff.
    Eigen::VectorXd a4(3), a5(4), a6(5), a7(6), e(6);
    a4 << 44.0 / 45, -56.0 / 15, 32.0 / 9;
    a5 << 19372.0 / 6561, -25360.0 / 2187, 64448.0 / 6561, -212.0 / 729;
    a6 << 9017.0 / 3168, -355.0 / 33, 46732.0 / 5247, 49.0 / 176, -5103.0 / 18656;
    a7 << 35.0 / 384, 0, 500.0 / 1113, 125.0 / 192, -2187.0 / 6784, 11.0 / 84;
    e << 71.0 / 57600, -1.0 / 40, -71.0 / 16695, 71.0 / 1920, -17253.0 / 339200, 22.0 / 525;

    int state_size = q.size();
    Eigen::VectorXd k1(state_size), k2(state_size), k3(state_size),
                    k4(state_size), k5(state_size), k6(state_size), q5(state_size), error_est(state_size);

    k1.noalias() = std::get<0>(dyn(t_now, q_now, links, joints, hydrodyn));

    int nRej = 0;
    // timer
    auto startTime = std::chrono::steady_clock::now();  

    while (t_now < te) {
        // check timeout
        auto curTime = std::chrono::steady_clock::now();
        double elapsed_seconds = std::chrono::duration<double>(curTime - startTime).count();
        if (elapsed_seconds > 2.0) {
            throw std::runtime_error("Simulation step timed out!");
        }

        if (t_now + h > te) {
            h = te - t_now;
        }

        // RK45
        k2.noalias() = std::get<0>(dyn(t_now + h / 5, q_now + h * k1 / 5, links, joints, hydrodyn));
        k3.noalias() = std::get<0>(dyn(t_now + 3 * h / 10, q_now + h * (3 * k1 + 9 * k2) / 40, links, joints, hydrodyn));
        k4.noalias() = std::get<0>(dyn(t_now + 4 * h / 5, q_now + h * (a4(0) * k1 + a4(1) * k2 + a4(2) * k3), links, joints, hydrodyn));
        k5.noalias() = std::get<0>(dyn(t_now + 8 * h / 9, q_now + h * (a5(0) * k1 + a5(1) * k2 + a5(2) * k3 + a5(3) * k4), links, joints, hydrodyn));
        k6.noalias() = std::get<0>(dyn(t_now + h, q_now + h * (a6(0) * k1 + a6(1) * k2 + a6(2) * k3 + a6(3) * k4 + a6(4) * k5), links, joints, hydrodyn));

        q5.noalias() = q_now + h * (a7(0) * k1 + a7(2) * k3 + a7(3) * k4 + a7(4) * k5 + a7(5) * k6);
        k2.noalias() = std::get<0>(dyn(t_now + h, q5, links, joints, hydrodyn));

        error_est.noalias() = h * (e(0) * k1 + e(1) * k2 + e(2) * k3 + e(3) * k4 + e(4) * k5 + e(5) * k6);
        double error_norm = error_est.lpNorm<Eigen::Infinity>();

        if (error_norm < tol) {
            t_now += h;
            q_now = q5;
            k1.swap(k2);
            nRej = 0;
        } else {
            if (++nRej > 1000) {
                throw std::runtime_error("Failed to solve!");
            }
        }

        h *= 0.9 * std::min(std::pow(tol / (error_norm + 1e-12), 0.2), 10.0);
    }

    return q_now;
}
