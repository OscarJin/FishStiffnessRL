#include "link.h"
#include <numeric>
#include <cmath>
#include "math_utils.h"
#include <stdexcept>

Link::Link(int num_links, double l_tot, const Eigen::VectorXd& lengths, const Fluid& fluid, const Geometry& geometry)
    : N(num_links), L(l_tot), length(lengths), fluid(fluid), geometry(geometry), foil(num_links, false), _step_size(0.001) {
    // check model
    bool err = (length.size() != N) || (std::abs(L - length.sum()) > 1e-3);
    if (err) {
        throw std::runtime_error("Error: Link parameter mismatch!");
    }

    // precompute integral segments
    l_seg.reserve(N);
    R_seg.reserve(N);
    r_seg.reserve(N);
    p_seg.reserve(N);
    for (int i = 0; i < N; i++) {
        double l_start = (i == 0) ? 0.0 : length.head(i).sum();
        double l_end = l_start + length(i);
        int _num_steps = static_cast<int>((l_end - l_start) / _step_size) + 1;
        Eigen::VectorXd ls = Eigen::VectorXd::LinSpaced(_num_steps, l_start, l_end);
        l_seg.push_back(ls);
        Eigen::VectorXd Rs = geometry.calc_R(ls);
        Eigen::VectorXd rs = geometry.calc_r(ls);
        R_seg.push_back(Rs);
        r_seg.push_back(rs);
        Eigen::VectorXd ps = M_PI * (2 * (Rs.array().square() + rs.array().square())).sqrt();
        p_seg.push_back(ps);
    }

    // compute properties
    Eigen::VectorXd massOriginal = calcBodyMass();
    mass = massOriginal.replicate(1, 2) + calcAddedMass();
    mTotal = massOriginal.sum();
    lc = calcLc();
    Izz = calcBodyIzz() + calcAddedIzz();
    xNext = length - lc;
    
    foil[N - 1] = true;
    foil_area = calcFoilArea();
}

Eigen::VectorXd Link::calcBodyMass() const {
    Eigen::VectorXd _mass(N);

    for (int i = 0; i < N; i++) {
        Eigen::VectorXd ls = l_seg[i];

        Eigen::VectorXd Rs = R_seg[i];
        Eigen::VectorXd rs = r_seg[i];
        Eigen::VectorXd dm = M_PI * (Rs.array() * rs.array()) * fluid.rho;
        _mass(i) = math_utils::trapz(dm, _step_size);
    }

    return _mass;
}

Eigen::MatrixXd Link::calcAddedMass() const {
    Eigen::VectorXd _body_mass = calcBodyMass();
    Eigen::MatrixXd added_mass = Eigen::MatrixXd::Zero(N, 2);

    for (int i = 0; i < N - 1; i++) {
        Eigen::VectorXd ls = l_seg[i];
        Eigen::VectorXd Rs = R_seg[i];

        double b = Rs.mean();
        double a = 3 * _body_mass(i) / (4 * fluid.rho * M_PI * std::pow(b, 2));

        if (b < a) {
            double e = std::sqrt(1 - std::pow(b / a, 2));
            double a0 = 2 * (1 - std::pow(e, 2)) / std::pow(e, 3) * (0.5 * std::log((1 + e) / (1 - e)) - e);
            double b0 = 1 / std::pow(e, 2) - (1 - std::pow(e, 2)) / (2 * std::pow(e, 3)) * std::log((1 + e) / (1 - e));

            double k1 = a0 / (2 - a0);
            double k2 = b0 / (2 - b0);

            added_mass(i, 0) = k1 * _body_mass(i);
            added_mass(i, 1) = k2 * _body_mass(i);
        } else {
            double e = std::sqrt(1 - std::pow(a / b, 2));
            double a0 = 2 * (1 - std::pow(e, 2)) / std::pow(e, 3) * (0.5 * std::log((1 + e) / (1 - e)) - e);
            double b0 = 1 / std::pow(e, 2) - (1 - std::pow(e, 2)) / (2 * std::pow(e, 3)) * std::log((1 + e) / (1 - e));

            double k1 = a0 / (2 - a0);
            double k2 = b0 / (2 - b0);

            added_mass(i, 0) = k2 * _body_mass(i);
            added_mass(i, 1) = k1 * _body_mass(i);
        }
    }

    return added_mass;
}

Eigen::VectorXd Link::calcLc() const {
    Eigen::VectorXd _lc = Eigen::VectorXd::Zero(N);
    Eigen::VectorXd _body_mass = calcBodyMass();

    for (int i = 0; i < N; i++) {
        double l_start = (i == 0) ? 0.0 : length.head(i).sum();
        Eigen::VectorXd ls = l_seg[i];

        Eigen::VectorXd Rs = R_seg[i];
        Eigen::VectorXd rs = r_seg[i];

        Eigen::VectorXd dm = fluid.rho * M_PI * ls.cwiseProduct(Rs).cwiseProduct(rs);
        _lc(i) = math_utils::trapz(dm, _step_size) / _body_mass(i) - l_start;
    }

    return _lc;
}

Eigen::VectorXd Link::calcBodyIzz() const {
    Eigen::VectorXd _Izz = Eigen::VectorXd::Zero(N);

    for (int i = 0; i < N; i++) {
        double l_start = (i == 0) ? 0.0 : length.head(i).sum();
        Eigen::VectorXd ls = l_seg[i];

        Eigen::VectorXd Rs = R_seg[i];
        Eigen::VectorXd rs = r_seg[i];
        Eigen::VectorXd As = M_PI * Rs.cwiseProduct(rs);

        double lci = lc(i) + l_start;
        Eigen::VectorXd _tmp = 0.25 * rs.array().pow(2) + (ls.array() - lci).array().pow(2);
        Eigen::VectorXd dI = fluid.rho * As.cwiseProduct(_tmp);
        _Izz(i) = math_utils::trapz(dI, _step_size);
    }

    return _Izz;
}

Eigen::VectorXd Link::calcAddedIzz() const {
    Eigen::VectorXd _body_mass = calcBodyMass();
    Eigen::VectorXd added_izz = Eigen::VectorXd::Zero(N);

    for (int i = 0; i < N - 1; i++) {
        Eigen::VectorXd ls = l_seg[i];
        Eigen::VectorXd Rs = R_seg[i];

        double b = Rs.mean();
        double a = 3 * _body_mass(i) / (4 * fluid.rho * M_PI * std::pow(b, 2));
        double e = std::sqrt(1 - std::pow(std::min(a, b) / std::max(a, b), 2));
        double a0 = 2 * (1 - std::pow(e, 2)) / std::pow(e, 3) * (0.5 * std::log((1 + e) / (1 - e)) - e);
        double b0 = 1 / std::pow(e, 2) - (1 - std::pow(e, 2)) / (2 * std::pow(e, 3)) * std::log((1 + e) / (1 - e));
        
        double k3 = std::pow(e, 4) * (b0 - a0) / ((2 - std::pow(e, 2)) * (2 * std::pow(e, 2) - (2 - std::pow(e, 2)) * (b0 - a0)));
        double Jf = _body_mass(i) * (std::pow(a, 2) + std::pow(b, 2)) / 5.0;
        added_izz(i) = k3 * Jf;
    }

    return added_izz;
}

double Link::calcFoilArea() const {
    Eigen::VectorXd ls = l_seg.back();
    Eigen::VectorXd Rs = R_seg.back();
    return math_utils::trapz(Rs, _step_size);
}
