#include "dynamics.h"
#include <stdexcept>

std::pair<Eigen::Vector2d, double> calcBodyDrag(int i, const Link &links, const HydroDyn& hydrodyn, double theta, const Eigen::Vector2d &vV)
{   
    double cosTheta = std::cos(theta);
    double sinTheta = std::sin(theta);
    // compute velocity in the local frame
    double vxi = vV(0) * cosTheta + vV(1) * sinTheta;
    double vyi = -vV(0) * sinTheta + vV(1) * cosTheta;

    // define integral segments
    double l_start = (i == 0) ? 0.0 : links.length.head(i).sum();
    // Eigen::VectorXd ls = links.l_seg[i];

    // Eigen::VectorXd Rs = links.R_seg[i];
    // Eigen::VectorXd rs = links.r_seg[i];
    // Eigen::VectorXd ps = links.p_seg[i];
    auto &ls = links.l_seg[i];
    auto &Rs = links.R_seg[i];
    auto &rs = links.r_seg[i];
    auto &ps = links.p_seg[i];

    // force and torque
    Eigen::MatrixXd df(2, ls.size());
    Eigen::VectorXd dT(ls.size());

    Eigen::Vector3d ri, dTi;
    Eigen::Vector2d dfi;

    for (size_t ind = 0; ind < ls.size(); ind++) {
        dfi(0) = 0.5 * links.fluid.rho * hydrodyn.Cf(i) * ps(ind) * std::abs(vxi) * vxi;
        dfi(1) = 0.5 * links.fluid.rho * hydrodyn.Cd(i) * (2 * Rs(ind)) * std::abs(vyi) * vyi;

        // transform to global frame
        df(0, ind) = dfi(0) * cosTheta - dfi(1) * sinTheta;
        df(1, ind) = dfi(0) * sinTheta + dfi(1) * cosTheta;

        ri << (ls(ind) - links.lc(i) - l_start) * cosTheta,
              (ls(ind) - links.lc(i) - l_start) * sinTheta,
              0.0;
        dTi.noalias() = ri.cross(Eigen::Vector3d(df(0, ind), df(1, ind), 0));
        dT(ind) = dTi(2);
    }

    // integral
    Eigen::Vector2d dragF;
    dragF(0) = math_utils::trapz(df.row(0).transpose(), links._step_size);
    dragF(1) = math_utils::trapz(df.row(1).transpose(), links._step_size);
    double dragT = math_utils::trapz(dT, links._step_size);

    return {dragF, dragT};
}

Eigen::Vector2d calcThrust(double sx, double x_next, const Eigen::Vector2d &vV, double theta, double omega, double rho)
{
    double cosTheta = std::cos(theta);
    double sinTheta = std::sin(theta);
    // added mass
    double beta = 1.0;
    double ma = 0.25 * beta * rho * M_PI * sx * sx;

    // calculate velocity at penducle
    double vx = vV(0) - x_next * omega * sinTheta;
    double vy = vV(1) + x_next * omega * cosTheta;
    
    double vw = -vx * sinTheta + vy * cosTheta;     // perpendicular to link
    double vr = vx * cosTheta + vy * sinTheta;      // parallel to link

    // calculate thrust force components
    double Fr = -0.5 * ma * (vw * vw);
    double Fw = -ma * vw * vr;

    // transform force to global frame
    Eigen::Vector2d thrustF;
    thrustF(0) = Fr * cosTheta - Fw * sinTheta;
    thrustF(1) = Fr * sinTheta + Fw * cosTheta;

    return thrustF;
}

std::pair<Eigen::Vector3d, Eigen::Vector3d> calcFoilLiftDrag(double theta, double rho, const Eigen::Vector2d &vV, double foil_area,const HydroDyn& hydrodyn)
{
    if (vV.norm() < 1e-6) {
        return std::pair<Eigen::Vector3d, Eigen::Vector3d>(Eigen::Vector3d::Zero(), Eigen::Vector3d::Zero());
    }

    double cosTheta = std::cos(theta);
    double sinTheta = std::sin(theta);

    // compute direction vectors
    Eigen::Vector2d v_hat = -vV.normalized();
    Eigen::Vector2d nt(-sinTheta, cosTheta);  // normal direction of the tail

    // compute angle of attack
    double alpha = std::abs(std::asin(nt.dot(v_hat)));

    // compute lift force magnitude
    Eigen::Vector2d coeffs = hydrodyn.computeCaudalFinLiftDrag(alpha);
    double vV_sq_norm = vV.squaredNorm();
    double FL_t = 0.5 * rho * coeffs(0) * vV_sq_norm * foil_area;
    
    Eigen::Vector2d dirF = (nt.dot(v_hat) > 0) ? Eigen::Vector2d(-vV * std::sin(alpha) - nt) : Eigen::Vector2d(-vV * std::sin(alpha) + nt);

    Eigen::Vector3d foil_lift = (dirF.norm() < 1e-6) ? Eigen::Vector3d::Zero() : Eigen::Vector3d(FL_t * dirF.normalized()(0), FL_t * dirF.normalized()(1), 0);

    // compute drag force
    double FD_t = 0.5 * rho * coeffs(1) * vV_sq_norm * foil_area;
    Eigen::Vector3d foil_drag(-FD_t * v_hat(0), -FD_t * v_hat(1), 0);

    return {foil_lift, foil_drag};
}

std::pair<Eigen::SparseMatrix<double>, Eigen::VectorXd> assembleChainDynamicMatrix(
    double t, const Link &links, const Joint &joints, 
    const Eigen::VectorXd &thetas, const Eigen::VectorXd &omegas, 
    const Eigen::MatrixXd &Fs, const Eigen::VectorXd &Ts)
{
    int _N = links.N;
    int size = _N * 2 + (_N - 1) * 3 + 1;

    std::vector<Eigen::Triplet<double>> triplets;
    Eigen::VectorXd B = Eigen::VectorXd::Zero(size);

    auto addTriplet = [&](int i, int j, double val) {
        triplets.emplace_back(i, j, val);
    };

    Eigen::VectorXd sins = thetas.array().sin();
    Eigen::VectorXd coss = thetas.array().cos();

    // F = ma
    for (int i = 0; i < _N; i++) {
        addTriplet(i, i, links.mass(i, 0));
        addTriplet(_N + i, _N + i, links.mass(i, 1));
        B(i) = Fs(i, 0);
        B(_N + i) = Fs(i, 1);
    }

    for (int i = 0; i < _N - 1; i++) {
        // interactive forces
        addTriplet(i, 3 * _N + i, -1);
        addTriplet(i + 1, 3 * _N + i, 1);

        addTriplet(_N + i, 4 * _N - 1 + i, -1);
        addTriplet(_N + i + 1, 4 * _N - 1 + i, 1);

        // rotation
        if (i == 0) {
            addTriplet(2 * _N, 2 * _N, links.Izz(0));
            addTriplet(2 * _N, 2 * _N + 1, links.Izz(1));
            addTriplet(2 * _N, 3 * _N, links.xNext(0) * sins(0) + links.lc(1) * sins(1));
            addTriplet(2 * _N, 4 * _N - 1, -links.xNext(0) * coss(0) - links.lc(1) * coss(1));
        } else {
            addTriplet(2 * _N + i, 2 * _N + i + 1, links.Izz(i + 1));
            addTriplet(2 * _N + i - 1, 3 * _N + i, links.xNext(i) * sins(i));
            addTriplet(2 * _N + i, 3 * _N + i, links.lc(i + 1) * sins(i + 1));
            addTriplet(2 * _N + i - 1, 4 * _N - 1 + i, -links.xNext(i) * coss(i));
            addTriplet(2 * _N + i, 4 * _N - 1 + i, -links.lc(i + 1) * coss(i + 1));
        }

        // constraints
        addTriplet(3 * _N - 1 + i, i, 1);
        addTriplet(3 * _N - 1 + i, i + 1, -1);
        addTriplet(3 * _N - 1 + i, 2 * _N + i, -links.xNext(i) * sins(i));
        addTriplet(3 * _N - 1 + i, 2 * _N + i + 1, -links.lc(i + 1) * sins(i + 1));
        B(3 * _N - 1 + i) = (omegas(i) * omegas(i)) * links.xNext(i) * coss(i) +
                            (omegas(i + 1) * omegas(i + 1)) * links.lc(i + 1) * coss(i + 1);
        
        addTriplet(4 * _N - 2 + i, _N + i, 1);
        addTriplet(4 * _N - 2 + i, _N + i + 1, -1);
        addTriplet(4 * _N - 2 + i, 2 * _N + i, links.xNext(i) * coss(i));
        addTriplet(4 * _N - 2 + i, 2 * _N + i + 1, links.lc(i + 1) * coss(i + 1));
        B(4 * _N - 2 + i) = (omegas(i) * omegas(i)) * links.xNext(i) * sins(i) +
                            (omegas(i + 1) * omegas(i + 1)) * links.lc(i + 1) * sins(i + 1);
        

    }

    B.segment(2 * _N, _N - 1) = Ts;

    // system input
    addTriplet(size - 1, _N * 2, -1);
    addTriplet(size - 1, _N * 2 + 1, 1);
    B(size - 1) = -joints.amp * std::pow(2 * M_PI * joints.freq, 2) * std::sin(2 * M_PI * joints.freq * t + joints.phase);

    Eigen::SparseMatrix<double> M(size, size);
    M.setFromTriplets(triplets.begin(), triplets.end());

    return {M, B};
}

std::tuple<Eigen::VectorXd, Eigen::Vector2d, double, Eigen::VectorXd> dyn(
    double t, 
    const Eigen::VectorXd &q, 
    const Link &links, 
    const Joint &joints, 
    const HydroDyn &hydrodyn) 
{
    int _N = links.N;
    if (q.size() != 6 * _N) {
        throw std::runtime_error("q size error!");
    }
    Eigen::VectorXd vxs = q.segment(2 * _N, _N);
    Eigen::VectorXd vys = q.segment(3 * _N, _N);
    Eigen::VectorXd thetas = q.segment(4 * _N, _N);
    Eigen::VectorXd omegas = q.segment(5 * _N, _N);

    /* HYDRODYNAMICS */
    Eigen::MatrixXd Fs = Eigen::MatrixXd::Zero(_N, 2);
    Eigen::VectorXd Ts = Eigen::VectorXd::Zero(_N - 1);
    double headT = 0.0;

    Eigen::Vector2d vV, dragF, thrustF;
    Eigen::Vector3d thrustT, tailF, tailT;
    for (int i = 0; i < _N; i++) {
        vV.noalias() = links.fluid.flowV - Eigen::Vector2d(vxs(i), vys(i));

        if (!links.foil[i]) {
            // drag
            // Eigen::Vector2d dragF;
            double dragT;
            std::tie(dragF, dragT) = calcBodyDrag(i, links, hydrodyn, thetas(i), vV);
            Fs.row(i).noalias() += dragF.transpose();
            if (i == 0) {
                Ts(i) += dragT;
                headT += dragT;
            } else {
                Ts(i - 1) += dragT;
            }
            // thrust
            if (i == _N - 2) {
                double sx = 2 * links.geometry.calc_R(links.length.head(i + 1).sum());
                thrustF.noalias() = calcThrust(sx, links.xNext(i), -vV - Eigen::Vector2d(vxs(0), vys(0)), thetas(i), omegas(i), links.fluid.rho);
                thrustT.noalias() = links.xNext(i) * Eigen::Vector3d(std::cos(thetas(i)), std::sin(thetas(i)), 0).cross(Eigen::Vector3d(thrustF(0), thrustF(1), 0));
                Fs.row(i).noalias() += thrustF.transpose();
                if (i == 0) {
                    Ts(i) += thrustT(2);
                    headT += thrustT(2);
                } else {
                    Ts(i - 1) += thrustT(2);
                }
            } 
        } else {
            // lift and drag on foil (caudal fin)
            std::pair<Eigen::Vector3d, Eigen::Vector3d> foilF = calcFoilLiftDrag(thetas(i), links.fluid.rho, vV, links.foil_area, hydrodyn);
            tailF.noalias() = foilF.first + foilF.second;

            double foil_center = links.length(i) / 3.0 - links.lc(i);
            tailT.noalias() = foil_center * Eigen::Vector3d(std::cos(thetas(i)), std::sin(thetas(i)), 0).cross(tailF);

            Fs.row(i).noalias() += tailF.head(2).transpose();
            Ts(i - 1) += tailT(2);
        }
    }

    /* JOINT TORQUES */
    for (int i = 1; i < joints.nJoint; ++i) {
        double jointT = joints.k(i) * (thetas(i + 1) - thetas(i)) + joints.mu(i) * (omegas(i + 1) - omegas(i));
        Ts(i - 1) += jointT;
        Ts(i) -= jointT;
    }

    /* SYSTEM DYNAMICS */
    // Eigen::MatrixXd M;
    // Eigen::VectorXd B;
    // std::tie(M, B) = assembleChainDynamicMatrix(t, links, joints, thetas, omegas, Fs, Ts);
    // Eigen::ColPivHouseholderQR<Eigen::MatrixXd> solver(M);
    // if (!solver.isInvertible()) {
    //     throw std::runtime_error("Sigular matrix!");
    // }
    
    auto [M, B] = assembleChainDynamicMatrix(t, links, joints, thetas, omegas, Fs, Ts);
    Eigen::SparseLU<Eigen::SparseMatrix<double>> solver;
    solver.analyzePattern(M);
    solver.setPivotThreshold(0.1);
    solver.factorize(M);
    if(solver.info() != Eigen::Success) {
        throw std::runtime_error("Matrix decomposition failed!");
    }
    Eigen::VectorXd delta = solver.solve(B);

    Eigen::VectorXd axs = delta.segment(0, _N);
    Eigen::VectorXd ays = delta.segment(_N, _N);
    Eigen::VectorXd alphas = delta.segment(2 * _N, _N);
    Eigen::VectorXd F_in = delta.segment(3 * _N, 2 * joints.nJoint);

    Eigen::VectorXd dq(6 * _N);
    dq << vxs, vys, axs, ays, omegas, alphas;

    return {dq, thrustF, headT, F_in};
}
