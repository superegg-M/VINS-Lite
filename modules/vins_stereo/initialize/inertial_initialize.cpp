//
// Created by Cain on 2024/4/28.
//

#include "../estimator.h"
#include "../vertex/vertex_acc_bias.h"
#include "../vertex/vertex_gyro_bias.h"
#include "../vertex/vertex_velocity.h"
#include "../edge/edge_align_linear.h"
#include "../edge/edge_align.h"

#define USE_STREAM

namespace vins {
    using namespace graph_optimization;
    using namespace std;

#ifdef USE_STREAM
    bool Estimator::align_visual_to_imu() {
        if (_stream.size() < 2) {
            return false;
        }

        /* 非线性问题, 以线性问题的结果作为初值, 进一步优化出v, alpha, R0, ba, bg */
        Problem nonlinear_problem;  // 非线性求解

        // 顶点
        double scale[1] {1.};
        double spherical[4] {0};
        double ba[3] {0};
        double bg[3] {0};
        VertexScale* vertex_scale = new VertexScale(scale);  ///< 尺度因子
        VertexSpherical* vertex_q_wb0 = new VertexSpherical(spherical);  ///< 重力方向
        VertexAccBias* vertex_ba = new VertexAccBias(ba);   ///< 加速度偏移
        VertexGyroBias* vertex_bg = new VertexGyroBias(bg);   ///< 角速度偏移
        vector<VertexVelocity*> vertex_v_vec;   ///< 速度
        vertex_v_vec.reserve(_stream.size());
        for (auto it = _stream.begin(); it != _stream.end(); ++it) {
            vertex_v_vec.emplace_back(new VertexVelocity(it->first->state + Frame::STATE::VX));
        }

        // 把顶点加入到problem中
        nonlinear_problem.add_vertex(vertex_scale);
        nonlinear_problem.add_vertex(vertex_q_wb0);
        nonlinear_problem.add_vertex(vertex_ba);
        nonlinear_problem.add_vertex(vertex_bg);
        nonlinear_problem.add_vertex(vertex_v_vec[0]);

        // 线性计算 Ax=b
#if NUM_OF_CAM > 1
        constexpr static unsigned long scale_size = 0;
#else
        constexpr static double scale_norm = 100.;
        constexpr static unsigned long scale_size = 1;
#endif
        unsigned long n_state = scale_size + 3 + 3 * _stream.size();
        MatXX A(MatXX::Zero(n_state, n_state));
        MatXX b(MatXX::Zero(n_state, 1));

        // 遍历 stream
        unsigned long i = 0;
        auto frame_i_it = _stream.begin();
        for (auto frame_j_it = _stream.begin(); frame_j_it != _stream.end(); ++frame_j_it) {
            if (frame_i_it == frame_j_it){
                ++i;
                continue;
            }

            auto frame_i = frame_i_it->first;
            auto frame_j = frame_j_it->first;

            Qd q_i = frame_i->q();
            Vec3 tij = frame_j->p() - frame_i->p();
            Qd qij = (frame_i->q().inverse() * frame_j->q()).normalized();

            /* 非线性问题的边 */
            EdgeAlign* nonlinear_edge = new EdgeAlign(frame_j_it->second.get(), tij, qij, q_i);
            nonlinear_edge->add_vertex(vertex_scale);
            nonlinear_edge->add_vertex(vertex_q_wb0);
            nonlinear_edge->add_vertex(vertex_v_vec[i - 1]);
            nonlinear_edge->add_vertex(vertex_v_vec[i]);
            nonlinear_edge->add_vertex(vertex_ba);
            nonlinear_edge->add_vertex(vertex_bg);

            // 加入到problem
            nonlinear_problem.add_edge(nonlinear_edge);
            nonlinear_problem.add_vertex(vertex_v_vec[i]);

            // Ax = b
            double dt = frame_j_it->second->get_sum_dt();
            Mat33 R_0i_T = q_i.toRotationMatrix().transpose();

            MatXX A_tmp(MatXX::Zero(6, 9 + scale_size));
            MatXX b_tmp(MatXX::Zero(6, 1));

            A_tmp.block<3, 3>(3, scale_size) = -R_0i_T * dt;
            A_tmp.block<3, 3>(0, scale_size) =  0.5 * dt * A_tmp.block<3, 3>(3, scale_size);
            A_tmp.block<3, 3>(3, 3 + scale_size) = -R_0i_T;
            A_tmp.block<3, 3>(0, 3 + scale_size) = dt * A_tmp.block<3, 3>(3, 3 + scale_size);
            A_tmp.block<3, 3>(3, 6 + scale_size) = R_0i_T;

            b_tmp.topRows<3>() = frame_j_it->second->get_delta_p();
            b_tmp.bottomRows<3>() = frame_j_it->second->get_delta_v();

#if NUM_OF_CAM > 1
            b_tmp.topRows<3>() -= R_0i_T * tij;
#else
            A_tmp.block<3, 1>(0, 0) = R_0i_T * tij / scale_norm;
#endif

            auto &&cov = frame_j_it->second->get_covariance();
            Mat66 cov_tmp;
            cov_tmp.block<3, 3>(0, 0) = cov.block<3, 3>(0, 0);
            cov_tmp.block<3, 3>(0, 3) = cov.block<3, 3>(0, 6);
            cov_tmp.block<3, 3>(3, 3) = cov.block<3, 3>(6, 6);
            cov_tmp.block<3, 3>(3, 0) = cov.block<3, 3>(6, 0);
            auto &&cov_tmp_ldlt = cov_tmp.ldlt();
            MatXX ATA = A_tmp.transpose() * cov_tmp_ldlt.solve(A_tmp);
            MatXX ATb = A_tmp.transpose() * cov_tmp_ldlt.solve(b_tmp);

            unsigned long index = scale_size + 3 + 3 * (i - 1);
            A.block<3 + scale_size, 3 + scale_size>(0, 0) += ATA.block<3 + scale_size, 3 + scale_size>(0, 0);
            A.block<3 + scale_size, 6>(0, index) += ATA.block<3 + scale_size, 6>(0, 3 + scale_size);
            A.block<6, 3 + scale_size>(index, 0) += ATA.block<6, 3 + scale_size>(3 + scale_size, 0);
            A.block<6, 6>(index, index) += ATA.block<6, 6>(3 + scale_size, 3 + scale_size);
            b.block<3 + scale_size, 1>(0, 0) += ATb.block<3 + scale_size, 1>(0, 0);
            b.block<6, 1>(index, 0) += ATb.block<6, 1>(3 + scale_size, 0);

            ++i;
            frame_i_it = frame_j_it;
        }
        // 线性问题求解
        VecX x = A.ldlt().solve(b);

        // 通过scale和g_b0的模值判断解是否可用
#if NUM_OF_CAM > 1
        double scale_est = 1.;
#else
        double scale_est = x[0] / scale_norm;
#endif
        Vec3 v_nav_est = x.segment<3>(scale_size);
        Vec3 v_nav_true = IMUIntegration::get_gravity();
        double v_nav_est2 = v_nav_est.squaredNorm();
        double v_nav_true2 = v_nav_true.squaredNorm();

#ifdef PRINT_INFO
        std::cout << "linear scale = " << scale_est << std::endl;
#endif
        if (scale_est < 0.) {    // 尺度必须大于0
            delete vertex_scale;
            delete vertex_q_wb0;
            delete vertex_ba;
            delete vertex_bg;
            for (auto &vertex_v : vertex_v_vec) {
                delete vertex_v;
            }
            for (auto &edge : nonlinear_problem.edges()) {
                delete edge;
            }
            return false;
        }
        if (v_nav_est2 > 1.1 * 1.1 * v_nav_true2 || v_nav_est2 < 0.9 * 0.9 * v_nav_true2) { // 重力加速度的模值与先验差异不得大于10%
            delete vertex_scale;
            delete vertex_q_wb0;
            delete vertex_ba;
            delete vertex_bg;
            for (auto &vertex_v : vertex_v_vec) {
                delete vertex_v;
            }
            for (auto &edge : nonlinear_problem.edges()) {
                delete edge;
            }
            return false;
        }

        // 把线性问题的求解结果当成非线性问题的初值
        double norm2 = sqrt(v_nav_est.squaredNorm() * v_nav_true.squaredNorm());
        double cos_psi = v_nav_est.dot(v_nav_true);
        Vec3 sin_psi = v_nav_est.cross(v_nav_true);
        Qd q0_est(norm2 + cos_psi, sin_psi(0), sin_psi(1), sin_psi(2));
        q0_est.normalize();
        vertex_q_wb0->q() = q0_est;
        vertex_scale->scale() = scale_est;

        for (i = 0; i < vertex_v_vec.size(); ++i) {
            vertex_v_vec[i]->v() = x.segment<3>(scale_size + 3 + 3 * i) / scale_est;
        }

        // 求解非线性问题
        nonlinear_problem.set_solver_type(graph_optimization::Problem::SolverType::LEVENBERG_MARQUARDT);
        vertex_ba->set_fixed();
        // vertex_bg->set_fixed();
#if NUM_OF_CAM > 1
        vertex_scale->set_fixed();
#endif
        nonlinear_problem.solve(50);

        // 把求解后的结果赋值到顶点中
        scale_est = vertex_scale->scale();
        Qd q_wb0 = vertex_q_wb0->q();

        // // 把yaw设为0
        // Eigen::Vector3d x_axis = {1., 0., 0.};
        // Eigen::Vector3d x_axis_est = q_wb0.toRotationMatrix().col(0);
        // double x_norm2 = sqrt(x_axis_est.head<2>().squaredNorm() * x_axis.head<2>().squaredNorm());
        // double x_sin_psi = x_axis_est(0) * x_axis(1) - x_axis_est(1) * x_axis(0);
        // double x_cos_psi = x_axis_est(0) * x_axis(0) + x_axis_est(1) * x_axis(1);
        // Eigen::Quaterniond dq_heading(x_norm2 + x_cos_psi, 0., 0., x_sin_psi);
        // dq_heading.normalize();
        // q_wb0 = (dq_heading * q_wb0).normalized();

        // 把yaw设为0
        Eigen::Vector3d ypr_0 = {0., 0., 0.};
        Eigen::Vector3d ypr_oldest = Utility::R2ypr(q_wb0.toRotationMatrix());
        double dyaw = ypr_0.x() - ypr_oldest.x();
        Eigen::Matrix3d dR = Utility::ypr2R(Eigen::Vector3d(dyaw, 0, 0));
        if (abs(abs(ypr_0.y()) - 90.) < 1. || abs(abs(ypr_oldest.y()) - 90.) < 1.) {
            dR = (q_wb0 * _vertex_pose_vec[0].q().inverse()).toRotationMatrix();
        }
        q_wb0 = Eigen::Quaterniond(dR) * q_wb0;

        // stream
        for (auto &frame_it : _stream) {
            auto frame_i = frame_it.first;
            frame_i->p() = q_wb0 * frame_i->p() * scale_est;
            frame_i->q() = q_wb0 * frame_i->q();

            frame_i->v() = q_wb0 * frame_i->v() * scale_est;
            frame_i->ba() = vertex_ba->ba();
            frame_i->bg() = vertex_bg->bg();
        }

        // 把尺度补充到逆深度中
        for (auto &landmark_it : _landmarks) {
            if (landmark_it.second->is_outlier) {
                continue;
            }
            if (!landmark_it.second->is_triangulated) {
                continue;
            }
            landmark_it.second->inv_depth /= scale_est;
        }
#ifdef PRINT_INFO
        std::cout << "nonlinear scale = " << scale_est << std::endl;
#endif

#ifdef PRINT_INFO
        std::cout << "done align_visual_to_imu" << std::endl;
        for (i = 0; i < _sliding_window.size(); ++i) {
            std::cout << "i = " << i << ":" << std::endl;
            std::cout << "q = " << _sliding_window[i].first->q().w() << ", ";
            std::cout << _sliding_window[i].first->q().x() << ", ";
            std::cout << _sliding_window[i].first->q().y() << ", ";
            std::cout << _sliding_window[i].first->q().z() << std::endl;
            std::cout << "t = " << _sliding_window[i].first->p().transpose() << std::endl;
            std::cout << "v = " << _sliding_window[i].first->v().transpose() << std::endl;
            std::cout << "ba = " << _sliding_window[i].first->ba().transpose() << std::endl;
            std::cout << "bg = " << _sliding_window[i].first->bg().transpose() << std::endl;
        }
        std::cout << "i = " << _sliding_window.size() << ":" << std::endl;
        std::cout << "q = " << _frame->q().w() << ", ";
        std::cout << _frame->q().x() << ", ";
        std::cout << _frame->q().y() << ", ";
        std::cout << _frame->q().z() << std::endl;
        std::cout << "t = " << _frame->p().transpose() << std::endl;
        std::cout << "v = " << _frame->v().transpose() << std::endl;
        std::cout << "ba = " << _frame->ba().transpose() << std::endl;
        std::cout << "bg = " << _frame->bg().transpose() << std::endl;
#endif
        delete vertex_scale;
        delete vertex_q_wb0;
        delete vertex_ba;
        delete vertex_bg;
        for (auto &vertex_v : vertex_v_vec) {
            delete vertex_v;
        }
        for (auto &edge : nonlinear_problem.edges()) {
            delete edge;
        }
        return true;
    }
#else
    bool Estimator::align_visual_to_imu() {
        if (_sliding_window.size() < 2) {
            return false;
        }

        // 线性与非线性问题的公共顶点
        vector<shared_ptr<VertexVelocity>> vertex_v_vec;   ///< 速度
        vertex_v_vec.reserve(_sliding_window.size() + 1);
        for (auto it = _sliding_window.begin(); it != _sliding_window.end(); ++it) {
            vertex_v_vec.emplace_back(make_shared<VertexVelocity>(it->first->state + Frame::STATE::VX));
        }
        vertex_v_vec.emplace_back(make_shared<VertexVelocity>(_frame->state + Frame::STATE::VX));

        /* 非线性问题, 以线性问题的结果作为初值, 进一步优化出v, alpha, R0, ba, bg */
        Problem nonlinear_problem;  // 非线性求解

        // 顶点
        double scale[1] {1.};
        double spherical[4] {0};
        double ba[3] {0};
        double bg[3] {0};
        shared_ptr<VertexScale> vertex_scale = make_shared<VertexScale>(scale);  // 尺度因子
        shared_ptr<VertexSpherical> vertex_q_wb0 = make_shared<VertexSpherical>(spherical);  // 重力方向
        shared_ptr<VertexAccBias> vertex_ba = make_shared<VertexAccBias>(ba);   // 加速度偏移
        shared_ptr<VertexGyroBias> vertex_bg = make_shared<VertexGyroBias>(bg);   // 角速度偏移

        // 把顶点加入到problem中
        nonlinear_problem.add_vertex(vertex_scale);
        nonlinear_problem.add_vertex(vertex_q_wb0);
        nonlinear_problem.add_vertex(vertex_ba);
        nonlinear_problem.add_vertex(vertex_bg);
        nonlinear_problem.add_vertex(vertex_v_vec[0]);

        // 线性计算 Ax=b
#if NUM_OF_CAM > 1
        constexpr static unsigned long scale_size = 0;
#else
        constexpr static double scale_norm = 100.;
        constexpr static unsigned long scale_size = 1;
#endif
        unsigned long n_state = scale_size + 3 + 3 * (_sliding_window.size() + 1);
        MatXX A(MatXX::Zero(n_state, n_state));
        MatXX b(MatXX::Zero(n_state, 1));

        // 遍历windows
        for (unsigned long i = 1; i < _sliding_window.size(); ++i) {
            auto frame_i = _sliding_window[i - 1].first;
            auto frame_j = _sliding_window[i].first;

            Qd q_i = frame_i->q();
            Vec3 tij = frame_j->p() - frame_i->p();
            Qd qij = (frame_i->q().inverse() * frame_j->q()).normalized();

            /* 非线性问题的边 */
            shared_ptr<EdgeAlign> nonlinear_edge = make_shared<EdgeAlign>(_sliding_window[i].second,
                                                                          tij,
                                                                          qij,
                                                                          q_i);
            nonlinear_edge->add_vertex(vertex_scale);
            nonlinear_edge->add_vertex(vertex_q_wb0);
            nonlinear_edge->add_vertex(vertex_v_vec[i - 1]);
            nonlinear_edge->add_vertex(vertex_v_vec[i]);
            nonlinear_edge->add_vertex(vertex_ba);
            nonlinear_edge->add_vertex(vertex_bg);

            // 加入到problem
            nonlinear_problem.add_edge(nonlinear_edge);
            nonlinear_problem.add_vertex(vertex_v_vec[i]);

            // Ax = b
            double dt = _sliding_window[i].second->get_sum_dt();
            Mat33 R_0i_T = q_i.toRotationMatrix().transpose();

            MatXX A_tmp(MatXX::Zero(6, 9 + scale_size));
            MatXX b_tmp(MatXX::Zero(6, 1));

            A_tmp.block<3, 3>(3, scale_size) = -R_0i_T * dt;
            A_tmp.block<3, 3>(0, scale_size) =  0.5 * dt * A_tmp.block<3, 3>(3, scale_size);
            A_tmp.block<3, 3>(3, 3 + scale_size) = -R_0i_T;
            A_tmp.block<3, 3>(0, 3 + scale_size) = dt * A_tmp.block<3, 3>(3, 3 + scale_size);
            A_tmp.block<3, 3>(3, 6 + scale_size) = R_0i_T;

            b_tmp.topRows<3>() = _sliding_window[i].second->get_delta_p();
            b_tmp.bottomRows<3>() = _sliding_window[i].second->get_delta_v();

#if NUM_OF_CAM > 1
            b_tmp.topRows<3>() -= R_0i_T * tij;
#else
            A_tmp.block<3, 1>(0, 0) = R_0i_T * tij / scale_norm;
#endif

            auto &&cov = _sliding_window[i].second->get_covariance();
            Mat66 cov_tmp;
            cov_tmp.block<3, 3>(0, 0) = cov.block<3, 3>(0, 0);
            cov_tmp.block<3, 3>(0, 3) = cov.block<3, 3>(0, 6);
            cov_tmp.block<3, 3>(3, 3) = cov.block<3, 3>(6, 6);
            cov_tmp.block<3, 3>(3, 0) = cov.block<3, 3>(6, 0);
            auto &&cov_tmp_ldlt = cov_tmp.ldlt();
            MatXX ATA = A_tmp.transpose() * cov_tmp_ldlt.solve(A_tmp);
            MatXX ATb = A_tmp.transpose() * cov_tmp_ldlt.solve(b_tmp);

            unsigned long index = scale_size + 3 + 3 * (i - 1);
            A.block<3 + scale_size, 3 + scale_size>(0, 0) += ATA.block<3 + scale_size, 3 + scale_size>(0, 0);
            A.block<3 + scale_size, 6>(0, index) += ATA.block<3 + scale_size, 6>(0, 3 + scale_size);
            A.block<6, 3 + scale_size>(index, 0) += ATA.block<6, 3 + scale_size>(3 + scale_size, 0);
            A.block<6, 6>(index, index) += ATA.block<6, 6>(3 + scale_size, 3 + scale_size);
            b.block<3 + scale_size, 1>(0, 0) += ATb.block<3 + scale_size, 1>(0, 0);
            b.block<6, 1>(index, 0) += ATb.block<6, 1>(3 + scale_size, 0);
        }

        // 当前imu
        {
            // 提取位姿
            auto frame_i = _sliding_window.back().first;

            Qd q_i = frame_i->q();
            Vec3 tij = _frame->p() - frame_i->p();
            Qd qij = (frame_i->q().inverse() * _frame->q()).normalized();

            /* 非线性问题的边 */
            shared_ptr<EdgeAlign> nonlinear_edge = make_shared<EdgeAlign>(_pre_integral_window,
                                                                          tij,
                                                                          qij,
                                                                          q_i);
            nonlinear_edge->add_vertex(vertex_scale);
            nonlinear_edge->add_vertex(vertex_q_wb0);
            nonlinear_edge->add_vertex(vertex_v_vec[_sliding_window.size() - 1]);
            nonlinear_edge->add_vertex(vertex_v_vec[_sliding_window.size()]);
            nonlinear_edge->add_vertex(vertex_ba);
            nonlinear_edge->add_vertex(vertex_bg);

            // 加入到problem
            nonlinear_problem.add_edge(nonlinear_edge);
            nonlinear_problem.add_vertex(vertex_v_vec[_sliding_window.size()]);

            // Ax = b
            double dt = _pre_integral_window->get_sum_dt();
            Mat33 R_0i_T = q_i.toRotationMatrix().transpose();

            MatXX A_tmp(MatXX::Zero(6, 9 + scale_size));
            MatXX b_tmp(MatXX::Zero(6, 1));

            A_tmp.block<3, 3>(3, scale_size) = -R_0i_T * dt;
            A_tmp.block<3, 3>(0, scale_size) =  0.5 * dt * A_tmp.block<3, 3>(3, scale_size);
            A_tmp.block<3, 3>(3, 3 + scale_size) = -R_0i_T;
            A_tmp.block<3, 3>(0, 3 + scale_size) = dt * A_tmp.block<3, 3>(3, 3 + scale_size);
            A_tmp.block<3, 3>(3, 6 + scale_size) = R_0i_T;

            b_tmp.topRows<3>() = _pre_integral_window->get_delta_p();
            b_tmp.bottomRows<3>() = _pre_integral_window->get_delta_v();

#if NUM_OF_CAM > 1
            b_tmp.topRows<3>() -= R_0i_T * tij;
#else
            A_tmp.block<3, 1>(0, 0) = R_0i_T * tij / scale_norm;
#endif

            auto &&cov = _pre_integral_window->get_covariance();
            Mat66 cov_tmp;
            cov_tmp.block<3, 3>(0, 0) = cov.block<3, 3>(0, 0);
            cov_tmp.block<3, 3>(0, 3) = cov.block<3, 3>(0, 6);
            cov_tmp.block<3, 3>(3, 3) = cov.block<3, 3>(6, 6);
            cov_tmp.block<3, 3>(3, 0) = cov.block<3, 3>(6, 0);
            auto &&cov_tmp_ldlt = cov_tmp.ldlt();
            MatXX ATA = A_tmp.transpose() * cov_tmp_ldlt.solve(A_tmp);
            MatXX ATb = A_tmp.transpose() * cov_tmp_ldlt.solve(b_tmp);

            unsigned long index = scale_size + 3 + 3 * (_sliding_window.size() - 1);
            A.block<3 + scale_size, 3 + scale_size>(0, 0) += ATA.block<3 + scale_size, 3 + scale_size>(0, 0);
            A.block<3 + scale_size, 6>(0, index) += ATA.block<3 + scale_size, 6>(0, 3 + scale_size);
            A.block<6, 3 + scale_size>(index, 0) += ATA.block<6, 3 + scale_size>(3 + scale_size, 0);
            A.block<6, 6>(index, index) += ATA.block<6, 6>(3 + scale_size, 3 + scale_size);
            b.block<3 + scale_size, 1>(0, 0) += ATb.block<3 + scale_size, 1>(0, 0);
            b.block<6, 1>(index, 0) += ATb.block<6, 1>(3 + scale_size, 0);
        }
        // 线性问题求解
        VecX x = A.ldlt().solve(b);

        // 通过scale和g_b0的模值判断解是否可用
#if NUM_OF_CAM > 1
        double scale_est = 1.;
#else
        double scale_est = x[0] / scale_norm;
#endif
        Vec3 v_nav_est = x.segment<3>(scale_size);
        Vec3 v_nav_true = IMUIntegration::get_gravity();
        double v_nav_est2 = v_nav_est.squaredNorm();
        double v_nav_true2 = v_nav_true.squaredNorm();
#ifdef PRINT_INFO
        std::cout << "linear scale = " << scale_est << std::endl;
#endif
        if (scale_est < 0.) {    // 尺度必须大于0
            return false;
        }
        if (v_nav_est2 > 1.1 * 1.1 * v_nav_true2 || v_nav_est2 < 0.9 * 0.9 * v_nav_true2) { // 重力加速度的模值与先验差异不得大于10%
            return false;
        }

        // 把线性问题的求解结果当成非线性问题的初值
        double norm2 = sqrt(v_nav_est.squaredNorm() * v_nav_true.squaredNorm());
        double cos_psi = v_nav_est.dot(v_nav_true);
        Vec3 sin_psi = v_nav_est.cross(v_nav_true);
        Qd q0_est(norm2 + cos_psi, sin_psi(0), sin_psi(1), sin_psi(2));
        q0_est.normalize();
        vertex_q_wb0->q() = q0_est;
        vertex_scale->scale() = scale_est;

        for (unsigned long i = 0; i < vertex_v_vec.size(); ++i) {
            vertex_v_vec[i]->v() = x.segment<3>(scale_size + 3 + 3 * i) / scale_est;
        }

        // 求解非线性问题
        nonlinear_problem.set_solver_type(graph_optimization::Problem::SolverType::LEVENBERG_MARQUARDT);
        vertex_ba->set_fixed();
        vertex_bg->set_fixed();
#if NUM_OF_CAM > 1
        vertex_scale->set_fixed();
#endif
        nonlinear_problem.solve(30);

        // 把求解后的结果赋值到顶点中
        scale_est = vertex_scale->scale();
        Qd q_wb0 = vertex_q_wb0->q();

        // windows
        for (unsigned long i = 0; i < _sliding_window.size(); ++i) {
            auto frame_i = _sliding_window[i].first;

            frame_i->p() = q_wb0 * frame_i->p() * scale_est;
            frame_i->q() = q_wb0 * frame_i->q();

            frame_i->v() = q_wb0 * frame_i->v() * scale_est;
            frame_i->ba() = vertex_ba->ba();
            frame_i->bg() = vertex_bg->bg();
        }

        // 当前imu
        {
            _frame->p() = q_wb0 * _frame->p() * scale_est;
            _frame->q() = q_wb0 * _frame->q();

            _frame->v() = q_wb0 * _frame->v() * scale_est;
            _frame->ba() = vertex_ba->ba();
            _frame->bg() = vertex_bg->bg();
        }

        // 把尺度补充到逆深度中
        for (auto &landmark_it : _landmarks) {
            if (landmark_it.second->is_outlier) {
                continue;
            }
            if (!landmark_it.second->is_triangulated) {
                continue;
            }
            landmark_it.second->inv_depth /= scale_est;
        }
#ifdef PRINT_INFO
        std::cout << "nonlinear scale = " << scale_est << std::endl;
#endif

#ifdef PRINT_INFO
        std::cout << "done align_visual_to_imu" << std::endl;
        for (unsigned long i = 0; i < _sliding_window.size(); ++i) {
            std::cout << "i = " << i << ":" << std::endl;
            std::cout << "q = " << _sliding_window[i].first->q().w() << ", ";
            std::cout << _sliding_window[i].first->q().x() << ", ";
            std::cout << _sliding_window[i].first->q().y() << ", ";
            std::cout << _sliding_window[i].first->q().z() << std::endl;
            std::cout << "t = " << _sliding_window[i].first->p().transpose() << std::endl;
            std::cout << "v = " << _sliding_window[i].first->v().transpose() << std::endl;
            std::cout << "ba = " << _sliding_window[i].first->ba().transpose() << std::endl;
            std::cout << "bg = " << _sliding_window[i].first->bg().transpose() << std::endl;
        }
        std::cout << "i = " << _sliding_window.size() << ":" << std::endl;
        std::cout << "q = " << _frame->q().w() << ", ";
        std::cout << _frame->q().x() << ", ";
        std::cout << _frame->q().y() << ", ";
        std::cout << _frame->q().z() << std::endl;
        std::cout << "t = " << _frame->p().transpose() << std::endl;
        std::cout << "v = " << _frame->v().transpose() << std::endl;
        std::cout << "ba = " << _frame->ba().transpose() << std::endl;
        std::cout << "bg = " << _frame->bg().transpose() << std::endl;
#endif
        return true;
    }
#endif
}