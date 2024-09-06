//
// Created by Cain on 2024/4/28.
//

#include "../estimator.h"

#include <array>
#include <memory>
#include <random>
#include <iostream>
#include <ostream>
#include <fstream>

#define USE_QR

namespace vins {
    using namespace graph_optimization;
    using namespace std;

    unsigned long Estimator::local_triangulate_with(const std::shared_ptr<Frame> &frame_i, bool enforce) {
#if NUM_OF_CAM < 2
        return 0;
#endif
        TicToc tri_t;

        unsigned long max_num_points = frame_i->features.size() * (NUM_OF_CAM - 1);
        vector<vector<const Vec3*>> match_pairs;
        match_pairs.reserve(max_num_points);
        vector<std::weak_ptr<Landmark>> landmarks;
        landmarks.reserve(max_num_points);

        // 找出匹配对
        for (auto &feature_i : frame_i->features) {
            if (!enforce && (feature_i.second->landmark.lock()->is_triangulated || feature_i.second->landmark.lock()->is_outlier)) {
                continue;
            }

            if (feature_i.second->points.size() < 2) {
                continue;
            }

            unsigned long landmark_id = feature_i.first;
            if (_landmarks.find(landmark_id) == _landmarks.end()) {
                std::cerr << "Error: feature not in feature_map when running pnp" << std::endl;
                continue;
            }

            match_pairs.emplace_back(feature_i.second->points.size());
            for (unsigned int j = 0; j < feature_i.second->points.size(); ++j) {
                match_pairs.back()[j] = &feature_i.second->points[j];
            }
            landmarks.emplace_back(feature_i.second->landmark);
        }

        unsigned long num_points = match_pairs.size();
        if (num_points == 0) {
            return 0;
        }

        unsigned long num_triangulated_landmarks = 0;
#ifdef USE_OPENMP
#pragma omp parallel for num_threads(NUM_THREADS) reduction(+:num_triangulated_landmarks)
#endif
        for (unsigned long k = 0; k < num_points; ++k) {
            // 三角化
#ifdef USE_QR
            Eigen::VectorXd A(2 * match_pairs[k].size() - 2, 1);
            Eigen::VectorXd b(2 * match_pairs[k].size() - 2, 1);
            const Eigen::Vector3d &point_i = *match_pairs[k][0];
            for (unsigned int j = 1; j < match_pairs[k].size(); ++j) {
                const Eigen::Vector3d &point_j = *match_pairs[k][j];
                Eigen::Vector3d f = (_q_ic[j].inverse() * _q_ic[0]) * point_i;
                Eigen::Vector3d t = _q_ic[j].inverse() * (_t_ic[0] - _t_ic[j]);

                A(2 * j - 2) = f.x() - f.z() * point_j.x();
                A(2 * j - 1) = f.y() - f.z() * point_j.y();
                b(2 * j - 2) = t.z() * point_j.x() - t.x();
                b(2 * j - 1) = t.z() * point_j.y() - t.y();
            }
//            double inverse_depth = 1. / A.fullPivHouseholderQr().solve(b)[0];
            double inverse_depth = A.squaredNorm() / A.dot(b);
#else
            Eigen::MatrixXd svd_A(2 * match_pairs[k].size(), 4);
            Eigen::Matrix<double, 3, 4> P;

            const Eigen::Vector3d &point_i = *match_pairs[k][0];
            P.leftCols<3>().setIdentity();
            P.rightCols<1>().setZero();
            svd_A.row(0) = point_i[0] * P.row(2) - P.row(0);
            svd_A.row(1) = point_i[1] * P.row(2) - P.row(1);
            for (unsigned j = 1; j < match_pairs[k].size(); ++j) {
                const Eigen::Vector3d &point_j = *match_pairs[k][j];
                P.leftCols<3>() = (_q_ic[j].inverse() * _q_ic[0]).toRotationMatrix();
                P.rightCols<1>() = _q_ic[j].inverse() * (_t_ic[0] - _t_ic[j]);
                svd_A.row(2 * j) = point_j[0] * P.row(2) - P.row(0);
                svd_A.row(2 * j + 1) = point_j[1] * P.row(2) - P.row(1);
            }

            // 最小二乘计算深度
            Eigen::Vector4d svd_V = Eigen::JacobiSVD<Eigen::MatrixXd>(svd_A, Eigen::ComputeThinV).matrixV().rightCols<1>();
            double inverse_depth {svd_V[3] / svd_V[2]};
#endif
            // 检查深度
            if (inverse_depth < 0.) {
                landmarks[k].lock()->is_outlier = true;
                landmarks[k].lock()->is_triangulated = false;
                continue;
            }
            for (unsigned j = 1; j < match_pairs[k].size(); ++j) {
                Vec3 p_cj = _q_ic[j].inverse() * (_q_ic[0] * point_i / inverse_depth + _t_ic[0] - _t_ic[j]);
                if (p_cj.z() < 0.) {
                    landmarks[k].lock()->is_outlier = true;
                    landmarks[k].lock()->is_triangulated = false;
                    break;
                }
            }
            if (landmarks[k].lock()->is_outlier) {
                continue;
            }

            landmarks[k].lock()->is_outlier = false;
            landmarks[k].lock()->is_triangulated = true;
            landmarks[k].lock()->inv_depth = inverse_depth;

            ++num_triangulated_landmarks;
        }
        return num_triangulated_landmarks;
    }



    unsigned long Estimator::local_triangulate_with(const std::shared_ptr<Frame> &frame_i, const std::shared_ptr<Frame> &frame_j, bool enforce) {
        TicToc tri_t;
        if (frame_i.get() == frame_j.get()) {
            return 0;
        }

        unsigned long max_num_points = max(frame_i->features.size(), frame_j->features.size());
        vector<pair<const Vec3*, const Vec3*>> match_pairs;
        match_pairs.reserve(max_num_points);
        vector<std::weak_ptr<Landmark>> landmarks;
        landmarks.reserve(max_num_points);

        // 找出匹配对
        for (auto &feature_i : frame_i->features) {
            if (!enforce && (feature_i.second->landmark.lock()->is_triangulated || feature_i.second->landmark.lock()->is_outlier)) {
                continue;
            }

            unsigned long landmark_id = feature_i.first;
            auto &&feature_j = frame_j->features.find(landmark_id);
            if (feature_j == frame_j->features.end()) {
                continue;
            }

            if (_landmarks.find(landmark_id) == _landmarks.end()) {
                std::cerr << "Error: feature not in feature_map when running pnp" << std::endl;
                continue;
            }

            match_pairs.emplace_back(&feature_i.second->points[0], &feature_j->second->points[0]);
            landmarks.emplace_back(feature_i.second->landmark);
        }

        unsigned long num_points = match_pairs.size();
        if (num_points == 0) {
            return 0;
        }

        // frame_i的信息
        Eigen::Vector3d t_wci_w = frame_i->p() + frame_i->q() * _t_ic[0];
        Eigen::Quaterniond q_wci = frame_i->q() * _q_ic[0];

        // frame_j的信息
        Eigen::Vector3d t_wcj_w = frame_j->p() + frame_j->q() * _t_ic[0];
        Eigen::Quaterniond q_wcj = frame_j->q() * _q_ic[0];

        unsigned long num_triangulated_landmarks = 0;
#ifdef USE_OPENMP
#pragma omp parallel for num_threads(NUM_THREADS) reduction(+:num_triangulated_landmarks)
#endif
        for (unsigned long k = 0; k < num_points; ++k) {
            // 三角化
#ifdef USE_QR
            Eigen::Matrix<double, 2, 1> A;
            Eigen::Matrix<double, 2, 1> b;

            const Eigen::Vector3d &point_i = *match_pairs[k].first;
            const Eigen::Vector3d &point_j = *match_pairs[k].second;
            Eigen::Vector3d f = (q_wcj.inverse() * q_wci) * point_i;
            Eigen::Vector3d t = q_wcj.inverse() * (t_wci_w - t_wcj_w);

            A(0, 0) = f.x() - f.z() * point_j.x();
            A(1, 0) = f.y() - f.z() * point_j.y();
            b(0) = t.z() * point_j.x() - t.x();
            b(1) = t.z() * point_j.y() - t.y();
//            double inverse_depth = 1. / A.fullPivHouseholderQr().solve(b)[0];
            double inverse_depth = A.squaredNorm() / A.dot(b);
#else
            Eigen::MatrixXd svd_A(4, 4);
            Eigen::Matrix<double, 3, 4> P;
            Eigen::Vector3d f;

            P.leftCols<3>().setIdentity();
            P.rightCols<1>().setZero();
            f = *match_pairs[k].first;
            svd_A.row(0) = f[0] * P.row(2) - P.row(0);
            svd_A.row(1) = f[1] * P.row(2) - P.row(1);

            P.leftCols<3>() = (q_wcj.inverse() * q_wci).toRotationMatrix();
            P.rightCols<1>() = q_wcj.inverse() * (t_wci_w - t_wcj_w);
            f = *match_pairs[k].second;
            svd_A.row(2) = f[0] * P.row(2) - P.row(0);
            svd_A.row(3) = f[1] * P.row(2) - P.row(1);

            // 最小二乘计算深度
            Eigen::Vector4d svd_V = Eigen::JacobiSVD<Eigen::MatrixXd>(svd_A, Eigen::ComputeThinV).matrixV().rightCols<1>();
            double inverse_depth {svd_V[3] / svd_V[2]};
#endif
            // 检查深度
            if (inverse_depth < 0.) {
                landmarks[k].lock()->is_outlier = true;
                landmarks[k].lock()->is_triangulated = false;
                continue;
            }
            Vec3 p_cj = q_wcj.inverse() * (q_wci * point_i / inverse_depth + t_wci_w - t_wcj_w);
            if (p_cj.z() < 0.) {
                landmarks[k].lock()->is_outlier = true;
                landmarks[k].lock()->is_triangulated = false;
                continue;
            }

            landmarks[k].lock()->is_outlier = false;
            landmarks[k].lock()->is_triangulated = true;
            landmarks[k].lock()->inv_depth = inverse_depth;

            ++num_triangulated_landmarks;
        }
        return num_triangulated_landmarks;
    }

    bool Estimator::local_triangulate_feature(const std::shared_ptr<Landmark> &landmark, bool enforce) {
        if (!enforce && (landmark->is_triangulated || landmark->is_outlier)) {
            return false;
        }
        TicToc tri_t;

        // 优先使用双目三角化
        auto &feature_i = landmark->observations.front();   ///< frame_i的信息
        if (feature_i.second->points.size() > 1) {
            // 双目三角化
            Eigen::Vector3d &point_i = feature_i.second->points[0];
#ifdef USE_QR
            Eigen::VectorXd A(2 * (feature_i.second->points.size() - 1), 1);
            Eigen::VectorXd b(2 * (feature_i.second->points.size() - 1), 1);
            for (size_t j = 1; j < feature_i.second->points.size(); ++j) {
                // 右目的信息
                const Eigen::Vector3d &point_j = feature_i.second->points[j];
                Eigen::Vector3d f = (_q_ic[j].inverse() * _q_ic[0]) * point_i;
                Eigen::Vector3d t = _q_ic[j].inverse() * (_t_ic[0] - _t_ic[j]);

                A(2 * j - 2) = f.x() - f.z() * point_j.x();
                A(2 * j - 1) = f.y() - f.z() * point_j.y();
                b(2 * j - 2) = t.z() * point_j.x() - t.x();
                b(2 * j - 1) = t.z() * point_j.y() - t.y();
            }
//            double inverse_depth = 1. / A.fullPivHouseholderQr().solve(b)[0];
            double inverse_depth = A.squaredNorm() / A.dot(b);
#else
            Eigen::MatrixXd svd_A(2 * feature_i.second->points.size(), 4);
            Eigen::Matrix<double, 3, 4> P;

            P.leftCols<3>().setIdentity();
            P.rightCols<1>().setZero();
            svd_A.row(0) = point_i[0] * P.row(2) - P.row(0);
            svd_A.row(1) = point_i[1] * P.row(2) - P.row(1);

            for (size_t j = 1; j < feature_i.second->points.size(); ++j) {
                // 右目的信息
                P.leftCols<3>() = (_q_ic[j].inverse() * _q_ic[0]).toRotationMatrix();
                P.rightCols<1>() = _q_ic[j].inverse() * (_t_ic[0] - _t_ic[j]);

                Eigen::Vector3d &point_j = feature_i.second->points[j];
                svd_A.row(2 * j) = point_j[0] * P.row(2) - P.row(0);
                svd_A.row(2 * j + 1) = point_j[1] * P.row(2) - P.row(1);
            }

            // 最小二乘计算深度
            Eigen::Vector4d svd_V = Eigen::JacobiSVD<Eigen::MatrixXd>(svd_A, Eigen::ComputeThinV).matrixV().rightCols<1>();
            double inverse_depth {svd_V[3] / svd_V[2]};
#endif
            // 检查深度
            if (inverse_depth < 0.) {
                landmark->is_triangulated = false;
                landmark->is_outlier = true;
                return false;
            }

            for (size_t j = 1; j < feature_i.second->points.size(); ++j) {
                // 右目的信息
                Vec3 p_cj = _q_ic[j].inverse() * (_q_ic[0] * point_i / inverse_depth + _t_ic[0] - _t_ic[j]);
                if (p_cj.z() < 0.) {
                    landmark->is_triangulated = false;
                    landmark->is_outlier = true;
                    return false;
                }
            }

            landmark->is_triangulated = true;
            landmark->is_outlier = false;
            landmark->inv_depth = inverse_depth;
            return true;
        } else {
            // 单目三角化
            if (landmark->observations.size() < 2) {
                return false;
            }

            Eigen::Vector3d t_wci_w = feature_i.first->p() + feature_i.first->q() * _t_ic[0];
            Eigen::Quaterniond q_wci = feature_i.first->q() * _q_ic[0];
            Eigen::Vector3d &point_i = feature_i.second->points[0];
#ifdef USE_QR
            Eigen::VectorXd A(2 * (landmark->observations.size() - 1), 1);
            Eigen::VectorXd b(2 * (landmark->observations.size() - 1), 1);
            // TODO: 使用多线程?
            for (size_t j = 1; j < landmark->observations.size(); ++j) {
                // frame_j的信息
                auto &feature_j = landmark->observations[j];
                Eigen::Vector3d t_wcj_w = feature_j.first->p() + feature_j.first->q() * _t_ic[0];
                Eigen::Quaterniond q_wcj = feature_j.first->q() * _q_ic[0];

                Eigen::Vector3d &point_j = feature_j.second->points[0];
                Eigen::Vector3d f = (q_wcj.inverse() * q_wci) * point_i;
                Eigen::Vector3d t = q_wcj.inverse() * (t_wci_w - t_wcj_w);

                A(2 * j - 2) = f.x() - f.z() * point_j.x();
                A(2 * j - 1) = f.y() - f.z() * point_j.y();
                b(2 * j - 2) = t.z() * point_j.x() - t.x();
                b(2 * j - 1) = t.z() * point_j.y() - t.y();
            }
//        double inverse_depth = 1. / A.fullPivHouseholderQr().solve(b)[0];
            double inverse_depth = A.squaredNorm() / A.dot(b);
#else
            Eigen::MatrixXd svd_A(2 * landmark->observations.size(), 4);
            Eigen::Matrix<double, 3, 4> P;

            P.leftCols<3>().setIdentity();
            P.rightCols<1>().setZero();
            f = feature_i.second->points[0];
            svd_A.row(0) = point_i[0] * P.row(2) - P.row(0);
            svd_A.row(1) = point_i[1] * P.row(2) - P.row(1);

            for (size_t j = 1; j < landmark->observations.size(); ++j) {
                // frame_j的信息
                auto &feature_j = landmark->observations[j];
                Eigen::Vector3d t_wcj_w = feature_j.first->p() + feature_j.first->q() * _t_ic[0];
                Eigen::Quaterniond q_wcj = feature_j.first->q() * _q_ic[0];

                P.leftCols<3>() = (q_wcj.inverse() * q_wci).toRotationMatrix();
                P.rightCols<1>() = q_wcj.inverse() * (t_wci_w - t_wcj_w);

                Eigen::Vector3d &point_j = feature_j.second->points[0];
                svd_A.row(2 * j) = point_j[0] * P.row(2) - P.row(0);
                svd_A.row(2 * j + 1) = point_j[1] * P.row(2) - P.row(1);
            }

            // 最小二乘计算深度
            Eigen::Vector4d svd_V = Eigen::JacobiSVD<Eigen::MatrixXd>(svd_A, Eigen::ComputeThinV).matrixV().rightCols<1>();
            double inverse_depth {svd_V[3] / svd_V[2]};
#endif
            // 检查深度
            if (inverse_depth < 0.) {
                landmark->is_triangulated = false;
                landmark->is_outlier = true;
                return false;
            }
            for (size_t j = 1; j < landmark->observations.size(); ++j) {
                // frame_j的信息
                auto &feature_j = landmark->observations[j];
                Eigen::Vector3d t_wcj_w = feature_j.first->p() + feature_j.first->q() * _t_ic[0];
                Eigen::Quaterniond q_wcj = feature_j.first->q() * _q_ic[0];

                Vec3 p_cj = q_wcj.inverse() * (q_wci * feature_i.second->points[0] / inverse_depth + t_wci_w - t_wcj_w);
                if (p_cj.z() < 0.) {
                    landmark->is_triangulated = false;
                    landmark->is_outlier = true;
                    return false;
                }
            }

            landmark->is_triangulated = true;
            landmark->is_outlier = false;
            landmark->inv_depth = inverse_depth;
            return true;
        }
    }
}