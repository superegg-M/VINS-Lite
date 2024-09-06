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

    unsigned long Estimator::global_triangulate_with(const std::shared_ptr<Frame> &frame_i, bool enforce) {
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

        // 所有相机的信息
        Eigen::Vector3d t_wc_w[NUM_OF_CAM];
        Eigen::Matrix3d r_wc[NUM_OF_CAM];
        for (unsigned int j = 0; j < NUM_OF_CAM; ++j) {
            t_wc_w[j] = frame_i->p() + frame_i->q() * _t_ic[j];
            r_wc[j] = (frame_i->q() * _q_ic[j]).toRotationMatrix();
        }

        unsigned long num_triangulated_landmarks = 0;
#ifdef USE_OPENMP
#pragma omp parallel for num_threads(NUM_THREADS) reduction(+:num_triangulated_landmarks)
#endif
        for (unsigned long k = 0; k < num_points; ++k) {
#ifdef USE_QR
            Eigen::MatrixXd A(2 * match_pairs[k].size(), 3);
            Eigen::VectorXd b(2 * match_pairs[k].size(), 1);
            for (unsigned int j = 0; j < match_pairs[k].size(); ++j) {
                auto &point_j = *match_pairs[k][j];
                Vec3 t1 = r_wc[j].transpose() * t_wc_w[j];
                A.row(2 * j) = (point_j.x() * r_wc[j].col(2) - r_wc[j].col(0)).transpose();
                A.row(2 * j + 1) = (point_j.y() * r_wc[j].col(2) - r_wc[j].col(1)).transpose();
                b(2 * j) = point_j.x() * t1.z() - t1.x();
                b(2 * j + 1) = point_j.y() * t1.z() - t1.y();
            }

            // QR分解求解最小二乘问题有更高的数值精度
            auto && A_qr = A.fullPivHouseholderQr();
            Vec3 position = A_qr.solve(b);
#else
            Eigen::MatrixXd svd_A(2 * match_pairs[k].size(), 4);
            Eigen::Matrix<double, 3, 4> P;
            Eigen::Vector3d f;
            for (unsigned int j = 0; j < match_pairs[k].size(); ++j) {
                P.leftCols<3>() = r_wc[j].transpose();
                P.rightCols<1>() = -r_wc[j].transpose() * t_wc_w[j];
                auto &point_j = *match_pairs[k][j];
                svd_A.row(2 * j) = point_j[0] * P.row(2) - P.row(0);
                svd_A.row(2 * j + 1) = point_j[1] * P.row(2) - P.row(1);
            }

            // 最小二乘计算世界坐标
            Eigen::Vector4d svd_V = Eigen::JacobiSVD<Eigen::MatrixXd>(svd_A, Eigen::ComputeThinV).matrixV().rightCols<1>();
            Vec3 position = svd_V.head<3>() / svd_V[3];
#endif
            // 检查深度
            for (unsigned int j = 0; j < match_pairs[k].size(); ++j) {
                Vec3 p_cj = r_wc[j].transpose() * (position - t_wc_w[j]);
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
            landmarks[k].lock()->position = position;

            ++num_triangulated_landmarks;
        }
//#ifdef PRINT_INFO
//        std::cout << "global 2d_to_3d takes " << tri_t.toc() << " ms" << std::endl;
//#endif
        return num_triangulated_landmarks;
    }



    unsigned long Estimator::global_triangulate_with(const std::shared_ptr<Frame> &frame_i, const std::shared_ptr<Frame> &frame_j, bool enforce) {
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
        Eigen::Matrix3d r_wci = (frame_i->q() * _q_ic[0]).toRotationMatrix();

        // frame_j的信息
        Eigen::Vector3d t_wcj_w = frame_j->p() + frame_j->q() * _t_ic[0];
        Eigen::Matrix3d r_wcj = (frame_j->q() * _q_ic[0]).toRotationMatrix();

        unsigned long num_triangulated_landmarks = 0;
#ifdef USE_OPENMP
#pragma omp parallel for num_threads(NUM_THREADS) reduction(+:num_triangulated_landmarks)
#endif
        for (unsigned long k = 0; k < num_points; ++k) {
#ifdef USE_QR
            Eigen::Matrix<double, 4, 3> A;
            Eigen::Vector4d b;

            auto &point_1 = *match_pairs[k].first;
            Vec3 t1 = r_wci.transpose() * t_wci_w;
            A.row(0) = (point_1.x() * r_wci.col(2) - r_wci.col(0)).transpose();
            A.row(1) = (point_1.y() * r_wci.col(2) - r_wci.col(1)).transpose();
            b(0) = point_1.x() * t1.z() - t1.x();
            b(1) = point_1.y() * t1.z() - t1.y();

            auto &point_2 = *match_pairs[k].second;
            Vec3 t2 = r_wcj.transpose() * t_wcj_w;
            A.row(2) = (point_2.x() * r_wcj.col(2) - r_wcj.col(0)).transpose();
            A.row(3) = (point_2.y() * r_wcj.col(2) - r_wcj.col(1)).transpose();
            b(2) = point_2.x() * t2.z() - t2.x();
            b(3) = point_2.y() * t2.z() - t2.y();

            // QR分解求解最小二乘问题有更高的数值精度
            auto && A_qr = A.fullPivHouseholderQr();
            Vec3 position = A_qr.solve(b);
#else
            Eigen::Matrix<double, 4, 4> svd_A;
            Eigen::Matrix<double, 3, 4> P;
            Eigen::Vector3d f;

            P.leftCols<3>() = r_wci.transpose();
            P.rightCols<1>() = -r_wci.transpose() * t_wci_w;
            f = *match_pairs[k].first;
            svd_A.row(0) = f[0] * P.row(2) - P.row(0);
            svd_A.row(1) = f[1] * P.row(2) - P.row(1);

            P.leftCols<3>() = r_wcj.transpose();
            P.rightCols<1>() = -r_wcj.transpose() * t_wcj_w;
            f = *match_pairs[k].second;
            svd_A.row(2) = f[0] * P.row(2) - P.row(0);
            svd_A.row(3) = f[1] * P.row(2) - P.row(1);

            // 最小二乘计算世界坐标
            Eigen::Vector4d svd_V = Eigen::JacobiSVD<Eigen::MatrixXd>(svd_A, Eigen::ComputeThinV).matrixV().rightCols<1>();
            Vec3 position = svd_V.head<3>() / svd_V[3];
#endif

            // 检查深度
            Vec3 p_ci = r_wci.transpose() * (position - t_wci_w);
            if (p_ci.z() < 0.) {
                landmarks[k].lock()->is_outlier = true;
                landmarks[k].lock()->is_triangulated = false;
                continue;
            }

            Vec3 p_cj = r_wcj.transpose() * (position - t_wcj_w);
            if (p_cj.z() < 0.) {
                landmarks[k].lock()->is_outlier = true;
                landmarks[k].lock()->is_triangulated = false;
                continue;
            }

            landmarks[k].lock()->is_outlier = false;
            landmarks[k].lock()->is_triangulated = true;
            landmarks[k].lock()->position = position;

            ++num_triangulated_landmarks;
        }
//#ifdef PRINT_INFO
//        std::cout << "global 2d_to_3d takes " << tri_t.toc() << " ms" << std::endl;
//#endif
        return num_triangulated_landmarks;
    }



    bool Estimator::global_triangulate_feature(const std::shared_ptr<Landmark> &landmark, bool enforce) {
        if (!enforce && (landmark->is_triangulated || landmark->is_outlier)) {
            return false;
        }
        TicToc tri_t;

        // 优先使用双目三角化
        auto &feature_i = landmark->observations.front();   ///< frame_i的信息
        if (feature_i.second->points.size() > 1) {
            // 双目三角化
#ifdef USE_QR
            Eigen::MatrixXd A(2 * feature_i.second->points.size(), 3);
            Eigen::VectorXd b(2 * feature_i.second->points.size(), 1);
            for (unsigned long j = 0; j < feature_i.second->points.size(); ++j) {
                // 每一目的信息
                Eigen::Vector3d t_wcj_w = feature_i.first->p() + feature_i.first->q() * _t_ic[j];
                Eigen::Matrix3d r_wcj = (feature_i.first->q() * _q_ic[j]).toRotationMatrix();

                auto &point_j = feature_i.second->points[j];
                Vec3 t2 = r_wcj.transpose() * t_wcj_w;
                A.row(2 * j) = (point_j.x() * r_wcj.col(2) - r_wcj.col(0)).transpose();
                A.row(2 * j + 1) = (point_j.y() * r_wcj.col(2) - r_wcj.col(1)).transpose();
                b(2 * j) = point_j.x() * t2.z() - t2.x();
                b(2 * j + 1) = point_j.y() * t2.z() - t2.y();
            }
            // QR分解求解最小二乘问题有更高的数值精度
            auto && A_qr = A.fullPivHouseholderQr();
            Vec3 position = A_qr.solve(b);
#else
            Eigen::MatrixXd svd_A(2 * feature_i.second->points.size(), 4);
            Eigen::Matrix<double, 3, 4> P;
            for (unsigned long j = 0; j < feature_i.second->points.size(); ++j) {
                // 每一目的信息
                Eigen::Vector3d t_wcj_w = feature_i.first->p() + feature_i.first->q() * _t_ic[j];
                Eigen::Matrix3d r_wcj = (feature_i.first->q() * _q_ic[j]).toRotationMatrix();

                P.leftCols<3>() = r_wcj.transpose();
                P.rightCols<1>() = -r_wcj.transpose() * t_wcj_w;

                auto &point_j = feature_i.second->points[0];
                svd_A.row(2 * j) = point_j[0] * P.row(2) - P.row(0);
                svd_A.row(2 * j + 1) = point_j[1] * P.row(2) - P.row(1);
            }
            // 最小二乘计算世界坐标
            Eigen::Vector4d svd_V = Eigen::JacobiSVD<Eigen::MatrixXd>(svd_A, Eigen::ComputeThinV).matrixV().rightCols<1>();
            Vec3 position = svd_V.head<3>() / svd_V[3];
#endif
            // 检查深度
            for (unsigned long j = 0; j < feature_i.second->points.size(); ++j) {
                // 每一目的信息
                Eigen::Vector3d t_wcj_w = feature_i.first->p() + feature_i.first->q() * _t_ic[j];
                Eigen::Matrix3d r_wcj = (feature_i.first->q() * _q_ic[j]).toRotationMatrix();

                Vec3 p_cj = r_wcj.transpose() * (position - t_wcj_w);
                if (p_cj.z() < 0.) {
                    landmark->is_triangulated = false;
                    landmark->is_outlier = true;
                    return false;
                }
            }

            landmark->is_triangulated = true;
            landmark->is_outlier = false;
            landmark->position = position;

//#ifdef PRINT_INFO
//        std::cout << "global 2d_to_3d takes " << tri_t.toc() << " ms" << std::endl;
//#endif
            return true;

        } else {
            // 单目三角化
            if (landmark->observations.size() < 2) {
                return false;
            }

#ifdef USE_QR
            Eigen::MatrixXd A(2 * landmark->observations.size(), 3);
            Eigen::VectorXd b(2 * landmark->observations.size(), 1);
            // TODO: 使用多线程?
            for (unsigned long j = 0; j < landmark->observations.size(); ++j) {
                // frame_j的信息
                auto &feature_j = landmark->observations[j];
                Eigen::Vector3d t_wcj_w = feature_j.first->p() + feature_j.first->q() * _t_ic[0];
                Eigen::Matrix3d r_wcj = (feature_j.first->q() * _q_ic[0]).toRotationMatrix();

                auto &point_2 = feature_j.second->points[0];
                Vec3 t2 = r_wcj.transpose() * t_wcj_w;
                A.row(2 * j) = (point_2.x() * r_wcj.col(2) - r_wcj.col(0)).transpose();
                A.row(2 * j + 1) = (point_2.y() * r_wcj.col(2) - r_wcj.col(1)).transpose();
                b(2 * j) = point_2.x() * t2.z() - t2.x();
                b(2 * j + 1) = point_2.y() * t2.z() - t2.y();
            }
            // QR分解求解最小二乘问题有更高的数值精度
            auto && A_qr = A.fullPivHouseholderQr();
            Vec3 position = A_qr.solve(b);
#else
            Eigen::MatrixXd svd_A(2 * landmark->observations.size(), 4);
            Eigen::Matrix<double, 3, 4> P;
            Eigen::Vector3d f;
            for (unsigned long j = 0; j < landmark->observations.size(); ++j) {
                // 左目的信息
                auto &feature_j = landmark->observations[j];
                Eigen::Vector3d t_wcj_w = feature_j.first->p() + feature_j.first->q() * _t_ic[0];
                Eigen::Matrix3d r_wcj = (feature_j.first->q() * _q_ic[0]).toRotationMatrix();

                P.leftCols<3>() = r_wcj.transpose();
                P.rightCols<1>() = -r_wcj.transpose() * t_wcj_w;

                f = feature_j.second->points[0];
                svd_A.row(2 * j) = f[0] * P.row(2) - P.row(0);
                svd_A.row(2 * j + 1) = f[1] * P.row(2) - P.row(1);
            }
            // 最小二乘计算世界坐标
            Eigen::Vector4d svd_V = Eigen::JacobiSVD<Eigen::MatrixXd>(svd_A, Eigen::ComputeThinV).matrixV().rightCols<1>();
            Vec3 position = svd_V.head<3>() / svd_V[3];
#endif
            // 检查深度
            for (unsigned long j = 0; j < landmark->observations.size(); ++j) {
                // frame_j的信息
                auto &feature_j = landmark->observations[j];
                Eigen::Vector3d t_wcj_w = feature_j.first->p() + feature_j.first->q() * _t_ic[0];
                Eigen::Matrix3d r_wcj = (feature_j.first->q() * _q_ic[0]).toRotationMatrix();

                Vec3 p_cj = r_wcj.transpose() * (position - t_wcj_w);
                if (p_cj.z() < 0.) {
                    landmark->is_triangulated = false;
                    landmark->is_outlier = true;
                    return false;
                }
            }

            landmark->is_triangulated = true;
            landmark->is_outlier = false;
            landmark->position = position;

//#ifdef PRINT_INFO
//        std::cout << "global 2d_to_3d takes " << tri_t.toc() << " ms" << std::endl;
//#endif
            return true;
        }


    }
}