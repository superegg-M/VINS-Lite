//
// Created by Cain on 2024/4/28.
//

#include "../estimator.h"

#include "graph_optimization/eigen_types.h"

#include <array>
#include <memory>
#include <random>
#include <iostream>
#include <ostream>
#include <fstream>

namespace vins {
    using namespace graph_optimization;
    using namespace std;

    bool Estimator::iter_pnp(const std::shared_ptr<Frame> &frame_i, Qd *q_wi_init, Vec3 *t_wi_init, unsigned int num_iters) {
        constexpr static unsigned long th_count = 6;
        constexpr static double th_e2 = 3.841;
        constexpr static unsigned int num_fix = 5;
        TicToc pnp_t;

        // 读取3d, 2d点
        vector<const Vec3*> p_w;
        vector<unsigned long> p_w_id;
        vector<const Vec3*> uv;
        p_w.reserve(frame_i->features.size());
        p_w_id.reserve(frame_i->features.size());
        uv.reserve(frame_i->features.size());
        for (auto &feature : frame_i->features) {
            auto &&landmark_it = _landmarks.find(feature.first);
            if (landmark_it == _landmarks.end()) {
                // 如果传入的是 _stream 中的 frame, 会存在 landmark 已不在 _landmarks 中的情况
//                std::cerr << "Error: feature not in feature_map when running pnp" << std::endl;
                continue;
            }

            if (landmark_it->second->is_outlier) {
                continue;
            }

            if (!landmark_it->second->is_triangulated) {
                continue;
            }

            p_w.emplace_back(&landmark_it->second->position);
            p_w_id.emplace_back(landmark_it->first);
            uv.emplace_back(&feature.second->points[0]);
        }

        // 特征点个数必须大于一定数量
        if (p_w.size() < th_count) {
            frame_i->is_initialized = false;
            return false;
        }

        // 初始化
        auto &&q_wi = frame_i->q();
        auto &&t_wi = frame_i->p();
        if (q_wi_init) {
            q_wi = *q_wi_init;
        }
        if (t_wi_init) {
            t_wi = *t_wi_init;
        }

        // TODO: 应该使用RANSAC
        for (unsigned int n = 0; n < num_iters; ++n) {
            Mat66 H;
            Vec6 b;
            H.setZero();
            b.setZero();
            for (unsigned long k = 0; k < p_w.size(); ++k) {
                Vec3 p_imu_i = q_wi.inverse() * (*p_w[k] - t_wi);
                Vec3 p_camera_i = _q_ic[0].inverse() * (p_imu_i - _t_ic[0]);
                double inv_depth_i = 1. / p_camera_i.z();

                // 重投影误差
                Vec2 e = (p_camera_i * inv_depth_i).head<2>() - uv[k]->head<2>();

                // 误差对投影点的偏导
                Mat23 dr_dpci;
                dr_dpci << inv_depth_i, 0., -p_camera_i[0] * inv_depth_i * inv_depth_i,
                        0., inv_depth_i, -p_camera_i[1] * inv_depth_i * inv_depth_i;

                // 投影点对imu位姿的偏导
                Eigen::Matrix<double, 3, 6> dpci_dpose_i;
                Mat33 R_ic = _q_ic[0].toRotationMatrix();
                dpci_dpose_i.leftCols<3>() = -R_ic.transpose() * q_wi.inverse();
                dpci_dpose_i.rightCols<3>() = R_ic.transpose() * Sophus::SO3d::hat(p_imu_i);

                // Jacobian
                Eigen::Matrix<double, 2, 6> jacobian_pose_i;
                jacobian_pose_i = dr_dpci * dpci_dpose_i;

                H += jacobian_pose_i.transpose() * jacobian_pose_i;
                b -= jacobian_pose_i.transpose() * e;
            }

            auto H_ldlt = H.ldlt();

            // 修复GN无解的情况
            if (H_ldlt.info() != Eigen::Success) {
                Vec6 lambda;
                for (unsigned int i = 0; i < 6; ++i) {
                    lambda[i] = min(max(H(i, i) * 1e-5, 1e-6), 1e6);
                }

                double v = 2.;
                for (unsigned int m = 0; m < num_fix; ++m) {
                    for (unsigned int i = 0; i < 6; ++i) {
                        H(i, i) += v * lambda[i];
                    }
                    H_ldlt = H.ldlt();
                    if (H_ldlt.info() == Eigen::Success){
                        break;
                    }
                    v *= 2.;
                }
            }

            // 只有成功才能进行更新
            if (H_ldlt.info() == Eigen::Success) {
                Vec6 delta = H_ldlt.solve(b);
                t_wi += delta.head(3);
                q_wi *= Sophus::SO3d::exp(delta.tail(3)).unit_quaternion();
                q_wi.normalize();
            } else {
                frame_i->is_initialized = false;
                return false;
            }
        }

        // 判断landmark是否outlier
        unsigned long outlier_count = 0;
        for (unsigned long k = 0; k < p_w.size(); ++k) {
            auto id = p_w_id[k];

            Vec3 p_imu_i = q_wi.inverse() * (*p_w[k] - t_wi);
            Vec3 p_camera_i = _q_ic[0].inverse() * (p_imu_i - _t_ic[0]);
            double inv_depth_i = 1. / p_camera_i.z();
            // 逆深度
            if (inv_depth_i < 0.) {
                _landmarks[id]->is_outlier = true;
            }

            // 重投影误差
            Vec2 e = (p_camera_i * inv_depth_i).head<2>() - uv[k]->head<2>();
            double e2 = e.squaredNorm();
            if (e2 > th_e2) {
                _landmarks[id]->is_outlier = true;
                ++outlier_count;
            }
        }

        if (outlier_count * 100 > p_w.size() * 30) {
            frame_i->is_initialized = false;
            return false;
        }

#ifdef PRINT_INFO
        std::cout << "p_w.size() = " << p_w.size() << std::endl;
        std::cout << "iter_pnp takes " << pnp_t.toc() << " ms" << std::endl;
        std::cout << "iter_pnp: q = " << q_wi.w() << ", " << q_wi.x() << ", " << q_wi.y() << ", " << q_wi.z() << std::endl;
        std::cout << "iter_pnp: t = " << t_wi.transpose() << std::endl;
#endif
        frame_i->is_initialized = true;
        return true;
    }



    bool Estimator::iter_pnp_local(const std::shared_ptr<Frame> &frame_i, Qd *q_wi_init, Vec3 *t_wi_init, unsigned int num_iters) {
        constexpr static unsigned long th_count = 6;
        constexpr static double th_e2 = 3.841;
        constexpr static unsigned int num_fix = 5;
        TicToc pnp_t;

        // 读取3d, 2d点
        vector<const Vec3*> p_w;
        vector<unsigned long> p_w_id;
        vector<const Vec3*> uv;
        p_w.reserve(frame_i->features.size());
        p_w_id.reserve(frame_i->features.size());
        uv.reserve(frame_i->features.size());
        for (auto &feature : frame_i->features) {
            auto &&landmark_it = _landmarks.find(feature.first);
            if (landmark_it == _landmarks.end()) {
                // 如果传入的是 _stream 中的 frame, 会存在 landmark 已不在 _landmarks 中的情况
//                std::cerr << "Error: feature not in feature_map when running pnp" << std::endl;
                continue;
            }

            if (landmark_it->second->is_outlier) {
                continue;
            }

            if (!landmark_it->second->is_triangulated) {
                continue;
            }

            auto &observations = landmark_it->second->observations;
            if (observations.size() < 2) {
                continue;
            }

            // 计算特征点的世界坐标
            auto observation_host = observations.front();
            Vec3 p_camera_host = observation_host.second->points[0] / landmark_it->second->inv_depth;
            Vec3 p_imu_host = _q_ic[0] * p_camera_host + _t_ic[0];
            landmark_it->second->position = observation_host.first->q() * p_imu_host + observation_host.first->p();

            p_w.emplace_back(&landmark_it->second->position);
            p_w_id.emplace_back(landmark_it->first);
            uv.emplace_back(&feature.second->points[0]);
        }

        // 特征点个数必须大于一定数量
        if (p_w.size() < th_count) {
            frame_i->is_initialized = false;
            return false;
        }

        // 初始化
        auto &&q_wi = frame_i->q();
        auto &&t_wi = frame_i->p();
        if (q_wi_init) {
            q_wi = *q_wi_init;
        }
        if (t_wi_init) {
            t_wi = *t_wi_init;
        }

        // TODO: 应该使用RANSAC
        for (unsigned int n = 0; n < num_iters; ++n) {
            Mat66 H;
            Vec6 b;
            H.setZero();
            b.setZero();
            for (unsigned long k = 0; k < p_w.size(); ++k) {
                Vec3 p_imu_i = q_wi.inverse() * (*p_w[k] - t_wi);
                Vec3 p_camera_i = _q_ic[0].inverse() * (p_imu_i - _t_ic[0]);
                double inv_depth_i = 1. / p_camera_i.z();

                // 重投影误差
                Vec2 e = (p_camera_i * inv_depth_i).head<2>() - uv[k]->head<2>();

                // 误差对投影点的偏导
                Mat23 dr_dpci;
                dr_dpci << inv_depth_i, 0., -p_camera_i[0] * inv_depth_i * inv_depth_i,
                        0., inv_depth_i, -p_camera_i[1] * inv_depth_i * inv_depth_i;

                // 投影点对imu位姿的偏导
                Eigen::Matrix<double, 3, 6> dpci_dpose_i;
                Mat33 R_ic = _q_ic[0].toRotationMatrix();
                dpci_dpose_i.leftCols<3>() = -R_ic.transpose() * q_wi.inverse();
                dpci_dpose_i.rightCols<3>() = R_ic.transpose() * Sophus::SO3d::hat(p_imu_i);

                // Jacobian
                Eigen::Matrix<double, 2, 6> jacobian_pose_i;
                jacobian_pose_i = dr_dpci * dpci_dpose_i;

                H += jacobian_pose_i.transpose() * jacobian_pose_i;
                b -= jacobian_pose_i.transpose() * e;
            }

            auto H_ldlt = H.ldlt();

            // 修复GN无解的情况
            if (H_ldlt.info() != Eigen::Success) {
                Vec6 lambda;
                for (unsigned int i = 0; i < 6; ++i) {
                    lambda[i] = min(max(H(i, i) * 1e-5, 1e-6), 1e6);
                }

                double v = 2.;
                for (unsigned int m = 0; m < num_fix; ++m) {
                    for (unsigned int i = 0; i < 6; ++i) {
                        H(i, i) += v * lambda[i];
                    }
                    H_ldlt = H.ldlt();
                    if (H_ldlt.info() == Eigen::Success){
                        break;
                    }
                    v *= 2.;
                }
            }

            // 只有成功才能进行更新
            if (H_ldlt.info() == Eigen::Success) {
                Vec6 delta = H_ldlt.solve(b);
                t_wi += delta.head(3);
                q_wi *= Sophus::SO3d::exp(delta.tail(3)).unit_quaternion();
                q_wi.normalize();
            } else {
                frame_i->is_initialized = false;
                return false;
            }
        }

        // 判断landmark是否outlier
        unsigned long outlier_count = 0;
        for (unsigned long k = 0; k < p_w.size(); ++k) {
            auto id = p_w_id[k];

            Vec3 p_imu_i = q_wi.inverse() * (*p_w[k] - t_wi);
            Vec3 p_camera_i = _q_ic[0].inverse() * (p_imu_i - _t_ic[0]);
            double inv_depth_i = 1. / p_camera_i.z();
            // 逆深度
            if (inv_depth_i < 0.) {
                _landmarks[id]->is_outlier = true;
            }

            // 重投影误差
            Vec2 e = (p_camera_i * inv_depth_i).head<2>() - uv[k]->head<2>();
            double e2 = e.squaredNorm();
            if (e2 > th_e2) {
                _landmarks[id]->is_outlier = true;
                ++outlier_count;
            }
        }

        if (outlier_count * 100 > p_w.size() * 30) {
            frame_i->is_initialized = false;
            return false;
        }

#ifdef PRINT_INFO
        std::cout << "p_w.size() = " << p_w.size() << std::endl;
        std::cout << "iter_pnp takes " << pnp_t.toc() << " ms" << std::endl;
        std::cout << "iter_pnp: q = " << q_wi.w() << ", " << q_wi.x() << ", " << q_wi.y() << ", " << q_wi.z() << std::endl;
        std::cout << "iter_pnp: t = " << t_wi.transpose() << std::endl;
#endif
        frame_i->is_initialized = true;
        return true;
    }
}