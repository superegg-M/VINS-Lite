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

    bool Estimator::mlpnp(const std::shared_ptr<Frame> &frame_i, unsigned int batch_size, unsigned int num_batches) {
        constexpr static unsigned long th_count = 6;    ///< MLPnP至少需要6个点
        constexpr static double th = 5.991;
        TicToc pnp_t;

        // MLPnP至少需要6个点
        if (batch_size < th_count) {
            batch_size = th_count;
        }

        // 读取3d, 2d点
        vector<const Vec3*> p_w;
        vector<const Vec3*> uv;
        p_w.reserve(frame_i->features.size());
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
            uv.emplace_back(&feature.second->points[0]);
        }
        unsigned long num_points = uv.size();
        // 特征点个数必须大于一定数量
        if (num_points < th_count) {
            frame_i->is_initialized = false;
            return false;
        }

        // 计算零空间
        vector<Eigen::Matrix<double, 3, 2>> null_space(num_points);
        for (unsigned long i = 0; i < num_points; ++i) {
            auto &&svd = (*uv[i]).jacobiSvd(Eigen::ComputeFullU);
            null_space[i] = svd.matrixU().rightCols<2>();
        }

        // 构造随机index batch, 用于RANSAC
        std::random_device rd;
        std::mt19937 gen(rd());
        vector<vector<unsigned long>> point_indices_set(num_batches);
        for (auto &point_indices : point_indices_set) {
            point_indices.resize(batch_size);
        }

        vector<unsigned long> local_index_map(batch_size);
        vector<unsigned long> global_index_map(num_points);
        for (unsigned long k = 0; k < num_points; ++k) {
            global_index_map[k] = k;
        }

        for (unsigned int n = 0; n < num_batches; ++n) {
            for (unsigned int k = 0; k < batch_size; ++k) {
                std::uniform_int_distribution<unsigned int> dist(0, global_index_map.size() - 1);
                unsigned int rand_i = dist(gen);
                auto index = global_index_map[rand_i];
                point_indices_set[n][k] = index;
                local_index_map[k] = index;

                global_index_map[rand_i] = global_index_map.back();
                global_index_map.pop_back();
            }

            for (unsigned int k = 0; k < batch_size; ++k) {
                global_index_map.emplace_back(local_index_map[k]);
            }
        }

        // RANSAC ML PnP
        unsigned int best_iter = 0;
        vector<Qd> q(num_batches);
        vector<Mat33> R(num_batches);
        vector<Vec3> t(num_batches);
        vector<double> score(num_batches);
        vector<unsigned long > num_inlier(num_batches);
        vector<vector<bool>> is_outlier(num_batches);
        for (auto &is : is_outlier) {
            is.resize(num_points);
        }

        MatXX A;    ///< 用于求解ML PnP
        A.resize(2 * batch_size, 12);
        for (unsigned int n = 0; n < num_batches; ++n) {
            // 构造A矩阵
            for (unsigned int i = 0; i < batch_size; ++i) {
                unsigned long index = point_indices_set[n][i];
                auto &N = null_space[index];
                auto &p = *p_w[index];
                A.row(2 * i) << N(0, 0) * p.x(), N(1, 0) * p.x(), N(2, 0) * p.x(),
                        N(0, 0) * p.y(), N(1, 0) * p.y(), N(2, 0) * p.y(),
                        N(0, 0) * p.z(), N(1, 0) * p.z(), N(2, 0) * p.z(),
                        N(0, 0), N(1, 0), N(2, 0);
                A.row(2 * i + 1) << N(0, 1) * p.x(), N(1, 1) * p.x(), N(2, 1) * p.x(),
                        N(0, 1) * p.y(), N(1, 1) * p.y(), N(2, 1) * p.y(),
                        N(0, 1) * p.z(), N(1, 1) * p.z(), N(2, 1) * p.z(),
                        N(0, 1), N(1, 1), N(2, 1);
            }

            // 对A进行SVD, 求出A*x = 0, ||x||_2 = 1的最优解
            auto &&A_svd = A.jacobiSvd(Eigen::ComputeFullV);
            Eigen::Matrix<double, 12, 1> ans = A_svd.matrixV().rightCols<1>();

            // 求R, t
            R[n] << ans[0], ans[1], ans[2],
                    ans[3], ans[4], ans[5],
                    ans[6], ans[7], ans[8];
            double scale = 1. / pow(R[n].col(0).norm() * R[n].col(1).norm() * R[n].col(2).norm(), 1. / 3.);

            auto &&R_svd = R[n].jacobiSvd(Eigen::ComputeFullU | Eigen::ComputeFullV);
            R[n] = R_svd.matrixU() * R_svd.matrixV().transpose();
            t[n] = R[n] * (ans.bottomRows<3>() * (-scale));

            if (R[n].determinant() < 0.) {
                R[n] = -R[n];
                t[n] = -t[n];
            }

            // GN, 进一步优化R, t
            q[n] = Qd(R[n]);
            Mat66 H;
            Vec6 b;
            for (unsigned int j = 0; j < 5; ++j) {
                H.setZero();
                b.setZero();
                for (unsigned int i = 0; i < batch_size; ++i) {
                    unsigned long index = point_indices_set[n][i];
                    auto &N = null_space[index];
                    auto &p = *p_w[index];

                    Vec3 p_c = R[n].transpose() * (p - t[n]);
                    Vec2 e = N.transpose() * p_c;

                    Eigen::Matrix<double, 2, 6> J;
                    J.leftCols<3>() = -N.transpose() * R[n].transpose();
                    J.rightCols<3>() = N.transpose() * Sophus::SO3d::hat(p_c);

                    H += J.transpose() * J;
                    b -= J.transpose() * e;
                }

                auto &&H_ldlt = H.ldlt();
                if (H_ldlt.info() == Eigen::Success) {
                    Vec6 delta_x = H_ldlt.solve(b);
                    t[n] += delta_x.topRows<3>();
                    q[n] *= Sophus::SO3d::exp(delta_x.bottomRows<3>()).unit_quaternion();
                    q[n].normalize();
                    R[n] = q[n].toRotationMatrix();
                }
            }

            // 遍历所有点, 计算得分
            score[n] = 0.;
            num_inlier[n] = 0;
            for (unsigned long k = 0; k < num_points; ++k) {
                Vec3 p_c = R[n].transpose() * (*p_w[k] - t[n]);

                // 深度必须为正
                if (p_c.z() > 0.) {
                    is_outlier[n][k] = false;
                } else {
                    is_outlier[n][k] = true;
                    continue;
                }

                // chi2必须小于阈值
                double e_u = p_c.x() / p_c.z() - uv[k]->x();
                double e_v = p_c.y() / p_c.z() - uv[k]->y();
                double e2 = e_u * e_u + e_v * e_v;
                if (e2 < th) {
                    is_outlier[n][k] = false;
                    score[n] += th - e2;
                } else {
                    is_outlier[n][k] = true;
                    continue;
                }

                // 记录内点数
                ++num_inlier[n];
            }

            // 记录得分最高的那次迭代
            if (score[n] > score[best_iter]) {
                best_iter = n;
            }
        }

        // camera系转到imu系
        q[best_iter] = q[best_iter] * _q_ic[0].inverse();
        t[best_iter] -= q[best_iter] * _t_ic[0];

        // 把结果保存到frame中
        frame_i->p() = t[best_iter];
        frame_i->q() = q[best_iter];

#ifdef PRINT_INFO
        std::cout << "mlpnp: takse " << pnp_t.toc() << " ms" << std::endl;
        std::cout << "mlpnp: inlier points = " << num_inlier[best_iter] << std::endl;
        std::cout << "mlpnp: score = " << score[best_iter] << std::endl;
        std::cout << "mlpnp: q = " << q[best_iter].w() << ", " << q[best_iter].x() << ", " << q[best_iter].y() << ", " << q[best_iter].z() << std::endl;
        std::cout << "mlpnp: t = " << t[best_iter].transpose() << std::endl;
#endif
        frame_i->is_initialized = true;
        return true;
    }
}