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
#include <opencv2/opencv.hpp>

namespace vins {
    using namespace graph_optimization;
    using namespace std;

    bool Estimator::compute_essential_matrix(Mat33 &R, Vec3 &t, const std::shared_ptr<Frame> &frame_i, const std::shared_ptr<Frame> &frame_j,
                                             bool is_init_landmark, unsigned int max_iters) {
        constexpr static unsigned long th_count = 15;
        unsigned long max_num_points = max(frame_i->features.size(), frame_j->features.size());
        vector<cv::Point2f> pts1, pts2;
        pts1.reserve(max_num_points);
        pts2.reserve(max_num_points);
        vector<unsigned long> landmark_ids;
        landmark_ids.reserve(max_num_points);

        // 找出匹配对, 同时计算平均视差
        double average_parallax = 0.;
        for (auto &feature_i : frame_i->features) {
            unsigned long landmark_id = feature_i.first;
            auto &&feature_j = frame_j->features.find(landmark_id);
            if (feature_j == frame_j->features.end()) {
                continue;
            }
            pts1.emplace_back(feature_i.second->points[0].x(), feature_i.second->points[0].y());
            pts2.emplace_back(feature_j->second->points[0].x(), feature_j->second->points[0].y());
            landmark_ids.emplace_back(landmark_id);

            // 计算视差
            double du = pts1.back().x - pts2.back().x;
            double dv = pts1.back().y - pts2.back().y;
            average_parallax += max(abs(du), abs(dv));
        }

        // 匹配对必须大于一定数量
        unsigned long num_points = landmark_ids.size();
        if (num_points < th_count) {
            return false;
        }

        // 平均视差必须大于一定值
        average_parallax /= double(num_points);
#ifdef PRINT_INFO
        std::cout << "average_parallax = " << average_parallax << std::endl;
#endif
        if (average_parallax * 460. < 30.) {
            return false;
        }

        // 调用openCV计算本质矩阵
        cv::Mat is_inliers;
        cv::Mat Et = cv::findFundamentalMat(pts1, pts2, cv::FM_RANSAC, 0.3 / 460, 0.99, is_inliers);  // 行主序
        Eigen::Map<Eigen::Matrix<double, 3, 3, Eigen::ColMajor>> best_E(Et.ptr<double>(0)); // 列主序
        unsigned long num_inliers = static_cast<unsigned long>(cv::sum(is_inliers)[0]);

        // 内点必须大于一定值
        if (10 * num_inliers < 9 * th_count) {
            return false;
        }

        Eigen::JacobiSVD<Mat33> E_svd(best_E, Eigen::ComputeFullU | Eigen::ComputeFullV);
        Mat33 V = E_svd.matrixV();
        Mat33 U1 = E_svd.matrixU();

        if (V.determinant() < 0.) {
            V *= -1.;
        }
        if (U1.determinant() < 0.) {
            U1 *= -1.;
        }

        Vec3 t1 = U1.col(2);
        t1 = t1 / t1.norm();
        Vec3 t2 = -t1;

        U1.col(0).swap(U1.col(1));
        Mat33 U2 = U1;
        U1.col(1) *= -1.;
        U2.col(0) *= -1.;

        Mat33 R1 = U1 * V.transpose();
        Mat33 R2 = U2 * V.transpose();

        // 进行三角化，通过深度筛选出正确的R, t
        auto tri = [&](const cv::Point2f &point_i, const cv::Point2f &point_j, const Mat33 &R, const Vec3 &t, Vec3 &p) -> bool {
            Vec3 RTt = R.transpose() * t;
            Mat43 A;
            A.row(0) << 1., 0., -point_i.x;
            A.row(1) << 0., 1., -point_i.y;
            A.row(2) = R.col(0).transpose() - point_j.x * R.col(2).transpose();
            A.row(3) = R.col(1).transpose() - point_j.y * R.col(2).transpose();
            Vec4 b;
            b << 0., 0., RTt[0] - RTt[2] * point_j.x, RTt[1] - RTt[2] * point_j.y;

//            Mat33 ATA = A.transpose() * A;
//            Vec3 ATb = A.transpose() * b;
//            auto &&ATA_ldlt = ATA.ldlt();
//            if (ATA_ldlt.info() == Eigen::Success) {
//                p = ATA_ldlt.solve(ATb);
//                return true;
//            } else {
//                return false;
//            }

            // 使用QR分解求解最小二乘问题，这样数值精度更加稳定
            p = A.fullPivHouseholderQr().solve(b);
            return true;
        };

        auto tri_all_points = [&](const Mat33 &R, const Vec3 &t, vector<pair<bool, Vec3>> &points) -> unsigned long {
            unsigned long succeed_count = 0;
            for (unsigned long k = 0; k < num_points; ++k) {
                if (!is_inliers.at<uchar>(k)) {
                    continue;
                }

                points[k].first = tri(pts1[k], pts2[k], R, t, points[k].second);
                if (!points[k].first) {
                    continue;
                }

                points[k].first = points[k].second[2] > 0.;
//                std::cout << "points[k].second[2] = " << points[k].second[2] << std::endl;
                if (!points[k].first) {
                    continue;
                }

                Vec3 pj = R.transpose() * (points[k].second - t);
                points[k].first = pj[2] > 0.;
//                std::cout << "pj[2] = " << pj[2] << std::endl;
                if (!points[k].first) {
                    continue;
                }

                ++succeed_count;
            }
            return succeed_count;
        };

        vector<pair<bool, Vec3>> points_w[4];
        unsigned long succeed_points[4];
        points_w[0].resize(num_points);
        points_w[1].resize(num_points);
        points_w[2].resize(num_points);
        points_w[3].resize(num_points);

        succeed_points[0] = tri_all_points(R1, t1, points_w[0]);
        succeed_points[1] = tri_all_points(R1, t2, points_w[1]);
        succeed_points[2] = tri_all_points(R2, t1, points_w[2]);
        succeed_points[3] = tri_all_points(R2, t2, points_w[3]);

        unsigned long max_succeed_points = max(succeed_points[0], max(succeed_points[1], max(succeed_points[2], succeed_points[3])));
        unsigned long min_succeed_points = 9 * num_inliers / 10; // 至少要超过90%的点成功被三角化

        if (max_succeed_points < min_succeed_points) {
#ifdef PRINT_INFO
            std::cout << "max_succeed_points = " << max_succeed_points << ", min_succeed_points = " << min_succeed_points << std::endl;
            std::cout << "succeed_points[0] = " << succeed_points[0] << std::endl;
            std::cout << "succeed_points[1] = " << succeed_points[1] << std::endl;
            std::cout << "succeed_points[2] = " << succeed_points[2] << std::endl;
            std::cout << "succeed_points[3] = " << succeed_points[3] << std::endl;
#endif
            return false;
        }

        unsigned long lim_succeed_points = 7 * max_succeed_points / 10;
        unsigned long num_similar = 0;  // 记录有多少组解使得70%的点都能三角化
        if (succeed_points[0] > lim_succeed_points) {
            ++num_similar;
        }
        if (succeed_points[1] > lim_succeed_points) {
            ++num_similar;
        }
        if (succeed_points[2] > lim_succeed_points) {
            ++num_similar;
        }
        if (succeed_points[3] > lim_succeed_points) {
            ++num_similar;
        }
        if (num_similar > 1) {  // 不允许超过1组解使得70%的点都能三角化
#ifdef PRINT_INFO
            std::cout << "num_similar > 1" << std::endl;
#endif
            return false;
        }

        unsigned int which_case;
        if (succeed_points[0] == max_succeed_points) {
            which_case = 0;
            R = R1;
            t = t1;
        } else if (succeed_points[1] == max_succeed_points) {
            which_case = 1;
            R = R1;
            t = t2;
        } else if (succeed_points[2] == max_succeed_points) {
            which_case = 2;
            R = R2;
            t = t1;
        } else {
            which_case = 3;
            R = R2;
            t = t2;
        }

        // 转到imu坐标系
        Qd q12;
        Vec3 t12;
        q12 = _q_ic[0] * R * _q_ic[0].inverse();
        t12 = _q_ic[0] * t - q12 * _t_ic[0] + _t_ic[0];

        R = q12;
        t = t12;

        // 把三角化的结果赋值给landmark
        if (is_init_landmark) {
            for (unsigned long k = 0; k < num_points; ++k) {
                auto &&landmark = _landmarks.find(landmark_ids[k]);
                if (landmark == _landmarks.end()) {
                    continue;
                }

                // 没有outliers以及深度为正的点才会进行赋值
                if (is_inliers.at<uchar>(k) && points_w[which_case][k].first) {
                    // 转到frame_i的imu系
                    landmark->second->position = _q_ic[0] * points_w[which_case][k].second + _t_ic[0];

                    landmark->second->is_triangulated = true;
                    landmark->second->is_outlier = false;
                } else {
                    landmark->second->is_triangulated = false;
                    landmark->second->is_outlier = true;
                }
            }
        }

        return true;
    }
}
