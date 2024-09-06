//
// Created by Cain on 2024/9/3.
//

#include "landmark.h"

namespace vins {
    bool Landmark::from_global_to_local(const Eigen::Quaterniond &q_ic, const Eigen::Vector3d &t_ic) {
        if (is_outlier) {
            return false;
        }
        if (!is_triangulated) {
            return false;
        }

        if (observations.empty()) {
            return false;
        }

        // 转到local系
        auto frame_host = observations.front().first;
        auto &point = observations.front().second->points[0];
        Eigen::Vector3d p_imu = frame_host->q().inverse() * (position - frame_host->p());
        inv_depth = point.squaredNorm() / point.dot(q_ic.inverse() * (p_imu - t_ic));

        return true;
    }

    bool Landmark::from_local_to_global(const Eigen::Quaterniond &q_ic, const Eigen::Vector3d &t_ic) {
        if (is_outlier) {
            return false;
        }
        if (!is_triangulated) {
            return false;
        }

        if (observations.empty()) {
            return false;
        }

        // 转到global系
        auto frame_host = observations.front().first;
        auto &point = observations.front().second->points[0];
        Eigen::Vector3d p_imu = q_ic * (point / inv_depth) + t_ic;
        position = frame_host->q() * p_imu + frame_host->p();

        return true;
    }
}