//
// Created by Cain on 2024/1/2.
//

#include "backend/vertex_pose.h"

void graph_optimization::VertexPose::plus(const VecX &delta) {
    p() += delta.head<3>();
    q() *= Sophus::SO3d::exp(delta.tail<3>()).unit_quaternion();
    q().normalized();
}

void graph_optimization::VertexPose::plus(double *delta) {
    p() += Eigen::Map<Eigen::Vector3d>(delta);
    q() *= Sophus::SO3d::exp(Eigen::Map<Eigen::Vector3d>(delta + 3)).unit_quaternion();
    q().normalized();
}
