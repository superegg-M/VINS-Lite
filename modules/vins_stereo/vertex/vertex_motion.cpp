//
// Created by Cain on 2024/8/27.
//

#include "vertex_motion.h"

void graph_optimization::VertexMotion::plus(const VecX &delta) {
    v() += delta.head<3>();
    ba() += delta.segment<3>(3);
    bg() += delta.tail<3>();
}

void graph_optimization::VertexMotion::plus(double *delta) {
    v() += Eigen::Map<Eigen::Vector3d>(delta);
    ba() += Eigen::Map<Eigen::Vector3d>(delta + 3);
    bg() += Eigen::Map<Eigen::Vector3d>(delta + 6);
}