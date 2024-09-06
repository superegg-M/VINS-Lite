//
// Created by Cain on 2024/8/27.
//

#include "vertex_point3d.h"

void graph_optimization::VertexPoint3d::plus(const VecX &delta) {
    point() += delta;
}

void graph_optimization::VertexPoint3d::plus(double *delta) {
    point() += Eigen::Map<Eigen::Vector3d>(delta);
}