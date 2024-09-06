//
// Created by Cain on 2024/4/12.
//

#include "vertex_quaternion.h"

graph_optimization::VertexQuaternion::VertexQuaternion(double *data) : Vertex(data, 4, 3) {
    q().setIdentity();
}

void graph_optimization::VertexQuaternion::plus(const VecX &delta) {
    q() *= Sophus::SO3d::exp(delta).unit_quaternion();
    q().normalized();
}

void graph_optimization::VertexQuaternion::plus(double *delta) {
    q() *= Sophus::SO3d::exp(Eigen::Map<Eigen::Vector3d>(delta)).unit_quaternion();
    q().normalized();
}