//
// Created by Cain on 2024/4/12.
//

#include "vertex_spherical.h"

graph_optimization::VertexSpherical::VertexSpherical(double *data) : Vertex(data, 4, 2) {
    q().setIdentity();
}

void graph_optimization::VertexSpherical::plus(const VecX &delta) {
    q() *= Sophus::SO3d::exp(Vec3(delta[0], delta[1], 0.)).unit_quaternion();
    q().normalized();
}

void graph_optimization::VertexSpherical::plus(double *delta) {
    q() *= Sophus::SO3d::exp(Vec3(delta[0], delta[1], 0.)).unit_quaternion();
    q().normalized();
}