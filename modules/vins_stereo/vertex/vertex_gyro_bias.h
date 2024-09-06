//
// Created by Cain on 2024/4/24.
//

#ifndef GRAPH_OPTIMIZATION_VERTEX_GYRO_BIAS_H
#define GRAPH_OPTIMIZATION_VERTEX_GYRO_BIAS_H

#include "vertex_point3d.h"

namespace graph_optimization {
    class VertexGyroBias : public VertexPoint3d {
    public:
        EIGEN_MAKE_ALIGNED_OPERATOR_NEW;
        VertexGyroBias() : VertexPoint3d() {}
        explicit VertexGyroBias(double *data) : VertexPoint3d(data) {}

        std::string type_info() const override { return "VertexGyroBias"; }
        Eigen::Map<Eigen::Vector3d> bg() { return point(); }
    };
}

#endif //GRAPH_OPTIMIZATION_VERTEX_GYRO_BIAS_H
