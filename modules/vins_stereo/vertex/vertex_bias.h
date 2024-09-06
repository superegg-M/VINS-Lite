//
// Created by Cain on 2024/4/24.
//

#ifndef GRAPH_OPTIMIZATION_VERTEX_BIAS_H
#define GRAPH_OPTIMIZATION_VERTEX_BIAS_H

#include "vertex_point3d.h"

namespace graph_optimization {
    class VertexBias : public VertexPoint3d {
    public:
        EIGEN_MAKE_ALIGNED_OPERATOR_NEW;
        VertexBias() : VertexPoint3d() {}
        explicit VertexBias(double *data) : VertexPoint3d(data) {}

        std::string type_info() const override { return "VertexBias"; }
        Eigen::Map<Eigen::Vector3d> bias() { return point(); }
    };
}

#endif //GRAPH_OPTIMIZATION_VERTEX_BIAS_H
