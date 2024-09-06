//
// Created by Cain on 2024/4/24.
//

#ifndef GRAPH_OPTIMIZATION_VERTEX_ACC_BIAS_H
#define GRAPH_OPTIMIZATION_VERTEX_ACC_BIAS_H

#include "vertex_point3d.h"

namespace graph_optimization {
    class VertexAccBias : public VertexPoint3d {
    public:
        EIGEN_MAKE_ALIGNED_OPERATOR_NEW;
        VertexAccBias() : VertexPoint3d(nullptr) {}
        explicit VertexAccBias(double *data) : VertexPoint3d(data) {}

        std::string type_info() const override { return "VertexAccBias"; }
        Eigen::Map<Eigen::Vector3d> ba() { return point(); }
    };
}

#endif //GRAPH_OPTIMIZATION_VERTEX_ACC_BIAS_H
