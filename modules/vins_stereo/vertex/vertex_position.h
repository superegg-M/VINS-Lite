//
// Created by Cain on 2024/8/29.
//

#ifndef GRAPH_OPTIMIZATION_VERTEX_POSITION_H
#define GRAPH_OPTIMIZATION_VERTEX_POSITION_H

#include "vertex_point3d.h"

namespace graph_optimization {
    class VertexPosition : public VertexPoint3d {
    public:
        EIGEN_MAKE_ALIGNED_OPERATOR_NEW;
        VertexPosition() : VertexPoint3d() {}
        explicit VertexPosition(double *data) : VertexPoint3d(data) {}

        std::string type_info() const override { return "VertexPosition"; }
        Eigen::Map<Eigen::Vector3d> p() { return point(); }
    };
}

#endif //GRAPH_OPTIMIZATION_VERTEX_POSITION_H
