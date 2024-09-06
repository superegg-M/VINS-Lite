//
// Created by Cain on 2024/4/24.
//

#ifndef GRAPH_OPTIMIZATION_VERTEX_VELOCITY_H
#define GRAPH_OPTIMIZATION_VERTEX_VELOCITY_H

#include "vertex_point3d.h"

namespace graph_optimization {
    class VertexVelocity : public VertexPoint3d {
    public:
        EIGEN_MAKE_ALIGNED_OPERATOR_NEW;
        VertexVelocity(double *data) : VertexPoint3d(data) {}

        std::string type_info() const override { return "VertexVelocity"; }
        Eigen::Map<Eigen::Vector3d> v() { return point(); }
    };
}


#endif //GRAPH_OPTIMIZATION_VERTEX_VELOCITY_H
