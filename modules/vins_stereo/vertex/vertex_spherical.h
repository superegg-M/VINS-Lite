//
// Created by Cain on 2024/4/12.
//

#ifndef GRAPH_OPTIMIZATION_VERTEX_SPHERICAL_H
#define GRAPH_OPTIMIZATION_VERTEX_SPHERICAL_H

#include "sophus/so3.hpp"
#include "graph_optimization/vertex.h"

namespace graph_optimization {
    class VertexSpherical : public Vertex {
    public:
        EIGEN_MAKE_ALIGNED_OPERATOR_NEW;
        VertexSpherical() : Vertex(nullptr, 4, 2) {};
        explicit VertexSpherical(double *data);

        void plus(const VecX &delta) override;
        void plus(double *delta) override;

        std::string type_info() const override { return "VertexSpherical"; }
        Eigen::Map<Eigen::Quaterniond> q() { return Eigen::Map<Eigen::Quaterniond>(_parameters); }
    };
}

#endif //GRAPH_OPTIMIZATION_VERTEX_SPHERICAL_H
