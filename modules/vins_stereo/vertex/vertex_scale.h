//
// Created by Cain on 2024/4/11.
//

#ifndef GRAPH_OPTIMIZATION_VERTEX_SCALE_H
#define GRAPH_OPTIMIZATION_VERTEX_SCALE_H

#include "sophus/so3.hpp"
#include "graph_optimization/vertex.h"

namespace graph_optimization {
    class VertexScale : public Vertex {
    public:
        EIGEN_MAKE_ALIGNED_OPERATOR_NEW;
        VertexScale() : Vertex(nullptr, 1) {}
        explicit VertexScale(double *data);

        void plus(const VecX &delta) override;
        void plus(double *delta) override;

        std::string type_info() const override { return "VertexScale"; }
        double &scale() { return _parameters[0]; }
    };
}

#endif //GRAPH_OPTIMIZATION_VERTEX_SCALE_H
