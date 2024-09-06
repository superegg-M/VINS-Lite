//
// Created by Cain on 2024/1/2.
//

#ifndef GRAPH_OPTIMIZATION_VERTEX_INVERSE_DEPTH_H
#define GRAPH_OPTIMIZATION_VERTEX_INVERSE_DEPTH_H

//#include <lib/graph_optimization/vertex.h>
#include "graph_optimization/vertex.h"

namespace graph_optimization {
    class VertexInverseDepth : public Vertex {
    public:
        EIGEN_MAKE_ALIGNED_OPERATOR_NEW;
        VertexInverseDepth() : Vertex(nullptr, 1) {}
        explicit VertexInverseDepth(double *data) : Vertex(data, 1) {}

        void plus(const VecX &delta) override;
        void plus(double *delta) override;
        std::string type_info() const override { return "VertexInverseDepth"; }

        double &inverse_depth() { return _parameters[0]; }
        double depth() const { return 1. / _parameters[0]; }
    };
}

#endif //GRAPH_OPTIMIZATION_VERTEX_INVERSE_DEPTH_H
