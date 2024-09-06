//
// Created by Cain on 2024/4/25.
//

#ifndef GRAPH_OPTIMIZATION_VERTEX_VECTOR_H
#define GRAPH_OPTIMIZATION_VERTEX_VECTOR_H

#include "graph_optimization/vertex.h"

namespace graph_optimization {
    template<unsigned int N>
    class VertexVector : public Vertex {
    public:
        EIGEN_MAKE_ALIGNED_OPERATOR_NEW;
        VertexVector() : Vertex(nullptr, N) {}
        explicit VertexVector(double *data) : Vertex(data, N) {}

        void plus(const VecX &delta) override { vector() += delta; }

        void plus(double *delta) override { vector() += Eigen::Map<Eigen::Matrix<double, N, 1>>(delta); }

        std::string type_info() const override { return "VertexVector" + std::string(N); }
        Eigen::Map<Eigen::Matrix<double, N, 1>> vector() { return Eigen::Map<Eigen::Matrix<double, N, 1>>(_parameters); }
    };
}

#endif //GRAPH_OPTIMIZATION_VERTEX_VECTOR_H
