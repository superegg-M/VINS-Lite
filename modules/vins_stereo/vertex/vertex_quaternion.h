//
// Created by Cain on 2024/4/12.
//

#ifndef GRAPHO_PTIMIZATION_VERTEX_QUATERNION_H
#define GRAPHO_PTIMIZATION_VERTEX_QUATERNION_H

#include "sophus/so3.hpp"
#include "graph_optimization/vertex.h"

namespace graph_optimization {
    class VertexQuaternion : public Vertex {
    public:
        EIGEN_MAKE_ALIGNED_OPERATOR_NEW;
        VertexQuaternion() : Vertex(nullptr, 4, 3) {};
        explicit VertexQuaternion(double *data);

        void plus(const VecX &delta) override;
        void plus(double *delta) override;

        std::string type_info() const override { return "VertexQuaternion"; }
        Eigen::Map<Eigen::Quaterniond> q() { return Eigen::Map<Eigen::Quaterniond>(_parameters); }
    };
}

#endif //GRAPHO_PTIMIZATION_VERTEX_QUATERNION_H
