//
// Created by Cain on 2024/1/2.
//

#ifndef GRAPH_OPTIMIZATION_VERTEX_POINT_3D_H
#define GRAPH_OPTIMIZATION_VERTEX_POINT_3D_H

#include "vertex.h"

namespace graph_optimization {
    class VertexPoint3d : public Vertex {
    public:
        EIGEN_MAKE_ALIGNED_OPERATOR_NEW;
        VertexPoint3d() : Vertex(nullptr, 3) {}
        explicit VertexPoint3d(double *data) : Vertex(data, 3) {}

        void plus(const VecX &delta) override;
        void plus(double *delta) override;
        std::string type_info() const override { return "VertexPoint3d"; }

        Eigen::Map<Eigen::Vector3d> point() { return Eigen::Map<Eigen::Vector3d>(_parameters); }
    };
}

#endif //GRAPH_OPTIMIZATION_VERTEX_POINT_3D_H
