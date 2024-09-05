//
// Created by Cain on 2024/1/2.
//

#ifndef GRAPH_OPTIMIZATION_VERTEX_MOTION_H
#define GRAPH_OPTIMIZATION_VERTEX_MOTION_H

#include "vertex.h"

namespace graph_optimization {
    class VertexMotion : public Vertex {
    public:
        EIGEN_MAKE_ALIGNED_OPERATOR_NEW;
        VertexMotion() : Vertex(nullptr, 9) {} // v, ba, bg
        explicit VertexMotion(double *data) : Vertex(data, 9) {} // v, ba, bg

        void plus(const VecX &delta) override;
        void plus(double *delta) override;
        std::string type_info() const override { return "VertexMotion"; }

        Eigen::Map<Eigen::Vector3d> v() { return Eigen::Map<Eigen::Vector3d>(_parameters); }
        Eigen::Map<Eigen::Vector3d> ba() { return Eigen::Map<Eigen::Vector3d>(_parameters + 3); }
        Eigen::Map<Eigen::Vector3d> bg() { return Eigen::Map<Eigen::Vector3d>(_parameters + 6); }
    };
}

#endif //GRAPH_OPTIMIZATION_VERTEX_MOTION_H
