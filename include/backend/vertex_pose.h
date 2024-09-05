//
// Created by Cain on 2024/1/2.
//

#ifndef GRAPH_OPTIMIZATION_VERTEX_POSE_H
#define GRAPH_OPTIMIZATION_VERTEX_POSE_H

#include "vertex.h"
#include "../thirdparty/Sophus/sophus/so3.hpp"

namespace graph_optimization {
    class VertexPose : public Vertex {
    public:
        EIGEN_MAKE_ALIGNED_OPERATOR_NEW;
        VertexPose() : Vertex(nullptr, 7, 6) {}
        explicit VertexPose(double *data) : Vertex(data, 7, 6) {}

        void plus(const VecX &delta) override;
        void plus(double *delta) override;
        std::string type_info() const override { return "VertexPose"; }

        Eigen::Map<Eigen::Vector3d> p() { return Eigen::Map<Eigen::Vector3d>(_parameters); }
        Eigen::Map<Eigen::Quaterniond> q() { return Eigen::Map<Eigen::Quaterniond>(_parameters + 3); }
    };
}

#endif //GRAPH_OPTIMIZATION_VERTEX_POSE_H
