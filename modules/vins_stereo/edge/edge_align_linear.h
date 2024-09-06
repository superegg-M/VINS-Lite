//
// Created by Cain on 2024/4/25.
//

#ifndef GRAPH_OPTIMIZATION_EDGE_ALIGN_LINEAR_H
#define GRAPH_OPTIMIZATION_EDGE_ALIGN_LINEAR_H

#include "graph_optimization/edge.h"
#include "../imu_integration.h"

namespace graph_optimization {
    class EdgeAlignLinear : public Edge {
    public:
        EIGEN_MAKE_ALIGNED_OPERATOR_NEW;
        explicit EdgeAlignLinear(vins::IMUIntegration* imu_integration, const Vec3 &t_ij, const Qd &q_ij, const Qd &q_0i);

        std::string type_info() const override { return "EdgeAlignLinear"; }   ///< 返回边的类型信息
        void compute_residual() override;   ///< 计算残差
        void compute_jacobians() override;  ///< 计算雅可比

        void set_norm_scale(double norm_scale) { _norm_scale = norm_scale; }
        double get_norm_scale() const { return _norm_scale; }

    protected:
        vins::IMUIntegration* _imu_integration;
        Vec3 _t_ij;
        Qd _q_ij;
        Qd _q_0i;
        double _norm_scale = 1.;
    };
}

#endif //GRAPH_OPTIMIZATION_EDGE_ALIGN_LINEAR_H
