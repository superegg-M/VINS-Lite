//
// Created by Cain on 2024/3/10.
//

#ifndef GRAPH_OPTIMIZATION_EDGE_IMU_H
#define GRAPH_OPTIMIZATION_EDGE_IMU_H

//#include <lib/graph_optimization/edge.h>
#include "graph_optimization/edge.h"
#include "../imu_integration.h"

namespace graph_optimization {
    class EdgeImu : public Edge {
    public:
        EIGEN_MAKE_ALIGNED_OPERATOR_NEW;
        EdgeImu();
        explicit EdgeImu(vins::IMUIntegration* imu_integration);

        std::string type_info() const override { return "EdgeImu"; }   ///< 返回边的类型信息
        void compute_residual() override;   ///< 计算残差
        void compute_jacobians() override;  ///< 计算雅可比

        vins::IMUIntegration*& imu_integration() { return _imu_integration; }

    protected:
        vins::IMUIntegration* _imu_integration {nullptr};
        static Eigen::Matrix<double, 4, 4> get_quat_left(const Qd &q);
        static Eigen::Matrix<double, 4, 4> get_quat_right(const Qd &q);
    };
}

#endif //GRAPH_OPTIMIZATION_EDGE_IMU_H