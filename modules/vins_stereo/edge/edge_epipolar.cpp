//
// Created by Cain on 2024/4/12.
//

#include "edge_epipolar.h"
#include "../vertex/vertex_quaternion.h"
#include "../vertex/vertex_spherical.h"
#include "../vertex/vertex_scale.h"
#include "../vertex/vertex_scale.h"

void graph_optimization::EdgeEpipolar::compute_residual() {
    // 姿态
    auto q = Eigen::Map<Eigen::Quaterniond>(_vertices[0]->parameters());

    // 位移
    auto t = Eigen::Map<Eigen::Quaterniond>(_vertices[1]->parameters()).toRotationMatrix().col(2);

    // 误差
    _residual[0] = _p1.dot(t.cross(q.toRotationMatrix() * _p2));
//    _residual[0] = _p2.dot(q.toRotationMatrix() * _p1.cross(t));
}

void graph_optimization::EdgeEpipolar::compute_jacobians() {
    // 姿态
    auto q = Eigen::Map<Eigen::Quaterniond>(_vertices[0]->parameters());
    Mat33 r = q.toRotationMatrix();

    // 位移
    auto qt = Eigen::Map<Eigen::Quaterniond>(_vertices[1]->parameters());
    Mat33 rt = qt.toRotationMatrix();
    Vec3 t = rt.col(2);

    Eigen::Matrix<double, 1, 3> jacobian_q;
    jacobian_q = -_p1.transpose() * Sophus::SO3d::hat(t) * r * Sophus::SO3d::hat(_p2);
//    jacobian_q = _p2.transpose() * Sophus::SO3d::hat(r.transpose() * _p1.cross(t));

    Eigen::Matrix<double, 1, 3> dr_dt = -_p1.transpose() * Sophus::SO3d::hat(r * _p2);
//    Eigen::Matrix<double, 1, 3> dr_dt = _p2.transpose() * r.transpose() * Sophus::SO3d::hat(_p1);
    Eigen::Matrix<double, 3, 2> dt_dqt;
    dt_dqt.col(0) = -rt.col(1);
    dt_dqt.col(1) = rt.col(0);
    Eigen::Matrix<double, 1, 2> jacobian_qt = dr_dt * dt_dqt;

    _jacobians[0] = jacobian_q;
    _jacobians[1] = jacobian_qt;
}