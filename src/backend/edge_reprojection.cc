//
// Created by Cain on 2024/1/2.
//

#include <iostream>
#include "../thirdparty/Sophus/sophus/se3.hpp"
#include "utility/utility.h"

#include "backend/edge_reprojection.h"
#include "backend/vertex_pose.h"
#include "backend/vertex_motion.h"
#include "backend/vertex_inverse_depth.h"
#include "backend/vertex_point3d.h"


void graph_optimization::EdgeReprojection::compute_residual() {
    double inv_depth_i = _vertices[0]->get_parameters()[0];

    const auto &params_i = _vertices[1]->get_parameters();
    Vec3 t_i = params_i.head<3>();
    Qd q_i {params_i[6], params_i[3], params_i[4], params_i[5]};

    const auto &params_j = _vertices[2]->get_parameters();
    Vec3 t_j = params_j.head<3>();
    Qd q_j = {params_j[6], params_j[3], params_j[4], params_j[5]};

    const auto &param_ext = _vertices[3]->get_parameters();
    Qd qic(param_ext[6], param_ext[3], param_ext[4], param_ext[5]);
    Vec3 tic = param_ext.head<3>();

    Vec3 p_camera_i = _pt_i / inv_depth_i;
    Vec3 p_imu_i = qic * p_camera_i + tic;
    Vec3 p_world = q_i * p_imu_i + t_i;
    Vec3 p_imu_j = q_j.inverse() * (p_world - t_j);
    Vec3 p_camera_j = qic.inverse() * (p_imu_j - tic);

    double inv_depth_j = 1. / p_camera_j.z();
    _residual = (p_camera_j * inv_depth_j).head<2>() - _pt_j.head<2>();
}

void graph_optimization::EdgeReprojection::compute_jacobians() {
    double inv_depth_i = _vertices[0]->get_parameters()[0];

    const auto &params_i = _vertices[1]->get_parameters();
    Vec3 t_i = params_i.head<3>();
    Qd q_i {params_i[6], params_i[3], params_i[4], params_i[5]};
    Mat33 R_i = q_i.toRotationMatrix();

    const auto &params_j = _vertices[2]->get_parameters();
    Vec3 t_j = params_j.head<3>();
    Qd q_j = {params_j[6], params_j[3], params_j[4], params_j[5]};
    Mat33 R_j = q_j.toRotationMatrix();

    const auto &param_ext = _vertices[3]->get_parameters();
    Qd qic(param_ext[6], param_ext[3], param_ext[4], param_ext[5]);
    Vec3 tic = param_ext.head<3>();
    Mat33 R_ic = qic.toRotationMatrix();

    Vec3 p_camera_i = _pt_i / inv_depth_i;
    Vec3 p_imu_i = R_ic * p_camera_i + tic;
    Vec3 p_world = R_i * p_imu_i + t_i;
    Vec3 p_imu_j = R_j.transpose() * (p_world - t_j);
    Vec3 p_camera_j = R_ic.transpose() * (p_imu_j - tic);

    double inv_depth_j = 1. / p_camera_j.z();

    Mat23 dr_dpcj;
    dr_dpcj << inv_depth_j, 0., -p_camera_j[0] * inv_depth_j * inv_depth_j,
               0., inv_depth_j, -p_camera_j[1] * inv_depth_j * inv_depth_j;

    Eigen::Matrix<double, 3, 6> dpcj_dposi;
    dpcj_dposi.leftCols<3>() = R_ic.transpose() * R_j.transpose();
    dpcj_dposi.rightCols<3>() = -dpcj_dposi.leftCols<3>() * R_i * Sophus::SO3d::hat(p_imu_i);

    Eigen::Matrix<double, 2, 6> jacobian_pose_i;
    jacobian_pose_i = dr_dpcj * dpcj_dposi;

    Eigen::Matrix<double, 3, 6> dpcj_dposj;
    dpcj_dposj.leftCols<3>() = -dpcj_dposi.leftCols<3>();
    dpcj_dposj.rightCols<3>() = R_ic.transpose() * Sophus::SO3d::hat(p_imu_j);

    Eigen::Matrix<double, 2, 6> jacobian_pose_j;
    jacobian_pose_j = dr_dpcj * dpcj_dposj;

    
    Eigen::Matrix<double, 3, 6> dpcj_dpos_ex;
    dpcj_dpos_ex.leftCols<3>() = R_ic.transpose() * (R_j.transpose() * R_i - Eigen::Matrix3d::Identity());
    Eigen::Matrix3d tmp_r = dpcj_dposi.leftCols<3>() * R_i * R_ic;
    dpcj_dpos_ex.rightCols<3>() = Sophus::SO3d::hat(tmp_r * p_camera_i) - tmp_r * Sophus::SO3d::hat(p_camera_i) +
                                  Sophus::SO3d::hat(R_ic.transpose() * (R_j.transpose() * (R_i * tic + p_imu_i - p_imu_j) - tic));

    Eigen::Matrix<double, 2, 6> jacobian_pose_ex;
    jacobian_pose_ex.leftCols<6>() = dr_dpcj * dpcj_dpos_ex;

    Eigen::Matrix<double, 2, 1> jacobian_inv_depth_i;
    jacobian_inv_depth_i = dr_dpcj * tmp_r * p_camera_i / (-inv_depth_i);

    _jacobians[0] = jacobian_inv_depth_i;
    _jacobians[1] = jacobian_pose_i;
    _jacobians[2] = jacobian_pose_j;
    _jacobians[3] = jacobian_pose_ex;
}

void graph_optimization::EdgeReprojectionPoint3d::compute_residual() {
    Vec3 p_world = _vertices[0]->get_parameters();

    const auto &params_i = _vertices[1]->get_parameters();
    Vec3 t_i = params_i.head<3>();
    Qd q_i {params_i[6], params_i[3], params_i[4], params_i[5]};

    Vec3 p_imu_i = q_i.inverse() * (p_world - t_i);
    Vec3 p_camera_i = _qic.inverse() * (p_imu_i - _tic);

    double inv_depth_i = 1. / p_camera_i.z();

    _residual = (p_camera_i * inv_depth_i).head<2>() - _pt_i.head<2>();
}

void graph_optimization::EdgeReprojectionPoint3d::compute_jacobians() {
    Mat33 R_ic = _qic.toRotationMatrix();

    Vec3 p_world = _vertices[0]->get_parameters();

    const auto &params_i = _vertices[1]->get_parameters();
    Vec3 t_i = params_i.head<3>();
    Qd q_i {params_i[6], params_i[3], params_i[4], params_i[5]};
    Mat33 R_i = q_i.toRotationMatrix();

    Vec3 p_imu_i = q_i.inverse() * (p_world - t_i);
    Vec3 p_camera_i = _qic.inverse() * (p_imu_i - _tic);

    double inv_depth_i = 1. / p_camera_i.z();

    Mat23 dr_dpci;
    dr_dpci << inv_depth_i, 0., -p_camera_i[0] * inv_depth_i * inv_depth_i,
               0., inv_depth_i, -p_camera_i[1] * inv_depth_i * inv_depth_i;

    Mat33 dpci_dpw = R_ic.transpose() * R_i.transpose();

    Eigen::Matrix<double, 2, 3> jacobian_point3d;
    jacobian_point3d = dr_dpci * dpci_dpw;

    Eigen::Matrix<double, 3, 6> dpci_dposi;
    dpci_dposi.leftCols<3>() = -dpci_dpw;
    dpci_dposi.rightCols<3>() = R_ic.transpose() * Sophus::SO3d::hat(p_imu_i);

    Eigen::Matrix<double, 2, 6> jacobian_pose_i;
    jacobian_pose_i = dr_dpci * dpci_dposi;

    _jacobians[0] = jacobian_point3d;
    _jacobians[1] = jacobian_pose_i;
}
