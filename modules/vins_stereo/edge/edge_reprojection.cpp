//
// Created by Cain on 2024/1/2.
//

#include <iostream>
#include "edge_reprojection.h"
#include "../vertex/vertex_inverse_depth.h"
#include "../vertex/vertex_pose.h"
#include "../vertex/vertex_point3d.h"
#include "../vertex/vertex_motion.h"

void graph_optimization::EdgeReprojectionTwoImuOneCameras::compute_residual() {
    double inv_depth_i = _vertices[0]->get_parameters()[0];

    auto t_i = Eigen::Map<Eigen::Vector3d>(_vertices[1]->parameters());
    auto q_i = Eigen::Map<Eigen::Quaterniond>(_vertices[1]->parameters() + 3);

    auto t_j = Eigen::Map<Eigen::Vector3d>(_vertices[2]->parameters());
    auto q_j = Eigen::Map<Eigen::Quaterniond>(_vertices[2]->parameters() + 3);

    auto tic = Eigen::Map<Eigen::Vector3d>(_vertices[3]->parameters());
    auto qic = Eigen::Map<Eigen::Quaterniond>(_vertices[3]->parameters() + 3);

    Vec3 p_camera_i = _pt_i / inv_depth_i;
    Vec3 p_imu_i = qic * p_camera_i + tic;
    Vec3 p_world = q_i * p_imu_i + t_i;
    Vec3 p_imu_j = q_j.inverse() * (p_world - t_j);
    Vec3 p_camera_j = qic.inverse() * (p_imu_j - tic);

    double inv_depth_j = 1. / p_camera_j.z();
    _residual = (p_camera_j * inv_depth_j).head<2>() - _pt_j.head<2>();
}

void graph_optimization::EdgeReprojectionTwoImuOneCameras::compute_jacobians() {
    double inv_depth_i = _vertices[0]->get_parameters()[0];

    auto t_i = Eigen::Map<Eigen::Vector3d>(_vertices[1]->parameters());
    auto q_i = Eigen::Map<Eigen::Quaterniond>(_vertices[1]->parameters() + 3);
    Mat33 R_i = q_i.toRotationMatrix();

    auto t_j = Eigen::Map<Eigen::Vector3d>(_vertices[2]->parameters());
    auto q_j = Eigen::Map<Eigen::Quaterniond>(_vertices[2]->parameters() + 3);
    Mat33 R_j = q_j.toRotationMatrix();

    auto tic = Eigen::Map<Eigen::Vector3d>(_vertices[3]->parameters());
    auto qic = Eigen::Map<Eigen::Quaterniond>(_vertices[3]->parameters() + 3);
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
    Eigen::Matrix3d tmp_r = dpcj_dposi.leftCols<3>() * R_i;     // R_c1i0
    dpcj_dpos_ex.leftCols<3>() = tmp_r - R_ic.transpose();
    tmp_r = tmp_r * R_ic;   // R_c1c0
    dpcj_dpos_ex.rightCols<3>() = Sophus::SO3d::hat(p_camera_j) - tmp_r * Sophus::SO3d::hat(p_camera_i);

    Eigen::Matrix<double, 2, 6> jacobian_pose_ex;
    jacobian_pose_ex.leftCols<6>() = dr_dpcj * dpcj_dpos_ex;

    Eigen::Matrix<double, 2, 1> jacobian_inv_depth_i;
    jacobian_inv_depth_i = dr_dpcj * tmp_r * p_camera_i / (-inv_depth_i);

    _jacobians[0] = jacobian_inv_depth_i;
    _jacobians[1] = jacobian_pose_i;
    _jacobians[2] = jacobian_pose_j;
    _jacobians[3] = jacobian_pose_ex;
}




void graph_optimization::EdgeReprojectionOneImuTwoCameras::compute_residual() {
    double inv_depth_0 = _vertices[0]->get_parameters()[0];

    Vec3 t_ic0_i = Eigen::Map<Eigen::Vector3d>(_vertices[1]->parameters());
    auto q_ic0 = Eigen::Map<Eigen::Quaterniond>(_vertices[1]->parameters() + 3);

    auto t_ic1_i = Eigen::Map<Eigen::Vector3d>(_vertices[2]->parameters());
    auto q_ic1 = Eigen::Map<Eigen::Quaterniond>(_vertices[2]->parameters() + 3);

    Qd q_c1c0 = (q_ic1.inverse() * q_ic0).normalized();
    Vec3 t_c1c0_c1 = q_ic1.inverse() * (t_ic0_i - t_ic1_i);

    Vec3 p1 = q_c1c0 * (_pt_0 / inv_depth_0) + t_c1c0_c1;

    double inv_depth_1 = 1. / p1.z();
    _residual = p1.head<2>() * inv_depth_1 - _pt_1.head<2>();
}

void graph_optimization::EdgeReprojectionOneImuTwoCameras::compute_jacobians() {
    double inv_depth_0 = _vertices[0]->get_parameters()[0];

    Vec3 t_ic0_i = Eigen::Map<Eigen::Vector3d>(_vertices[1]->parameters());
    auto q_ic0 = Eigen::Map<Eigen::Quaterniond>(_vertices[1]->parameters() + 3);

    auto t_ic1_i = Eigen::Map<Eigen::Vector3d>(_vertices[2]->parameters());
    auto q_ic1 = Eigen::Map<Eigen::Quaterniond>(_vertices[2]->parameters() + 3);

    Qd q_c1c0 = (q_ic1.inverse() * q_ic0).normalized();
    Vec3 t_c1c0_c1 = q_ic1.inverse() * (t_ic0_i - t_ic1_i);

    Vec3 p0 = _pt_0 / inv_depth_0;
    Vec3 p1 = q_c1c0 * p0 + t_c1c0_c1;

    double inv_depth_1 = 1. / p1.z();

    Mat23 dr_dp1;
    dr_dp1 << inv_depth_1, 0., -p1.x() * inv_depth_1 * inv_depth_1,
              0., inv_depth_1, -p1.y() * inv_depth_1 * inv_depth_1;

    Eigen::Matrix<double, 3, 6> dp1_dpose0;
    dp1_dpose0.leftCols<3>() = q_ic1.toRotationMatrix().transpose();
    dp1_dpose0.rightCols<3>() = -q_c1c0.toRotationMatrix() * Sophus::SO3d::hat(p0);

    Eigen::Matrix<double, 2, 6> jacobian_pose_0;
    jacobian_pose_0 = dr_dp1 * dp1_dpose0;

    Eigen::Matrix<double, 3, 6> dp1_dpose1;
    dp1_dpose1.leftCols<3>() = -dp1_dpose0.leftCols<3>();
    dp1_dpose1.rightCols<3>() = Sophus::SO3d::hat(p1);

    Eigen::Matrix<double, 2, 6> jacobian_pose_1;
    jacobian_pose_1 = dr_dp1 * dp1_dpose1;

    Eigen::Matrix<double, 2, 1> jacobian_inv_depth_0;
    jacobian_inv_depth_0 = dr_dp1 * (q_c1c0 * p0) / (-inv_depth_0);

    _jacobians[0] = jacobian_inv_depth_0;
    _jacobians[1] = jacobian_pose_0;
    _jacobians[2] = jacobian_pose_1;
}




void graph_optimization::EdgeReprojectionTwoImuTwoCameras::compute_residual() {
    double inv_depth_i = _vertices[0]->get_parameters()[0];

    auto t_i = Eigen::Map<Eigen::Vector3d>(_vertices[1]->parameters());
    auto q_i = Eigen::Map<Eigen::Quaterniond>(_vertices[1]->parameters() + 3);

    auto t_j = Eigen::Map<Eigen::Vector3d>(_vertices[2]->parameters());
    auto q_j = Eigen::Map<Eigen::Quaterniond>(_vertices[2]->parameters() + 3);

    auto t_ic0 = Eigen::Map<Eigen::Vector3d>(_vertices[3]->parameters());
    auto q_ic0 = Eigen::Map<Eigen::Quaterniond>(_vertices[3]->parameters() + 3);

    auto t_ic1 = Eigen::Map<Eigen::Vector3d>(_vertices[4]->parameters());
    auto q_ic1 = Eigen::Map<Eigen::Quaterniond>(_vertices[4]->parameters() + 3);

    Vec3 p_camera_i = _pt_i / inv_depth_i;
    Vec3 p_imu_i = q_ic0 * p_camera_i + t_ic0;
    Vec3 p_world = q_i * p_imu_i + t_i;
    Vec3 p_imu_j = q_j.inverse() * (p_world - t_j);
    Vec3 p_camera_j = q_ic1.inverse() * (p_imu_j - t_ic1);

    double inv_depth_j = 1. / p_camera_j.z();
    _residual = (p_camera_j * inv_depth_j).head<2>() - _pt_j.head<2>();
}

void graph_optimization::EdgeReprojectionTwoImuTwoCameras::compute_jacobians() {
    double inv_depth_i = _vertices[0]->get_parameters()[0];

    auto t_i = Eigen::Map<Eigen::Vector3d>(_vertices[1]->parameters());
    auto q_i = Eigen::Map<Eigen::Quaterniond>(_vertices[1]->parameters() + 3);
    Mat33 R_i = q_i.toRotationMatrix();

    auto t_j = Eigen::Map<Eigen::Vector3d>(_vertices[2]->parameters());
    auto q_j = Eigen::Map<Eigen::Quaterniond>(_vertices[2]->parameters() + 3);
    Mat33 R_j = q_j.toRotationMatrix();

    auto t_ic0 = Eigen::Map<Eigen::Vector3d>(_vertices[3]->parameters());
    auto q_ic0 = Eigen::Map<Eigen::Quaterniond>(_vertices[3]->parameters() + 3);
    Mat33 R_ic0 = q_ic0.toRotationMatrix();

    auto t_ic1 = Eigen::Map<Eigen::Vector3d>(_vertices[4]->parameters());
    auto q_ic1 = Eigen::Map<Eigen::Quaterniond>(_vertices[4]->parameters() + 3);
    Mat33 R_ic1 = q_ic1.toRotationMatrix();

    Vec3 p_camera_i = _pt_i / inv_depth_i;
    Vec3 p_imu_i = R_ic0 * p_camera_i + t_ic0;
    Vec3 p_world = R_i * p_imu_i + t_i;
    Vec3 p_imu_j = R_j.transpose() * (p_world - t_j);
    Vec3 p_camera_j = R_ic1.transpose() * (p_imu_j - t_ic1);

    double inv_depth_j = 1. / p_camera_j.z();

    Mat23 dr_dpcj;
    dr_dpcj << inv_depth_j, 0., -p_camera_j[0] * inv_depth_j * inv_depth_j,
            0., inv_depth_j, -p_camera_j[1] * inv_depth_j * inv_depth_j;

    Eigen::Matrix<double, 3, 6> dpcj_dposi;
    dpcj_dposi.leftCols<3>() = R_ic1.transpose() * R_j.transpose();
    dpcj_dposi.rightCols<3>() = -dpcj_dposi.leftCols<3>() * R_i * Sophus::SO3d::hat(p_imu_i);

    Eigen::Matrix<double, 2, 6> jacobian_pose_i;
    jacobian_pose_i = dr_dpcj * dpcj_dposi;

    Eigen::Matrix<double, 3, 6> dpcj_dposj;
    dpcj_dposj.leftCols<3>() = -dpcj_dposi.leftCols<3>();
    dpcj_dposj.rightCols<3>() = R_ic1.transpose() * Sophus::SO3d::hat(p_imu_j);

    Eigen::Matrix<double, 2, 6> jacobian_pose_j;
    jacobian_pose_j = dr_dpcj * dpcj_dposj;

    Eigen::Matrix<double, 3, 6> dpcj_dpos_ex0;
    dpcj_dpos_ex0.leftCols<3>() = dpcj_dposi.leftCols<3>() * R_i;
    Eigen::Matrix3d R_c1c0 = dpcj_dpos_ex0.leftCols<3>() * R_ic0;
    dpcj_dpos_ex0.rightCols<3>() = -R_c1c0 * Sophus::SO3d::hat(p_camera_i);

    Eigen::Matrix<double, 2, 6> jacobian_pose_ex0;
    jacobian_pose_ex0.leftCols<6>() = dr_dpcj * dpcj_dpos_ex0;

    Eigen::Matrix<double, 3, 6> dpcj_dpos_ex1;
    dpcj_dpos_ex1.leftCols<3>() = -R_ic1.transpose();
    dpcj_dpos_ex1.rightCols<3>() = Sophus::SO3d::hat(p_camera_j);

    Eigen::Matrix<double, 2, 6> jacobian_pose_ex1;
    jacobian_pose_ex1.leftCols<6>() = dr_dpcj * dpcj_dpos_ex1;

    Eigen::Matrix<double, 2, 1> jacobian_inv_depth_i;
    jacobian_inv_depth_i = dr_dpcj * R_c1c0 * p_camera_i / (-inv_depth_i);

    _jacobians[0] = jacobian_inv_depth_i;
    _jacobians[1] = jacobian_pose_i;
    _jacobians[2] = jacobian_pose_j;
    _jacobians[3] = jacobian_pose_ex0;
    _jacobians[4] = jacobian_pose_ex1;
}



void graph_optimization::EdgeReprojectionPoint3d::compute_residual() {
    // 特征点的世界坐标
    auto p_world = Eigen::Map<Eigen::Vector3d>(_vertices[0]->parameters());

    // imu的位姿
    auto t_i = Eigen::Map<Eigen::Vector3d>(_vertices[1]->parameters());
    auto q_i = Eigen::Map<Eigen::Quaterniond>(_vertices[1]->parameters());

    // 相机的外参
    auto t_ic = Eigen::Map<Eigen::Vector3d>(_vertices[2]->parameters());
    auto q_ic = Eigen::Map<Eigen::Quaterniond>(_vertices[2]->parameters());

    // 重投影误差
    Vec3 p_imu_i = q_i.inverse() * (p_world - t_i);
    Vec3 p_camera_i = q_ic.inverse() * (p_imu_i - t_ic);

    // 逆深度
    double inv_depth_i = 1. / p_camera_i.z();

    // 误差
    _residual = (p_camera_i * inv_depth_i).head<2>() - _pt_i.head<2>();
}

void graph_optimization::EdgeReprojectionPoint3d::compute_jacobians() {
    // 特征点的世界坐标
    auto p_world = Eigen::Map<Eigen::Vector3d>(_vertices[0]->parameters());

    // imu的位姿
    auto t_i = Eigen::Map<Eigen::Vector3d>(_vertices[1]->parameters());
    auto q_i = Eigen::Map<Eigen::Quaterniond>(_vertices[1]->parameters());
    Mat33 R_i = q_i.toRotationMatrix();

    // 相机的外参
    auto t_ic = Eigen::Map<Eigen::Vector3d>(_vertices[2]->parameters());
    auto q_ic = Eigen::Map<Eigen::Quaterniond>(_vertices[2]->parameters());
    Mat33 R_ic = q_ic.toRotationMatrix();

    // 重投影
    Vec3 p_imu_i = q_i.inverse() * (p_world - t_i);
    Vec3 p_camera_i = q_ic.inverse() * (p_imu_i - t_ic);

    // 逆深度
    double inv_depth_i = 1. / p_camera_i.z();

    // 误差对投影点的偏导
    Mat23 dr_dpci;
    dr_dpci << inv_depth_i, 0., -p_camera_i[0] * inv_depth_i * inv_depth_i,
            0., inv_depth_i, -p_camera_i[1] * inv_depth_i * inv_depth_i;

    // 投影点对特征点世界坐标的偏导
    Mat33 dpci_dpw = R_ic.transpose() * R_i.transpose();

    Eigen::Matrix<double, 2, 3> jacobian_point3d;
    jacobian_point3d = dr_dpci * dpci_dpw;

    // 投影点对imu位姿的偏导
    Eigen::Matrix<double, 3, 6> dpci_dpose_i;
    dpci_dpose_i.leftCols<3>() = -dpci_dpw;
    dpci_dpose_i.rightCols<3>() = R_ic.transpose() * Sophus::SO3d::hat(p_imu_i);

    Eigen::Matrix<double, 2, 6> jacobian_pose_i;
    jacobian_pose_i = dr_dpci * dpci_dpose_i;

    // 投影点对相机外参的偏导
    Eigen::Matrix<double, 3, 6> dpci_dpose_ext;
    dpci_dpose_ext.leftCols<3>() = -R_ic.transpose();
    dpci_dpose_ext.rightCols<3>() = Sophus::SO3d::hat(p_camera_i);

    Eigen::Matrix<double, 2, 6> jacobian_pose_ext;
    jacobian_pose_ext = dr_dpci * dpci_dpose_ext;

    _jacobians[0] = jacobian_point3d;
    _jacobians[1] = jacobian_pose_i;
    _jacobians[2] = jacobian_pose_ext;
}