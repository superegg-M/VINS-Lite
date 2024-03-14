#include "backend/vertex_pose.h"
#include "backend/vertex_speedbias.h"
#include "backend/edge_imu.h"

#include <iostream>

namespace myslam {
namespace backend {
using Sophus::SO3d;

Vec3 EdgeImu::gravity_ = Vec3(0, 0, 9.8);

void EdgeImu::ComputeResidual() {
//     VecX param_0 = verticies_[0]->Parameters();
//     Qd Qi(param_0[6], param_0[3], param_0[4], param_0[5]);
//     Vec3 Pi = param_0.head<3>();

//     VecX param_1 = verticies_[1]->Parameters();
//     Vec3 Vi = param_1.head<3>();
//     Vec3 Bai = param_1.segment(3, 3);
//     Vec3 Bgi = param_1.tail<3>();

//     VecX param_2 = verticies_[2]->Parameters();
//     Qd Qj(param_2[6], param_2[3], param_2[4], param_2[5]);
//     Vec3 Pj = param_2.head<3>();

//     VecX param_3 = verticies_[3]->Parameters();
//     Vec3 Vj = param_3.head<3>();
//     Vec3 Baj = param_3.segment(3, 3);
//     Vec3 Bgj = param_3.tail<3>();

//     residual_ = pre_integration_->evaluate(Pi, Qi, Vi, Bai, Bgi,
//                               Pj, Qj, Vj, Baj, Bgj);
// //    Mat1515 sqrt_info  = Eigen::LLT< Mat1515 >(pre_integration_->covariance.inverse()).matrixL().transpose();
//     SetInformation(pre_integration_->covariance.inverse());


    auto &&pose_i = verticies_[0]->Parameters();
	auto &&motion_i = verticies_[1]->Parameters();
	auto &&pose_j = verticies_[2]->Parameters();
	auto &&motion_j = verticies_[3]->Parameters();

	Vec3 p_i = pose_i.head<3>();
	Qd q_i(pose_i[6], pose_i[3], pose_i[4], pose_i[5]);
	Vec3 v_i = motion_i.head<3>();
	Vec3 ba_i = motion_i.segment(3, 3);
	Vec3 bg_i = motion_i.tail<3>();

	Vec3 p_j = pose_j.head<3>();
	Qd q_j(pose_j[6], pose_j[3], pose_j[4], pose_j[5]);
	Vec3 v_j = motion_j.head<3>();
	Vec3 ba_j = motion_j.segment(3, 3);
	Vec3 bg_j = motion_j.tail<3>();

	Eigen::Matrix<double, 3, 3> R_i_w = q_i.inverse().toRotationMatrix();
	auto &&dt = pre_integration_->get_sum_dt();

	Eigen::Vector3d delta_ba_i = ba_i - pre_integration_->get_ba();
	Eigen::Vector3d delta_bg_i = bg_i - pre_integration_->get_bg();
	pre_integration_->correct(delta_ba_i, delta_bg_i);

	residual_.head<3>() = R_i_w * (p_j - (p_i + v_i * dt + (0.5 * dt * dt) * IMUIntegration::get_gravity())) - pre_integration_->get_delta_p();
	residual_.segment<3>(3) = 2. * (pre_integration_->get_delta_q().inverse() * (q_i.inverse() * q_j)).vec();
	residual_.segment<3>(6) = R_i_w * (v_j - (v_i + dt * IMUIntegration::get_gravity())) - pre_integration_->get_delta_v();
	residual_.segment<3>(9) = ba_j - ba_i;
	residual_.segment<3>(12) = bg_j - bg_i;

	SetInformation(pre_integration_->get_covariance().inverse());
}

void EdgeImu::ComputeJacobians() {
//     VecX param_0 = verticies_[0]->Parameters();
//     Qd Qi(param_0[6], param_0[3], param_0[4], param_0[5]);
//     Vec3 Pi = param_0.head<3>();

//     VecX param_1 = verticies_[1]->Parameters();
//     Vec3 Vi = param_1.head<3>();
//     Vec3 Bai = param_1.segment(3, 3);
//     Vec3 Bgi = param_1.tail<3>();

//     VecX param_2 = verticies_[2]->Parameters();
//     Qd Qj(param_2[6], param_2[3], param_2[4], param_2[5]);
//     Vec3 Pj = param_2.head<3>();

//     VecX param_3 = verticies_[3]->Parameters();
//     Vec3 Vj = param_3.head<3>();
//     Vec3 Baj = param_3.segment(3, 3);
//     Vec3 Bgj = param_3.tail<3>();

//     double sum_dt = pre_integration_->sum_dt;
//     Eigen::Matrix3d dp_dba = pre_integration_->jacobian.template block<3, 3>(O_P, O_BA);
//     Eigen::Matrix3d dp_dbg = pre_integration_->jacobian.template block<3, 3>(O_P, O_BG);

//     Eigen::Matrix3d dq_dbg = pre_integration_->jacobian.template block<3, 3>(O_R, O_BG);

//     Eigen::Matrix3d dv_dba = pre_integration_->jacobian.template block<3, 3>(O_V, O_BA);
//     Eigen::Matrix3d dv_dbg = pre_integration_->jacobian.template block<3, 3>(O_V, O_BG);

//     if (pre_integration_->jacobian.maxCoeff() > 1e8 || pre_integration_->jacobian.minCoeff() < -1e8)
//     {
//         // ROS_WARN("numerical unstable in preintegration");
//     }

// //    if (jacobians[0])
//     {
//         Eigen::Matrix<double, 15, 6, Eigen::RowMajor> jacobian_pose_i;
//         jacobian_pose_i.setZero();

//         jacobian_pose_i.block<3, 3>(O_P, O_P) = -Qi.inverse().toRotationMatrix();
//         jacobian_pose_i.block<3, 3>(O_P, O_R) = Utility::skewSymmetric(Qi.inverse() * (0.5 * G * sum_dt * sum_dt + Pj - Pi - Vi * sum_dt));

// #if 0
//         jacobian_pose_i.block<3, 3>(O_R, O_R) = -(Qj.inverse() * Qi).toRotationMatrix();
// #else
//         Eigen::Quaterniond corrected_delta_q = pre_integration_->delta_q * Utility::deltaQ(dq_dbg * (Bgi - pre_integration_->linearized_bg));
//         jacobian_pose_i.block<3, 3>(O_R, O_R) = -(Utility::Qleft(Qj.inverse() * Qi) * Utility::Qright(corrected_delta_q)).bottomRightCorner<3, 3>();
// #endif

//         jacobian_pose_i.block<3, 3>(O_V, O_R) = Utility::skewSymmetric(Qi.inverse() * (G * sum_dt + Vj - Vi));
// //        jacobian_pose_i = sqrt_info * jacobian_pose_i;

//         if (jacobian_pose_i.maxCoeff() > 1e8 || jacobian_pose_i.minCoeff() < -1e8)
//         {
//         //     ROS_WARN("numerical unstable in preintegration");
//         }
//         jacobians_[0] = jacobian_pose_i;
//     }
// //    if (jacobians[1])
//     {
//         Eigen::Matrix<double, 15, 9, Eigen::RowMajor> jacobian_speedbias_i;
//         jacobian_speedbias_i.setZero();
//         jacobian_speedbias_i.block<3, 3>(O_P, O_V - O_V) = -Qi.inverse().toRotationMatrix() * sum_dt;
//         jacobian_speedbias_i.block<3, 3>(O_P, O_BA - O_V) = -dp_dba;
//         jacobian_speedbias_i.block<3, 3>(O_P, O_BG - O_V) = -dp_dbg;

// #if 0
//         jacobian_speedbias_i.block<3, 3>(O_R, O_BG - O_V) = -dq_dbg;
// #else
//         //Eigen::Quaterniond corrected_delta_q = pre_integration->delta_q * Utility::deltaQ(dq_dbg * (Bgi - pre_integration->linearized_bg));
//         //jacobian_speedbias_i.block<3, 3>(O_R, O_BG - O_V) = -Utility::Qleft(Qj.inverse() * Qi * corrected_delta_q).bottomRightCorner<3, 3>() * dq_dbg;
//         jacobian_speedbias_i.block<3, 3>(O_R, O_BG - O_V) = -Utility::Qleft(Qj.inverse() * Qi * pre_integration_->delta_q).bottomRightCorner<3, 3>() * dq_dbg;
// #endif

//         jacobian_speedbias_i.block<3, 3>(O_V, O_V - O_V) = -Qi.inverse().toRotationMatrix();
//         jacobian_speedbias_i.block<3, 3>(O_V, O_BA - O_V) = -dv_dba;
//         jacobian_speedbias_i.block<3, 3>(O_V, O_BG - O_V) = -dv_dbg;

//         jacobian_speedbias_i.block<3, 3>(O_BA, O_BA - O_V) = -Eigen::Matrix3d::Identity();

//         jacobian_speedbias_i.block<3, 3>(O_BG, O_BG - O_V) = -Eigen::Matrix3d::Identity();

// //        jacobian_speedbias_i = sqrt_info * jacobian_speedbias_i;
//         jacobians_[1] = jacobian_speedbias_i;
//     }
// //    if (jacobians[2])
//     {
//         Eigen::Matrix<double, 15, 6, Eigen::RowMajor> jacobian_pose_j;
//         jacobian_pose_j.setZero();

//         jacobian_pose_j.block<3, 3>(O_P, O_P) = Qi.inverse().toRotationMatrix();
// #if 0
//         jacobian_pose_j.block<3, 3>(O_R, O_R) = Eigen::Matrix3d::Identity();
// #else
//         Eigen::Quaterniond corrected_delta_q = pre_integration_->delta_q * Utility::deltaQ(dq_dbg * (Bgi - pre_integration_->linearized_bg));
//         jacobian_pose_j.block<3, 3>(O_R, O_R) = Utility::Qleft(corrected_delta_q.inverse() * Qi.inverse() * Qj).bottomRightCorner<3, 3>();
// #endif

// //        jacobian_pose_j = sqrt_info * jacobian_pose_j;
//         jacobians_[2] = jacobian_pose_j;

//     }
// //    if (jacobians[3])
//     {
//         Eigen::Matrix<double, 15, 9, Eigen::RowMajor> jacobian_speedbias_j;
//         jacobian_speedbias_j.setZero();

//         jacobian_speedbias_j.block<3, 3>(O_V, O_V - O_V) = Qi.inverse().toRotationMatrix();

//         jacobian_speedbias_j.block<3, 3>(O_BA, O_BA - O_V) = Eigen::Matrix3d::Identity();

//         jacobian_speedbias_j.block<3, 3>(O_BG, O_BG - O_V) = Eigen::Matrix3d::Identity();

// //        jacobian_speedbias_j = sqrt_info * jacobian_speedbias_j;
//         jacobians_[3] = jacobian_speedbias_j;

//     }

	auto pose_i = verticies_[0]->Parameters();
	auto motion_i = verticies_[1]->Parameters();
	auto pose_j = verticies_[2]->Parameters();
	auto motion_j = verticies_[3]->Parameters();

	Vec3 p_i = pose_i.head<3>();
	Qd q_i(pose_i[6], pose_i[3], pose_i[4], pose_i[5]);
	Vec3 v_i = motion_i.head<3>();
	Vec3 ba_i = motion_i.segment(3, 3);
	Vec3 bg_i = motion_i.tail<3>();

	Vec3 p_j = pose_j.head<3>();
	Qd q_j(pose_j[6], pose_j[3], pose_j[4], pose_j[5]);
	Vec3 v_j = motion_j.head<3>();
	Vec3 ba_j = motion_j.segment(3, 3);
	Vec3 bg_j = motion_j.tail<3>();

	Eigen::Matrix<double, 3, 3> R_i_w = q_i.inverse().toRotationMatrix();
	auto &&dt = pre_integration_->get_sum_dt();
	Eigen::Vector3d delta_ba_i = ba_i - pre_integration_->get_ba();
	Eigen::Vector3d delta_bg_i = bg_i - pre_integration_->get_bg();
	pre_integration_->correct(delta_ba_i, delta_bg_i);
	auto &&dr_dbg = pre_integration_->get_dr_dbg();
	auto &&dp_dba = pre_integration_->get_dp_dba();
	auto &&dp_dbg = pre_integration_->get_dp_dbg();
	auto &&dv_dba = pre_integration_->get_dv_dba();
	auto &&dv_dbg = pre_integration_->get_dv_dbg();

	// jacobian[0]: 15x6, (e_p, e_r, e_v, e_ba, e_bg) x (p_i, r_i)
	jacobians_[0] = Eigen::Matrix<double, 15, 6>::Zero();
	// e_p, p_i
	jacobians_[0].block<3, 3>(0, 0) = -R_i_w;
	// e_p, r_i
	jacobians_[0].block<3, 3>(0, 3) = Sophus::SO3d::hat(R_i_w * (p_j - (p_i + dt * v_i + (0.5 * dt * dt) * IMUIntegration::get_gravity())));
	// e_r, r_i
	jacobians_[0].block<3, 3>(3, 3) = -get_quat_left(q_j.inverse() * q_i).bottomRows<3>() * get_quat_right(pre_integration_->get_delta_q()).rightCols<3>();
	// e_v, r_i
	jacobians_[0].block<3, 3>(6, 3) = Sophus::SO3d::hat((R_i_w * (v_j - (v_i + dt * IMUIntegration::get_gravity()))));

	// jacobian[1]: 15x9, (e_p, e_r, e_v, e_ba, e_bg) x (v_i, ba_i, bg_i)
	jacobians_[1] = Eigen::Matrix<double, 15, 9>::Zero();
	// e_p, v_i
	jacobians_[1].block<3, 3>(0, 0) = -R_i_w * dt;
	// e_p, ba_i
	jacobians_[1].block<3, 3>(0, 3) = -dp_dba;
	// e_p, bg_i
	jacobians_[1].block<3, 3>(0, 6) = -dp_dbg;
	// e_r, bg_i
	jacobians_[1].block<3, 3>(3, 6) = -get_quat_left(q_j.inverse() * q_i * pre_integration_->get_delta_q()).bottomRightCorner<3, 3>() * dr_dbg;
	// e_v, v_i
	jacobians_[1].block<3, 3>(6, 0) = -R_i_w;
	// e_v, ba_i
	jacobians_[1].block<3, 3>(6, 3) = -dv_dba;
	// e_v, bg_i
	jacobians_[1].block<3, 3>(6, 6) = -dv_dbg;
	// e_ba, ba_i
	jacobians_[1].block<3, 3>(9, 3) = -Eigen::Matrix3d::Identity();
	// e_bg, bg_i
	jacobians_[1].block<3, 3>(12, 6) = -Eigen::Matrix3d::Identity();

	// jacobian[2]: 15x6, (e_p, e_r, e_v, e_ba, e_bg) x (p_j, r_j)
	jacobians_[2] = Eigen::Matrix<double, 15, 6>::Zero();
	// e_p, p_j
	jacobians_[2].block<3, 3>(0, 0) = R_i_w;
	// e_r, r_j
	jacobians_[2].block<3, 3>(3, 3) = get_quat_left(pre_integration_->get_delta_q().inverse() * q_i.inverse() * q_j).bottomRightCorner<3, 3>();

	// jacobian[3]: 15x9, (e_p, e_r, e_v, e_ba, e_bg) x (v_j, ba_j, bg_j)
	jacobians_[3] = Eigen::Matrix<double, 15, 9>::Zero();
	// e_v, v_j
	jacobians_[3].block<3, 3>(6, 0) = R_i_w;
	// e_ba, ba_j
	jacobians_[3].block<3, 3>(9, 3) = Eigen::Matrix3d::Identity();
	// e_bg, bg_j
	jacobians_[3].block<3, 3>(12, 6) = Eigen::Matrix3d::Identity();
}

Eigen::Matrix<double, 4, 4> EdgeImu::get_quat_left(const Qd &q) {
	Eigen::Matrix<double, 4, 4> m;
	m(0, 0) = q.w();
	m.block<3, 1>(1, 0) = q.vec();
	m.block<1, 3>(0, 1) = -q.vec().transpose();
	// m.block<3, 3>(1, 1) << q.w(), -q.z(), q.y(),
	// 										q.z(), q.w(), -q.x(),
	// 										-q.y(), q.x(), q.w();
	m(1, 1) = q.w();
	m(1, 2) = -q.z();
	m(1, 3) = q.y();
	m(2, 1) = q.z();
	m(2, 2) = q.w();
	m(2, 3) = -q.x();
	m(3, 1) = -q.y();
	m(3, 2) = q.x();
	m(3, 3) = q.w();
	return m;
}

Eigen::Matrix<double, 4, 4> EdgeImu::get_quat_right(const Qd &q) {
	Eigen::Matrix<double, 4, 4> m;
	m(0, 0) = q.w();
	m.block<3, 1>(1, 0) = q.vec();
	m.block<1, 3>(0, 1) = -q.vec().transpose();
	// m.block<3, 3>(1, 1) << q.w(), q.z(), -q.y(),
	// 										-q.z(), q.w(), q.x(),
	// 										q.y(), -q.x(), q.w();
	m(1, 1) = q.w();
	m(1, 2) = q.z();
	m(1, 3) = -q.y();
	m(2, 1) = -q.z();
	m(2, 2) = q.w();
	m(2, 3) = q.x();
	m(3, 1) = q.y();
	m(3, 2) = -q.x();
	m(3, 3) = q.w();
	return m;
}

}
}