//
// Created by gaoxiang19 on 19-1-7.
//
#include "backend/imu_integration.h"
#include <iostream>
#include "../../include/parameters.h"

using Sophus::SO3d;

namespace myslam {
namespace backend {

using namespace Eigen;
Vec3 IMUIntegration::_gravity = {0., 0., -G(2)};

IMUIntegration::IMUIntegration(const Vec3 &acc_init, const Vec3 &gyro_init, const Vec3 &ba, const Vec3 &bg) 
: _acc_init(acc_init), _gyro_init(gyro_init), _acc_last(acc_init), _gyro_last(gyro_init), _ba(ba), _bg(bg) {
    // _jacobian.setIdentity();
    // _covariance.setZero();
    // _sum_dt = 0.;
    // _delta_p.setZero();
    // _delta_v.setZero();
    // _delta_q.setIdentity();
    double gyro_var = GYR_N * GYR_N;
    double acc_var = ACC_N * ACC_N;
    double bg_var = GYR_W * GYR_W;
    double ba_var = ACC_W * ACC_W;
    _noise_measurement << acc_var, acc_var, acc_var,
                          gyro_var, gyro_var, gyro_var,
                          acc_var, acc_var, acc_var,
                          gyro_var, gyro_var, gyro_var;
    _noise_random_walk << ba_var, ba_var, ba_var,
                          bg_var, bg_var, bg_var;

    _noise = Eigen::Matrix<double, 18, 18>::Zero();
    _noise.block<3, 3>(0, 0) =  (ACC_N * ACC_N) * Eigen::Matrix3d::Identity();
    _noise.block<3, 3>(3, 3) =  (GYR_N * GYR_N) * Eigen::Matrix3d::Identity();
    _noise.block<3, 3>(6, 6) =  (ACC_N * ACC_N) * Eigen::Matrix3d::Identity();
    _noise.block<3, 3>(9, 9) =  (GYR_N * GYR_N) * Eigen::Matrix3d::Identity();
    _noise.block<3, 3>(12, 12) =  (ACC_W * ACC_W) * Eigen::Matrix3d::Identity();
    _noise.block<3, 3>(15, 15) =  (GYR_W * GYR_W) * Eigen::Matrix3d::Identity();

    _N << ACC_N * ACC_N, ACC_N * ACC_N, ACC_N * ACC_N,
          GYR_N * GYR_N, GYR_N * GYR_N, GYR_N * GYR_N,
          ACC_N * ACC_N, ACC_N * ACC_N, ACC_N * ACC_N,
          GYR_N * GYR_N, GYR_N * GYR_N, GYR_N * GYR_N,
          ACC_W * ACC_W, ACC_W * ACC_W, ACC_W * ACC_W,
          GYR_W * GYR_W, GYR_W * GYR_W, GYR_W * GYR_W;
}

void IMUIntegration::push_back(double dt, const Vec3 &acc, const Vec3 &gyro) {
    _dt_buf.emplace_back(dt);
    _acc_buf.emplace_back(acc);
    _gyro_buf.emplace_back(gyro);
    propagate(dt, acc, gyro);
}

void IMUIntegration::propagate(double dt, const Vec3 &acc, const Vec3 &gyro) {
    // _dt = dt;

    // Vector3d un_acc_0 = _delta_q * (_acc_last - _ba);
    // Vector3d un_gyr = 0.5 * (_gyro_last + gyro) - _bg;
    // Quaterniond result_delta_q = _delta_q * Quaterniond(1, un_gyr(0) * _dt / 2, un_gyr(1) * _dt / 2, un_gyr(2) * _dt / 2);
    // result_delta_q.normalize();
    // Vector3d un_acc_1 = result_delta_q * (acc - _ba);
    // Vector3d un_acc = 0.5 * (un_acc_0 + un_acc_1);
    // Vector3d result_delta_p = _delta_p + _delta_v * _dt + 0.5 * un_acc * _dt * _dt;
    // Vector3d result_delta_v = _delta_v + un_acc * _dt;      

    // Vector3d w_x = 0.5 * (_gyro_last + gyro) - _bg;
    // Vector3d a_0_x = _acc_last - _ba;
    // Vector3d a_1_x = acc - _ba;
    // Matrix3d R_w_x, R_a_0_x, R_a_1_x;

    // R_w_x<<0, -w_x(2), w_x(1),
    //     w_x(2), 0, -w_x(0),
    //     -w_x(1), w_x(0), 0;
    // R_a_0_x<<0, -a_0_x(2), a_0_x(1),
    //     a_0_x(2), 0, -a_0_x(0),
    //     -a_0_x(1), a_0_x(0), 0;
    // R_a_1_x<<0, -a_1_x(2), a_1_x(1),
    //     a_1_x(2), 0, -a_1_x(0),
    //     -a_1_x(1), a_1_x(0), 0;

    // MatrixXd F = MatrixXd::Zero(15, 15);
    // F.block<3, 3>(0, 0) = Matrix3d::Identity();
    // F.block<3, 3>(0, 3) = -0.25 * _delta_q.toRotationMatrix() * R_a_0_x * _dt * _dt + 
    //                         -0.25 * result_delta_q.toRotationMatrix() * R_a_1_x * (Matrix3d::Identity() - R_w_x * _dt) * _dt * _dt;
    // F.block<3, 3>(0, 6) = MatrixXd::Identity(3,3) * _dt;
    // F.block<3, 3>(0, 9) = -0.25 * (_delta_q.toRotationMatrix() + result_delta_q.toRotationMatrix()) * _dt * _dt;
    // F.block<3, 3>(0, 12) = -0.25 * result_delta_q.toRotationMatrix() * R_a_1_x * _dt * _dt * -_dt;
    // F.block<3, 3>(3, 3) = Matrix3d::Identity() - R_w_x * _dt;
    // F.block<3, 3>(3, 12) = -1.0 * MatrixXd::Identity(3,3) * _dt;
    // F.block<3, 3>(6, 3) = -0.5 * _delta_q.toRotationMatrix() * R_a_0_x * _dt + 
    //                         -0.5 * result_delta_q.toRotationMatrix() * R_a_1_x * (Matrix3d::Identity() - R_w_x * _dt) * _dt;
    // F.block<3, 3>(6, 6) = Matrix3d::Identity();
    // F.block<3, 3>(6, 9) = -0.5 * (_delta_q.toRotationMatrix() + result_delta_q.toRotationMatrix()) * _dt;
    // F.block<3, 3>(6, 12) = -0.5 * result_delta_q.toRotationMatrix() * R_a_1_x * _dt * -_dt;
    // F.block<3, 3>(9, 9) = Matrix3d::Identity();
    // F.block<3, 3>(12, 12) = Matrix3d::Identity();
    // //cout<<"A"<<endl<<A<<endl;

    // MatrixXd V = MatrixXd::Zero(15,18);
    // V.block<3, 3>(0, 0) =  0.25 * _delta_q.toRotationMatrix() * _dt * _dt;
    // V.block<3, 3>(0, 3) =  0.25 * -result_delta_q.toRotationMatrix() * R_a_1_x  * _dt * _dt * 0.5 * _dt;
    // V.block<3, 3>(0, 6) =  0.25 * result_delta_q.toRotationMatrix() * _dt * _dt;
    // V.block<3, 3>(0, 9) =  V.block<3, 3>(0, 3);
    // V.block<3, 3>(3, 3) =  0.5 * MatrixXd::Identity(3,3) * _dt;
    // V.block<3, 3>(3, 9) =  0.5 * MatrixXd::Identity(3,3) * _dt;
    // V.block<3, 3>(6, 0) =  0.5 * _delta_q.toRotationMatrix() * _dt;
    // V.block<3, 3>(6, 3) =  0.5 * -result_delta_q.toRotationMatrix() * R_a_1_x  * _dt * 0.5 * _dt;
    // V.block<3, 3>(6, 6) =  0.5 * result_delta_q.toRotationMatrix() * _dt;
    // V.block<3, 3>(6, 9) =  V.block<3, 3>(6, 3);
    // V.block<3, 3>(9, 12) = MatrixXd::Identity(3,3) * _dt;
    // V.block<3, 3>(12, 15) = MatrixXd::Identity(3,3) * _dt;

    // //step_jacobian = F;
    // //step_V = V;
    // _jacobian = F * _jacobian;
    // _covariance = F * _covariance * F.transpose() + V * _noise * V.transpose();

    // double half_dt = 0.5 * dt;
    // auto delta_r_last = _delta_q.toRotationMatrix();
    // auto delta_r = result_delta_q.toRotationMatrix();
    // auto dr = Quaterniond(1, un_gyr(0) * _dt / 2, un_gyr(1) * _dt / 2, un_gyr(2) * _dt / 2).normalized().toRotationMatrix();
    // Mat33 jr_dt = Sophus::SO3d::JacobianR(un_gyr * _dt ) * dt;
    // auto dr_dbg_last = _dr_dbg;
    // _dr_dbg = dr.transpose() * _dr_dbg - jr_dt;

    // auto delta_r_a_hat_last = delta_r_last * R_a_0_x;
    // auto delta_r_a_hat = delta_r * R_a_1_x;
    // auto dv_dba_last = _dv_dba;
    // auto dv_dbg_last = _dv_dbg;
    // _dv_dba -= half_dt * (delta_r_last + delta_r);
    // _dv_dbg -= half_dt * (delta_r_a_hat_last * dr_dbg_last + delta_r_a_hat * _dr_dbg);

    // _dp_dba += half_dt * (dv_dba_last + _dv_dba);
    // _dp_dbg += half_dt * (dv_dbg_last + _dv_dbg);

    // // std::cout << "1 : " << std::endl;
    // // std::cout << _dr_dbg << std::endl;
    // // std::cout << "2 : " << std::endl;
    // // std::cout << _jacobian.block<3, 3>(3, 12) << std::endl;

    // // _dr_dbg = _jacobian.block<3, 3>(3, 12);
    // // _dv_dba = _jacobian.block<3, 3>(6, 9);
    // // _dv_dbg = _jacobian.block<3, 3>(6, 12);
    // // _dp_dba = _jacobian.block<3, 3>(0, 9);
    // // _dp_dbg = _jacobian.block<3, 3>(0, 12);


    // _delta_p = result_delta_p;
    // _delta_q = result_delta_q;
    // _delta_v = result_delta_v;
    // _sum_dt += dt;
    // _acc_last = acc;
    // _gyro_last = gyro;  


    _dt = dt;

    // 去偏移
    Vec3 a = acc - _ba;
    Vec3 a_last = _acc_last - _ba;
    Vec3 w = gyro - _bg;
    Vec3 w_last = _gyro_last - _bg;

    // 预积分(中值积分)
    Vec3 w_mid = 0.5 * (w_last + w);
    Vec3 d_axis_angle = w_mid * dt;
    Qd dq = Sophus::SO3d::exp(d_axis_angle).unit_quaternion();
    auto delta_q_last = _delta_q;
    _delta_q *= dq;
    _delta_q.normalize();

    auto delta_r_last = delta_q_last.toRotationMatrix();
    auto delta_r = _delta_q.toRotationMatrix();
    auto dr = dq.toRotationMatrix();

    Vec3 a_mid = 0.5 * (a_last + dr * a);
    Vec3 d_vel = delta_r_last * a_mid * dt;
    auto delta_v_last = _delta_v;
    _delta_v += d_vel;

    Vec3 delta_v_mid = 0.5 * (delta_v_last + _delta_v);
    Vec3 d_pos = delta_v_mid * dt;
    _delta_p += d_pos;

    _sum_dt += dt;

    // 预积分关于偏移的雅可比
    double half_dt = 0.5 * dt;
    Mat33 jr_dt = Sophus::SO3d::JacobianR(d_axis_angle) * dt;
    auto dr_dbg_last = _dr_dbg;
    _dr_dbg = dr.transpose() * _dr_dbg - jr_dt;

    auto delta_r_a_hat_last = delta_r_last * Sophus::SO3d::hat(a_last);
    auto delta_r_a_hat = delta_r * Sophus::SO3d::hat(a);
    auto dv_dba_last = _dv_dba;
    auto dv_dbg_last = _dv_dbg;
    _dv_dba -= half_dt * (delta_r_last + delta_r);
    _dv_dbg -= half_dt * (delta_r_a_hat_last * dr_dbg_last + delta_r_a_hat * _dr_dbg);

    _dp_dba += half_dt * (dv_dba_last + _dv_dba);
    _dp_dbg += half_dt * (dv_dbg_last + _dv_dbg);

    // 噪声迭代
    // // _A.block<3, 3>(6, 3).noalias() = -half_dt * (delta_r_a_hat_last + delta_r_a_hat * dr.transpose());
    // // _A.block<3, 3>(6, 9).noalias() = -half_dt * (delta_r_last + delta_r);
    // // _A.block<3, 3>(6, 12).noalias() = half_dt * delta_r_a_hat * jr_dt;
    // _A00.block<3, 3>(6, 3).noalias() = -half_dt * (delta_r_a_hat_last + delta_r_a_hat * dr.transpose());
    // _A01.block<3, 3>(6, 0).noalias() = -half_dt * (delta_r_last + delta_r);
    // _A01.block<3, 3>(6, 3).noalias() = half_dt * delta_r_a_hat * jr_dt;

    // // _A.block<3, 3>(0, 3).noalias() = half_dt * _A.block<3, 3>(6, 3);
    // // _A.block<3, 3>(0, 6).noalias() = Mat33::Identity() * dt;
    // // _A.block<3, 3>(0, 9).noalias() = half_dt * _A.block<3, 3>(6, 9);
    // // _A.block<3, 3>(0, 12).noalias() = half_dt * _A.block<3, 3>(6, 12);
    // _A00.block<3, 3>(0, 3).noalias() = half_dt * _A00.block<3, 3>(6, 3);
    // _A00.block<3, 3>(0, 6).noalias() = Mat33::Identity() * dt;
    // _A01.block<3, 3>(0, 0).noalias() = half_dt * _A01.block<3, 3>(6, 0);
    // _A01.block<3, 3>(0, 3).noalias() = half_dt * _A01.block<3, 3>(6, 3);
    
    // // _A.block<3, 3>(3, 3).noalias() = dr.transpose();
    // // _A.block<3, 3>(3, 12).noalias() = -jr_dt;
    // _A00.block<3, 3>(3, 3).noalias() = dr.transpose();
    // _A01.block<3, 3>(3, 3).noalias() = -jr_dt;

    // // _B.block<3, 3>(6, 0).noalias() = -half_dt * delta_r_last;
    // // _B.block<3, 3>(6, 3).noalias() = 0.5 * half_dt * delta_r_a_hat * jr_dt;
    // // _B.block<3, 3>(6, 6).noalias() = -half_dt * delta_r;
    // // _B.block<3, 3>(6, 9).noalias() = _B.block<3, 3>(6, 3);
    // // _B.block<3, 3>(6, 12).noalias() = _B.block<3, 3>(6, 6) * dt;
    // // _B.block<3, 3>(6, 15).noalias() = _B.block<3, 3>(6, 9) * dt;
    // _B00.block<3, 3>(6, 0).noalias() = -half_dt * delta_r_last;
    // _B00.block<3, 3>(6, 3).noalias() = 0.5 * half_dt * delta_r_a_hat * jr_dt;
    // _B00.block<3, 3>(6, 6).noalias() = -half_dt * delta_r;
    // _B00.block<3, 3>(6, 9).noalias() = _B00.block<3, 3>(6, 3);
    // _B01.block<3, 3>(6, 0).noalias() = _B00.block<3, 3>(6, 6) * dt;
    // _B01.block<3, 3>(6, 3).noalias() = _B00.block<3, 3>(6, 9) * dt;

    // // _B.block<3, 3>(0, 0).noalias() = half_dt * _B.block<3, 3>(6, 0);
    // // _B.block<3, 3>(0, 3).noalias() = half_dt * _B.block<3, 3>(6, 3);
    // // _B.block<3, 3>(0, 6).noalias() = half_dt * _B.block<3, 3>(6, 6);
    // // _B.block<3, 3>(0, 9).noalias() = half_dt * _B.block<3, 3>(6, 9);
    // // _B.block<3, 3>(0, 12).noalias() = half_dt * _B.block<3, 3>(6, 12);
    // // _B.block<3, 3>(0, 15).noalias() = half_dt * _B.block<3, 3>(6, 15);
    // _B00.block<3, 3>(0, 0).noalias() = half_dt * _B00.block<3, 3>(6, 0);
    // _B00.block<3, 3>(0, 3).noalias() = half_dt * _B00.block<3, 3>(6, 3);
    // _B00.block<3, 3>(0, 6).noalias() = half_dt * _B00.block<3, 3>(6, 6);
    // _B00.block<3, 3>(0, 9).noalias() = half_dt * _B00.block<3, 3>(6, 9);
    // _B01.block<3, 3>(0, 0).noalias() = half_dt * _B01.block<3, 3>(6, 0);
    // _B01.block<3, 3>(0, 3).noalias() = half_dt * _B01.block<3, 3>(6, 3);

    // // _B.block<3, 3>(3, 3).noalias() = -0.5 * jr_dt;
    // // _B.block<3, 3>(3, 9).noalias() = _B.block<3, 3>(3, 3);
    // // _B.block<3, 3>(3, 15).noalias() = _B.block<3, 3>(3, 9) * dt;
    // _B00.block<3, 3>(3, 3).noalias() = -0.5 * jr_dt;
    // _B00.block<3, 3>(3, 9).noalias() = _B00.block<3, 3>(3, 3);
    // _B01.block<3, 3>(3, 3).noalias() = _B00.block<3, 3>(3, 9) * dt;

    // // _B.block<3, 3>(9, 12).noalias() = Mat33::Identity() * dt;

    // // _B.block<3, 3>(12, 15).noalias() = Mat33::Identity() * dt;

    // // _covariance = _A * _covariance * _A.transpose() + _B * _N.asDiagonal() * _B.transpose();
    // Eigen::Matrix<double, 9, 9> tmp = _A00 * _covariance.block<9, 6>(0, 9) * _A01.transpose();
    // _covariance.block<9, 9>(0, 0) = _A00 * _covariance.block<9, 9>(0, 0) * _A00.transpose()
    //                                 + _A01 * _covariance.block<6, 6>(9, 9) * _A01.transpose()
    //                                 + tmp + tmp.transpose()
    //                                 + _B00 * _noise_measurement.asDiagonal() * _B00.transpose()
    //                                 + _B01 * _noise_random_walk.asDiagonal() * _B01.transpose();
    // _covariance.block<9, 6>(0, 9) = _A00 * _covariance.block<9, 6>(0, 9)
    //                                 + _A01 * _covariance.block<6, 6>(9, 9)
    //                                 + _B01 * (_noise_random_walk * _dt).asDiagonal();
    // _covariance.block<6, 9>(9, 0).noalias() = _covariance.block<9, 6>(0, 9).transpose();  
    // for (unsigned int i = 0; i < 6; ++i) {
    //     _covariance(i + 9, i + 9) += _noise_random_walk(i) * dt * dt;
    // }

    calculate_APAT(delta_r, delta_r_last, delta_r_a_hat, delta_r_a_hat_last, dr, jr_dt);                                                          

    for (unsigned int i = 0; i < 9; ++i) {
        _covariance(i, i) += 1e-6 * dt * dt;
    }

    // Eigen::Matrix<double, 9, 6> AP01 = _A * _covariance.block<9, 6>(0, 9);
    // Eigen::Matrix<double, 9, 6> BP11 = _B * _covariance.block<6, 6>(9, 9);
    // Eigen::Matrix<double, 9, 9> AP01BT = AP01 * _B.transpose();
    // _covariance.block<9, 6>(0, 9).noalias() = AP01 + BP11;
    // _covariance.block<6, 9>(9, 0).noalias() = _covariance.block<9, 6>(0, 9).transpose();
    // // BP11 += _B * (0.5 * _noise_measurement).asDiagonal();
    // _covariance.block<9, 9>(0, 0) = _A * _covariance.block<9, 9>(0, 0) * _A.transpose()
    //                                 + BP11 * _B.transpose() + (AP01BT + AP01BT.transpose())
    //                                 + _B  * (0.5 * _noise_measurement).asDiagonal() * _B.transpose();
    // for (unsigned int i = 0; i < 6; ++i) {
    //     unsigned int j = i + 9;
    //     _covariance(j, j) += (_noise_random_walk(i) * dt * dt);
    // }

    // 记录上一时刻数据
    _acc_last = acc;
    _gyro_last = gyro;
}

void IMUIntegration::repropagate(const Eigen::Vector3d &ba, const Eigen::Vector3d &bg) {
    _ba = ba;
    _bg = bg;
    reset();
    // _sum_dt = 0.0;
    // _acc_last = _acc_init;
    // _gyro_last = _gyro_init;
    // _delta_p.setZero();
    // _delta_q.setIdentity();
    // _delta_v.setZero();
    // _ba = ba;
    // _bg = bg;
    // _jacobian.setIdentity();
    // _covariance.setZero();
    for (size_t i = 0; i < _dt_buf.size(); ++i) {
        propagate(_dt_buf[i], _acc_buf[i], _gyro_buf[i]);
    }
}

void IMUIntegration::correct(const Vec3 &delta_ba, const Vec3 &delta_bg) {
    _delta_r *= Sophus::SO3d::exp(get_dr_dbg() * delta_bg);
    
    _delta_q *= Utility::deltaQ(get_dr_dbg() * delta_bg);
    _delta_q.normalize();
    _delta_p += get_dp_dba() * delta_ba + get_dp_dbg() * delta_bg;
    _delta_v += get_dv_dba() * delta_ba + get_dv_dbg() * delta_bg;
    _ba += delta_ba;
    _bg += delta_bg;

    // _delta_p_corr = _delta_p + _dp_dba * delta_ba + _dp_dbg * delta_bg;
    // _delta_r_corr = _delta_r * Sophus::SO3d::exp(_dr_dbg * delta_bg);
    // _delta_v_corr = _delta_v + _dv_dba * delta_ba + _dv_dbg * delta_bg;
    // _ba_corr = _ba + delta_ba;
    // _bg_corr = _bg + delta_bg;
}

void IMUIntegration::reset() {
    _acc_last = _acc_init;
    _gyro_last = _gyro_init;

    _sum_dt = 0;
    _delta_r = Sophus::SO3d();  // dR
    _delta_v.setZero();    // dv
    _delta_p.setZero();    // dp

    // jacobian w.r.t bg and ba
    _dr_dbg.setZero();
    _dv_dbg.setZero();
    _dv_dba.setZero();
    _dp_dbg.setZero();
    _dp_dba.setZero();

    // noise propagation
    _covariance.setZero();
    _A.setIdentity();
    _B.setZero();

    _jacobian.setIdentity();
    _delta_q.setIdentity();
}

void IMUIntegration::calculate_APAT(const Mat33 &delta_r, const Mat33 &delta_r_last, 
                                    const Mat33 &delta_r_a_hat, const Mat33 &delta_r_a_hat_last,
                                    const Mat33 &dr, const Mat33 &jr_dt) {
    double delta_r00 = delta_r(0, 0), delta_r01 = delta_r(0, 1), delta_r02 = delta_r(0, 2);   
    double delta_r10 = delta_r(1, 0), delta_r11 = delta_r(1, 1), delta_r12 = delta_r(1, 2);       
    double delta_r20 = delta_r(2, 0), delta_r21 = delta_r(2, 1), delta_r22 = delta_r(2, 2);          
    double delta_r_last00 = delta_r_last(0, 0), delta_r_last01 = delta_r_last(0, 1), delta_r_last02 = delta_r_last(0, 2);   
    double delta_r_last10 = delta_r_last(1, 0), delta_r_last11 = delta_r_last(1, 1), delta_r_last12 = delta_r_last(1, 2);       
    double delta_r_last20 = delta_r_last(2, 0), delta_r_last21 = delta_r_last(2, 1), delta_r_last22 = delta_r_last(2, 2); 
    double delta_r_a_hat00 = delta_r_a_hat(0, 0), delta_r_a_hat01 = delta_r_a_hat(0, 1), delta_r_a_hat02 = delta_r_a_hat(0, 2);   
    double delta_r_a_hat10 = delta_r_a_hat(1, 0), delta_r_a_hat11 = delta_r_a_hat(1, 1), delta_r_a_hat12 = delta_r_a_hat(1, 2);       
    double delta_r_a_hat20 = delta_r_a_hat(2, 0), delta_r_a_hat21 = delta_r_a_hat(2, 1), delta_r_a_hat22 = delta_r_a_hat(2, 2);          
    double delta_r_a_hat_last00 = delta_r_a_hat_last(0, 0), delta_r_a_hat_last01 = delta_r_a_hat_last(0, 1), delta_r_a_hat_last02 = delta_r_a_hat_last(0, 2);   
    double delta_r_a_hat_last10 = delta_r_a_hat_last(1, 0), delta_r_a_hat_last11 = delta_r_a_hat_last(1, 1), delta_r_a_hat_last12 = delta_r_a_hat_last(1, 2);       
    double delta_r_a_hat_last20 = delta_r_a_hat_last(2, 0), delta_r_a_hat_last21 = delta_r_a_hat_last(2, 1), delta_r_a_hat_last22 = delta_r_a_hat_last(2, 2);                 
    double dr00 = dr(0, 0), dr01 = dr(0, 1), dr02 = dr(0, 2);   
    double dr10 = dr(1, 0), dr11 = dr(1, 1), dr12 = dr(1, 2);       
    double dr20 = dr(2, 0), dr21 = dr(2, 1), dr22 = dr(2, 2);    
    double jr_dt00 = jr_dt(0, 0), jr_dt01 = jr_dt(0, 1), jr_dt02 = jr_dt(0, 2);   
    double jr_dt10 = jr_dt(1, 0), jr_dt11 = jr_dt(1, 1), jr_dt12 = jr_dt(1, 2);       
    double jr_dt20 = jr_dt(2, 0), jr_dt21 = jr_dt(2, 1), jr_dt22 = jr_dt(2, 2);    
    double p33 = _covariance(3, 3);
    double p44 = _covariance(4, 4);
    double p55 = _covariance(5, 5);
    double na_m = _noise_measurement(0);
    double ng_m = _noise_measurement(3);
    double na_w = _noise_random_walk(0);
    double ng_w = _noise_random_walk(3);

    // Equations for covariance matrix prediction, without process noise!
    static double var[836];
    // Equations  or covariance matrix prediction, without process noise!
    var[0] = _covariance(6,6)*_dt;
    var[1] = delta_r01 + delta_r_last01;
    var[2] = pow(_dt, 2);
    var[3] = 0.25 *var[2];
    var[4] = var[1]*var[3];
    var[5] = _covariance(10,6)*var[4];
    var[6] = delta_r02 + delta_r_last02;
    var[7] = var[3]*var[6];
    var[8] = _covariance(11,6)*var[7];
    var[9] = delta_r00 + delta_r_last00;
    var[10] = var[3]*var[9];
    var[11] = _covariance(9,6)*var[10];
    var[12] = delta_r_a_hat00*jr_dt00 + delta_r_a_hat01*jr_dt10 + delta_r_a_hat02*jr_dt20;
    var[13] = var[12]*var[3];
    var[14] = _covariance(12,6)*var[13];
    var[15] = delta_r_a_hat00*jr_dt01 + delta_r_a_hat01*jr_dt11 + delta_r_a_hat02*jr_dt21;
    var[16] = var[15]*var[3];
    var[17] = _covariance(13,6)*var[16];
    var[18] = delta_r_a_hat00*jr_dt02 + delta_r_a_hat01*jr_dt12 + delta_r_a_hat02*jr_dt22;
    var[19] = var[18]*var[3];
    var[20] = _covariance(14,6)*var[19];
    var[21] = delta_r_a_hat00*dr00 + delta_r_a_hat01*dr01 + delta_r_a_hat02*dr02 + delta_r_a_hat_last00;
    var[22] = var[21]*var[3];
    var[23] = _covariance(6,3)*var[22];
    var[24] = delta_r_a_hat00*dr10 + delta_r_a_hat01*dr11 + delta_r_a_hat02*dr12 + delta_r_a_hat_last01;
    var[25] = var[24]*var[3];
    var[26] = _covariance(6,4)*var[25];
    var[27] = delta_r_a_hat00*dr20 + delta_r_a_hat01*dr21 + delta_r_a_hat02*dr22 + delta_r_a_hat_last02;
    var[28] = var[27]*var[3];
    var[29] = _covariance(6,5)*var[28];
    var[30] = pow(var[12], 2);
    var[31] = ng_w*var[30];
    var[32] = pow(_dt, 6);
    var[33] = 0.015625 *var[32];
    var[34] = pow(var[15], 2);
    var[35] = ng_w*var[33];
    var[36] = pow(var[18], 2);
    var[37] = ng_m*var[30];
    var[38] = pow(_dt, 4);
    var[39] = 0.03125 *var[38];
    var[40] = ng_m*var[39];
    var[41] = _covariance(9,6)*_dt;
    var[42] = _covariance(10,9)*var[4];
    var[43] = _covariance(11,9)*var[7];
    var[44] = _covariance(9,9)*var[10];
    var[45] = _covariance(12,9)*var[13];
    var[46] = _covariance(13,9)*var[16];
    var[47] = _covariance(14,9)*var[19];
    var[48] = _covariance(9,3)*var[22];
    var[49] = _covariance(9,4)*var[25];
    var[50] = _covariance(9,5)*var[28];
    var[51] = -_covariance(9,0) - var[41] + var[42] + var[43] + var[44] - var[45] - var[46] - var[47] + var[48] + var[49] + var[50];
    var[52] = _covariance(10,6)*_dt;
    var[53] = _covariance(10,10)*var[4];
    var[54] = _covariance(10,9)*var[10];
    var[55] = _covariance(11,10)*var[7];
    var[56] = _covariance(12,10)*var[13];
    var[57] = _covariance(13,10)*var[16];
    var[58] = _covariance(14,10)*var[19];
    var[59] = _covariance(10,3)*var[22];
    var[60] = _covariance(10,4)*var[25];
    var[61] = _covariance(10,5)*var[28];
    var[62] = -_covariance(10,0) - var[52] + var[53] + var[54] + var[55] - var[56] - var[57] - var[58] + var[59] + var[60] + var[61];
    var[63] = _covariance(11,6)*_dt;
    var[64] = _covariance(11,10)*var[4];
    var[65] = _covariance(11,11)*var[7];
    var[66] = _covariance(11,9)*var[10];
    var[67] = _covariance(12,11)*var[13];
    var[68] = _covariance(13,11)*var[16];
    var[69] = _covariance(14,11)*var[19];
    var[70] = _covariance(11,3)*var[22];
    var[71] = _covariance(11,4)*var[25];
    var[72] = _covariance(11,5)*var[28];
    var[73] = -_covariance(11,0) - var[63] + var[64] + var[65] + var[66] - var[67] - var[68] - var[69] + var[70] + var[71] + var[72];
    var[74] = _covariance(6,3)*_dt;
    var[75] = _covariance(10,3)*var[4] + _covariance(11,3)*var[7] - _covariance(12,3)*var[13] - _covariance(13,3)*var[16] - _covariance(14,3)*var[19] - _covariance(3,0) + _covariance(3,3)*var[22] + _covariance(4,3)*var[25] + _covariance(5,3)*var[28] + _covariance(9,3)*var[10] - var[74];
    var[76] = _covariance(6,4)*_dt;
    var[77] = _covariance(10,4)*var[4] + _covariance(11,4)*var[7] - _covariance(12,4)*var[13] - _covariance(13,4)*var[16] - _covariance(14,4)*var[19] - _covariance(4,0) + _covariance(4,3)*var[22] + _covariance(4,4)*var[25] + _covariance(5,4)*var[28] + _covariance(9,4)*var[10] - var[76];
    var[78] = _covariance(6,5)*_dt;
    var[79] = _covariance(10,5)*var[4] + _covariance(11,5)*var[7] - _covariance(12,5)*var[13] - _covariance(13,5)*var[16] - _covariance(14,5)*var[19] - _covariance(5,0) + _covariance(5,3)*var[22] + _covariance(5,4)*var[25] + _covariance(5,5)*var[28] + _covariance(9,5)*var[10] - var[78];
    var[80] = _covariance(12,6)*_dt;
    var[81] = _covariance(12,10)*var[4];
    var[82] = _covariance(12,11)*var[7];
    var[83] = _covariance(12,9)*var[10];
    var[84] = _covariance(12,12)*var[13];
    var[85] = _covariance(13,12)*var[16];
    var[86] = _covariance(14,12)*var[19];
    var[87] = _covariance(12,3)*var[22];
    var[88] = _covariance(12,4)*var[25];
    var[89] = _covariance(12,5)*var[28];
    var[90] = -_covariance(12,0) - var[80] + var[81] + var[82] + var[83] - var[84] - var[85] - var[86] + var[87] + var[88] + var[89];
    var[91] = _covariance(13,6)*_dt;
    var[92] = _covariance(13,10)*var[4];
    var[93] = _covariance(13,11)*var[7];
    var[94] = _covariance(13,9)*var[10];
    var[95] = _covariance(13,12)*var[13];
    var[96] = _covariance(13,13)*var[16];
    var[97] = _covariance(14,13)*var[19];
    var[98] = _covariance(13,3)*var[22];
    var[99] = _covariance(13,4)*var[25];
    var[100] = _covariance(13,5)*var[28];
    var[101] = -_covariance(13,0) + var[100] - var[91] + var[92] + var[93] + var[94] - var[95] - var[96] - var[97] + var[98] + var[99];
    var[102] = _covariance(14,6)*_dt;
    var[103] = _covariance(14,10)*var[4];
    var[104] = _covariance(14,11)*var[7];
    var[105] = _covariance(14,9)*var[10];
    var[106] = _covariance(14,12)*var[13];
    var[107] = _covariance(14,13)*var[16];
    var[108] = _covariance(14,14)*var[19];
    var[109] = _covariance(14,3)*var[22];
    var[110] = _covariance(14,4)*var[25];
    var[111] = _covariance(14,5)*var[28];
    var[112] = -_covariance(14,0) - var[102] + var[103] + var[104] + var[105] - var[106] - var[107] - var[108] + var[109] + var[110] + var[111];
    var[113] = 0.125 *var[38];
    var[114] = na_m*var[113] + 0.0625 *na_w*var[32];
    var[115] = delta_r_a_hat10*jr_dt00 + delta_r_a_hat11*jr_dt10 + delta_r_a_hat12*jr_dt20;
    var[116] = var[115]*var[12];
    var[117] = delta_r_a_hat10*jr_dt01 + delta_r_a_hat11*jr_dt11 + delta_r_a_hat12*jr_dt21;
    var[118] = var[117]*var[15];
    var[119] = delta_r_a_hat10*jr_dt02 + delta_r_a_hat11*jr_dt12 + delta_r_a_hat12*jr_dt22;
    var[120] = var[119]*var[18];
    var[121] = _covariance(7,6)*_dt;
    var[122] = _covariance(10,7)*var[4];
    var[123] = _covariance(11,7)*var[7];
    var[124] = _covariance(9,7)*var[10];
    var[125] = _covariance(12,7)*var[13];
    var[126] = _covariance(13,7)*var[16];
    var[127] = _covariance(14,7)*var[19];
    var[128] = _covariance(7,3)*var[22];
    var[129] = _covariance(7,4)*var[25];
    var[130] = _covariance(7,5)*var[28];
    var[131] = delta_r10 + delta_r_last10;
    var[132] = var[3]*var[51];
    var[133] = delta_r11 + delta_r_last11;
    var[134] = var[3]*var[62];
    var[135] = delta_r12 + delta_r_last12;
    var[136] = var[3]*var[73];
    var[137] = var[3]*var[90];
    var[138] = var[101]*var[3];
    var[139] = var[112]*var[3];
    var[140] = delta_r_a_hat10*dr00 + delta_r_a_hat11*dr01 + delta_r_a_hat12*dr02 + delta_r_a_hat_last10;
    var[141] = var[3]*var[75];
    var[142] = delta_r_a_hat10*dr10 + delta_r_a_hat11*dr11 + delta_r_a_hat12*dr12 + delta_r_a_hat_last11;
    var[143] = var[3]*var[77];
    var[144] = delta_r_a_hat10*dr20 + delta_r_a_hat11*dr21 + delta_r_a_hat12*dr22 + delta_r_a_hat_last12;
    var[145] = var[3]*var[79];
    var[146] = delta_r_a_hat20*jr_dt00 + delta_r_a_hat21*jr_dt10 + delta_r_a_hat22*jr_dt20;
    var[147] = var[12]*var[146];
    var[148] = delta_r_a_hat20*jr_dt01 + delta_r_a_hat21*jr_dt11 + delta_r_a_hat22*jr_dt21;
    var[149] = var[148]*var[15];
    var[150] = delta_r_a_hat20*jr_dt02 + delta_r_a_hat21*jr_dt12 + delta_r_a_hat22*jr_dt22;
    var[151] = var[150]*var[18];
    var[152] = _covariance(8,6)*_dt;
    var[153] = _covariance(10,8)*var[4];
    var[154] = _covariance(11,8)*var[7];
    var[155] = _covariance(9,8)*var[10];
    var[156] = _covariance(12,8)*var[13];
    var[157] = _covariance(13,8)*var[16];
    var[158] = _covariance(14,8)*var[19];
    var[159] = _covariance(8,3)*var[22];
    var[160] = _covariance(8,4)*var[25];
    var[161] = _covariance(8,5)*var[28];
    var[162] = delta_r20 + delta_r_last20;
    var[163] = delta_r21 + delta_r_last21;
    var[164] = delta_r22 + delta_r_last22;
    var[165] = delta_r_a_hat20*dr00 + delta_r_a_hat21*dr01 + delta_r_a_hat22*dr02 + delta_r_a_hat_last20;
    var[166] = delta_r_a_hat20*dr10 + delta_r_a_hat21*dr11 + delta_r_a_hat22*dr12 + delta_r_a_hat_last21;
    var[167] = delta_r_a_hat20*dr20 + delta_r_a_hat21*dr21 + delta_r_a_hat22*dr22 + delta_r_a_hat_last22;
    var[168] = 0.125 *var[2];
    var[169] = ng_m*var[168];
    var[170] = jr_dt00*var[12];
    var[171] = jr_dt01*var[15];
    var[172] = jr_dt02*var[18];
    var[173] = 0.0625 *var[38];
    var[174] = ng_w*var[173];
    var[175] = jr_dt10*var[12];
    var[176] = jr_dt11*var[15];
    var[177] = jr_dt12*var[18];
    var[178] = jr_dt20*var[12];
    var[179] = jr_dt21*var[15];
    var[180] = jr_dt22*var[18];
    var[181] = pow(_dt, 5);
    var[182] = 0.03125 *var[181];
    var[183] = ng_w*var[182];
    var[184] = pow(_dt, 3);
    var[185] = 0.0625 *var[184];
    var[186] = ng_m*var[185];
    var[187] = 0.5 *_dt;
    var[188] = var[187]*var[51];
    var[189] = var[187]*var[62];
    var[190] = var[187]*var[73];
    var[191] = var[187]*var[75];
    var[192] = var[187]*var[77];
    var[193] = var[187]*var[79];
    var[194] = var[187]*var[90];
    var[195] = var[101]*var[187];
    var[196] = var[112]*var[187];
    var[197] = 0.25 *var[184];
    var[198] = na_m*var[197] + 0.125 *na_w*var[181];
    var[199] = var[116]*var[183] + var[116]*var[186] + var[118]*var[183] + var[118]*var[186] + var[120]*var[183] + var[120]*var[186] + var[121];
    var[200] = var[147]*var[183] + var[147]*var[186] + var[149]*var[183] + var[149]*var[186] + var[151]*var[183] + var[151]*var[186] + var[152];
    var[201] = 0.25 *na_w*var[38];
    var[202] = -var[201];
    var[203] = ng_w*var[113];
    var[204] = _covariance(7,7)*_dt;
    var[205] = var[133]*var[3];
    var[206] = _covariance(10,7)*var[205];
    var[207] = var[135]*var[3];
    var[208] = _covariance(11,7)*var[207];
    var[209] = var[131]*var[3];
    var[210] = _covariance(9,7)*var[209];
    var[211] = var[115]*var[3];
    var[212] = _covariance(12,7)*var[211];
    var[213] = var[117]*var[3];
    var[214] = _covariance(13,7)*var[213];
    var[215] = var[119]*var[3];
    var[216] = _covariance(14,7)*var[215];
    var[217] = var[140]*var[3];
    var[218] = _covariance(7,3)*var[217];
    var[219] = var[142]*var[3];
    var[220] = _covariance(7,4)*var[219];
    var[221] = var[144]*var[3];
    var[222] = _covariance(7,5)*var[221];
    var[223] = pow(var[115], 2);
    var[224] = pow(var[117], 2);
    var[225] = pow(var[119], 2);
    var[226] = _covariance(9,7)*_dt;
    var[227] = _covariance(10,9)*var[205];
    var[228] = _covariance(11,9)*var[207];
    var[229] = _covariance(9,9)*var[209];
    var[230] = _covariance(12,9)*var[211];
    var[231] = _covariance(13,9)*var[213];
    var[232] = _covariance(14,9)*var[215];
    var[233] = _covariance(9,3)*var[217];
    var[234] = _covariance(9,4)*var[219];
    var[235] = _covariance(9,5)*var[221];
    var[236] = -_covariance(9,1) - var[226] + var[227] + var[228] + var[229] - var[230] - var[231] - var[232] + var[233] + var[234] + var[235];
    var[237] = _covariance(10,7)*_dt;
    var[238] = _covariance(10,10)*var[205];
    var[239] = _covariance(10,9)*var[209];
    var[240] = _covariance(11,10)*var[207];
    var[241] = _covariance(12,10)*var[211];
    var[242] = _covariance(13,10)*var[213];
    var[243] = _covariance(14,10)*var[215];
    var[244] = _covariance(10,3)*var[217];
    var[245] = _covariance(10,4)*var[219];
    var[246] = _covariance(10,5)*var[221];
    var[247] = -_covariance(10,1) - var[237] + var[238] + var[239] + var[240] - var[241] - var[242] - var[243] + var[244] + var[245] + var[246];
    var[248] = _covariance(11,7)*_dt;
    var[249] = _covariance(11,10)*var[205];
    var[250] = _covariance(11,11)*var[207];
    var[251] = _covariance(11,9)*var[209];
    var[252] = _covariance(12,11)*var[211];
    var[253] = _covariance(13,11)*var[213];
    var[254] = _covariance(14,11)*var[215];
    var[255] = _covariance(11,3)*var[217];
    var[256] = _covariance(11,4)*var[219];
    var[257] = _covariance(11,5)*var[221];
    var[258] = -_covariance(11,1) - var[248] + var[249] + var[250] + var[251] - var[252] - var[253] - var[254] + var[255] + var[256] + var[257];
    var[259] = _covariance(7,3)*_dt;
    var[260] = _covariance(10,3)*var[205] + _covariance(11,3)*var[207] - _covariance(12,3)*var[211] - _covariance(13,3)*var[213] - _covariance(14,3)*var[215] - _covariance(3,1) + _covariance(3,3)*var[217] + _covariance(4,3)*var[219] + _covariance(5,3)*var[221] + _covariance(9,3)*var[209] - var[259];
    var[261] = _covariance(7,4)*_dt;
    var[262] = _covariance(10,4)*var[205] + _covariance(11,4)*var[207] - _covariance(12,4)*var[211] - _covariance(13,4)*var[213] - _covariance(14,4)*var[215] - _covariance(4,1) + _covariance(4,3)*var[217] + _covariance(4,4)*var[219] + _covariance(5,4)*var[221] + _covariance(9,4)*var[209] - var[261];
    var[263] = _covariance(7,5)*_dt;
    var[264] = _covariance(10,5)*var[205] + _covariance(11,5)*var[207] - _covariance(12,5)*var[211] - _covariance(13,5)*var[213] - _covariance(14,5)*var[215] - _covariance(5,1) + _covariance(5,3)*var[217] + _covariance(5,4)*var[219] + _covariance(5,5)*var[221] + _covariance(9,5)*var[209] - var[263];
    var[265] = _covariance(12,7)*_dt;
    var[266] = _covariance(12,10)*var[205];
    var[267] = _covariance(12,11)*var[207];
    var[268] = _covariance(12,9)*var[209];
    var[269] = _covariance(12,12)*var[211];
    var[270] = _covariance(13,12)*var[213];
    var[271] = _covariance(14,12)*var[215];
    var[272] = _covariance(12,3)*var[217];
    var[273] = _covariance(12,4)*var[219];
    var[274] = _covariance(12,5)*var[221];
    var[275] = -_covariance(12,1) - var[265] + var[266] + var[267] + var[268] - var[269] - var[270] - var[271] + var[272] + var[273] + var[274];
    var[276] = _covariance(13,7)*_dt;
    var[277] = _covariance(13,10)*var[205];
    var[278] = _covariance(13,11)*var[207];
    var[279] = _covariance(13,9)*var[209];
    var[280] = _covariance(13,12)*var[211];
    var[281] = _covariance(13,13)*var[213];
    var[282] = _covariance(14,13)*var[215];
    var[283] = _covariance(13,3)*var[217];
    var[284] = _covariance(13,4)*var[219];
    var[285] = _covariance(13,5)*var[221];
    var[286] = -_covariance(13,1) - var[276] + var[277] + var[278] + var[279] - var[280] - var[281] - var[282] + var[283] + var[284] + var[285];
    var[287] = _covariance(14,7)*_dt;
    var[288] = _covariance(14,10)*var[205];
    var[289] = _covariance(14,11)*var[207];
    var[290] = _covariance(14,9)*var[209];
    var[291] = _covariance(14,12)*var[211];
    var[292] = _covariance(14,13)*var[213];
    var[293] = _covariance(14,14)*var[215];
    var[294] = _covariance(14,3)*var[217];
    var[295] = _covariance(14,4)*var[219];
    var[296] = _covariance(14,5)*var[221];
    var[297] = -_covariance(14,1) - var[287] + var[288] + var[289] + var[290] - var[291] - var[292] - var[293] + var[294] + var[295] + var[296];
    var[298] = var[115]*var[146];
    var[299] = var[117]*var[148];
    var[300] = var[119]*var[150];
    var[301] = _covariance(8,7)*_dt;
    var[302] = _covariance(10,8)*var[205];
    var[303] = _covariance(11,8)*var[207];
    var[304] = _covariance(9,8)*var[209];
    var[305] = _covariance(12,8)*var[211];
    var[306] = _covariance(13,8)*var[213];
    var[307] = _covariance(14,8)*var[215];
    var[308] = _covariance(8,3)*var[217];
    var[309] = _covariance(8,4)*var[219];
    var[310] = _covariance(8,5)*var[221];
    var[311] = var[162]*var[236];
    var[312] = var[163]*var[247];
    var[313] = var[164]*var[258];
    var[314] = var[146]*var[275];
    var[315] = var[148]*var[286];
    var[316] = var[150]*var[297];
    var[317] = var[165]*var[260];
    var[318] = var[166]*var[262];
    var[319] = var[167]*var[264];
    var[320] = jr_dt00*var[115];
    var[321] = jr_dt01*var[117];
    var[322] = jr_dt02*var[119];
    var[323] = jr_dt10*var[115];
    var[324] = jr_dt11*var[117];
    var[325] = jr_dt12*var[119];
    var[326] = jr_dt20*var[115];
    var[327] = jr_dt21*var[117];
    var[328] = jr_dt22*var[119];
    var[329] = var[187]*var[236];
    var[330] = var[187]*var[247];
    var[331] = var[187]*var[258];
    var[332] = var[187]*var[260];
    var[333] = var[187]*var[262];
    var[334] = var[187]*var[264];
    var[335] = var[187]*var[275];
    var[336] = var[187]*var[286];
    var[337] = var[187]*var[297];
    var[338] = var[183]*var[298] + var[183]*var[299] + var[183]*var[300] + var[186]*var[298] + var[186]*var[299] + var[186]*var[300] + var[301];
    var[339] = _covariance(8,8)*_dt;
    var[340] = var[163]*var[3];
    var[341] = _covariance(10,8)*var[340];
    var[342] = var[164]*var[3];
    var[343] = _covariance(11,8)*var[342];
    var[344] = var[162]*var[3];
    var[345] = _covariance(9,8)*var[344];
    var[346] = var[146]*var[3];
    var[347] = _covariance(12,8)*var[346];
    var[348] = var[148]*var[3];
    var[349] = _covariance(13,8)*var[348];
    var[350] = var[150]*var[3];
    var[351] = _covariance(14,8)*var[350];
    var[352] = var[165]*var[3];
    var[353] = _covariance(8,3)*var[352];
    var[354] = var[166]*var[3];
    var[355] = _covariance(8,4)*var[354];
    var[356] = var[167]*var[3];
    var[357] = _covariance(8,5)*var[356];
    var[358] = pow(var[146], 2);
    var[359] = pow(var[148], 2);
    var[360] = pow(var[150], 2);
    var[361] = _covariance(9,8)*_dt;
    var[362] = _covariance(10,9)*var[340];
    var[363] = _covariance(11,9)*var[342];
    var[364] = _covariance(9,9)*var[344];
    var[365] = _covariance(12,9)*var[346];
    var[366] = _covariance(13,9)*var[348];
    var[367] = _covariance(14,9)*var[350];
    var[368] = _covariance(9,3)*var[352];
    var[369] = _covariance(9,4)*var[354];
    var[370] = _covariance(9,5)*var[356];
    var[371] = -_covariance(9,2) - var[361] + var[362] + var[363] + var[364] - var[365] - var[366] - var[367] + var[368] + var[369] + var[370];
    var[372] = _covariance(10,8)*_dt;
    var[373] = _covariance(10,10)*var[340];
    var[374] = _covariance(10,9)*var[344];
    var[375] = _covariance(11,10)*var[342];
    var[376] = _covariance(12,10)*var[346];
    var[377] = _covariance(13,10)*var[348];
    var[378] = _covariance(14,10)*var[350];
    var[379] = _covariance(10,3)*var[352];
    var[380] = _covariance(10,4)*var[354];
    var[381] = _covariance(10,5)*var[356];
    var[382] = -_covariance(10,2) - var[372] + var[373] + var[374] + var[375] - var[376] - var[377] - var[378] + var[379] + var[380] + var[381];
    var[383] = _covariance(11,8)*_dt;
    var[384] = _covariance(11,10)*var[340];
    var[385] = _covariance(11,11)*var[342];
    var[386] = _covariance(11,9)*var[344];
    var[387] = _covariance(12,11)*var[346];
    var[388] = _covariance(13,11)*var[348];
    var[389] = _covariance(14,11)*var[350];
    var[390] = _covariance(11,3)*var[352];
    var[391] = _covariance(11,4)*var[354];
    var[392] = _covariance(11,5)*var[356];
    var[393] = -_covariance(11,2) - var[383] + var[384] + var[385] + var[386] - var[387] - var[388] - var[389] + var[390] + var[391] + var[392];
    var[394] = _covariance(8,3)*_dt;
    var[395] = _covariance(10,3)*var[340] + _covariance(11,3)*var[342] - _covariance(12,3)*var[346] - _covariance(13,3)*var[348] - _covariance(14,3)*var[350] - _covariance(3,2) + _covariance(3,3)*var[352] + _covariance(4,3)*var[354] + _covariance(5,3)*var[356] + _covariance(9,3)*var[344] - var[394];
    var[396] = _covariance(8,4)*_dt;
    var[397] = _covariance(10,4)*var[340] + _covariance(11,4)*var[342] - _covariance(12,4)*var[346] - _covariance(13,4)*var[348] - _covariance(14,4)*var[350] - _covariance(4,2) + _covariance(4,3)*var[352] + _covariance(4,4)*var[354] + _covariance(5,4)*var[356] + _covariance(9,4)*var[344] - var[396];
    var[398] = _covariance(8,5)*_dt;
    var[399] = _covariance(10,5)*var[340] + _covariance(11,5)*var[342] - _covariance(12,5)*var[346] - _covariance(13,5)*var[348] - _covariance(14,5)*var[350] - _covariance(5,2) + _covariance(5,3)*var[352] + _covariance(5,4)*var[354] + _covariance(5,5)*var[356] + _covariance(9,5)*var[344] - var[398];
    var[400] = _covariance(12,8)*_dt;
    var[401] = _covariance(12,10)*var[340];
    var[402] = _covariance(12,11)*var[342];
    var[403] = _covariance(12,9)*var[344];
    var[404] = _covariance(12,12)*var[346];
    var[405] = _covariance(13,12)*var[348];
    var[406] = _covariance(14,12)*var[350];
    var[407] = _covariance(12,3)*var[352];
    var[408] = _covariance(12,4)*var[354];
    var[409] = _covariance(12,5)*var[356];
    var[410] = -_covariance(12,2) - var[400] + var[401] + var[402] + var[403] - var[404] - var[405] - var[406] + var[407] + var[408] + var[409];
    var[411] = _covariance(13,8)*_dt;
    var[412] = _covariance(13,10)*var[340];
    var[413] = _covariance(13,11)*var[342];
    var[414] = _covariance(13,9)*var[344];
    var[415] = _covariance(13,12)*var[346];
    var[416] = _covariance(13,13)*var[348];
    var[417] = _covariance(14,13)*var[350];
    var[418] = _covariance(13,3)*var[352];
    var[419] = _covariance(13,4)*var[354];
    var[420] = _covariance(13,5)*var[356];
    var[421] = -_covariance(13,2) - var[411] + var[412] + var[413] + var[414] - var[415] - var[416] - var[417] + var[418] + var[419] + var[420];
    var[422] = _covariance(14,8)*_dt;
    var[423] = _covariance(14,10)*var[340];
    var[424] = _covariance(14,11)*var[342];
    var[425] = _covariance(14,9)*var[344];
    var[426] = _covariance(14,12)*var[346];
    var[427] = _covariance(14,13)*var[348];
    var[428] = _covariance(14,14)*var[350];
    var[429] = _covariance(14,3)*var[352];
    var[430] = _covariance(14,4)*var[354];
    var[431] = _covariance(14,5)*var[356];
    var[432] = -_covariance(14,2) - var[422] + var[423] + var[424] + var[425] - var[426] - var[427] - var[428] + var[429] + var[430] + var[431];
    var[433] = jr_dt00*var[146];
    var[434] = jr_dt01*var[148];
    var[435] = jr_dt02*var[150];
    var[436] = jr_dt10*var[146];
    var[437] = jr_dt11*var[148];
    var[438] = jr_dt12*var[150];
    var[439] = jr_dt20*var[146];
    var[440] = jr_dt21*var[148];
    var[441] = jr_dt22*var[150];
    var[442] = var[187]*var[371];
    var[443] = var[187]*var[382];
    var[444] = var[187]*var[393];
    var[445] = var[187]*var[395];
    var[446] = var[187]*var[397];
    var[447] = var[187]*var[399];
    var[448] = var[187]*var[410];
    var[449] = var[187]*var[421];
    var[450] = var[187]*var[432];
    var[451] = pow(jr_dt00, 2);
    var[452] = 0.5 *ng_m;
    var[453] = pow(jr_dt01, 2);
    var[454] = pow(jr_dt02, 2);
    var[455] = ng_w*var[2];
    var[456] = 0.25 *var[455];
    var[457] = _covariance(12,12)*jr_dt00;
    var[458] = _covariance(13,12)*jr_dt01;
    var[459] = _covariance(14,12)*jr_dt02;
    var[460] = _covariance(12,3)*dr00;
    var[461] = _covariance(12,4)*dr10;
    var[462] = _covariance(12,5)*dr20;
    var[463] = var[457] + var[458] + var[459] - var[460] - var[461] - var[462];
    var[464] = _covariance(13,12)*jr_dt00;
    var[465] = _covariance(13,13)*jr_dt01;
    var[466] = _covariance(14,13)*jr_dt02;
    var[467] = _covariance(13,3)*dr00;
    var[468] = _covariance(13,4)*dr10;
    var[469] = _covariance(13,5)*dr20;
    var[470] = var[464] + var[465] + var[466] - var[467] - var[468] - var[469];
    var[471] = _covariance(14,12)*jr_dt00;
    var[472] = _covariance(14,13)*jr_dt01;
    var[473] = _covariance(14,14)*jr_dt02;
    var[474] = _covariance(14,3)*dr00;
    var[475] = _covariance(14,4)*dr10;
    var[476] = _covariance(14,5)*dr20;
    var[477] = var[471] + var[472] + var[473] - var[474] - var[475] - var[476];
    var[478] = _covariance(12,3)*jr_dt00 + _covariance(13,3)*jr_dt01 + _covariance(14,3)*jr_dt02 - _covariance(3,3)*dr00 - _covariance(4,3)*dr10 - _covariance(5,3)*dr20;
    var[479] = _covariance(12,4)*jr_dt00 + _covariance(13,4)*jr_dt01 + _covariance(14,4)*jr_dt02 - _covariance(4,3)*dr00 - _covariance(4,4)*dr10 - _covariance(5,4)*dr20;
    var[480] = _covariance(12,5)*jr_dt00 + _covariance(13,5)*jr_dt01 + _covariance(14,5)*jr_dt02 - _covariance(5,3)*dr00 - _covariance(5,4)*dr10 - _covariance(5,5)*dr20;
    var[481] = jr_dt00*var[452];
    var[482] = jr_dt01*var[452];
    var[483] = jr_dt02*var[452];
    var[484] = jr_dt00*var[456];
    var[485] = jr_dt01*var[456];
    var[486] = jr_dt02*var[456];
    var[487] = 0.25 *_dt*ng_m;
    var[488] = 0.125 *ng_w*var[184];
    var[489] = _covariance(12,9)*jr_dt00;
    var[490] = _covariance(13,9)*jr_dt01;
    var[491] = _covariance(14,9)*jr_dt02;
    var[492] = _covariance(9,3)*dr00;
    var[493] = _covariance(9,4)*dr10;
    var[494] = _covariance(9,5)*dr20;
    var[495] = var[187]*(var[489] + var[490] + var[491] - var[492] - var[493] - var[494]);
    var[496] = _covariance(10,3)*dr00 + _covariance(10,4)*dr10 + _covariance(10,5)*dr20 - _covariance(12,10)*jr_dt00 - _covariance(13,10)*jr_dt01 - _covariance(14,10)*jr_dt02;
    var[497] = var[187]*var[496];
    var[498] = _covariance(11,3)*dr00 + _covariance(11,4)*dr10 + _covariance(11,5)*dr20 - _covariance(12,11)*jr_dt00 - _covariance(13,11)*jr_dt01 - _covariance(14,11)*jr_dt02;
    var[499] = var[187]*var[498];
    var[500] = var[187]*var[463];
    var[501] = var[187]*var[470];
    var[502] = var[187]*var[477];
    var[503] = var[187]*var[478];
    var[504] = var[187]*var[479];
    var[505] = var[187]*var[480];
    var[506] = 0.5 *var[455];
    var[507] = pow(jr_dt10, 2);
    var[508] = pow(jr_dt11, 2);
    var[509] = pow(jr_dt12, 2);
    var[510] = _covariance(12,12)*jr_dt10;
    var[511] = _covariance(13,12)*jr_dt11;
    var[512] = _covariance(14,12)*jr_dt12;
    var[513] = _covariance(12,3)*dr01;
    var[514] = _covariance(12,4)*dr11;
    var[515] = _covariance(12,5)*dr21;
    var[516] = var[510] + var[511] + var[512] - var[513] - var[514] - var[515];
    var[517] = _covariance(13,12)*jr_dt10;
    var[518] = _covariance(13,13)*jr_dt11;
    var[519] = _covariance(14,13)*jr_dt12;
    var[520] = _covariance(13,3)*dr01;
    var[521] = _covariance(13,4)*dr11;
    var[522] = _covariance(13,5)*dr21;
    var[523] = var[517] + var[518] + var[519] - var[520] - var[521] - var[522];
    var[524] = _covariance(14,12)*jr_dt10;
    var[525] = _covariance(14,13)*jr_dt11;
    var[526] = _covariance(14,14)*jr_dt12;
    var[527] = _covariance(14,3)*dr01;
    var[528] = _covariance(14,4)*dr11;
    var[529] = _covariance(14,5)*dr21;
    var[530] = var[524] + var[525] + var[526] - var[527] - var[528] - var[529];
    var[531] = _covariance(12,3)*jr_dt10 + _covariance(13,3)*jr_dt11 + _covariance(14,3)*jr_dt12 - _covariance(3,3)*dr01 - _covariance(4,3)*dr11 - _covariance(5,3)*dr21;
    var[532] = _covariance(12,4)*jr_dt10 + _covariance(13,4)*jr_dt11 + _covariance(14,4)*jr_dt12 - _covariance(4,3)*dr01 - _covariance(4,4)*dr11 - _covariance(5,4)*dr21;
    var[533] = _covariance(12,5)*jr_dt10 + _covariance(13,5)*jr_dt11 + _covariance(14,5)*jr_dt12 - _covariance(5,3)*dr01 - _covariance(5,4)*dr11 - _covariance(5,5)*dr21;
    var[534] = jr_dt10*jr_dt20;
    var[535] = jr_dt11*jr_dt21;
    var[536] = jr_dt12*jr_dt22;
    var[537] = _covariance(12,9)*jr_dt10;
    var[538] = _covariance(13,9)*jr_dt11;
    var[539] = _covariance(14,9)*jr_dt12;
    var[540] = _covariance(9,3)*dr01;
    var[541] = _covariance(9,4)*dr11;
    var[542] = _covariance(9,5)*dr21;
    var[543] = var[187]*(var[537] + var[538] + var[539] - var[540] - var[541] - var[542]);
    var[544] = _covariance(10,3)*dr01 + _covariance(10,4)*dr11 + _covariance(10,5)*dr21 - _covariance(12,10)*jr_dt10 - _covariance(13,10)*jr_dt11 - _covariance(14,10)*jr_dt12;
    var[545] = var[187]*var[544];
    var[546] = _covariance(11,3)*dr01 + _covariance(11,4)*dr11 + _covariance(11,5)*dr21 - _covariance(12,11)*jr_dt10 - _covariance(13,11)*jr_dt11 - _covariance(14,11)*jr_dt12;
    var[547] = var[187]*var[546];
    var[548] = var[187]*var[516];
    var[549] = var[187]*var[523];
    var[550] = var[187]*var[530];
    var[551] = var[187]*var[531];
    var[552] = var[187]*var[532];
    var[553] = var[187]*var[533];
    var[554] = pow(jr_dt20, 2);
    var[555] = pow(jr_dt21, 2);
    var[556] = pow(jr_dt22, 2);
    var[557] = _covariance(12,12)*jr_dt20;
    var[558] = _covariance(13,12)*jr_dt21;
    var[559] = _covariance(14,12)*jr_dt22;
    var[560] = _covariance(12,3)*dr02;
    var[561] = _covariance(12,4)*dr12;
    var[562] = _covariance(12,5)*dr22;
    var[563] = var[557] + var[558] + var[559] - var[560] - var[561] - var[562];
    var[564] = _covariance(13,12)*jr_dt20;
    var[565] = _covariance(13,13)*jr_dt21;
    var[566] = _covariance(14,13)*jr_dt22;
    var[567] = _covariance(13,3)*dr02;
    var[568] = _covariance(13,4)*dr12;
    var[569] = _covariance(13,5)*dr22;
    var[570] = var[564] + var[565] + var[566] - var[567] - var[568] - var[569];
    var[571] = _covariance(14,12)*jr_dt20;
    var[572] = _covariance(14,13)*jr_dt21;
    var[573] = _covariance(14,14)*jr_dt22;
    var[574] = _covariance(14,3)*dr02;
    var[575] = _covariance(14,4)*dr12;
    var[576] = _covariance(14,5)*dr22;
    var[577] = var[571] + var[572] + var[573] - var[574] - var[575] - var[576];
    var[578] = _covariance(12,3)*jr_dt20 + _covariance(13,3)*jr_dt21 + _covariance(14,3)*jr_dt22 - _covariance(3,3)*dr02 - _covariance(4,3)*dr12 - _covariance(5,3)*dr22;
    var[579] = _covariance(12,4)*jr_dt20 + _covariance(13,4)*jr_dt21 + _covariance(14,4)*jr_dt22 - _covariance(4,3)*dr02 - _covariance(4,4)*dr12 - _covariance(5,4)*dr22;
    var[580] = _covariance(12,5)*jr_dt20 + _covariance(13,5)*jr_dt21 + _covariance(14,5)*jr_dt22 - _covariance(5,3)*dr02 - _covariance(5,4)*dr12 - _covariance(5,5)*dr22;
    var[581] = _covariance(12,9)*jr_dt20;
    var[582] = _covariance(13,9)*jr_dt21;
    var[583] = _covariance(14,9)*jr_dt22;
    var[584] = _covariance(9,3)*dr02;
    var[585] = _covariance(9,4)*dr12;
    var[586] = _covariance(9,5)*dr22;
    var[587] = var[187]*(var[581] + var[582] + var[583] - var[584] - var[585] - var[586]);
    var[588] = _covariance(10,3)*dr02 + _covariance(10,4)*dr12 + _covariance(10,5)*dr22 - _covariance(12,10)*jr_dt20 - _covariance(13,10)*jr_dt21 - _covariance(14,10)*jr_dt22;
    var[589] = var[187]*var[588];
    var[590] = _covariance(11,3)*dr02 + _covariance(11,4)*dr12 + _covariance(11,5)*dr22 - _covariance(12,11)*jr_dt20 - _covariance(13,11)*jr_dt21 - _covariance(14,11)*jr_dt22;
    var[591] = var[187]*var[590];
    var[592] = var[187]*var[563];
    var[593] = var[187]*var[570];
    var[594] = var[187]*var[577];
    var[595] = var[187]*var[578];
    var[596] = var[187]*var[579];
    var[597] = var[187]*var[580];
    var[598] = 0.5 *var[12];
    var[599] = 0.5 *var[15];
    var[600] = 0.5 *var[18];
    var[601] = var[187]*var[1];
    var[602] = _covariance(10,9)*var[601];
    var[603] = var[187]*var[6];
    var[604] = _covariance(11,9)*var[603];
    var[605] = var[187]*var[9];
    var[606] = _covariance(9,9)*var[605];
    var[607] = var[12]*var[187];
    var[608] = _covariance(12,9)*var[607];
    var[609] = var[15]*var[187];
    var[610] = _covariance(13,9)*var[609];
    var[611] = var[187]*var[18];
    var[612] = _covariance(14,9)*var[611];
    var[613] = var[187]*var[21];
    var[614] = _covariance(9,3)*var[613];
    var[615] = var[187]*var[24];
    var[616] = _covariance(9,4)*var[615];
    var[617] = var[187]*var[27];
    var[618] = _covariance(9,5)*var[617];
    var[619] = -_covariance(9,6) + var[602] + var[604] + var[606] - var[608] - var[610] - var[612] + var[614] + var[616] + var[618];
    var[620] = _covariance(10,10)*var[601];
    var[621] = _covariance(10,9)*var[605];
    var[622] = _covariance(11,10)*var[603];
    var[623] = _covariance(12,10)*var[607];
    var[624] = _covariance(13,10)*var[609];
    var[625] = _covariance(14,10)*var[611];
    var[626] = _covariance(10,3)*var[613];
    var[627] = _covariance(10,4)*var[615];
    var[628] = _covariance(10,5)*var[617];
    var[629] = -_covariance(10,6) + var[620] + var[621] + var[622] - var[623] - var[624] - var[625] + var[626] + var[627] + var[628];
    var[630] = _covariance(11,10)*var[601];
    var[631] = _covariance(11,11)*var[603];
    var[632] = _covariance(11,9)*var[605];
    var[633] = _covariance(12,11)*var[607];
    var[634] = _covariance(13,11)*var[609];
    var[635] = _covariance(14,11)*var[611];
    var[636] = _covariance(11,3)*var[613];
    var[637] = _covariance(11,4)*var[615];
    var[638] = _covariance(11,5)*var[617];
    var[639] = -_covariance(11,6) + var[630] + var[631] + var[632] - var[633] - var[634] - var[635] + var[636] + var[637] + var[638];
    var[640] = _covariance(10,3)*var[601] + _covariance(11,3)*var[603] - _covariance(12,3)*var[607] - _covariance(13,3)*var[609] - _covariance(14,3)*var[611] + _covariance(3,3)*var[613] + _covariance(4,3)*var[615] + _covariance(5,3)*var[617] - _covariance(6,3) + _covariance(9,3)*var[605];
    var[641] = _covariance(10,4)*var[601] + _covariance(11,4)*var[603] - _covariance(12,4)*var[607] - _covariance(13,4)*var[609] - _covariance(14,4)*var[611] + _covariance(4,3)*var[613] + _covariance(4,4)*var[615] + _covariance(5,4)*var[617] - _covariance(6,4) + _covariance(9,4)*var[605];
    var[642] = _covariance(10,5)*var[601] + _covariance(11,5)*var[603] - _covariance(12,5)*var[607] - _covariance(13,5)*var[609] - _covariance(14,5)*var[611] + _covariance(5,3)*var[613] + _covariance(5,4)*var[615] + _covariance(5,5)*var[617] - _covariance(6,5) + _covariance(9,5)*var[605];
    var[643] = 0.5 *var[1];
    var[644] = 0.5 *var[6];
    var[645] = 0.5 *var[21];
    var[646] = 0.5 *var[24];
    var[647] = 0.5 *var[27];
    var[648] = 0.5 *var[9];
    var[649] = _covariance(12,10)*var[601];
    var[650] = _covariance(12,11)*var[603];
    var[651] = _covariance(12,9)*var[605];
    var[652] = _covariance(12,12)*var[607];
    var[653] = _covariance(13,12)*var[609];
    var[654] = _covariance(14,12)*var[611];
    var[655] = _covariance(12,3)*var[613];
    var[656] = _covariance(12,4)*var[615];
    var[657] = _covariance(12,5)*var[617];
    var[658] = -_covariance(12,6) + var[649] + var[650] + var[651] - var[652] - var[653] - var[654] + var[655] + var[656] + var[657];
    var[659] = _covariance(13,10)*var[601];
    var[660] = _covariance(13,11)*var[603];
    var[661] = _covariance(13,9)*var[605];
    var[662] = _covariance(13,12)*var[607];
    var[663] = _covariance(13,13)*var[609];
    var[664] = _covariance(14,13)*var[611];
    var[665] = _covariance(13,3)*var[613];
    var[666] = _covariance(13,4)*var[615];
    var[667] = _covariance(13,5)*var[617];
    var[668] = -_covariance(13,6) + var[659] + var[660] + var[661] - var[662] - var[663] - var[664] + var[665] + var[666] + var[667];
    var[669] = _covariance(14,10)*var[601];
    var[670] = _covariance(14,11)*var[603];
    var[671] = _covariance(14,9)*var[605];
    var[672] = _covariance(14,12)*var[607];
    var[673] = _covariance(14,13)*var[609];
    var[674] = _covariance(14,14)*var[611];
    var[675] = _covariance(14,3)*var[613];
    var[676] = _covariance(14,4)*var[615];
    var[677] = _covariance(14,5)*var[617];
    var[678] = -_covariance(14,6) + var[669] + var[670] + var[671] - var[672] - var[673] - var[674] + var[675] + var[676] + var[677];
    var[679] = 0.5 *na_m*var[2] + var[201];
    var[680] = var[187]*var[619];
    var[681] = var[187]*var[629];
    var[682] = var[187]*var[639];
    var[683] = var[187]*var[658];
    var[684] = var[187]*var[668];
    var[685] = var[187]*var[678];
    var[686] = var[187]*var[640];
    var[687] = var[187]*var[641];
    var[688] = var[187]*var[642];
    var[689] = -0.5 *na_w*var[184];
    var[690] = ng_w*var[197];
    var[691] = 0.5 *var[115];
    var[692] = 0.5 *var[117];
    var[693] = 0.5 *var[119];
    var[694] = var[133]*var[187];
    var[695] = _covariance(10,9)*var[694];
    var[696] = var[135]*var[187];
    var[697] = _covariance(11,9)*var[696];
    var[698] = var[131]*var[187];
    var[699] = _covariance(9,9)*var[698];
    var[700] = var[115]*var[187];
    var[701] = _covariance(12,9)*var[700];
    var[702] = var[117]*var[187];
    var[703] = _covariance(13,9)*var[702];
    var[704] = var[119]*var[187];
    var[705] = _covariance(14,9)*var[704];
    var[706] = var[140]*var[187];
    var[707] = _covariance(9,3)*var[706];
    var[708] = var[142]*var[187];
    var[709] = _covariance(9,4)*var[708];
    var[710] = var[144]*var[187];
    var[711] = _covariance(9,5)*var[710];
    var[712] = -_covariance(9,7) + var[695] + var[697] + var[699] - var[701] - var[703] - var[705] + var[707] + var[709] + var[711];
    var[713] = _covariance(10,10)*var[694];
    var[714] = _covariance(10,9)*var[698];
    var[715] = _covariance(11,10)*var[696];
    var[716] = _covariance(12,10)*var[700];
    var[717] = _covariance(13,10)*var[702];
    var[718] = _covariance(14,10)*var[704];
    var[719] = _covariance(10,3)*var[706];
    var[720] = _covariance(10,4)*var[708];
    var[721] = _covariance(10,5)*var[710];
    var[722] = -_covariance(10,7) + var[713] + var[714] + var[715] - var[716] - var[717] - var[718] + var[719] + var[720] + var[721];
    var[723] = _covariance(11,10)*var[694];
    var[724] = _covariance(11,11)*var[696];
    var[725] = _covariance(11,9)*var[698];
    var[726] = _covariance(12,11)*var[700];
    var[727] = _covariance(13,11)*var[702];
    var[728] = _covariance(14,11)*var[704];
    var[729] = _covariance(11,3)*var[706];
    var[730] = _covariance(11,4)*var[708];
    var[731] = _covariance(11,5)*var[710];
    var[732] = -_covariance(11,7) + var[723] + var[724] + var[725] - var[726] - var[727] - var[728] + var[729] + var[730] + var[731];
    var[733] = _covariance(10,3)*var[694] + _covariance(11,3)*var[696] - _covariance(12,3)*var[700] - _covariance(13,3)*var[702] - _covariance(14,3)*var[704] + _covariance(3,3)*var[706] + _covariance(4,3)*var[708] + _covariance(5,3)*var[710] - _covariance(7,3) + _covariance(9,3)*var[698];
    var[734] = _covariance(10,4)*var[694] + _covariance(11,4)*var[696] - _covariance(12,4)*var[700] - _covariance(13,4)*var[702] - _covariance(14,4)*var[704] + _covariance(4,3)*var[706] + _covariance(4,4)*var[708] + _covariance(5,4)*var[710] - _covariance(7,4) + _covariance(9,4)*var[698];
    var[735] = _covariance(10,5)*var[694] + _covariance(11,5)*var[696] - _covariance(12,5)*var[700] - _covariance(13,5)*var[702] - _covariance(14,5)*var[704] + _covariance(5,3)*var[706] + _covariance(5,4)*var[708] + _covariance(5,5)*var[710] - _covariance(7,5) + _covariance(9,5)*var[698];
    var[736] = 0.5 *var[133];
    var[737] = 0.5 *var[135];
    var[738] = 0.5 *var[140];
    var[739] = 0.5 *var[142];
    var[740] = 0.5 *var[144];
    var[741] = 0.5 *var[131];
    var[742] = _covariance(12,10)*var[694];
    var[743] = _covariance(12,11)*var[696];
    var[744] = _covariance(12,9)*var[698];
    var[745] = _covariance(12,12)*var[700];
    var[746] = _covariance(13,12)*var[702];
    var[747] = _covariance(14,12)*var[704];
    var[748] = _covariance(12,3)*var[706];
    var[749] = _covariance(12,4)*var[708];
    var[750] = _covariance(12,5)*var[710];
    var[751] = -_covariance(12,7) + var[742] + var[743] + var[744] - var[745] - var[746] - var[747] + var[748] + var[749] + var[750];
    var[752] = _covariance(13,10)*var[694];
    var[753] = _covariance(13,11)*var[696];
    var[754] = _covariance(13,9)*var[698];
    var[755] = _covariance(13,12)*var[700];
    var[756] = _covariance(13,13)*var[702];
    var[757] = _covariance(14,13)*var[704];
    var[758] = _covariance(13,3)*var[706];
    var[759] = _covariance(13,4)*var[708];
    var[760] = _covariance(13,5)*var[710];
    var[761] = -_covariance(13,7) + var[752] + var[753] + var[754] - var[755] - var[756] - var[757] + var[758] + var[759] + var[760];
    var[762] = _covariance(14,10)*var[694];
    var[763] = _covariance(14,11)*var[696];
    var[764] = _covariance(14,9)*var[698];
    var[765] = _covariance(14,12)*var[700];
    var[766] = _covariance(14,13)*var[702];
    var[767] = _covariance(14,14)*var[704];
    var[768] = _covariance(14,3)*var[706];
    var[769] = _covariance(14,4)*var[708];
    var[770] = _covariance(14,5)*var[710];
    var[771] = -_covariance(14,7) + var[762] + var[763] + var[764] - var[765] - var[766] - var[767] + var[768] + var[769] + var[770];
    var[772] = var[162]*var[187];
    var[773] = var[163]*var[187];
    var[774] = var[164]*var[187];
    var[775] = var[146]*var[187];
    var[776] = var[148]*var[187];
    var[777] = var[150]*var[187];
    var[778] = var[165]*var[187];
    var[779] = var[166]*var[187];
    var[780] = var[167]*var[187];
    var[781] = _covariance(10,9)*var[773];
    var[782] = _covariance(11,9)*var[774];
    var[783] = _covariance(9,9)*var[772];
    var[784] = _covariance(12,9)*var[775];
    var[785] = _covariance(13,9)*var[776];
    var[786] = _covariance(14,9)*var[777];
    var[787] = _covariance(9,3)*var[778];
    var[788] = _covariance(9,4)*var[779];
    var[789] = _covariance(9,5)*var[780];
    var[790] = _covariance(10,10)*var[773];
    var[791] = _covariance(10,9)*var[772];
    var[792] = _covariance(11,10)*var[774];
    var[793] = _covariance(12,10)*var[775];
    var[794] = _covariance(13,10)*var[776];
    var[795] = _covariance(14,10)*var[777];
    var[796] = _covariance(10,3)*var[778];
    var[797] = _covariance(10,4)*var[779];
    var[798] = _covariance(10,5)*var[780];
    var[799] = _covariance(11,10)*var[773];
    var[800] = _covariance(11,11)*var[774];
    var[801] = _covariance(11,9)*var[772];
    var[802] = _covariance(12,11)*var[775];
    var[803] = _covariance(13,11)*var[776];
    var[804] = _covariance(14,11)*var[777];
    var[805] = _covariance(11,3)*var[778];
    var[806] = _covariance(11,4)*var[779];
    var[807] = _covariance(11,5)*var[780];
    var[808] = _covariance(12,10)*var[773];
    var[809] = _covariance(12,11)*var[774];
    var[810] = _covariance(12,9)*var[772];
    var[811] = _covariance(12,12)*var[775];
    var[812] = _covariance(13,12)*var[776];
    var[813] = _covariance(14,12)*var[777];
    var[814] = _covariance(12,3)*var[778];
    var[815] = _covariance(12,4)*var[779];
    var[816] = _covariance(12,5)*var[780];
    var[817] = _covariance(13,10)*var[773];
    var[818] = _covariance(13,11)*var[774];
    var[819] = _covariance(13,9)*var[772];
    var[820] = _covariance(13,12)*var[775];
    var[821] = _covariance(13,13)*var[776];
    var[822] = _covariance(14,13)*var[777];
    var[823] = _covariance(13,3)*var[778];
    var[824] = _covariance(13,4)*var[779];
    var[825] = _covariance(13,5)*var[780];
    var[826] = _covariance(14,10)*var[773];
    var[827] = _covariance(14,11)*var[774];
    var[828] = _covariance(14,9)*var[772];
    var[829] = _covariance(14,12)*var[775];
    var[830] = _covariance(14,13)*var[776];
    var[831] = _covariance(14,14)*var[777];
    var[832] = _covariance(14,3)*var[778];
    var[833] = _covariance(14,4)*var[779];
    var[834] = _covariance(14,5)*var[780];
    var[835] = na_w*var[2];


    _covariance(0,0) = _covariance(0,0) - _covariance(10,0)*var[4] - _covariance(11,0)*var[7] + _covariance(12,0)*var[13] + _covariance(13,0)*var[16] + _covariance(14,0)*var[19] - _covariance(3,0)*var[22] - _covariance(4,0)*var[25] - _covariance(5,0)*var[28] + _covariance(6,0)*_dt - _covariance(9,0)*var[10] - _dt*(-_covariance(6,0) - var[0] + var[11] - var[14] - var[17] - var[20] + var[23] + var[26] + var[29] + var[5] + var[8]) - var[101]*var[16] + var[10]*var[51] - var[112]*var[19] + var[114] - var[13]*var[90] + var[22]*var[75] + var[25]*var[77] + var[28]*var[79] + var[31]*var[33] + var[34]*var[35] + var[34]*var[40] + var[35]*var[36] + var[36]*var[40] + var[37]*var[39] + var[4]*var[62] + var[73]*var[7];
    _covariance(0,1) = _covariance(1,0) - _covariance(10,1)*var[4] - _covariance(11,1)*var[7] + _covariance(12,1)*var[13] + _covariance(13,1)*var[16] + _covariance(14,1)*var[19] - _covariance(3,1)*var[22] - _covariance(4,1)*var[25] - _covariance(5,1)*var[28] + _covariance(6,1)*_dt - _covariance(9,1)*var[10] - _dt*(-_covariance(7,0) - var[121] + var[122] + var[123] + var[124] - var[125] - var[126] - var[127] + var[128] + var[129] + var[130]) - var[115]*var[137] + var[116]*var[35] + var[116]*var[40] - var[117]*var[138] + var[118]*var[35] + var[118]*var[40] - var[119]*var[139] + var[120]*var[35] + var[120]*var[40] + var[131]*var[132] + var[133]*var[134] + var[135]*var[136] + var[140]*var[141] + var[142]*var[143] + var[144]*var[145];
    _covariance(1,1) = _covariance(1,1) - _covariance(10,1)*var[205] - _covariance(11,1)*var[207] + _covariance(12,1)*var[211] + _covariance(13,1)*var[213] + _covariance(14,1)*var[215] - _covariance(3,1)*var[217] - _covariance(4,1)*var[219] - _covariance(5,1)*var[221] + _covariance(7,1)*_dt - _covariance(9,1)*var[209] - _dt*(-_covariance(7,1) - var[204] + var[206] + var[208] + var[210] - var[212] - var[214] - var[216] + var[218] + var[220] + var[222]) + var[114] + var[205]*var[247] + var[207]*var[258] + var[209]*var[236] - var[211]*var[275] - var[213]*var[286] - var[215]*var[297] + var[217]*var[260] + var[219]*var[262] + var[221]*var[264] + var[223]*var[35] + var[223]*var[40] + var[224]*var[35] + var[224]*var[40] + var[225]*var[35] + var[225]*var[40];
    _covariance(0,2) = -_covariance(10,2)*var[4] - _covariance(11,2)*var[7] + _covariance(12,2)*var[13] + _covariance(13,2)*var[16] + _covariance(14,2)*var[19] + _covariance(2,0) - _covariance(3,2)*var[22] - _covariance(4,2)*var[25] - _covariance(5,2)*var[28] + _covariance(6,2)*_dt - _covariance(9,2)*var[10] - _dt*(-_covariance(8,0) - var[152] + var[153] + var[154] + var[155] - var[156] - var[157] - var[158] + var[159] + var[160] + var[161]) + var[132]*var[162] + var[134]*var[163] + var[136]*var[164] - var[137]*var[146] - var[138]*var[148] - var[139]*var[150] + var[141]*var[165] + var[143]*var[166] + var[145]*var[167] + var[147]*var[35] + var[147]*var[40] + var[149]*var[35] + var[149]*var[40] + var[151]*var[35] + var[151]*var[40];
    _covariance(1,2) = -_covariance(10,2)*var[205] - _covariance(11,2)*var[207] + _covariance(12,2)*var[211] + _covariance(13,2)*var[213] + _covariance(14,2)*var[215] + _covariance(2,1) - _covariance(3,2)*var[217] - _covariance(4,2)*var[219] - _covariance(5,2)*var[221] + _covariance(7,2)*_dt - _covariance(9,2)*var[209] - _dt*(-_covariance(8,1) - var[301] + var[302] + var[303] + var[304] - var[305] - var[306] - var[307] + var[308] + var[309] + var[310]) + var[298]*var[35] + var[298]*var[40] + var[299]*var[35] + var[299]*var[40] + var[300]*var[35] + var[300]*var[40] + var[311]*var[3] + var[312]*var[3] + var[313]*var[3] - var[314]*var[3] - var[315]*var[3] - var[316]*var[3] + var[317]*var[3] + var[318]*var[3] + var[319]*var[3];
    _covariance(2,2) = -_covariance(10,2)*var[340] - _covariance(11,2)*var[342] + _covariance(12,2)*var[346] + _covariance(13,2)*var[348] + _covariance(14,2)*var[350] + _covariance(2,2) - _covariance(3,2)*var[352] - _covariance(4,2)*var[354] - _covariance(5,2)*var[356] + _covariance(8,2)*_dt - _covariance(9,2)*var[344] - _dt*(-_covariance(8,2) - var[339] + var[341] + var[343] + var[345] - var[347] - var[349] - var[351] + var[353] + var[355] + var[357]) + var[114] + var[340]*var[382] + var[342]*var[393] + var[344]*var[371] - var[346]*var[410] - var[348]*var[421] - var[350]*var[432] + var[352]*var[395] + var[354]*var[397] + var[356]*var[399] + var[358]*var[35] + var[358]*var[40] + var[359]*var[35] + var[359]*var[40] + var[35]*var[360] + var[360]*var[40];
    _covariance(0,3) = -dr00*var[75] - dr10*var[77] - dr20*var[79] + jr_dt00*var[90] + jr_dt01*var[101] + jr_dt02*var[112] - var[169]*var[170] - var[169]*var[171] - var[169]*var[172] - var[170]*var[174] - var[171]*var[174] - var[172]*var[174];
    _covariance(1,3) = -dr00*var[260] - dr10*var[262] - dr20*var[264] + jr_dt00*var[275] + jr_dt01*var[286] + jr_dt02*var[297] - var[169]*var[320] - var[169]*var[321] - var[169]*var[322] - var[174]*var[320] - var[174]*var[321] - var[174]*var[322];
    _covariance(2,3) = -dr00*var[395] - dr10*var[397] - dr20*var[399] + jr_dt00*var[410] + jr_dt01*var[421] + jr_dt02*var[432] - var[169]*var[433] - var[169]*var[434] - var[169]*var[435] - var[174]*var[433] - var[174]*var[434] - var[174]*var[435];
    _covariance(3,3) = -dr00*var[478] - dr10*var[479] - dr20*var[480] + jr_dt00*var[463] + jr_dt01*var[470] + jr_dt02*var[477] + var[451]*var[452] + var[451]*var[456] + var[452]*var[453] + var[452]*var[454] + var[453]*var[456] + var[454]*var[456];
    _covariance(0,4) = -dr01*var[75] - dr11*var[77] - dr21*var[79] + jr_dt10*var[90] + jr_dt11*var[101] + jr_dt12*var[112] - var[169]*var[175] - var[169]*var[176] - var[169]*var[177] - var[174]*var[175] - var[174]*var[176] - var[174]*var[177];
    _covariance(1,4) = -dr01*var[260] - dr11*var[262] - dr21*var[264] + jr_dt10*var[275] + jr_dt11*var[286] + jr_dt12*var[297] - var[169]*var[323] - var[169]*var[324] - var[169]*var[325] - var[174]*var[323] - var[174]*var[324] - var[174]*var[325];
    _covariance(2,4) = -dr01*var[395] - dr11*var[397] - dr21*var[399] + jr_dt10*var[410] + jr_dt11*var[421] + jr_dt12*var[432] - var[169]*var[436] - var[169]*var[437] - var[169]*var[438] - var[174]*var[436] - var[174]*var[437] - var[174]*var[438];
    _covariance(3,4) = -dr01*var[478] - dr11*var[479] - dr21*var[480] + jr_dt10*var[463] + jr_dt10*var[481] + jr_dt10*var[484] + jr_dt11*var[470] + jr_dt11*var[482] + jr_dt11*var[485] + jr_dt12*var[477] + jr_dt12*var[483] + jr_dt12*var[486];
    _covariance(4,4) = -dr01*var[531] - dr11*var[532] - dr21*var[533] + jr_dt10*var[516] + jr_dt11*var[523] + jr_dt12*var[530] + var[452]*var[507] + var[452]*var[508] + var[452]*var[509] + var[456]*var[507] + var[456]*var[508] + var[456]*var[509];
    _covariance(0,5) = -dr02*var[75] - dr12*var[77] - dr22*var[79] + jr_dt20*var[90] + jr_dt21*var[101] + jr_dt22*var[112] - var[169]*var[178] - var[169]*var[179] - var[169]*var[180] - var[174]*var[178] - var[174]*var[179] - var[174]*var[180];
    _covariance(1,5) = -dr02*var[260] - dr12*var[262] - dr22*var[264] + jr_dt20*var[275] + jr_dt21*var[286] + jr_dt22*var[297] - var[169]*var[326] - var[169]*var[327] - var[169]*var[328] - var[174]*var[326] - var[174]*var[327] - var[174]*var[328];
    _covariance(2,5) = -dr02*var[395] - dr12*var[397] - dr22*var[399] + jr_dt20*var[410] + jr_dt21*var[421] + jr_dt22*var[432] - var[169]*var[439] - var[169]*var[440] - var[169]*var[441] - var[174]*var[439] - var[174]*var[440] - var[174]*var[441];
    _covariance(3,5) = -dr02*var[478] - dr12*var[479] - dr22*var[480] + jr_dt20*var[463] + jr_dt20*var[481] + jr_dt20*var[484] + jr_dt21*var[470] + jr_dt21*var[482] + jr_dt21*var[485] + jr_dt22*var[477] + jr_dt22*var[483] + jr_dt22*var[486];
    _covariance(4,5) = -dr02*var[531] - dr12*var[532] - dr22*var[533] + jr_dt20*var[516] + jr_dt21*var[523] + jr_dt22*var[530] + var[452]*var[534] + var[452]*var[535] + var[452]*var[536] + var[456]*var[534] + var[456]*var[535] + var[456]*var[536];
    _covariance(5,5) = -dr02*var[578] - dr12*var[579] - dr22*var[580] + jr_dt20*var[563] + jr_dt21*var[570] + jr_dt22*var[577] + var[452]*var[554] + var[452]*var[555] + var[452]*var[556] + var[456]*var[554] + var[456]*var[555] + var[456]*var[556];
    _covariance(0,6) = _covariance(6,0) + var[0] - var[11] - var[12]*var[194] + var[14] - var[15]*var[195] + var[17] + var[182]*var[31] + var[183]*var[34] + var[183]*var[36] + var[185]*var[37] + var[186]*var[34] + var[186]*var[36] + var[188]*var[9] + var[189]*var[1] - var[18]*var[196] + var[190]*var[6] + var[191]*var[21] + var[192]*var[24] + var[193]*var[27] + var[198] + var[20] - var[23] - var[26] - var[29] - var[5] - var[8];
    _covariance(1,6) = -_covariance(10,6)*var[205] - _covariance(11,6)*var[207] + _covariance(12,6)*var[211] + _covariance(13,6)*var[213] + _covariance(14,6)*var[215] + _covariance(6,1) - _covariance(6,3)*var[217] - _covariance(6,4)*var[219] - _covariance(6,5)*var[221] - _covariance(9,6)*var[209] - var[12]*var[335] - var[15]*var[336] - var[18]*var[337] + var[199] + var[1]*var[330] + var[21]*var[332] + var[24]*var[333] + var[27]*var[334] + var[329]*var[9] + var[331]*var[6];
    _covariance(2,6) = -_covariance(10,6)*var[340] - _covariance(11,6)*var[342] + _covariance(12,6)*var[346] + _covariance(13,6)*var[348] + _covariance(14,6)*var[350] + _covariance(6,2) - _covariance(6,3)*var[352] - _covariance(6,4)*var[354] - _covariance(6,5)*var[356] - _covariance(9,6)*var[344] - var[12]*var[448] - var[15]*var[449] - var[18]*var[450] + var[1]*var[443] + var[200] + var[21]*var[445] + var[24]*var[446] + var[27]*var[447] + var[442]*var[9] + var[444]*var[6];
    _covariance(3,6) = -_covariance(12,6)*jr_dt00 - _covariance(13,6)*jr_dt01 - _covariance(14,6)*jr_dt02 + _covariance(6,3)*dr00 + _covariance(6,4)*dr10 + _covariance(6,5)*dr20 - var[12]*var[500] - var[15]*var[501] - var[170]*var[487] - var[170]*var[488] - var[171]*var[487] - var[171]*var[488] - var[172]*var[487] - var[172]*var[488] - var[18]*var[502] - var[1]*var[497] + var[21]*var[503] + var[24]*var[504] + var[27]*var[505] + var[495]*var[9] - var[499]*var[6];
    _covariance(4,6) = -_covariance(12,6)*jr_dt10 - _covariance(13,6)*jr_dt11 - _covariance(14,6)*jr_dt12 + _covariance(6,3)*dr01 + _covariance(6,4)*dr11 + _covariance(6,5)*dr21 - var[12]*var[548] - var[15]*var[549] - var[175]*var[487] - var[175]*var[488] - var[176]*var[487] - var[176]*var[488] - var[177]*var[487] - var[177]*var[488] - var[18]*var[550] - var[1]*var[545] + var[21]*var[551] + var[24]*var[552] + var[27]*var[553] + var[543]*var[9] - var[547]*var[6];
    _covariance(5,6) = -_covariance(12,6)*jr_dt20 - _covariance(13,6)*jr_dt21 - _covariance(14,6)*jr_dt22 + _covariance(6,3)*dr02 + _covariance(6,4)*dr12 + _covariance(6,5)*dr22 - var[12]*var[592] - var[15]*var[593] - var[178]*var[487] - var[178]*var[488] - var[179]*var[487] - var[179]*var[488] - var[180]*var[487] - var[180]*var[488] - var[18]*var[594] - var[1]*var[589] + var[21]*var[595] + var[24]*var[596] + var[27]*var[597] + var[587]*var[9] - var[591]*var[6];
    _covariance(6,6) = _covariance(6,6) + var[102]*var[600] + var[168]*var[37] + var[169]*var[34] + var[169]*var[36] + var[173]*var[31] + var[174]*var[34] + var[174]*var[36] - var[41]*var[648] - var[52]*var[643] + var[598]*var[80] + var[599]*var[91] + var[601]*var[629] + var[603]*var[639] + var[605]*var[619] - var[607]*var[658] - var[609]*var[668] - var[611]*var[678] + var[613]*var[640] + var[615]*var[641] + var[617]*var[642] - var[63]*var[644] - var[645]*var[74] - var[646]*var[76] - var[647]*var[78] + var[679];
    _covariance(0,7) = _covariance(7,0) - var[115]*var[194] - var[117]*var[195] - var[119]*var[196] - var[122] - var[123] - var[124] + var[125] + var[126] + var[127] - var[128] - var[129] - var[130] + var[131]*var[188] + var[133]*var[189] + var[135]*var[190] + var[140]*var[191] + var[142]*var[192] + var[144]*var[193] + var[199];
    _covariance(1,7) = _covariance(7,1) - var[115]*var[335] - var[117]*var[336] - var[119]*var[337] + var[131]*var[329] + var[133]*var[330] + var[135]*var[331] + var[140]*var[332] + var[142]*var[333] + var[144]*var[334] + var[183]*var[223] + var[183]*var[224] + var[183]*var[225] + var[186]*var[223] + var[186]*var[224] + var[186]*var[225] + var[198] + var[204] - var[206] - var[208] - var[210] + var[212] + var[214] + var[216] - var[218] - var[220] - var[222];
    _covariance(2,7) = -_covariance(10,7)*var[340] - _covariance(11,7)*var[342] + _covariance(12,7)*var[346] + _covariance(13,7)*var[348] + _covariance(14,7)*var[350] + _covariance(7,2) - _covariance(7,3)*var[352] - _covariance(7,4)*var[354] - _covariance(7,5)*var[356] - _covariance(9,7)*var[344] - var[115]*var[448] - var[117]*var[449] - var[119]*var[450] + var[131]*var[442] + var[133]*var[443] + var[135]*var[444] + var[140]*var[445] + var[142]*var[446] + var[144]*var[447] + var[338];
    _covariance(3,7) = -_covariance(12,7)*jr_dt00 - _covariance(13,7)*jr_dt01 - _covariance(14,7)*jr_dt02 + _covariance(7,3)*dr00 + _covariance(7,4)*dr10 + _covariance(7,5)*dr20 - var[115]*var[500] - var[117]*var[501] - var[119]*var[502] + var[131]*var[495] - var[133]*var[497] - var[135]*var[499] + var[140]*var[503] + var[142]*var[504] + var[144]*var[505] - var[320]*var[487] - var[320]*var[488] - var[321]*var[487] - var[321]*var[488] - var[322]*var[487] - var[322]*var[488];
    _covariance(4,7) = -_covariance(12,7)*jr_dt10 - _covariance(13,7)*jr_dt11 - _covariance(14,7)*jr_dt12 + _covariance(7,3)*dr01 + _covariance(7,4)*dr11 + _covariance(7,5)*dr21 - var[115]*var[548] - var[117]*var[549] - var[119]*var[550] + var[131]*var[543] - var[133]*var[545] - var[135]*var[547] + var[140]*var[551] + var[142]*var[552] + var[144]*var[553] - var[323]*var[487] - var[323]*var[488] - var[324]*var[487] - var[324]*var[488] - var[325]*var[487] - var[325]*var[488];
    _covariance(5,7) = -_covariance(12,7)*jr_dt20 - _covariance(13,7)*jr_dt21 - _covariance(14,7)*jr_dt22 + _covariance(7,3)*dr02 + _covariance(7,4)*dr12 + _covariance(7,5)*dr22 - var[115]*var[592] - var[117]*var[593] - var[119]*var[594] + var[131]*var[587] - var[133]*var[589] - var[135]*var[591] + var[140]*var[595] + var[142]*var[596] + var[144]*var[597] - var[326]*var[487] - var[326]*var[488] - var[327]*var[487] - var[327]*var[488] - var[328]*var[487] - var[328]*var[488];
    _covariance(6,7) = _covariance(7,6) - var[115]*var[683] + var[116]*var[169] + var[116]*var[174] - var[117]*var[684] + var[118]*var[169] + var[118]*var[174] - var[119]*var[685] + var[120]*var[169] + var[120]*var[174] + var[131]*var[680] + var[133]*var[681] + var[135]*var[682] + var[140]*var[686] + var[142]*var[687] + var[144]*var[688] - var[226]*var[648] - var[237]*var[643] - var[248]*var[644] - var[259]*var[645] - var[261]*var[646] - var[263]*var[647] + var[265]*var[598] + var[276]*var[599] + var[287]*var[600];
    _covariance(7,7) = _covariance(7,7) + var[169]*var[223] + var[169]*var[224] + var[169]*var[225] + var[174]*var[223] + var[174]*var[224] + var[174]*var[225] - var[226]*var[741] - var[237]*var[736] - var[248]*var[737] - var[259]*var[738] - var[261]*var[739] - var[263]*var[740] + var[265]*var[691] + var[276]*var[692] + var[287]*var[693] + var[679] + var[694]*var[722] + var[696]*var[732] + var[698]*var[712] - var[700]*var[751] - var[702]*var[761] - var[704]*var[771] + var[706]*var[733] + var[708]*var[734] + var[710]*var[735];
    _covariance(0,8) = _covariance(8,0) - var[146]*var[194] - var[148]*var[195] - var[150]*var[196] - var[153] - var[154] - var[155] + var[156] + var[157] + var[158] - var[159] - var[160] - var[161] + var[162]*var[188] + var[163]*var[189] + var[164]*var[190] + var[165]*var[191] + var[166]*var[192] + var[167]*var[193] + var[200];
    _covariance(1,8) = _covariance(8,1) + var[187]*var[311] + var[187]*var[312] + var[187]*var[313] - var[187]*var[314] - var[187]*var[315] - var[187]*var[316] + var[187]*var[317] + var[187]*var[318] + var[187]*var[319] - var[302] - var[303] - var[304] + var[305] + var[306] + var[307] - var[308] - var[309] - var[310] + var[338];
    _covariance(2,8) = _covariance(8,2) - var[146]*var[448] - var[148]*var[449] - var[150]*var[450] + var[162]*var[442] + var[163]*var[443] + var[164]*var[444] + var[165]*var[445] + var[166]*var[446] + var[167]*var[447] + var[183]*var[358] + var[183]*var[359] + var[183]*var[360] + var[186]*var[358] + var[186]*var[359] + var[186]*var[360] + var[198] + var[339] - var[341] - var[343] - var[345] + var[347] + var[349] + var[351] - var[353] - var[355] - var[357];
    _covariance(3,8) = -_covariance(12,8)*jr_dt00 - _covariance(13,8)*jr_dt01 - _covariance(14,8)*jr_dt02 + _covariance(8,3)*dr00 + _covariance(8,4)*dr10 + _covariance(8,5)*dr20 - var[146]*var[500] - var[148]*var[501] - var[150]*var[502] + var[162]*var[495] - var[163]*var[497] - var[164]*var[499] + var[165]*var[503] + var[166]*var[504] + var[167]*var[505] - var[433]*var[487] - var[433]*var[488] - var[434]*var[487] - var[434]*var[488] - var[435]*var[487] - var[435]*var[488];
    _covariance(4,8) = -_covariance(12,8)*jr_dt10 - _covariance(13,8)*jr_dt11 - _covariance(14,8)*jr_dt12 + _covariance(8,3)*dr01 + _covariance(8,4)*dr11 + _covariance(8,5)*dr21 - var[146]*var[548] - var[148]*var[549] - var[150]*var[550] + var[162]*var[543] - var[163]*var[545] - var[164]*var[547] + var[165]*var[551] + var[166]*var[552] + var[167]*var[553] - var[436]*var[487] - var[436]*var[488] - var[437]*var[487] - var[437]*var[488] - var[438]*var[487] - var[438]*var[488];
    _covariance(5,8) = -_covariance(12,8)*jr_dt20 - _covariance(13,8)*jr_dt21 - _covariance(14,8)*jr_dt22 + _covariance(8,3)*dr02 + _covariance(8,4)*dr12 + _covariance(8,5)*dr22 - var[146]*var[592] - var[148]*var[593] - var[150]*var[594] + var[162]*var[587] - var[163]*var[589] - var[164]*var[591] + var[165]*var[595] + var[166]*var[596] + var[167]*var[597] - var[439]*var[487] - var[439]*var[488] - var[440]*var[487] - var[440]*var[488] - var[441]*var[487] - var[441]*var[488];
    _covariance(6,8) = _covariance(8,6) - var[146]*var[683] + var[147]*var[169] + var[147]*var[174] - var[148]*var[684] + var[149]*var[169] + var[149]*var[174] - var[150]*var[685] + var[151]*var[169] + var[151]*var[174] + var[162]*var[680] + var[163]*var[681] + var[164]*var[682] + var[165]*var[686] + var[166]*var[687] + var[167]*var[688] - var[361]*var[648] - var[372]*var[643] - var[383]*var[644] - var[394]*var[645] - var[396]*var[646] - var[398]*var[647] + var[400]*var[598] + var[411]*var[599] + var[422]*var[600];
    _covariance(7,8) = _covariance(8,7) + var[169]*var[298] + var[169]*var[299] + var[169]*var[300] + var[174]*var[298] + var[174]*var[299] + var[174]*var[300] - var[361]*var[741] - var[372]*var[736] - var[383]*var[737] - var[394]*var[738] - var[396]*var[739] - var[398]*var[740] + var[400]*var[691] + var[411]*var[692] + var[422]*var[693] + var[712]*var[772] + var[722]*var[773] + var[732]*var[774] + var[733]*var[778] + var[734]*var[779] + var[735]*var[780] - var[751]*var[775] - var[761]*var[776] - var[771]*var[777];
    _covariance(8,8) = _covariance(8,8) + 0.5 *var[146]*var[400] + 0.5 *var[148]*var[411] + 0.5 *var[150]*var[422] - 0.5 *var[162]*var[361] - 0.5 *var[163]*var[372] - 0.5 *var[164]*var[383] - 0.5 *var[165]*var[394] - 0.5 *var[166]*var[396] - 0.5 *var[167]*var[398] + var[169]*var[358] + var[169]*var[359] + var[169]*var[360] + var[174]*var[358] + var[174]*var[359] + var[174]*var[360] + var[679] + var[772]*(-_covariance(9,8) + var[781] + var[782] + var[783] - var[784] - var[785] - var[786] + var[787] + var[788] + var[789]) + var[773]*(-_covariance(10,8) + var[790] + var[791] + var[792] - var[793] - var[794] - var[795] + var[796] + var[797] + var[798]) + var[774]*(-_covariance(11,8) + var[799] + var[800] + var[801] - var[802] - var[803] - var[804] + var[805] + var[806] + var[807]) - var[775]*(-_covariance(12,8) + var[808] + var[809] + var[810] - var[811] - var[812] - var[813] + var[814] + var[815] + var[816]) - var[776]*(-_covariance(13,8) + var[817] + var[818] + var[819] - var[820] - var[821] - var[822] + var[823] + var[824] + var[825]) - var[777]*(-_covariance(14,8) + var[826] + var[827] + var[828] - var[829] - var[830] - var[831] + var[832] + var[833] + var[834]) + var[778]*(_covariance(10,3)*var[773] + _covariance(11,3)*var[774] - _covariance(12,3)*var[775] - _covariance(13,3)*var[776] - _covariance(14,3)*var[777] + p33*var[778] + _covariance(4,3)*var[779] + _covariance(5,3)*var[780] - _covariance(8,3) + _covariance(9,3)*var[772]) + var[779]*(_covariance(10,4)*var[773] + _covariance(11,4)*var[774] - _covariance(12,4)*var[775] - _covariance(13,4)*var[776] - _covariance(14,4)*var[777] + _covariance(4,3)*var[778] + p44*var[779] + _covariance(5,4)*var[780] - _covariance(8,4) + _covariance(9,4)*var[772]) + var[780]*(_covariance(10,5)*var[773] + _covariance(11,5)*var[774] - _covariance(12,5)*var[775] - _covariance(13,5)*var[776] - _covariance(14,5)*var[777] + _covariance(5,3)*var[778] + _covariance(5,4)*var[779] + p55*var[780] - _covariance(8,5) + _covariance(9,5)*var[772]);
    _covariance(0,9) = _covariance(9,0) + var[202] + var[41] - var[42] - var[43] - var[44] + var[45] + var[46] + var[47] - var[48] - var[49] - var[50];
    _covariance(1,9) = _covariance(9,1) + var[226] - var[227] - var[228] - var[229] + var[230] + var[231] + var[232] - var[233] - var[234] - var[235];
    _covariance(2,9) = _covariance(9,2) + var[361] - var[362] - var[363] - var[364] + var[365] + var[366] + var[367] - var[368] - var[369] - var[370];
    _covariance(3,9) = -var[489] - var[490] - var[491] + var[492] + var[493] + var[494];
    _covariance(4,9) = -var[537] - var[538] - var[539] + var[540] + var[541] + var[542];
    _covariance(5,9) = -var[581] - var[582] - var[583] + var[584] + var[585] + var[586];
    _covariance(6,9) = _covariance(9,6) - var[602] - var[604] - var[606] + var[608] + var[610] + var[612] - var[614] - var[616] - var[618] + var[689];
    _covariance(7,9) = _covariance(9,7) - var[695] - var[697] - var[699] + var[701] + var[703] + var[705] - var[707] - var[709] - var[711];
    _covariance(8,9) = _covariance(9,8) - var[781] - var[782] - var[783] + var[784] + var[785] + var[786] - var[787] - var[788] - var[789];
    _covariance(9,9) = _covariance(9,9) + var[835];
    _covariance(0,10) = _covariance(10,0) + var[52] - var[53] - var[54] - var[55] + var[56] + var[57] + var[58] - var[59] - var[60] - var[61];
    _covariance(1,10) = _covariance(10,1) + var[202] + var[237] - var[238] - var[239] - var[240] + var[241] + var[242] + var[243] - var[244] - var[245] - var[246];
    _covariance(2,10) = _covariance(10,2) + var[372] - var[373] - var[374] - var[375] + var[376] + var[377] + var[378] - var[379] - var[380] - var[381];
    _covariance(3,10) = var[496];
    _covariance(4,10) = var[544];
    _covariance(5,10) = var[588];
    _covariance(6,10) = _covariance(10,6) - var[620] - var[621] - var[622] + var[623] + var[624] + var[625] - var[626] - var[627] - var[628];
    _covariance(7,10) = _covariance(10,7) + var[689] - var[713] - var[714] - var[715] + var[716] + var[717] + var[718] - var[719] - var[720] - var[721];
    _covariance(8,10) = _covariance(10,8) - var[790] - var[791] - var[792] + var[793] + var[794] + var[795] - var[796] - var[797] - var[798];
    _covariance(9,10) = _covariance(10,9);
    _covariance(10,10) = _covariance(10,10) + var[835];
    _covariance(0,11) = _covariance(11,0) + var[63] - var[64] - var[65] - var[66] + var[67] + var[68] + var[69] - var[70] - var[71] - var[72];
    _covariance(1,11) = _covariance(11,1) + var[248] - var[249] - var[250] - var[251] + var[252] + var[253] + var[254] - var[255] - var[256] - var[257];
    _covariance(2,11) = _covariance(11,2) + var[202] + var[383] - var[384] - var[385] - var[386] + var[387] + var[388] + var[389] - var[390] - var[391] - var[392];
    _covariance(3,11) = var[498];
    _covariance(4,11) = var[546];
    _covariance(5,11) = var[590];
    _covariance(6,11) = _covariance(11,6) - var[630] - var[631] - var[632] + var[633] + var[634] + var[635] - var[636] - var[637] - var[638];
    _covariance(7,11) = _covariance(11,7) - var[723] - var[724] - var[725] + var[726] + var[727] + var[728] - var[729] - var[730] - var[731];
    _covariance(8,11) = _covariance(11,8) + var[689] - var[799] - var[800] - var[801] + var[802] + var[803] + var[804] - var[805] - var[806] - var[807];

    _covariance(11,11) = _covariance(11,11) + var[835];
    _covariance(0,12) = _covariance(12,0) + var[12]*var[203] + var[80] - var[81] - var[82] - var[83] + var[84] + var[85] + var[86] - var[87] - var[88] - var[89];
    _covariance(1,12) = _covariance(12,1) + var[115]*var[203] + var[265] - var[266] - var[267] - var[268] + var[269] + var[270] + var[271] - var[272] - var[273] - var[274];
    _covariance(2,12) = _covariance(12,2) + var[146]*var[203] + var[400] - var[401] - var[402] - var[403] + var[404] + var[405] + var[406] - var[407] - var[408] - var[409];
    _covariance(3,12) = -jr_dt00*var[506] - var[457] - var[458] - var[459] + var[460] + var[461] + var[462];
    _covariance(4,12) = -jr_dt10*var[506] - var[510] - var[511] - var[512] + var[513] + var[514] + var[515];
    _covariance(5,12) = -jr_dt20*var[506] - var[557] - var[558] - var[559] + var[560] + var[561] + var[562];
    _covariance(6,12) = _covariance(12,6) + var[12]*var[690] - var[649] - var[650] - var[651] + var[652] + var[653] + var[654] - var[655] - var[656] - var[657];
    _covariance(7,12) = _covariance(12,7) + var[115]*var[690] - var[742] - var[743] - var[744] + var[745] + var[746] + var[747] - var[748] - var[749] - var[750];
    _covariance(8,12) = _covariance(12,8) + var[146]*var[690] - var[808] - var[809] - var[810] + var[811] + var[812] + var[813] - var[814] - var[815] - var[816];

    _covariance(12,12) = _covariance(12,12) + var[455];
    _covariance(0,13) = _covariance(13,0) - var[100] + var[15]*var[203] + var[91] - var[92] - var[93] - var[94] + var[95] + var[96] + var[97] - var[98] - var[99];
    _covariance(1,13) = _covariance(13,1) + var[117]*var[203] + var[276] - var[277] - var[278] - var[279] + var[280] + var[281] + var[282] - var[283] - var[284] - var[285];
    _covariance(2,13) = _covariance(13,2) + var[148]*var[203] + var[411] - var[412] - var[413] - var[414] + var[415] + var[416] + var[417] - var[418] - var[419] - var[420];
    _covariance(3,13) = -jr_dt01*var[506] - var[464] - var[465] - var[466] + var[467] + var[468] + var[469];
    _covariance(4,13) = -jr_dt11*var[506] - var[517] - var[518] - var[519] + var[520] + var[521] + var[522];
    _covariance(5,13) = -jr_dt21*var[506] - var[564] - var[565] - var[566] + var[567] + var[568] + var[569];
    _covariance(6,13) = _covariance(13,6) + var[15]*var[690] - var[659] - var[660] - var[661] + var[662] + var[663] + var[664] - var[665] - var[666] - var[667];
    _covariance(7,13) = _covariance(13,7) + var[117]*var[690] - var[752] - var[753] - var[754] + var[755] + var[756] + var[757] - var[758] - var[759] - var[760];
    _covariance(8,13) = _covariance(13,8) + var[148]*var[690] - var[817] - var[818] - var[819] + var[820] + var[821] + var[822] - var[823] - var[824] - var[825];

    _covariance(13,13) = _covariance(13,13) + var[455];
    _covariance(0,14) = _covariance(14,0) + var[102] - var[103] - var[104] - var[105] + var[106] + var[107] + var[108] - var[109] - var[110] - var[111] + var[18]*var[203];
    _covariance(1,14) = _covariance(14,1) + var[119]*var[203] + var[287] - var[288] - var[289] - var[290] + var[291] + var[292] + var[293] - var[294] - var[295] - var[296];
    _covariance(2,14) = _covariance(14,2) + var[150]*var[203] + var[422] - var[423] - var[424] - var[425] + var[426] + var[427] + var[428] - var[429] - var[430] - var[431];
    _covariance(3,14) = -jr_dt02*var[506] - var[471] - var[472] - var[473] + var[474] + var[475] + var[476];
    _covariance(4,14) = -jr_dt12*var[506] - var[524] - var[525] - var[526] + var[527] + var[528] + var[529];
    _covariance(5,14) = -jr_dt22*var[506] - var[571] - var[572] - var[573] + var[574] + var[575] + var[576];
    _covariance(6,14) = _covariance(14,6) + var[18]*var[690] - var[669] - var[670] - var[671] + var[672] + var[673] + var[674] - var[675] - var[676] - var[677];
    _covariance(7,14) = _covariance(14,7) + var[119]*var[690] - var[762] - var[763] - var[764] + var[765] + var[766] + var[767] - var[768] - var[769] - var[770];
    _covariance(8,14) = _covariance(14,8) + var[150]*var[690] - var[826] - var[827] - var[828] + var[829] + var[830] + var[831] - var[832] - var[833] - var[834];

    _covariance(14,14) = _covariance(14,14) + var[455];




    for (unsigned int i = 1; i < 15; ++i) {
        for (unsigned int j = 0; j < i; ++j) {
            _covariance(i, j) = _covariance(j, i);
        }
    }
}

}
}