//
// Created by gaoxiang19 on 19-1-7.
//
#include "backend/imu_integration.h"

using Sophus::SO3d;

namespace myslam {
namespace backend {

void IMUIntegration::Propagate(double dt, const Vec3 &acc, const Vec3 &gyr) {
    dt_buf_.emplace_back(dt);
    acc_buf_.emplace_back(acc);
    gyr_buf_.emplace_back(gyr);

    // 梯形积分
    static Vec3 acc_last_ {};
    static Vec3 gyr_last_ {};

    // 去偏移
    Vec3 a = acc - ba_;
    Vec3 a_last = acc_last_ - ba_;
    Vec3 w = gyr - bg_;
    Vec3 w_last = gyr_last_ - bg_;

    // 中值积分
    Vec3 w_mid = 0.5 * (w_last + w);
    Vec3 delta_axis_angle = w_mid * dt;
    Sophus::SO3d dR = Sophus::SO3d::exp(delta_axis_angle);
    auto delta_r_last = delta_r_.matrix();
    delta_r_ *= dR;

    Vec3 a_mid = 0.5 * (delta_r_last * a_last + delta_r_ * a);
    Vec3 delta_vel = a_mid * dt;
    auto delta_v_last = delta_v_;
    delta_v_ += delta_vel;

    Vec3 v_mid = 0.5 * (delta_v_last + delta_v_);
    Vec3 delta_pos = v_mid * dt;
    delta_p_ += delta_pos;

    double half_dt = 0.5 * dt;
    sum_dt_ += dt;

    // 雅可比
    auto delta_r = delta_r_.matrix();
    auto dr = dR.matrix();
    Mat33 jr_dt = Sophus::SO3d::JacobianR(delta_axis_angle) * dt;
    auto dr_dbg_last = dr_dbg_;
    dr_dbg_ = dr.transpose() * dr_dbg_ - jr_dt;

    auto delta_r_a_hat_last = delta_r_last * Sophus::SO3d::hat(a_last);
    auto delta_r_a_hat = delta_r * Sophus::SO3d::hat(a);
    auto dv_dba_last = dv_dba_;
    auto dv_dbg_last = dv_dbg_;
    dv_dba_ -= half_dt * (delta_r_last + delta_r);
    dv_dbg_ -= half_dt * (delta_r_a_hat_last * dr_dbg_last + delta_r_a_hat * dr_dbg_);

    dp_dba_ += half_dt * (dv_dba_last + dv_dba_);
    dp_dbg_ += half_dt * (dv_dbg_last + dv_dbg_);

    // 先验传递
    A_.block<3, 3>(0, 0).noalias() = dr.transpose();
    A_.block<3, 3>(3, 0).noalias() = -half_dt * (delta_r_a_hat_last + delta_r_a_hat * dr.transpose());
    A_.block<3, 3>(6, 0).noalias() = half_dt * A_.block<3, 3>(3, 0);
    A_.block<3, 3>(6, 3).noalias() = Mat33::Identity() * dt;

    B_.block<3, 3>(0, 0).noalias() = jr_dt;
    B_.block<3, 3>(3, 0).noalias() = -half_dt * delta_r_a_hat * B_.block<3, 3>(0, 0);
    B_.block<3, 3>(6, 0).noalias() = half_dt * B_.block<3, 3>(3, 0);
    B_.block<3, 3>(3, 3).noalias() = delta_r * dt;
    B_.block<3, 3>(6, 3).noalias() = half_dt * B_.block<3,  3>(3, 3);

    covariance_measurement_ = A_ * covariance_measurement_ * A_.transpose() + B_ * noise_measurement_ * B_.transpose();

    // 记录上一时刻数据
    acc_last_ = acc;
    gyr_last_ = gyr;
    

    // // 前向欧拉积分
    // Sophus::SO3d dR = Sophus::SO3d::exp((gyr - bg_) * dt);
    // delta_r_ = delta_r_ * dR;
    // delta_v_ += delta_r_ * (acc - ba_) * dt;
    // delta_p_ += delta_v_ * dt + 0.5 * (delta_r_ * (acc - ba_) * dt * dt);
    // sum_dt_ += dt;

    // // update jacobians w.r.t. bg and ba
    // dr_dbg_ -= delta_r_.inverse().matrix() * SO3d::JacobianR(((gyr - bg_) * dt)) * dt;
    // dv_dba_ -= delta_r_.matrix() * dt;
    // dv_dbg_ -= delta_r_.matrix() * SO3d::hat(acc - ba_) * dr_dbg_ * dt;
    // dp_dba_ += dv_dba_ * dt - 0.5 * delta_r_.matrix() * dt * dt;
    // dp_dbg_ += dv_dbg_ * dt - 0.5 * delta_r_.matrix() * SO3d::hat(acc - ba_) * dr_dbg_ * dt * dt;

    // // propagate noise
    // A_.block<3, 3>(0, 0) = dR.inverse().matrix();
    // B_.block<3, 3>(0, 0) = SO3d::JacobianR((gyr - bg_) * dt) * dt;

    // A_.block<3, 3>(3, 0) = -delta_r_.matrix() * SO3d::hat(acc - ba_) * dt;
    // A_.block<3, 3>(3, 3) = Mat33::Identity();
    // B_.block<3, 3>(3, 3) = delta_r_.matrix() * dt;

    // A_.block<3, 3>(6, 0) = -0.5 * delta_r_.matrix() * SO3d::hat(acc - ba_) * dt * dt;
    // A_.block<3, 3>(6, 3) = Mat33::Identity() * dt;
    // A_.block<3, 3>(6, 6) = Mat33::Identity();
    // B_.block<3, 3>(6, 3) = 0.5 * delta_r_.matrix() * dt * dt;

    covariance_measurement_ = A_ * covariance_measurement_ * A_.transpose() + B_ * noise_measurement_ * B_.transpose();
}

void IMUIntegration::Repropagate() {
    // backup imu data
    auto dt = dt_buf_;
    auto acc_buf = acc_buf_;
    auto gyr_buf = gyr_buf_;
    Reset();

    for (size_t i = 0; i < dt.size(); ++i) {
        Propagate(dt[i], acc_buf[i], gyr_buf[i]);
    }
}

void IMUIntegration::Correct(const Vec3 &delta_ba, const Vec3 &delta_bg) {
    delta_r_ = delta_r_ * SO3d::exp(dr_dbg_ * delta_bg);
    delta_v_ += dv_dba_ * delta_ba + dv_dbg_ * delta_bg;
    delta_p_ += dp_dba_ * delta_ba + dp_dbg_ * delta_bg;
}

}
}