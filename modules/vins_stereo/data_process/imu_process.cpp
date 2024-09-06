//
// Created by Cain on 2024/4/28.
//

#include "../estimator.h"

#include <array>
#include <memory>
#include <random>
#include <iostream>
#include <ostream>
#include <fstream>

namespace vins {
    using namespace graph_optimization;
    using namespace std;

    void Estimator::process_imu(double dt, const Vec3 &linear_acceleration, const Vec3 &angular_velocity) {
        static bool is_first = true;
        if (is_first) {
            is_first = false;
            _acc_latest = linear_acceleration;
            _gyro_latest = angular_velocity;

            _pre_integral_stream->set_acc_init(_acc_latest);
            _pre_integral_stream->set_gyro_init(_gyro_latest);
            _pre_integral_stream->set_acc_last(_acc_latest);
            _pre_integral_stream->set_gyro_last(_gyro_latest);

            _pre_integral_window->set_acc_init(_acc_latest);
            _pre_integral_window->set_gyro_init(_gyro_latest);
            _pre_integral_window->set_acc_last(_acc_latest);
            _pre_integral_window->set_gyro_last(_gyro_latest);
        }

        _pre_integral_stream->push_back(dt, linear_acceleration, angular_velocity);
        _pre_integral_window->push_back(dt, linear_acceleration, angular_velocity);
        Vec3 gyro_corr = 0.5 * (_gyro_latest + angular_velocity) - _frame->bg();
        Vec3 acc0_corr = _frame->q() * (_acc_latest - _frame->ba());
        auto delta_q = Sophus::SO3d::exp(gyro_corr * dt);
        _frame->q() *= delta_q.unit_quaternion();
        _frame->q().normalize();
        Vec3 acc1_corr = _frame->q() * (linear_acceleration - _frame->ba());
        Vec3 acc_corr = 0.5 * (acc0_corr + acc1_corr) + _g;
        _frame->p() += (0.5 * acc_corr * dt + _frame->v()) * dt;
        _frame->v() += acc_corr * dt;

        _acc_latest = linear_acceleration;
        _gyro_latest = angular_velocity;
    }
}