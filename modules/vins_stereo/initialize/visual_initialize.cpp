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

    bool Estimator::structure_from_motion() {
        // 因为尺度未知, 所以要把t_ic设为零
        _t_ic[0].setZero();

        // 找出第一个与当前imu拥有足够视差的imu, 同时利用对极几何计算t_i_curr, R_i_curr
        unsigned long imu_index;
        Mat33 r_i_curr;
        Vec3 t_i_curr;
        if (!search_relative_pose(r_i_curr, t_i_curr, imu_index)) {
#ifdef PRINT_INFO
            cout << "Not enough features or parallax; Move device around" << endl;
#endif
            _t_ic[0] = Eigen::Map<Eigen::Vector3d>(_ext_params_bp[0]);
            return false;
        }

        // 以frame_i为基准建立坐标系
        auto frame_i = _sliding_window[imu_index].first;
        frame_i->p().setZero();
        frame_i->q().setIdentity();
        frame_i->is_initialized = true;

        // 设置curr的位姿
        _frame->p() = t_i_curr;
        _frame->q() = r_i_curr;
        _frame->is_initialized = true;

        // 在利用对极几何计算relative_pose时，已经同时得到了global landmark
//        // 利用i和curr进行三角化, 计算特征点的世界坐标
//        global_triangulate_with(imu_i, _imu_node);
//        std::cout << "1" << std::endl;

        /*
         * 1. 对imu_index后面的点进行pnp, 计算R, t.
         * 2. 得到最新得到R, t后进行三角化, 计算只有在imu_j到imu_node中才出现的特征点的世界坐标, i < j < curr
         * 3. 利用历史R, t进行三角化, 计算只有在imu_i到imu_j中才出现的特征点的世界坐标, i < j < curr
         * */
        for (unsigned long j = imu_index + 1; j < _sliding_window.size(); ++j) {
            auto frame_j = _sliding_window[j].first;

            // pnp, 用 j - 1 的位姿作为 j 的初始位姿估计
            Vec3 t_wj  = _sliding_window[j - 1].first->p();
            Qd q_wj = _sliding_window[j - 1].first->q();
            // if (!iter_pnp(frame_j, &q_wj, &t_wj)) {
            //     _t_ic[0] = Eigen::Map<Eigen::Vector3d>(_ext_params_bp[0]);
            //     return false;
            // }
        //    epnp(frame_j);
//            mlpnp(frame_j);
//            dltpnp(frame_j);
           pnp(frame_j, &q_wj, &t_wj);

            // 三角化
            global_triangulate_with(frame_j, _frame);

            // 三角化
            global_triangulate_with(frame_i, frame_j);
        }

        /*
         * 0. 假设imu_index - 1与imu_index有共有的特征点, 并且已求得其世界坐标
         * 1. 对imu_index前面的点进行pnp, 计算R, t.
         * 2. 得到R, t后进行三角化, 计算只有在imu_j到imu_i中才出现的特征点的世界坐标, 0 <= j < i
         * */
        for (unsigned long k = 0; k < imu_index; ++k) {
            unsigned long j = imu_index - k - 1;
            auto frame_j = _sliding_window[j].first;

            // pnp, 用 j + 1 的位姿作为 j 的初始位姿估计
            Vec3 t_wj = _sliding_window[j + 1].first->p();
            Qd q_wj = _sliding_window[j + 1].first->q();
            // if (!iter_pnp(frame_j, &q_wj, &t_wj)) {
            //     _t_ic[0] = Eigen::Map<Eigen::Vector3d>(_ext_params_bp[0]);
            //     return false;
            // }
        //    epnp(frame_j);
//            mlpnp(frame_j);
//            dltpnp(frame_j);
           pnp(frame_j, &q_wj, &t_wj);

            // 三角化
            /*
             * 若按照vins-mono源代码中的操作:
             * global_triangulate_with(frame_j, frame_i);
             * 会出现 frame_j 与 frame_i 没有公视特征的情况
             * 这会导致 frame_j 中的特征点均无法进行三角化
             * 从而无法进行后续的 pnp
             * */
            global_triangulate_with(frame_j, frame_i); 
            // global_triangulate_with(frame_j, _sliding_window[j + 1].first);
        }

        // 遍历所有特征点, 对没有赋值的特征点进行三角化
        for (auto &landmark_it : _landmarks) {
            global_triangulate_feature(landmark_it.second);
        }

        // 把所有pose都调整到以第0帧为基准
        Vec3 t_0 = _sliding_window.front().first->p();
        Qd q_0 = _sliding_window.front().first->q();
        auto R_0 = q_0.toRotationMatrix();
        _sliding_window.front().first->p().setZero();
        _sliding_window.front().first->q().setIdentity();
        for (unsigned long k = 1; k < _sliding_window.size(); ++k) {
            auto frame_k = _sliding_window[k].first;
            frame_k->p() = R_0.transpose() * (frame_k->p() - t_0);
            frame_k->q() = (q_0.inverse() * frame_k->q()).normalized();
        }
        _frame->p() = R_0.transpose() * (_frame->p() - t_0);
        _frame->q() = (q_0.inverse() * _frame->q()).normalized();

        // 把所有landmark都调整到以第0帧为基准
        unsigned int num_landmarks = 0;
        for (auto &landmark_it : _landmarks) {
            auto landmark = landmark_it.second;
            if (landmark->is_outlier) {
                continue;
            }
            if (!landmark->is_triangulated) {
                continue;
            }
            landmark->position = R_0.transpose() * (landmark->position - t_0);
            ++num_landmarks;
        }
#ifdef PRINT_INFO
        std::cout << "number of useful landmarks = " << num_landmarks << std::endl;
#endif

//        // 总体的优化
//        // 固定住不参与优化的点
//        vector<shared_ptr<VertexPose>> fixed_poses;
//        fixed_poses.emplace_back(_vertex_ext[0]);
//        fixed_poses.emplace_back(_windows[imu_index]->vertex_pose);
//        fixed_poses.emplace_back(_imu_node->vertex_pose);
//
//        // Global Bundle Adjustment
//        global_bundle_adjustment(&fixed_poses);
//        // 把特征点从global转为local
//        for (auto &feature_it : _feature_map) {
//            auto feature_node = feature_it.second;
//            feature_node->from_global_to_local(_q_ic, _t_ic);
//        }

//        // 把特征点从global转为local
//        for (auto &feature_it : _feature_map) {
//            auto feature_node = feature_it.second;
//            feature_node->from_global_to_local(_q_ic, _t_ic);
//        }
//        // Local Bundle Adjustment
//        local_bundle_adjustment(&fixed_poses);

        // 为stream中的所有frame进行pnp
        unsigned int closest_index = 0;
        std::shared_ptr<Frame> closest_frame = _sliding_window[closest_index].first;
        for (auto &frame_it : _stream) {
            if (frame_it.first->time_us == closest_frame->time_us) {
                ++closest_index;
                closest_frame = closest_index < _sliding_window.size() ? _sliding_window[closest_index].first : _frame;
                continue;
            }

            if (frame_it.first->time_us > closest_frame->time_us) {
                ++closest_index;
                closest_frame = closest_index < _sliding_window.size() ? _sliding_window[closest_index].first : _frame;
            }

            Vec3 t_init  = closest_frame->p();
            Qd q_init = closest_frame->q();
            iter_pnp(frame_it.first, &q_init, &t_init);
        }

        // 转到local系
        for (auto &landmark_it : _landmarks) {
            landmark_it.second->from_global_to_local(_q_ic[0], _t_ic[0]);
        }

//        remove_outlier_landmarks();

#ifdef PRINT_INFO
        std::cout << "done structure_from_motion" << std::endl;
        for (unsigned long i = 0; i < _sliding_window.size(); ++i) {
            std::cout << "i = " << i << ":" << std::endl;
            std::cout << "q = " << _sliding_window[i].first->q().w() << ", ";
            std::cout << _sliding_window[i].first->q().x() << ", ";
            std::cout << _sliding_window[i].first->q().y() << ", ";
            std::cout << _sliding_window[i].first->q().z() << std::endl;
            std::cout << "t = " << _sliding_window[i].first->p().transpose() << std::endl;
            std::cout << "v = " << _sliding_window[i].first->v().transpose() << std::endl;
            std::cout << "ba = " << _sliding_window[i].first->ba().transpose() << std::endl;
            std::cout << "bg = " << _sliding_window[i].first->bg().transpose() << std::endl;
        }
        std::cout << "i = " << _sliding_window.size() << ":" << std::endl;
        std::cout << "q = " << _frame->q().w() << ", ";
        std::cout << _frame->q().x() << ", ";
        std::cout << _frame->q().y() << ", ";
        std::cout << _frame->q().z() << std::endl;
        std::cout << "t = " << _frame->p().transpose() << std::endl;
        std::cout << "v = " << _frame->v().transpose() << std::endl;
        std::cout << "ba = " << _frame->ba().transpose() << std::endl;
        std::cout << "bg = " << _frame->bg().transpose() << std::endl;
#endif
        _t_ic[0] = Eigen::Map<Eigen::Vector3d>(_ext_params_bp[0]);
        return true;
    }

    bool Estimator::search_relative_pose(Mat33 &r, Vec3 &t, unsigned long &imu_index) {
        TicToc t_r;

        for (unsigned long i = 0; i < _sliding_window.size(); ++i) {
            if (compute_essential_matrix(r, t, _sliding_window[i].first, _frame)) {
                imu_index = i;
#ifdef PRINT_INFO
                std::cout << "imu_index = " << imu_index << std::endl;
                std::cout << "find essential_matrix: " << t_r.toc() << std::endl;
#endif
                return true;
            }
        }

#ifdef PRINT_INFO
        std::cout << "find essential_matrix: " << t_r.toc() << " ms" << std::endl;
#endif
        return false;
    }

    bool Estimator::stereo_visual_initialize(Qd *q_wi_init, Vec3 *t_wi_init) {
        if (_stream.size() < 2) {
            return false;
        }

        Eigen::Vector3d t_wi = Eigen::Vector3d::Zero();
        Eigen::Quaterniond q_wi = Eigen::Quaterniond::Identity();
        if (t_wi_init) {
            t_wi = *t_wi_init;
        }
        if (q_wi_init) {
            q_wi = *q_wi_init;
        }

        _stream.front().first->p() = t_wi;
        _stream.front().first->q() = q_wi;
        _stream.front().first->is_initialized = true;

        // 双目三角化
        global_triangulate_with(_stream.front().first);

        // PnP, Triangulate, PnP, Triangulate, ...
        for (auto frame_it = ++_stream.begin(); frame_it != _stream.end(); ++frame_it) {
            // 单目PnP
            iter_pnp(frame_it->first, &q_wi, &t_wi);
            t_wi = frame_it->first->p();
            q_wi = frame_it->first->q();

            // 双目三角化
            global_triangulate_with(frame_it->first);
        }

        // 转到local系
        for (auto &landmark_it : _landmarks) {
            landmark_it.second->from_global_to_local(_q_ic[0], _t_ic[0]);
        }

#ifdef PRINT_INFO
        std::cout << "done stereo visual initialize" << std::endl;
        for (unsigned long i = 0; i < _sliding_window.size(); ++i) {
            std::cout << "i = " << i << ":" << std::endl;
            std::cout << "q = " << _sliding_window[i].first->q().w() << ", ";
            std::cout << _sliding_window[i].first->q().x() << ", ";
            std::cout << _sliding_window[i].first->q().y() << ", ";
            std::cout << _sliding_window[i].first->q().z() << std::endl;
            std::cout << "t = " << _sliding_window[i].first->p().transpose() << std::endl;
            std::cout << "v = " << _sliding_window[i].first->v().transpose() << std::endl;
            std::cout << "ba = " << _sliding_window[i].first->ba().transpose() << std::endl;
            std::cout << "bg = " << _sliding_window[i].first->bg().transpose() << std::endl;
        }
        std::cout << "i = " << _sliding_window.size() << ":" << std::endl;
        std::cout << "q = " << _frame->q().w() << ", ";
        std::cout << _frame->q().x() << ", ";
        std::cout << _frame->q().y() << ", ";
        std::cout << _frame->q().z() << std::endl;
        std::cout << "t = " << _frame->p().transpose() << std::endl;
        std::cout << "v = " << _frame->v().transpose() << std::endl;
        std::cout << "ba = " << _frame->ba().transpose() << std::endl;
        std::cout << "bg = " << _frame->bg().transpose() << std::endl;
#endif

        return true;
    }
}