//
// Created by Cain on 2024/1/11.
//

#include "estimator.h"
#include "vertex/vertex_inverse_depth.h"
#include "vertex/vertex_pose.h"
#include "vertex/vertex_motion.h"
#include "edge/edge_reprojection.h"
#include "edge/edge_imu.h"

#include <array>
#include <memory>
#include <random>
#include <iostream>
#include <ostream>
#include <fstream>

//#ifndef USE_OPENMP
//#define NUM_THREADS 1
//#endif

namespace vins {
    using namespace graph_optimization;
    using namespace std;

    Estimator::Estimator() {
        _project_sqrt_info = FOCAL_LENGTH / 1.5 * Eigen::Matrix2d::Identity();
        MatXX information = _project_sqrt_info.transpose() * _project_sqrt_info;

        _ext_params.resize(NUM_OF_CAM);
        for (auto &ext_param : _ext_params) {
            ext_param = new double[7]{0};
            ext_param[6] = 1.;
            _t_ic.emplace_back(ext_param);
            _q_ic.emplace_back(ext_param + 3);
        }

        _ext_params_bp.resize(NUM_OF_CAM);
        for (auto &ext_param_bp : _ext_params_bp) {
            ext_param_bp = new double[7]{0};
            ext_param_bp[6] = 1.;
        }

        // 外参顶点
        _vertex_ext_vec.resize(_ext_params.size());
        for (size_t i = 0; i < _vertex_ext_vec.size(); ++i) {
            _vertex_ext_vec[i].parameters() = _ext_params[i];
        }

        // pose顶点
        _vertex_pose_vec.resize(WINDOW_SIZE + 1);

        // motion顶点
        _vertex_motion_vec.resize(WINDOW_SIZE + 1);

        // landmark顶点
        _num_landmarks.resize(NUM_THREADS, 0);
        _vertex_landmarks_vec.resize(NUM_THREADS);
        for (auto &vertex_landmarks : _vertex_landmarks_vec) {
            vertex_landmarks.resize(NUM_OF_F);
        }

        // 预积分边
        _edge_imu.resize(WINDOW_SIZE + 1);
        for (auto &edge_imu : _edge_imu) {
            edge_imu.vertices().resize(4);
        }

        // 重投影边
        _num_edges_12.resize(NUM_THREADS, 0);
        _edges_12_vec.resize(NUM_THREADS);
        for (auto &edges_12 : _edges_12_vec) {
            edges_12.resize(NUM_OF_F * WINDOW_SIZE);
            for (auto &edge_12 : edges_12) {
                edge_12.vertices().resize(3);
                edge_12.set_information(information);
            }
        }

        _num_edges_21.resize(NUM_THREADS, 0);
        _edges_21_vec.resize(NUM_THREADS);
        for (auto &edges_21 : _edges_21_vec) {
            edges_21.resize(NUM_OF_F);
            for (auto &edge_21 : edges_21) {
                edge_21.vertices().resize(4);
                edge_21.set_information(information);
            }
        }

        _num_edges_22.resize(NUM_THREADS, 0);
        _edges_22_vec.resize(NUM_THREADS);
        for (auto &edges_22 : _edges_22_vec) {
            edges_22.resize(NUM_OF_F * WINDOW_SIZE);
            for (auto &edge_22 : edges_22) {
                edge_22.vertices().resize(5);
                edge_22.set_information(information);
            }
        }

        // 边缘化landmarks
        _marg_landmarks_vec.resize(NUM_THREADS);
        for (auto &marg_landmarks : _marg_landmarks_vec) {
            marg_landmarks.reserve(NUM_OF_F);
        }

        // 边缘化edges
        _marg_edges_vec.resize(NUM_THREADS);
        for (auto &marg_edges : _marg_edges_vec) {
            marg_edges.reserve(NUM_OF_F * WINDOW_SIZE);
        }

        // 边缘化landmarks
        _marg_landmarks.reserve(NUM_OF_F);

        // 边缘化edges
        _marg_edges.reserve(NUM_OF_F * WINDOW_SIZE);

        // 预积分
        _pre_integral_vec.resize(WINDOW_SIZE + 1);

        // 路标map
        _landmarks.reserve(NUM_OF_F);

        // 路标vector
        _landmarks_vector.reserve(NUM_OF_F);

        // 可重投影的路标
        _suitable_landmarks.reserve(NUM_OF_F);

        // 待删除的路标id
        _landmark_erase_id_vec.resize(NUM_THREADS);
        for (auto &landmark_erase_id : _landmark_erase_id_vec) {
            landmark_erase_id.reserve(NUM_OF_F);
        }

        _problem.set_solver_type(graph_optimization::Problem::SolverType::LEVENBERG_MARQUARDT);
    }

    Estimator::~Estimator() {
        for (auto &ext_param : _ext_params) {
            delete [] ext_param;
        }

        for (auto &ext_param_bp : _ext_params_bp) {
            delete [] ext_param_bp;
        }
    }

    void Estimator::set_ext_param(unsigned int index, const Eigen::Vector3d &t_ic, const Eigen::Quaterniond &q_ic) {
        if (index >= _ext_params.size()) {
            std::cerr << "在设置外参时, index 超出相机个数" << std::endl;
            return;
        }
        _t_ic[index] = t_ic;
        _q_ic[index] = q_ic;
        std::memcpy(_ext_params_bp[index], _ext_params[index], 7 * sizeof(double));
    }

    bool Estimator::initialize() {
#ifdef PRINT_INFO
        std::cout << "running initialize" << std::endl;
#endif
        TicToc t_sfm;
#if NUM_OF_CAM > 1
        if (stereo_visual_initialize()) {
#else
        if (structure_from_motion()) {
#endif
            if (align_visual_to_imu()) {
                // 移除outlier的landmarks
                remove_outlier_landmarks();

//                // 移除未三角化的landmarks
//                remove_untriangulated_landmarks();

#ifdef PRINT_INFO
                cout << "Initialization finish!" << endl;
#endif
                return true;
            }
        }

        return false;
    }

    void Estimator::reset_initialization_flags() {
#ifdef USE_OPENMP
#pragma omp parallel for num_threads(NUM_THREADS)
#endif
        for (size_t i = 0; i < _landmarks_vector.size(); ++i) {
            _landmarks_vector[i]->is_outlier = false;
            _landmarks_vector[i]->is_triangulated = false;
        }

        for (auto &frame_it : _stream) {
            frame_it.first->is_initialized = false;
        }
    }

    void Estimator::prepare_landmarks() {
        // 计算suitable landmark
        if (_landmarks_vector.capacity() < 2 * _landmarks.size()) {
            _landmarks_vector.reserve(3 * _landmarks.size());
        }
        _landmarks_vector.clear();

        if (_suitable_landmarks.capacity() < 2 * _landmarks.size()) {
            _suitable_landmarks.reserve(3 * _landmarks.size());
        }
        _suitable_landmarks.clear();

        for (auto &landmark_it : _landmarks) {
            auto landmark = landmark_it.second;
            _landmarks_vector.emplace_back(landmark);
            if (is_landmark_suitable(landmark_it.second)) {
                _suitable_landmarks.emplace_back(landmark);
            }
        }

        for (auto &landmark_erase_id : _landmark_erase_id_vec) {
            if (landmark_erase_id.capacity() < _landmarks.size()) {
                landmark_erase_id.reserve(2 * _landmarks.size());
            }
            landmark_erase_id.clear();
        }
    }

    void Estimator::backend() {
        prepare_landmarks();

        if (solver_flag == SolverFlag::INITIAL) {
            if (is_data_enough()) {
                if (initialize()) {
                    // 初始化后进行非线性优化
                    solver_flag = SolverFlag::OPTIMIZATION;
                    optimization();
                } else {
                    reset_initialization_flags();
                }
            }
        } else {
            optimization();
        }
        slide_window();
    }

    void Estimator::optimization() {
#ifdef PRINT_INFO
        std::cout << "running optimization" << std::endl;
#endif
        if (_sliding_window.empty()) {
            return;
        }

        TicToc t_triangulate, t_new_problem, t_solve_problem, t_optimization;
        double cost_triangulate, cost_new_problem, cost_solve_problem, cost_optimization;
        unsigned long num_imu_edges = 0, num_reproj_edges = 0;

        // 三角化
        t_triangulate.tic();
        for (auto &landmark : _suitable_landmarks) {
            local_triangulate_feature(landmark);
        }
        cost_triangulate = t_triangulate.toc();

        // 准备构建问题
        t_new_problem.tic();
        _problem.clear();

        _marg_landmarks.clear();
        for (auto &landmarks : _marg_landmarks_vec) {
            landmarks.clear();
        }

        _marg_edges.clear();
        for (auto &edges : _marg_edges_vec) {
            edges.clear();
        }

        for (auto &counter : _num_landmarks) {
            counter = 0;
        }
        for (auto &counter : _num_edges_12) {
            counter = 0;
        }
        for (auto &counter : _num_edges_21) {
            counter = 0;
        }
        for (auto &counter : _num_edges_22) {
            counter = 0;
        }

        // 记录被marg的imu
        unsigned int imu_marg;
        if (marginalization_flag == MarginalizationFlag::MARGIN_OLD) {
            imu_marg = 0;
        } else if (marginalization_flag == MarginalizationFlag::MARGIN_SECOND_NEW) {
            imu_marg = _sliding_window.size() - 1;
        } else {
            imu_marg = -1;
        }

        // 状态顶点与预积分存储在数组中，方便多线程的查询
        uint16_t frame_ordering = 0;
        for (auto it = _sliding_window.begin(); it != _sliding_window.end(); ++it) {
            it->first->ordering = frame_ordering;
            _vertex_pose_vec[frame_ordering].parameters() = it->first->state + Frame::STATE::PX;
            _vertex_motion_vec[frame_ordering].parameters() = it->first->state + Frame::STATE::VX;
            _pre_integral_vec[frame_ordering] = it->second.get();
            ++frame_ordering;
        }

        // 由于最新的frame还没加入到sliding_window中, 所以这里需要补充最新帧
        _frame->ordering = frame_ordering;
        _vertex_pose_vec[frame_ordering].parameters() = _frame->state + Frame::STATE::PX;
        _vertex_motion_vec[frame_ordering].parameters() = _frame->state + Frame::STATE::VX;
        _pre_integral_vec[frame_ordering] = _pre_integral_window.get();
        ++frame_ordering;

        // 外参
        for (auto &vertex_ext : _vertex_ext_vec) {
            vertex_ext.set_fixed();
            _problem.add_state_vertex(&vertex_ext);
        }

        // 顶点: frame
        for (unsigned int i = 0; i < frame_ordering; ++i) {
            _problem.add_state_vertex(&_vertex_pose_vec[i]);
#ifdef REDUCE_MOTION
            if (i + 2 >= frame_ordering) {
                _problem.add_state_vertex(&_vertex_motion_vec[i]);
            }
#else
            _problem.add_state_vertex(&_vertex_motion_vec[i]);
#endif
        }

#ifdef USE_IMU
        // 边: IMU预积分误差
#ifdef REDUCE_MOTION
        for (unsigned int i = frame_ordering - 1; i < frame_ordering; ++i) {
#else
        for (unsigned int i = 1; i < frame_ordering; ++i) {
#endif
            // 间隔太长的不考虑
            if (_pre_integral_vec[i]->get_sum_dt() > 10.0 || _pre_integral_vec[i]->get_dt_buf().empty()) {
                continue;
            }

            auto &imu_edge = _edge_imu[i];
            imu_edge.imu_integration() = _pre_integral_vec[i];
            imu_edge.set_vertex(&_vertex_pose_vec[i - 1], 0);
            imu_edge.set_vertex(&_vertex_motion_vec[i - 1], 1);
            imu_edge.set_vertex(&_vertex_pose_vec[i], 2);
            imu_edge.set_vertex(&_vertex_motion_vec[i], 3);
            _problem.add_imu_edge(&imu_edge);
            ++num_imu_edges;

            // 记录需要被边缘化的预积分边
            if (i == imu_marg || (i - 1) == imu_marg) {
                _marg_edges.emplace_back(&imu_edge);
            }
        }
#endif
        /*
         * 只有marg old的时候才会保留landmark相关的信息
         * 当marg new时则丢弃所有与landmark相关的信息
        */
        if (marginalization_flag == MarginalizationFlag::MARGIN_OLD) {
#ifdef USE_OPENMP
#pragma omp parallel for num_threads(NUM_THREADS)
            for (size_t n = 0; n < _suitable_landmarks.size(); ++n) {
                unsigned int thread_id = omp_get_thread_num();
#else
            for (size_t n = 0; n < _suitable_landmarks.size(); ++n) {
                unsigned int thread_id = 0;
#endif
                if (_num_landmarks[thread_id] >= NUM_OF_F) {
                    continue;
                }

                auto landmark = _suitable_landmarks[n];
                if (landmark->is_outlier || !landmark->is_triangulated) {
                    continue;
                }

                // 创建 landmark 顶点
                auto &vertex_landmark = _vertex_landmarks_vec[thread_id][_num_landmarks[thread_id]++];
                vertex_landmark.parameters() = &landmark->inv_depth;

                // 遍历所有的观测 (landmark所关联的frame), 计算视觉重投影误差
                auto &observations = landmark->observations;

                // host frame
                auto observation = observations.front();
                auto &point_i = observation.second->points[0];
                auto imu_i = observation.first->ordering;
                assert(imu_i + observations.size() - 1 <= _sliding_window.size());

                // 同frame下的左右目重投影误差
                for (size_t k = 1; k < observations.front().second->points.size(); ++k) {
                    auto &edge = _edges_12_vec[thread_id][_num_edges_12[thread_id]++];
                    edge.set_pt_0(point_i);
                    edge.set_pt_1(observation.second->points[k]);
                    edge.set_vertex(&vertex_landmark, 0);
                    edge.set_vertex(&_vertex_ext_vec[0], 1);
                    edge.set_vertex(&_vertex_ext_vec[k], 2);

                    if (imu_i == imu_marg) {
                        _marg_edges_vec[thread_id].emplace_back(&edge);
                    }
                }

                // 不同frame下的重投影误差
                for (unsigned int index = 1; index < observations.size(); ++index) {
                    unsigned int imu_j = imu_i + index;
                    observation = observations[index];
                    // 左目与左目
                    {
                        auto &edge = _edges_21_vec[thread_id][_num_edges_21[thread_id]++];
                        edge.set_pt_i(point_i);
                        edge.set_pt_j(observation.second->points[0]);
                        edge.set_vertex(&vertex_landmark, 0);
                        edge.set_vertex(&_vertex_pose_vec[imu_i], 1);
                        edge.set_vertex(&_vertex_pose_vec[imu_j], 2);
                        edge.set_vertex(&_vertex_ext_vec[0], 3);

                        if (imu_i == imu_marg || imu_j == imu_marg) {
                            _marg_edges_vec[thread_id].emplace_back(&edge);
                        }
                    }

                    // 左目与右目
                    for (size_t k = 1; k < observation.second->points.size(); ++k) {
                        auto &edge = _edges_22_vec[thread_id][_num_edges_22[thread_id]++];
                        edge.set_pt_i(point_i);
                        edge.set_pt_j(observation.second->points[k]);
                        edge.set_vertex(&vertex_landmark, 0);
                        edge.set_vertex(&_vertex_pose_vec[imu_i], 1);
                        edge.set_vertex(&_vertex_pose_vec[imu_j], 2);
                        edge.set_vertex(&_vertex_ext_vec[0], 3);
                        edge.set_vertex(&_vertex_ext_vec[k], 4);

                        if (imu_i == imu_marg || imu_j == imu_marg) {
                            _marg_edges_vec[thread_id].emplace_back(&edge);
                        }
                    }

                    if (imu_j == imu_marg) {
                        _marg_landmarks_vec[thread_id].emplace_back(&vertex_landmark);
                    }
                }
                if (imu_i == imu_marg) {
                    _marg_landmarks_vec[thread_id].emplace_back(&vertex_landmark);
                }
            }

            // 加顶点
            for (unsigned int n = 0; n < NUM_THREADS; ++n) {
                for (size_t k = 0; k < _num_landmarks[n]; ++k) {
                    _problem.add_landmark_vertex(&_vertex_landmarks_vec[n][k]);
                }
            }
            // 加边
            for (unsigned int n = 0; n < NUM_THREADS; ++n) {
                for (size_t k = 0; k < _num_edges_12[n]; ++k) {
                    _problem.add_reproj_edge(&_edges_12_vec[n][k]);
                    num_reproj_edges++;
                }
            }
            for (unsigned int n = 0; n < NUM_THREADS; ++n) {
                for (size_t k = 0; k < _num_edges_21[n]; ++k) {
                    _problem.add_reproj_edge(&_edges_21_vec[n][k]);
                    num_reproj_edges++;
                }
            }
            for (unsigned int n = 0; n < NUM_THREADS; ++n) {
                for (size_t k = 0; k < _num_edges_22[n]; ++k) {
                    _problem.add_reproj_edge(&_edges_22_vec[n][k]);
                    num_reproj_edges++;
                }
            }
            // 需要边缘化的landmark
            for (auto &landmarks : _marg_landmarks_vec) {
                for (auto &landmark : landmarks) {
                    _marg_landmarks.emplace_back(landmark);
                }
            }
            // 边缘化相关的边
            for (auto &edges : _marg_edges_vec) {
                for (auto &edge : edges) {
                    _marg_edges.emplace_back(edge);
                }
            }
        } else {
#ifdef USE_OPENMP
#pragma omp parallel for num_threads(NUM_THREADS)
            for (size_t n = 0; n < _suitable_landmarks.size(); ++n) {
                unsigned int thread_id = omp_get_thread_num();
#else
                for (size_t n = 0; n < _suitable_landmarks.size(); ++n) {
                unsigned int thread_id = 0;
#endif
                if (_num_landmarks[thread_id] >= NUM_OF_F) {
                    continue;
                }

                auto landmark = _suitable_landmarks[n];
                if (landmark->is_outlier || !landmark->is_triangulated) {
                    continue;
                }

                // 创建 landmark 顶点
                auto &vertex_landmark = _vertex_landmarks_vec[thread_id][_num_landmarks[thread_id]++];
                vertex_landmark.parameters() = &landmark->inv_depth;

                // 遍历所有的观测 (landmark所关联的frame), 计算视觉重投影误差
                auto &observations = landmark->observations;

                // host frame
                auto observation = observations.front();
                auto &point_i = observation.second->points[0];
                auto imu_i = observation.first->ordering;
                assert(imu_i + observations.size() - 1 <= _sliding_window.size());

                // 同frame下的左右目重投影误差
                for (size_t k = 1; k < observations.front().second->points.size(); ++k) {
                    auto &edge = _edges_12_vec[thread_id][_num_edges_12[thread_id]++];
                    edge.set_pt_0(point_i);
                    edge.set_pt_1(observation.second->points[k]);
                    edge.set_vertex(&vertex_landmark, 0);
                    edge.set_vertex(&_vertex_ext_vec[0], 1);
                    edge.set_vertex(&_vertex_ext_vec[k], 2);
                }

                // 不同frame下的重投影误差
                for (unsigned int index = 1; index < observations.size(); ++index) {
                    unsigned int imu_j = imu_i + index;
                    observation = observations[index];
                    // 左目与左目
                    {
                        auto &edge = _edges_21_vec[thread_id][_num_edges_21[thread_id]++];
                        edge.set_pt_i(point_i);
                        edge.set_pt_j(observation.second->points[0]);
                        edge.set_vertex(&vertex_landmark, 0);
                        edge.set_vertex(&_vertex_pose_vec[imu_i], 1);
                        edge.set_vertex(&_vertex_pose_vec[imu_j], 2);
                        edge.set_vertex(&_vertex_ext_vec[0], 3);
                    }

                    // 左目与右目
                    for (size_t k = 1; k < observation.second->points.size(); ++k) {
                        auto &edge = _edges_22_vec[thread_id][_num_edges_22[thread_id]++];
                        edge.set_pt_i(point_i);
                        edge.set_pt_j(observation.second->points[k]);
                        edge.set_vertex(&vertex_landmark, 0);
                        edge.set_vertex(&_vertex_pose_vec[imu_i], 1);
                        edge.set_vertex(&_vertex_pose_vec[imu_j], 2);
                        edge.set_vertex(&_vertex_ext_vec[0], 3);
                        edge.set_vertex(&_vertex_ext_vec[k], 4);
                    }
                }
            }

            // 加顶点
            for (unsigned int n = 0; n < NUM_THREADS; ++n) {
                for (size_t k = 0; k < _num_landmarks[n]; ++k) {
                    _problem.add_landmark_vertex(&_vertex_landmarks_vec[n][k]);
                }
            }
            // 加边
            for (unsigned int n = 0; n < NUM_THREADS; ++n) {
                for (size_t k = 0; k < _num_edges_12[n]; ++k) {
                    _problem.add_reproj_edge(&_edges_12_vec[n][k]);
                }
            }
            for (unsigned int n = 0; n < NUM_THREADS; ++n) {
                for (size_t k = 0; k < _num_edges_21[n]; ++k) {
                    _problem.add_reproj_edge(&_edges_21_vec[n][k]);
                }
            }
            for (unsigned int n = 0; n < NUM_THREADS; ++n) {
                for (size_t k = 0; k < _num_edges_22[n]; ++k) {
                    _problem.add_reproj_edge(&_edges_22_vec[n][k]);
                }
            }
        }
        cost_new_problem = t_new_problem.toc();

        t_solve_problem.tic();
        // 保留第0帧数的pose
        Eigen::Quaterniond q_0 = _vertex_pose_vec[0].q();
        Eigen::Vector3d p_0 = _vertex_pose_vec[0].p();
        /*
        * 由于marg old的计算量大于marg new
        * 所以marg old时图优化的迭代次数应该小于marg new时
        */
        if (marginalization_flag == MarginalizationFlag::MARGIN_OLD) {
            // _vertex_pose_vec[0].set_fixed();
            _problem.solve(4);
            // _vertex_pose_vec[0].set_fixed(false);
#ifdef PRINT_INFO
            std::cout << "done problem solve" << std::endl;
#endif
            if (is_sliding_window_full()) {
#ifdef REDUCE_MOTION
                _problem.marginalize(&_vertex_pose_vec[0], nullptr, _marg_landmarks, _marg_edges);
#else
                _problem.marginalize(&_vertex_pose_vec[0], &_vertex_motion_vec[0], _marg_landmarks, _marg_edges);
#endif
            }
        } else if (marginalization_flag == MarginalizationFlag::MARGIN_SECOND_NEW){
            // _vertex_pose_vec[0].set_fixed();
            _problem.solve(5);
            // _vertex_pose_vec[0].set_fixed(false);
#ifdef PRINT_INFO
            std::cout << "done problem solve" << std::endl;
#endif
            if (is_sliding_window_full()) {
                _problem.marginalize(&_vertex_pose_vec[_sliding_window.size() - 1], &_vertex_motion_vec[_sliding_window.size() - 1], _marg_landmarks, _marg_edges);
            }
        } else {
            // _vertex_pose_vec[0].set_fixed();
            _problem.solve(5);
            // _vertex_pose_vec[0].set_fixed(false);
#ifdef PRINT_INFO
            std::cout << "done problem solve" << std::endl;
#endif
        }

        // // 还原第0帧
        // Eigen::Vector3d v_nav = q_0.toRotationMatrix().col(0);
        // Eigen::Vector3d v_nav_est = _vertex_pose_vec[0].q().toRotationMatrix().col(0);
        // double norm2 = sqrt(v_nav_est.head<2>().squaredNorm() * v_nav.head<2>().squaredNorm());
        // double sin_psi = v_nav_est(0) * v_nav(1) - v_nav_est(1) * v_nav(0);
        // double cos_psi = v_nav_est(0) * v_nav(0) + v_nav_est(1) * v_nav(1);
        // Eigen::Quaterniond dq(norm2 + cos_psi, 0., 0., sin_psi);
        // dq.normalize();
        // Eigen::Matrix3d dR = dq.toRotationMatrix();
        // for (unsigned int i = 0; i < frame_ordering; ++i) {
        //     _vertex_pose_vec[i].q() = (dq * _vertex_pose_vec[i].q()).normalized();
        //     _vertex_pose_vec[i].p() = dR * (_vertex_pose_vec[i].p() - _vertex_pose_vec[0].p()) + p_0;
        //     _vertex_motion_vec[i].v() = dR * (_vertex_motion_vec[i].v());
        // }

        // 还原第0帧
        Eigen::Vector3d ypr_0 = Utility::R2ypr(q_0.toRotationMatrix());
        Eigen::Vector3d ypr_oldest = Utility::R2ypr(_vertex_pose_vec[0].q().toRotationMatrix());
        double dyaw = ypr_0.x() - ypr_oldest.x();
        Eigen::Matrix3d dR = Utility::ypr2R(Eigen::Vector3d(dyaw, 0, 0));
        if (abs(abs(ypr_0.y()) - 90.) < 1. || abs(abs(ypr_oldest.y()) - 90.) < 1.) {
            dR = (q_0 * _vertex_pose_vec[0].q().inverse()).toRotationMatrix();
        }
        auto dq = Eigen::Quaterniond(dR);
        // for (auto &frame : _stream) {
        //     frame.first->q() = (dq * frame.first->q()).normalized();
        //     frame.first->p() = dR * (frame.first->p() - _vertex_pose_vec[0].p()) + p_0;
        //     frame.first->v() = dR * frame.first->v();
        // }
        for (unsigned int i = 0; i < frame_ordering; ++i) {
            _vertex_pose_vec[i].q() = (dq * _vertex_pose_vec[i].q()).normalized();
            _vertex_pose_vec[i].p() = dR * (_vertex_pose_vec[i].p() - _vertex_pose_vec[0].p()) + p_0;
            _vertex_motion_vec[i].v() = dR * _vertex_motion_vec[i].v();
        }

        cost_solve_problem = t_solve_problem.toc();

        // 寻找超出阈值的重投误差对应的landmarks
        search_outlier_landmarks();

        // 移除深度值小于0的点
        remove_outlier_landmarks(true);

        cost_optimization = t_optimization.toc();

#ifdef PRINT_INFO
        std::cout << "num_imu_edges = " << num_imu_edges << std::endl;
        std::cout << "num_reproj_edges = " << num_reproj_edges << std::endl;
        std::cout << "done triangulate, cost = " << cost_triangulate << " ms" << std::endl;
        std::cout << "done new problem, cost = " << cost_new_problem << " ms" << std::endl;
        std::cout << "done solve problem, cost = " << cost_solve_problem << " ms" << std::endl;
        std::cout << "done optimization, cost = " << cost_optimization << " ms" << std::endl;

        if (_problem.get_h_prior().rows() > 0) {
            std::cout << "----------- update bprior -------------\n";
            std::cout << "             before: " << _problem.get_b_prior().norm() << std::endl;
            std::cout << "             after: " << _problem.get_b_prior().norm() << std::endl;
        }
#endif
    }

    void Estimator::slide_window() {
#ifdef PRINT_INFO       
        std::cout << "slide_window" << std::endl;
#endif        
        TicToc t_slide_window;

        // 滑窗若不满, 则不移除最老帧
        if (marginalization_flag == MarginalizationFlag::MARGIN_OLD) {
            // 移除stream中, 比sliding_window中最老帧还要老的帧
            auto time_us = _sliding_window.front().first->time_us;
            for (auto it = _stream.begin(); it != _stream.end(); ) {
                if (it->first->time_us > time_us) {
                    break;
                }
                // 删除
                it->first.reset();
                it->second.reset();
                it = _stream.erase(it);
            }
            // 由于VINS的landmark以深度为参数, 所以host frame改变时需要重新计算深度
            if (solver_flag == SolverFlag::OPTIMIZATION) {
                // 旧的host
                Eigen::Vector3d t_wci_w = _sliding_window.front().first->p() + _sliding_window.front().first->q() * _t_ic[0];
                Eigen::Quaterniond q_wci = _sliding_window.front().first->q() * _q_ic[0];
                // 以新的host, 重新计算深度
#ifdef USE_OPENMP
#pragma omp parallel for num_threads(NUM_THREADS)
                for (size_t n = 0; n < _landmarks_vector.size(); ++n) {
                    unsigned int i = omp_get_thread_num();
#else
                for (size_t n = 0; n < _landmarks_vector.size(); ++n) {
                    unsigned int i = 0;
#endif
                    auto landmark = _landmarks_vector[n];
                    if (landmark->observations.empty() || _sliding_window.empty()) {
                        continue;
                    }
                    if (landmark->observations.front().first.get() != _sliding_window.front().first.get()) {
                        continue;
                    }
                    // 若只在最后一帧被观测到, 则直接把landmark删除
                    if (landmark->observations.size() < 2) {
                        _landmark_erase_id_vec[i].emplace_back(landmark->id());
                        // 移除观测
                        landmark->observations.clear();
                        continue;
                    }
#ifndef DELAY_ERASE
                    // 若只在sliding_window中的最后两帧被观测到, 则直接把landmark删除
                    if (landmark->observations.size() == 2 && _sliding_window.size() > 1) { // _frame->features.find(landmark->id()) == _frame->features.end()
                        _landmark_erase_id_vec[i].emplace_back(landmark->id());
                        // 移除观测
                        landmark->observations.clear();
                        continue;
                    }
#endif
                    // 新的host
                    Eigen::Vector3d t_wcj_w = _sliding_window[1].first->p() + _sliding_window[1].first->q() * _t_ic[0];
                    Eigen::Quaterniond q_wcj = _sliding_window[1].first->q() * _q_ic[0];

                    // 从i重投影到j
                    Vec3 p_cif_ci = landmark->observations.back().second->points[0] / landmark->inv_depth;
                    Vec3 p_wf_w = q_wci * p_cif_ci + t_wci_w;
                    Vec3 p_cjf_cj = q_wcj.inverse() * (p_wf_w - t_wcj_w);
                    double depth = p_cjf_cj.z();
                    // TODO: 若深度不合理, 应该设置为outlier
                    if (depth < 0.01) {
                        landmark->is_outlier = true;
                    }
                    landmark->inv_depth = 1. / depth;

                    // 移除观测
                    landmark->observations.pop_front();
                }
            } else {
#ifdef USE_OPENMP
#pragma omp parallel for num_threads(NUM_THREADS)
                for (size_t n = 0; n < _landmarks_vector.size(); ++n) {
                    unsigned int i = omp_get_thread_num();
#else
                for (size_t n = 0; n < _landmarks_vector.size(); ++n) {
                    unsigned int i = 0;
#endif
                    auto landmark = _landmarks_vector[n];
                    if (landmark->observations.empty() || _sliding_window.empty()) {
                        continue;
                    }
                    if (landmark->observations.front().first.get() != _sliding_window.front().first.get()) {
                        continue;
                    }
                    // 若只在最后一帧被观测到, 则直接把landmark删除
                    if (landmark->observations.size() < 2) {
                        _landmark_erase_id_vec[i].emplace_back(landmark->id());
                        // 移除观测
                        landmark->observations.clear();
                        continue;
                    }
#ifndef DELAY_ERASE
                    // 若只在sliding_window中的最后两帧被观测到, 则直接把landmark删除
                    if (landmark->observations.size() == 2 && _sliding_window.size() > 1) {
                        _landmark_erase_id_vec[i].emplace_back(landmark->id());
                        // 移除观测
                        landmark->observations.clear();
                        continue;
                    }
#endif
                    // 移除观测
                    landmark->observations.pop_front();
                }
            }
            // 删除最老帧, 同时把最新帧加入到sliding_window中
            _sliding_window.front().second.reset();
            _sliding_window.pop_front();
            _sliding_window.emplace_back(_frame, _pre_integral_window);

            // TODO: 是否需要对 _stream 做处理 ? 因为 feature 事实上还存在于 _stream 的 frame 中
            // 删除不再被使用的landmark
            for (auto &landmark_erase_id : _landmark_erase_id_vec) {
                for (auto &id : landmark_erase_id) {
                    _landmarks.erase(id);
                    _sliding_window.front().first->features.erase(id);
                }
            }

        } else if (marginalization_flag == MarginalizationFlag::MARGIN_SECOND_NEW) {
#ifdef USE_OPENMP
#pragma omp parallel for num_threads(NUM_THREADS)
            for (size_t n = 0; n < _landmarks_vector.size(); ++n) {
                unsigned int i = omp_get_thread_num();
#else
            for (size_t n = 0; n < _landmarks_vector.size(); ++n) {
                unsigned int i = 0;
#endif
                auto landmark = _landmarks_vector[n];
                if (landmark->observations.empty() || _sliding_window.empty()) {
                    continue;
                }
                if (_sliding_window.back().first->features.find(landmark->id()) == _sliding_window.back().first->features.end()) {
                    continue;
                }

                auto obs_times = landmark->observations.size();
                // 若只在次新帧被观测到, 则直接删除
                if (obs_times < 2) {
                    _landmark_erase_id_vec[i].emplace_back(landmark->id());
                    // 移除观测
                    landmark->observations.clear();
                    continue;
                }
#ifndef DELAY_ERASE
                if (obs_times == 2 && landmark->observations.back().first.get() == _sliding_window.back().first.get()) {
                    _landmark_erase_id_vec[i].emplace_back(landmark->id());
                    // 移除观测
                    landmark->observations.clear();
                    continue;
                }
#endif
                // 移除观测
                landmark->observations[obs_times - 2] = landmark->observations.back();
                landmark->observations.pop_back();
            }

            // 叠加imu数据
            auto &dt_buf = _pre_integral_window->get_dt_buf();
            auto &acc_buf = _pre_integral_window->get_acc_buf();
            auto &gyro_buf = _pre_integral_window->get_gyro_buf();
            auto pre_integral = _sliding_window.back().second;
            for (size_t i = 0; i < _pre_integral_window->size(); ++i) {
                pre_integral->push_back(dt_buf[i], acc_buf[i], gyro_buf[i]);
            }
            _pre_integral_window.reset();

            // 删除次新帧, 同时把最新帧加入到sliding_window中
            _sliding_window.pop_back();
            _sliding_window.emplace_back(_frame, pre_integral);

            // 删除不再被使用的landmark
            for (auto &landmark_erase_id : _landmark_erase_id_vec) {
                for (auto &id : landmark_erase_id) {
                    _landmarks.erase(id);
                    _sliding_window[_sliding_window.size() - 2].first->features.erase(id);
                }
            }
        } else {
            _sliding_window.emplace_back(_frame, _pre_integral_window);
        }

//        for (auto &frame_it : _sliding_window) {
//            std::cout << frame_it.first->time_us << std::endl;
//        }

        _pre_integral_window = std::make_shared<vins::IMUIntegration>(_acc_latest, _gyro_latest,
                                                                      _sliding_window.back().first->ba(),
                                                                      _sliding_window.back().first->bg());

#ifdef PRINT_INFO
        std::cout << "t_slide_window = " << t_slide_window.toc() << std::endl;
#endif
    }

    bool Estimator::remove_outlier_landmarks(bool lazy) {
        unsigned long outlier_landmarks = 0;
        if (lazy) {
#ifdef USE_OPENMP
#pragma omp parallel for num_threads(NUM_THREADS) reduction(+:outlier_landmarks)
#endif
            for (unsigned long i = 0; i < _landmarks_vector.size(); ++i) {
                if (_landmarks_vector[i]->is_triangulated && _landmarks_vector[i]->inv_depth < 0.) {
                    _landmarks_vector[i]->is_outlier = true;
                    ++outlier_landmarks;
                }
                if (_landmarks_vector[i]->is_outlier) {
                    ++outlier_landmarks;
                }
            }
        } else {
            for (auto &landmark_erase_id : _landmark_erase_id_vec) {
                landmark_erase_id.clear();
            }
#ifdef USE_OPENMP
#pragma omp parallel for num_threads(NUM_THREADS) reduction(+:outlier_landmarks)
            for (unsigned long i = 0; i < _landmarks_vector.size(); ++i) {
                unsigned int index = omp_get_thread_num();
#else
            for (unsigned long i = 0; i < _landmarks_vector.size(); ++i) {
                unsigned int index = 0;
#endif
                if (_landmarks_vector[i]->is_triangulated && _landmarks_vector[i]->inv_depth < 0.) {
                    _landmarks_vector[i]->is_outlier = true;
                }
                if (_landmarks_vector[i]->is_outlier) {
                    _landmarks_vector[i]->observations.clear();
                    _landmark_erase_id_vec[index].emplace_back(_landmarks_vector[i]->id());
                    ++outlier_landmarks;
                }
            }

            // 删除outlier的landmark
            for (auto &landmark_erase_id : _landmark_erase_id_vec) {
                for (auto &id : landmark_erase_id) {
                    _landmarks.erase(id);
                    for (auto &frame : _sliding_window) {
                        frame.first->features.erase(id);
                    }
                }
            }

#ifdef PRINT_INFO
            std::cout << "There are " << outlier_landmarks << " outlier landmarks in " << _landmarks.size() << std::endl;
#endif
        }

        return true;
    }

    bool Estimator::remove_untriangulated_landmarks(bool lazy) {
        unsigned long untriangulated_landmarks = 0;
        if (lazy) {
#ifdef USE_OPENMP
#pragma omp parallel for num_threads(NUM_THREADS) reduction(+:untriangulated_landmarks)
#endif
            for (unsigned long i = 0; i < _landmarks_vector.size(); ++i) {
                if (_landmarks_vector[i]->is_triangulated) {
                    _landmarks_vector[i]->is_outlier = true;
                    ++untriangulated_landmarks;
                }
            }
        } else {
            for (auto &landmark_erase_id : _landmark_erase_id_vec) {
                landmark_erase_id.clear();
            }

#ifdef USE_OPENMP
#pragma omp parallel for num_threads(NUM_THREADS) reduction(+:untriangulated_landmarks)
            for (unsigned long i = 0; i < _landmarks_vector.size(); ++i) {
                unsigned int index = omp_get_thread_num();
#else
            for (unsigned long i = 0; i < _landmarks_vector.size(); ++i) {
                unsigned int index = 0;
#endif
                if (_landmarks_vector[i]->is_triangulated) {
                    _landmarks_vector[i]->observations.clear();
                    _landmark_erase_id_vec[index].emplace_back(_landmarks_vector[i]->id());
                    ++untriangulated_landmarks;
                }
            }

            // 删除untriangulated的landmark
            for (auto &landmark_erase_id : _landmark_erase_id_vec) {
                for (auto &id : landmark_erase_id) {
                    _landmarks.erase(id);
                    for (auto &frame : _sliding_window) {
                        frame.first->features.erase(id);
                    }
                }
            }
        }


#ifdef PRINT_INFO
        std::cout << "There are " << untriangulated_landmarks << " untriangulated landmarks in " << _landmarks.size() << std::endl;
#endif

        return true;
    }

    void Estimator::search_outlier_landmarks(unsigned int iteration) {
        if (_problem._reproj_edges.empty()) {
            return;
        }

        double chi2_th = 3.841;
        unsigned int cnt_outlier, cnt_inlier;
        for (unsigned int i = 0; i < iteration; ++i) {
            cnt_outlier = 0;
            cnt_inlier = 0;
            for (const auto &edge : _problem._reproj_edges) {
                if (edge->get_chi2() > chi2_th) {
                    ++cnt_outlier;
                } else {
                    ++cnt_inlier;
                }
            }

            double inlier_ratio = double(cnt_inlier) / double(cnt_outlier + cnt_inlier);
            if (inlier_ratio > 0.5) {
                break;
            } else {
                chi2_th *= 2.;
            }
        }

        for (const auto &edge : _problem._reproj_edges) {
            if (edge->get_chi2() > chi2_th) {
                edge->vertices()[0]->parameters()[0] = -1.;  // 把逆深度值设置为负数
            }
        }
    }

    std::vector<Eigen::Vector3d> Estimator::get_positions() const {
        std::vector<Eigen::Vector3d> positions;
        positions.reserve(_sliding_window.size());
        for (auto it = _sliding_window.begin(); it != _sliding_window.end(); ++it) {
            positions.emplace_back(it->first->p());
        }
        return positions;
    }
}