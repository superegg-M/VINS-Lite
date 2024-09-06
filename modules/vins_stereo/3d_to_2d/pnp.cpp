//
// Created by Cain on 2024/4/28.
//

#include "../estimator.h"

#include "graph_optimization/eigen_types.h"

#include <array>
#include <memory>
#include <random>
#include <iostream>
#include <ostream>
#include <fstream>

namespace vins {
    using namespace graph_optimization;
    using namespace std;

    bool Estimator::pnp(const std::shared_ptr<Frame> &frame_i, Qd *q_wi_init, Vec3 *t_wi_init, unsigned int num_iters) {
        constexpr static unsigned long th_count = 6;
        TicToc pnp_t;

        // 初始化
        if (q_wi_init) {
            frame_i->q() = *q_wi_init;
        }
        if (t_wi_init) {
            frame_i->p() = *t_wi_init;
        }

        // 相机外参
        Vec3 t_ic = _t_ic[0];
        Qd q_ic = _q_ic[0];

        // 问题构建
        unsigned int num_landmarks = 0;
        VertexPose* vertex_pose = new VertexPose(frame_i->state + Frame::STATE::PX);
        Problem problem;
        problem.add_vertex(vertex_pose);
        for (auto &feature : frame_i->features) {
            auto &&landmark_it = _landmarks.find(feature.first);
            if (landmark_it == _landmarks.end()) {
                // 如果传入的是 _stream 中的 frame, 会存在 landmark 已不在 _landmarks 中的情况
//                std::cerr << "Error: feature not in feature_map when running pnp" << std::endl;
                continue;
            }

            if (landmark_it->second->is_outlier) {
                continue;
            }

            if (!landmark_it->second->is_triangulated) {
                continue;
            }

            // 重投影edge
            EdgePnP* edge_pnp = new EdgePnP(feature.second->points[0], landmark_it->second->position);
            edge_pnp->set_translation_imu_from_camera(q_ic, t_ic);
            edge_pnp->add_vertex(vertex_pose);
            problem.add_edge(edge_pnp);

            ++num_landmarks;
        }

        // 特征点个数必须大于一定数量
        if (num_landmarks < th_count) {
            frame_i->is_initialized = false;
            delete vertex_pose;
            for (auto &edge : problem.edges()) {
                delete edge;
            }
            return false;
        }

        problem.set_solver_type(graph_optimization::Problem::SolverType::LEVENBERG_MARQUARDT);
        problem.solve(num_iters);

#ifdef PRINT_INFO
        std::cout << "pnp takes " << pnp_t.toc() << " ms" << std::endl;
        std::cout << "pnp: q = " << frame_i->q().w() << ", " << frame_i->q().x() << ", " << frame_i->q().y() << ", " << frame_i->q().z() << std::endl;
        std::cout << "pnp: t = " << frame_i->p().transpose() << std::endl;
#endif
        frame_i->is_initialized = true;
        delete vertex_pose;
        for (auto &edge : problem.edges()) {
            delete edge;
        }
        return true;
    }

    bool Estimator::pnp_local(const std::shared_ptr<Frame> &frame_i, Qd *q_wi_init, Vec3 *t_wi_init, unsigned int num_iters) {
        constexpr static unsigned long th_count = 6;
        TicToc pnp_t;

        // 初始化
        if (q_wi_init) {
            frame_i->q() = *q_wi_init;
        }
        if (t_wi_init) {
            frame_i->p() = *t_wi_init;
        }

        // 相机外参
        Vec3 t_ic = _t_ic[0];
        Qd q_ic = _q_ic[0];

        // 问题构建
        unsigned int num_landmarks = 0;
        VertexPose* vertex_pose = new VertexPose(frame_i->state + Frame::STATE::PX);
        Problem problem;
        problem.add_vertex(vertex_pose);
        for (auto &feature : frame_i->features) {
            auto &&landmark_it = _landmarks.find(feature.first);
            if (landmark_it == _landmarks.end()) {
                // 如果传入的是 _stream 中的 frame, 会存在 landmark 已不在 _landmarks 中的情况
//                std::cerr << "Error: feature not in feature_map when running pnp" << std::endl;
                continue;
            }

            if (landmark_it->second->is_outlier) {
                continue;
            }

            if (!landmark_it->second->is_triangulated) {
                continue;
            }

            auto &observations = landmark_it->second->observations;
            if (observations.size() < 2) {
                continue;
            }

            // 计算特征点的世界坐标
            auto observation_host = observations.front();
            Vec3 p_camera_host = observation_host.second->points[0] / landmark_it->second->inv_depth;
            Vec3 p_imu_host = _q_ic[0] * p_camera_host + _t_ic[0];
            landmark_it->second->position = observation_host.first->q() * p_imu_host + observation_host.first->p();

            // 重投影edge
            EdgePnP* edge_pnp = new EdgePnP(feature.second->points[0], landmark_it->second->position);
            edge_pnp->set_translation_imu_from_camera(q_ic, t_ic);
            edge_pnp->add_vertex(vertex_pose);
            problem.add_edge(edge_pnp);

            ++num_landmarks;
        }

        // 特征点个数必须大于一定数量
        if (num_landmarks < th_count) {
            frame_i->is_initialized = false;
            delete vertex_pose;
            for (auto &edge : problem.edges()) {
                delete edge;
            }
            return false;
        }

        problem.set_solver_type(graph_optimization::Problem::SolverType::LEVENBERG_MARQUARDT);
        problem.solve(num_iters);

#ifdef PRINT_INFO
        std::cout << "pnp takes " << pnp_t.toc() << " ms" << std::endl;
        std::cout << "pnp: q = " << frame_i->q().w() << ", " << frame_i->q().x() << ", " << frame_i->q().y() << ", " << frame_i->q().z() << std::endl;
        std::cout << "pnp: t = " << frame_i->p().transpose() << std::endl;
#endif
        frame_i->is_initialized = true;
        delete vertex_pose;
        for (auto &edge : problem.edges()) {
            delete edge;
        }
        return true;
    }
}