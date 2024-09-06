//
// Created by Cain on 2024/6/12.
//

#ifndef GRAPH_OPTIMIZATION_FRAME_H
#define GRAPH_OPTIMIZATION_FRAME_H

#include <iostream>
#include <memory>
#include <unordered_map>
#include <Eigen/Core>
#include <Eigen/Geometry>
#include "../parameters.h"
#include "feature.h"

namespace vins {
    class Frame;
    class Landmark;
    class Feature;

    class Frame {
    public:
        EIGEN_MAKE_ALIGNED_OPERATOR_NEW;
        enum STATE {
            PX = 0,
            PY,
            PZ,
            QX,
            QY,
            QZ,
            QW,
            VX,
            VY,
            VZ,
            BAX,
            BAY,
            BAZ,
            BGX,
            BGY,
            BGZ,

            DIM
        };

        Frame() {
            state[STATE::QW] = 1.;
        }

        explicit Frame(double *state_init) {
            std::memcpy(state, state_init, STATE::DIM * sizeof(double));
        }

        ~Frame() {
            delete [] state_first;
            state_first = nullptr;
        }

        void record_to_state_first() {
            if (!state_first) {
                state_first = new double [STATE::DIM];
                std::memcpy(state_first, state, STATE::DIM * sizeof(double));
            }
        }

        Eigen::Map<Eigen::Quaterniond> q() { return Eigen::Map<Eigen::Quaterniond>(state + STATE::QX); }
        Eigen::Map<Eigen::Vector3d> p() { return Eigen::Map<Eigen::Vector3d>(state + STATE::PX); }
        Eigen::Map<Eigen::Vector3d> v() { return Eigen::Map<Eigen::Vector3d>(state + STATE::VX); }
        Eigen::Map<Eigen::Vector3d> ba() { return Eigen::Map<Eigen::Vector3d>(state + STATE::BAX); }
        Eigen::Map<Eigen::Vector3d> bg() { return Eigen::Map<Eigen::Vector3d>(state + STATE::BGX); }

    public:
        bool is_initialized {false};
        bool is_key_frame {false};
        uint16_t ordering {0};
        uint32_t id {0};
        uint64_t time_us {0};

        // 状态
        double state[STATE::DIM] {0};     // q, t, v, ba, bg

        // 用于FEJ
        double *state_first {nullptr};     // q, t, v, ba, bg

        // 特征
        std::unordered_map<unsigned long, std::shared_ptr<Feature>> features;
    };
}

#endif //GRAPH_OPTIMIZATION_FRAME_H
