//
// Created by Cain on 2024/9/3.
//

#ifndef GRAPH_OPTIMIZATION_LANDMARK_H
#define GRAPH_OPTIMIZATION_LANDMARK_H

#include <iostream>
#include <memory>
#include <deque>
#include <Eigen/Core>
#include <Eigen/Geometry>
#include "../parameters.h"
#include "feature.h"
#include "frame.h"

namespace vins {
    class Frame;
    class Landmark;
    class Feature;

    class Landmark {
    public:
        EIGEN_MAKE_ALIGNED_OPERATOR_NEW;
        explicit Landmark(unsigned long id) : _id(id) {}

        unsigned long id() const { return _id; }
//        pair<unsigned long, FrameNode *> get_reference_frame() const { return imu_deque[0]->frames[0]; }
        bool from_global_to_local(const Eigen::Quaterniond &q_ic, const Eigen::Vector3d &t_ic);
        bool from_local_to_global(const Eigen::Quaterniond &q_ic, const Eigen::Vector3d &t_ic);

    public:
        bool is_triangulated {false};
        bool is_outlier {false};

        double inv_depth {0.};
        Eigen::Vector3d position {0., 0., 0.};

        std::deque<std::pair<std::shared_ptr<Frame>, std::shared_ptr<Feature>>> observations;

    private:
        const unsigned long _id;
    };
}

#endif //GRAPH_OPTIMIZATION_LANDMARK_H
