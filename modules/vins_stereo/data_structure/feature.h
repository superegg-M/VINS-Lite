//
// Created by Cain on 2024/6/12.
//

#ifndef GRAPH_OPTIMIZATION_FEATURE_H
#define GRAPH_OPTIMIZATION_FEATURE_H

#include <iostream>
#include <memory>
#include <vector>
#include <Eigen/Core>
#include <Eigen/Geometry>
#include "../parameters.h"

namespace vins {
    class Frame;
    class Landmark;
    class Feature;

    class Feature {
    public:
        EIGEN_MAKE_ALIGNED_OPERATOR_NEW;
        Feature();
    public:
        std::vector<Eigen::Vector3d> points;
        std::weak_ptr<Landmark> landmark;
        std::weak_ptr<Frame> frame;
    };
}

#endif //GRAPH_OPTIMIZATION_FEATURE_H
