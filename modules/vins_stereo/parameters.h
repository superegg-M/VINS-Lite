//
// Created by Cain on 2024/1/10.
//

#ifndef GRAPH_OPTIMIZATION_PARAMETERS_H
#define GRAPH_OPTIMIZATION_PARAMETERS_H

//#define USE_OPENMP
//#define NUM_THREADS 8
//#define PRINT_INFO
//#define NUM_OF_CAM 2

namespace vins {
    static constexpr unsigned NUM_OF_F = 1000;
    static constexpr unsigned int WINDOW_SIZE = 10;
    static constexpr double FOCAL_LENGTH = 460.;
    static constexpr double MIN_PARALLAX = 0.1;
    static constexpr double INIT_DEPTH = 0.1;
    static constexpr unsigned int SIZE_POSE = 7;
    static constexpr unsigned int SIZE_SPEEDBIAS = 9;
    static constexpr unsigned int SIZE_FEATURE = 1;
}

#endif //GRAPH_OPTIMIZATION_PARAMETERS_H
