//
// Created by Cain on 2024/5/23.
//

#ifndef GRAPH_OPTIMIZATION_COMMON_H
#define GRAPH_OPTIMIZATION_COMMON_H

namespace vins {
    union MarginFlags {
        struct {
            bool none : 1;
            bool margin_old : 1;
            bool margin_new : 1;
        } flags;
        unsigned char value {0};
    };

    union LandmarkType {
        struct {
            bool global_xyz : 1;
            bool global_inverse_depth : 1;
            bool anchored_xyz : 1;
            bool anchored_inverse_depth : 1;
            bool anchored_normalized_inverse_depth : 1;
        } flags;
        unsigned char value {0};
    };
}

#endif //GRAPH_OPTIMIZATION_COMMON_H
