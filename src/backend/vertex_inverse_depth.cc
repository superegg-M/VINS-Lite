//
// Created by Cain on 2024/8/27.
//

#include "backend/vertex_inverse_depth.h"

void graph_optimization::VertexInverseDepth::plus(const VecX &delta) {
    inverse_depth() += delta[0];
}

void graph_optimization::VertexInverseDepth::plus(double *delta) {
    inverse_depth() += delta[0];
}