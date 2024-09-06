//
// Created by Cain on 2024/4/11.
//

#include "vertex_scale.h"

graph_optimization::VertexScale::VertexScale(double *data) : Vertex(data, 1) {
    scale() = 1.;
}

void graph_optimization::VertexScale::plus(const graph_optimization::VecX &delta) {
    scale() *= exp(delta[0]);
}

void graph_optimization::VertexScale::plus(double *delta) {
    scale() *= exp(delta[0]);
}