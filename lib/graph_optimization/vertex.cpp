//
// Created by Cain on 2023/2/20.
//

#include "vertex.h"

namespace graph_optimization {
    unsigned long Vertex::_global_vertex_id = 0;

    Vertex::Vertex(double *data, unsigned long num_dimension, unsigned long local_dimension) {
        _dimension = num_dimension;
        _local_dimension = local_dimension ? local_dimension : num_dimension;
        _id = _global_vertex_id++;

        _parameters = data;
        _parameters_backup = new double [num_dimension];
    }

    Vertex::~Vertex() {
        delete [] _parameters_backup;
    }

    void Vertex::save_parameters() {
        saved_parameters = true;
        std::memcpy(_parameters_backup, _parameters, _dimension * sizeof(double));
    }

    bool Vertex::load_parameters() {
        if (saved_parameters) {
            saved_parameters = false;
            std::memcpy(_parameters, _parameters_backup, _dimension * sizeof(double));
            return true;
        }
        return false;
    }

}

