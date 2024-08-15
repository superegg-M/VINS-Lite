//
// Created by Cain on 2023/2/20.
//

#include <iostream>

#include "backend/edge.h"
#include "backend/vertex.h"

namespace graph_optimization {
    unsigned long Edge::_global_edge_id = 0;

    Edge::Edge(unsigned long residual_dimension, unsigned long num_vertices, const std::vector<std::string> &vertices_types, LossFunction::Type loss_function_type) {
        _residual.resize(residual_dimension, 1);
        _vertices.reserve(num_vertices);
        if (!vertices_types.empty()) {
            _vertices_types = vertices_types;
        }
        _jacobians.resize(num_vertices);
        _id = _global_edge_id++;

        _information.resize(residual_dimension, residual_dimension);
        _information.setIdentity();

        switch (loss_function_type) {
            case LossFunction::Type::HUBER:
                _loss_function = std::make_shared<HuberLoss>(1.);
                break;
            case LossFunction::Type::CAUCHY:
                _loss_function = std::make_shared<CauchyLoss>(1.);
                break;
            case LossFunction::Type::TUKEY:
                _loss_function = std::make_shared<TukeyLoss>(2.);
                break;
            default:
                _loss_function = std::make_shared<TrivialLoss>();
                break;
        }
    }

    void Edge::compute_chi2() {
        _chi2 = _residual.transpose() * _information * _residual;
        _rho = _loss_function->compute(_chi2);
    }

    void Edge::robust_information(double &drho, MatXX &info, VecX &res) const {
        if (_use_info) {
            res = _information * _residual;
            info = _rho[1] * _information;
            if(_rho[1] + 2. * _rho[2] * _chi2 > 0.) {
                info += ((2. * _rho[2]) * res) * res.transpose();
            }
            res *= _rho[1];

            drho = _rho[1];
        } else {
            res = _residual;
            info = _rho[1] * MatXX::Identity(_information.rows(), _information.cols());
            if(_rho[1] + 2. * _rho[2] * _chi2 > 0.) {
                info += ((2. * _rho[2]) * res) * res.transpose();
            }
            res *= _rho[1];

            drho = _rho[1];
        }
    }

    bool Edge::check_valid() {
        if (!_vertices_types.empty()) {
            // check type info
            for (size_t i = 0; i < _vertices.size(); ++i) {
                if (_vertices_types[i] != _vertices[i]->type_info()) {
                    std::cout << "Vertex type does not match, should be " << _vertices_types[i] <<
                              ", but set to " << _vertices[i]->type_info() << std::endl;
                    return false;
                }
            }
        }
        return true;
    }
}

