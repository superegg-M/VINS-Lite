//
// Created by Cain on 2023/12/28.
//

//#include <glog/logging.h>
#include <fstream>
#include <iostream>
#include <Eigen/Dense>
#include "../thirdparty/Sophus/sophus/se3.hpp"
#include <omp.h>

#include "backend/problem.h"

#define MULTIPLY_HESSIAN_USING_SELF_ADJOINT

using namespace std;

namespace graph_optimization {
    Problem::Problem() {
        logout_vector_size();
    }

    bool Problem::add_vertex(const std::shared_ptr<Vertex>& vertex) {
        _vertices.emplace_back(vertex);

        return true;
    }

    bool Problem::remove_vertex(const std::shared_ptr<Vertex>& vertex) {
        // TODO:
        return true;
    }

    bool Problem::add_edge(const shared_ptr<Edge>& edge) {
        _edges.emplace_back(edge);

        for (auto &vertex: edge->vertices()) {
            _vertex_to_edge.emplace(vertex->id(), edge);
        }
        return true;
    }

    bool Problem::remove_edge(const std::shared_ptr<Edge>& edge) {
        // TODO:
        return true;
    }

    bool Problem::solve(unsigned long iterations) {
        if (_edges.empty() || _vertices.empty()) {
            std::cerr << "\nCannot solve problem without edges or vertices" << std::endl;
            return false;
        }

        TicToc t_solve;

        _t_hessian_cost = 0.;
        _t_jacobian_cost = 0.;
        _t_chi2_cost = 0;
        _t_residual_cost = 0.;

        initialize_ordering(); // 统计优化变量的维数: _ordering_generic，为构建 hessian 矩阵做准备

        bool flag;
        switch (_solver_type) {
            case SolverType::STEEPEST_DESCENT:
                flag = calculate_steepest_descent(_delta_x_sd, iterations);
                break;
            case SolverType::GAUSS_NEWTON:
                flag = calculate_gauss_newton(_delta_x_gn, iterations);
                break;
            case SolverType::LEVENBERG_MARQUARDT:
                flag = calculate_levenberg_marquardt(_delta_x_lm, iterations);
                break;
            case SolverType::DOG_LEG:
                flag = calculate_dog_leg(_delta_x_dl, iterations);
                break;
            default:
                flag = calculate_levenberg_marquardt(_delta_x_lm, iterations);
                break;
        }

#ifdef PRINT_INFO
        std::cout << "problem solve cost: " << t_solve.toc() << " ms" << std::endl;
        std::cout << "update_hessian cost: " << _t_hessian_cost << " ms" << std::endl;
        std::cout << "update_jacobian cost: " << _t_jacobian_cost << " ms" << std::endl;
        std::cout << "update_chi2 cost: " << _t_chi2_cost << " ms" << std::endl;
        std::cout << "update_residual cost: " << _t_residual_cost << " ms" << std::endl;
#endif

        return flag;
    }

    void Problem::initialize_ordering() {
        // 每次重新计数
        _ordering_generic = 0;

        // Note: _vertices 是 map 类型的, 顺序是按照 id 号排序的
        // 统计带估计的所有变量的总维度
#ifdef USE_OPENMP
#pragma omp parallel for num_threads(NUM_THREADS) default(none)
#endif
        for (size_t i = 0; i < _vertices.size(); ++i) {
            _vertices[i]->set_ordering_id(_ordering_generic);
            _ordering_generic += _vertices[i]->local_dimension();  // 所有的优化变量总维数
        }
    }

    void Problem::update_hessian() {
        TicToc t_h;

        ulong size = _ordering_generic;
        MatXX H(MatXX::Zero(size, size));       ///< Hessian矩阵
        VecX b(VecX::Zero(size));       ///< 负梯度

#ifdef USE_OPENMP
        MatXX Hs[NUM_THREADS];       ///< Hessian矩阵
        VecX bs[NUM_THREADS];       ///< 负梯度
        for (unsigned int i = 0; i < NUM_THREADS; ++i) {
            Hs[i] = MatXX::Zero(size, size);
            bs[i] = VecX::Zero(size);
        }

        // 遍历每个残差，并计算他们的雅克比，得到最后的 H = J^T * J
//        omp_set_num_threads(NUM_THREADS);
#pragma omp parallel for num_threads(NUM_THREADS) default(none) shared(Hs, bs)
        for (size_t n = 0; n < _edges.size(); ++n) {//for (auto &edge: edges) {
            unsigned int index = omp_get_thread_num();

            auto &&edge = _edges[n];
            auto &&jacobians = edge->jacobians();
            auto &&verticies = edge->vertices();
            assert(jacobians.size() == verticies.size());

            // 计算edge的鲁棒权重
            double drho;
            MatXX robust_information;
            VecX robust_residual;
            edge->robust_information(drho, robust_information, robust_residual);
            for (size_t i = 0; i < verticies.size(); ++i) {
                auto &&v_i = verticies[i];
                if (v_i->is_fixed()) continue;    // Hessian 里不需要添加它的信息，也就是它的雅克比为 0

                auto &&jacobian_i = jacobians[i];
                ulong index_i = v_i->ordering_id();
                ulong dim_i = v_i->local_dimension();

                MatXX JtW = jacobian_i.transpose() * robust_information;
                for (size_t j = i; j < verticies.size(); ++j) {
                    auto &&v_j = verticies[j];
                    if (v_j->is_fixed()) continue;

                    auto &&jacobian_j = jacobians[j];
                    ulong index_j = v_j->ordering_id();
                    ulong dim_j = v_j->local_dimension();

//                    assert(v_j->ordering_id() != -1);
                    MatXX hessian = JtW * jacobian_j;   // TODO: 这里能继续优化, 因为J'*W*J也是对称矩阵
                    // 所有的信息矩阵叠加起来
                    Hs[index].block(index_i, index_j, dim_i, dim_j).noalias() += hessian;
                    if (j != i) {
                        // 对称的下三角
                        Hs[index].block(index_j, index_i, dim_j, dim_i).noalias() += hessian.transpose();
                    }
                }
                bs[index].segment(index_i, dim_i).noalias() -= jacobian_i.transpose() * robust_residual;
            }
        }

        for (unsigned int i = 0; i < NUM_THREADS; ++i) {
            H += Hs[i];
            b += bs[i];

            // _t_hessian_cost += t_cost[i];
        }
#else
        // 遍历每个残差，并计算他们的雅克比，得到最后的 H = J^T * J
        for (auto &edge: edges) {
            auto &&jacobians = edge->jacobians();
            auto &&verticies = edge->vertices();
            assert(jacobians.size() == verticies.size());

            // 计算edge的鲁棒权重
            double drho;
            MatXX robust_information;
            VecX robust_residual;
            edge.second->robust_information(drho, robust_information, robust_residual);
            for (size_t i = 0; i < verticies.size(); ++i) {
                auto &&v_i = verticies[i];
                if (v_i->is_fixed()) continue;    // Hessian 里不需要添加它的信息，也就是它的雅克比为 0

                auto &&jacobian_i = jacobians[i];
                ulong index_i = v_i->ordering_id();
                ulong dim_i = v_i->local_dimension();

                MatXX JtW = jacobian_i.transpose() * robust_information;
                for (size_t j = i; j < verticies.size(); ++j) {
                    auto &&v_j = verticies[j];
                    if (v_j->is_fixed()) continue;

                    auto &&jacobian_j = jacobians[j];
                    ulong index_j = v_j->ordering_id();
                    ulong dim_j = v_j->local_dimension();

                    assert(v_j->ordering_id() != -1);
                    MatXX hessian = JtW * jacobian_j;   // TODO: 这里能继续优化, 因为J'*W*J也是对称矩阵
                    // 所有的信息矩阵叠加起来
                    H.block(index_i, index_j, dim_i, dim_j).noalias() += hessian;
                    if (j != i) {
                        // 对称的下三角
                        H.block(index_j, index_i, dim_j, dim_i).noalias() += hessian.transpose();
                    }
                }
                b.segment(index_i, dim_i).noalias() -= jacobian_i.transpose() * robust_residual;
            }

        }
#endif
        _hessian = H;
        _b = b;

        // 叠加先验
        add_prior_to_hessian();

//        _delta_x = VecX::Zero(size);  // initial delta_x = 0_n;

        _t_hessian_cost += t_h.toc();
    }

    void Problem::add_prior_to_hessian() {
        if(_h_prior.rows() > 0) {
            MatXX H_prior_tmp = _h_prior;
            VecX b_prior_tmp = _b_prior;

            // 对所有没有被fix的顶点添加先验信息
#ifdef USE_OPENMP
#pragma omp parallel for num_threads(NUM_THREADS) default(none) shared(H_prior_tmp, b_prior_tmp)
#endif
            for (size_t n = 0; n < _vertices.size(); ++n) {
                auto &&vertex = _vertices[n];
                if (vertex->is_fixed()) {
                    ulong idx = vertex->ordering_id();
                    ulong dim = vertex->local_dimension();
                    H_prior_tmp.block(idx,0, dim, H_prior_tmp.cols()).setZero();
                    H_prior_tmp.block(0,idx, H_prior_tmp.rows(), dim).setZero();
                    b_prior_tmp.segment(idx,dim).setZero();
//                std::cout << " fixed prior, set the Hprior and bprior part to zero, idx: "<<idx <<" dim: "<<dim<<std::endl;
                }
            }
            _hessian += H_prior_tmp;
            _b += b_prior_tmp;
        }
    }

    void Problem::initialize_lambda() {
        double max_diagonal = 0.;
        ulong size = _hessian.cols();
        assert(_hessian.rows() == _hessian.cols() && "Hessian is not square");
        for (ulong i = 0; i < size; ++i) {
            max_diagonal = std::max(fabs(_hessian(i, i)), max_diagonal);
        }

        double tau = 1e-5;  // 1e-5
        _current_lambda = tau * max_diagonal;
        _current_lambda = std::min(std::max(_current_lambda, _lambda_min), _lambda_max);

        _diag_lambda = tau * _hessian.diagonal();
        for (int i = 0; i < _hessian.rows(); ++i) {
            _diag_lambda(i) = std::min(std::max(_diag_lambda(i), _lambda_min), _lambda_max);
        }
    }

    double Problem::calculate_hessian_norm_square(const VecX &x) {
#ifdef HESSIAN_NORM_SQUARE_USING_GRAPH
        double norm2 = 0.;

#ifdef USE_OPENMP
#pragma omp parallel for num_threads(NUM_THREADS) reduction(+:norm2)
        for (size_t n = 0; n < _edges_vector.size(); ++n) {
            auto &&edge = _edges_vector[n];
            auto &&jacobians = edge.second->jacobians();
            auto &&verticies = edge.second->vertices();
            assert(jacobians.size() == verticies.size());

            // 计算edge的鲁棒权重
            double drho;
            MatXX robust_information;
            VecX robust_residual;
            edge.second->robust_information(drho, robust_information, robust_residual);
            for (size_t i = 0; i < verticies.size(); ++i) {
                auto &&v_i = verticies[i];
                if (v_i->is_fixed()) continue;    // Hessian 里不需要添加它的信息，也就是它的雅克比为 0

                auto &&jacobian_i = jacobians[i];
                ulong index_i = v_i->ordering_id();
                ulong dim_i = v_i->local_dimension();

                VecX Jx = jacobian_i * x.segment(index_i, dim_i);
                norm2 += Jx.transpose() * robust_information * Jx;
            }
        }
#else 
    for (auto &edge: _edges) {
            auto &&jacobians = edge.second->jacobians();
            auto &&verticies = edge.second->vertices();
            assert(jacobians.size() == verticies.size());

            // 计算edge的鲁棒权重
            double drho;
            MatXX robust_information;
            VecX robust_residual;
            edge.second->robust_information(drho, robust_information, robust_residual);
            for (size_t i = 0; i < verticies.size(); ++i) {
                auto &&v_i = verticies[i];
                if (v_i->is_fixed()) continue;    // Hessian 里不需要添加它的信息，也就是它的雅克比为 0

                auto &&jacobian_i = jacobians[i];
                ulong index_i = v_i->ordering_id();
                ulong dim_i = v_i->local_dimension();

                

                VecX Jx = jacobian_i * x.segment(index_i, dim_i);
                norm2 += Jx.transpose() * robust_information * Jx;
            }
        }
#endif

        return norm2;
#else
        return x.dot(multiply_hessian(x));
#endif
    }

    VecX Problem::multiply_hessian(const VecX &x) {
#ifdef MULTIPLY_HESSIAN_USING_SELF_ADJOINT
        auto &&hessian = _hessian.selfadjointView<Eigen::Upper>();
        return hessian * x;
#else
        return _hessian * x;
#endif
    }

    /*
    * Solve Hx = b, we can use PCG iterative method or use sparse Cholesky
    */
    bool Problem::solve_linear_system(VecX &delta_x) {
        MatXX H = _hessian;
//            for (unsigned i = 0; i < _hessian.rows(); ++i) {
//                H(i, i) += _current_lambda;
//            }
        H += _diag_lambda.asDiagonal();
        auto && H_ldlt = H.ldlt();
        if (H_ldlt.info() == Eigen::Success) {
            delta_x = H_ldlt.solve(_b);
            return true;
        } else {
            return false;
        }
    }

    void Problem::update_states(const VecX &delta_x) {
#ifdef USE_OPENMP
#pragma omp parallel for num_threads(NUM_THREADS) default(none) shared(delta_x)
#endif
        for (size_t n = 0; n < _vertices.size(); ++n) {
            auto &&vertex = _vertices[n];
            vertex->save_parameters();

            VecX delta = delta_x.segment(vertex->ordering_id(), vertex->local_dimension());

            // 所有的参数 x 叠加一个增量  x_{k+1} = x_{k} + delta_x
            vertex->plus(delta);
        }

        // 利用 delta_x 更新先验信息
        update_prior(delta_x);
    }

    void Problem::update_prior(const VecX &delta_x) {
        if (_b_prior.rows() > 0) {
            _b_prior_bp = _b_prior;
            _b_prior -= _h_prior * delta_x;
        }
    }

    void Problem::update_residual() {
        TicToc t_r;

#ifdef USE_OPENMP
#pragma omp parallel for num_threads(NUM_THREADS) default(none)
#endif
        for (size_t n = 0; n < _edges.size(); ++n) {
            _edges[n]->compute_residual();
        }

        _t_residual_cost += t_r.toc();
    }

    void Problem::update_jacobian() {
        TicToc t_j;

#ifdef USE_OPENMP
#pragma omp parallel for num_threads(NUM_THREADS) default(none)
#endif
        for (size_t n = 0; n < _edges.size(); ++n) {
            _edges[n]->compute_jacobians();
        }

        _t_jacobian_cost += t_j.toc();
    }

    void Problem::update_chi2() {
        TicToc t_c;

        _chi2 = 0.;
#ifdef USE_OPENMP
#pragma omp parallel for num_threads(NUM_THREADS) default(none) reduction(+:_chi2)
#endif
        for (size_t n = 0; n < _edges.size(); ++n) {
            _edges[n]->compute_chi2();
            _chi2 += _edges[n]->get_robust_chi2();
        }
        _chi2 *= 0.5;

        _t_chi2_cost += t_c.toc();
    }

    void Problem::rollback_states(const VecX &delta_x) {
#ifdef USE_OPENMP
#pragma omp parallel for num_threads(NUM_THREADS) default(none)
#endif
        for (size_t n = 0; n < _vertices.size(); ++n) {
            auto &&vertex = _vertices[n];
            vertex->load_parameters();
        }
        _b_prior = _b_prior_bp;
    }

    void Problem::save_hessian_diagonal_elements() {
        ulong size = _hessian.cols();
        assert(_hessian.rows() == _hessian.cols() && "Hessian is not square");
        _hessian_diag.resize(size);
        for (ulong i = 0; i < size; ++i) {
            _hessian_diag(i) = _hessian(i, i);
        }
    }

    void Problem::load_hessian_diagonal_elements() {
        ulong size = _hessian.cols();
        assert(_hessian.rows() == _hessian.cols() && "Hessian is not square");
        assert(size == _hessian_diag.size() && "Hessian dimension is wrong");
        for (ulong i = 0; i < size; ++i) {
            _hessian(i, i) = _hessian_diag(i);
        }
    }

    /** @brief conjugate gradient with perconditioning
*
*  the jacobi PCG method
*
*/
    VecX Problem::PCG_solver(const MatXX &A, const VecX &b, unsigned long max_iter) {
        assert(A.rows() == A.cols() && "PCG solver ERROR: A is not a square matrix");
        TicToc t_h;
        unsigned long rows = b.rows();
        unsigned long n = max_iter ? max_iter : rows;
        double threshold = 1e-6 * b.norm();

        VecX x(VecX::Zero(rows));
        VecX M = A.diagonal();
        VecX r(-b);
        VecX y(r.array() / M.array());
        VecX p(-y);
        VecX Ap = A * p;
        double ry = r.dot(y);
        double ry_prev = ry;
        double alpha = ry / p.dot(Ap);
        double beta;
        x += alpha * p;
        r += alpha * Ap;

        unsigned long i = 0;
        while (r.norm() > threshold && ++i < n) {
            y = r.array() / M.array();
            ry = r.dot(y);
            beta = ry / ry_prev;
            ry_prev = ry;
            p = -y + beta * p;

            Ap = A * p;
            alpha = ry / p.dot(Ap);
            x += alpha * p;
            r += alpha * Ap;
        }

        _t_PCG_solve_cost = t_h.toc();
        return x;
    }

//    void Problem::compute_prior() {
//
//    }


    void Problem::extend_prior_hessian_size(ulong dim) {
        ulong size = _h_prior.rows() + dim;
        _h_prior.conservativeResize(size, size);
        _b_prior.conservativeResize(size);

        _b_prior.tail(dim).setZero();
        _h_prior.rightCols(dim).setZero();
        _h_prior.bottomRows(dim).setZero();
    }

    bool Problem::check_ordering() {
        unsigned long current_ordering = 0;
        for (size_t n = 0; n < _vertices.size(); ++n) {
            if (_vertices[n]->ordering_id() != current_ordering) {
                return false;
            }
            current_ordering += _vertices[n]->local_dimension();
        }
        return true;
    }

    std::vector<std::shared_ptr<Edge>> Problem::get_connected_edges(const std::shared_ptr<Vertex>& vertex) {
        vector<shared_ptr<Edge>> edges;
        auto range = _vertex_to_edge.equal_range(vertex->id());
        for (auto iter = range.first; iter != range.second; ++iter) {
            edges.emplace_back(iter->second);
        }
        return edges;
    }

    void Problem::logout_vector_size() {
        // LOG(INFO) <<l
        //           "1 problem::LogoutVectorSize verticies_:" << verticies_.size() <<
        //           " edges:" << edges_.size();
    }
}