//
// Created by Cain on 2024/8/13.
//

#include <iostream>
#include "utility/utility.h"
#include "../../thirdparty/Sophus/sophus/se3.hpp"
#include "../../../include/parameters.h"
#include <omp.h>

#include "backend/tiny/problem_slam.h"

// #define USE_PCG_SOLVER


namespace graph_optimization {
    using namespace std;

    /*
    * marginalize 所有和 frame 相连的 edge: imu factor, projection factor
    * 假设vertex_pose和vertex_motion, 分别排序在vertex的倒数第二与倒数第一位
    * */
    bool TinySLAMProblem::marginalize(const std::shared_ptr<Vertex>& vertex_pose, 
                                      const std::shared_ptr<Vertex>& vertex_motion,
                                      const std::vector<std::shared_ptr<Vertex>> &vertices_landmark,
                                      const std::vector<std::shared_ptr<Edge>> &edges) {
        // 计算所需marginalize的edge的hessian
        ulong state_dim = _ordering_states;
        ulong landmark_size = vertices_landmark.size();
        ulong cols = state_dim + landmark_size;

        // 为H加上eps, 避免出现对角线元素为0的情况
        MatXX h_state_landmark = VecX::Constant(cols, 1, 1e-6).asDiagonal();
        VecX b_state_landmark(VecX::Zero(cols));

        // 修改landmark的ordering_id, 方便hessian的计算
        for (unsigned i = 0; i < landmark_size; ++i) {
            vertices_landmark[i]->set_ordering_id(state_dim + i);
        }

#ifdef USE_OPENMP
        MatXX Hs[NUM_THREADS];       ///< Hessian矩阵
        VecX bs[NUM_THREADS];       ///< 负梯度
        for (unsigned int i = 0; i < NUM_THREADS; ++i) {
            Hs[i] = MatXX::Zero(cols, cols);
            bs[i] = VecX::Zero(cols);
        }

#pragma omp parallel for num_threads(NUM_THREADS)
        for (size_t n = 0; n < edges.size(); ++n) {
            unsigned int index = omp_get_thread_num();

            auto &edge = edges[n];
            auto &&jacobians = edge->jacobians();
            auto &&vertices = edge->vertices();
            assert(jacobians.size() == vertices.size());

            // 计算edge的鲁棒权重
            double drho;
            MatXX robust_information;
            VecX robust_residual;
            edge->robust_information(drho, robust_information, robust_residual);

            for (size_t i = 0; i < vertices.size(); ++i) {
                auto &&v_i = vertices[i];
                if (v_i->is_fixed()) continue;

                auto &&jacobian_i = jacobians[i];
                ulong index_i = v_i->ordering_id();
                ulong dim_i = v_i->local_dimension();

                MatXX JtW = jacobian_i.transpose() * robust_information;
                for (size_t j = i; j < vertices.size(); ++j) {
                    auto &&v_j = vertices[j];
                    if (v_j->is_fixed()) continue;

                    auto &&jacobian_j = jacobians[j];
                    ulong index_j = v_j->ordering_id();
                    ulong dim_j = v_j->local_dimension();

                    MatXX hessian = JtW * jacobian_j;

//                    assert(hessian.rows() == v_i->local_dimension() && hessian.cols() == v_j->local_dimension());
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
            h_state_landmark += Hs[i];
            b_state_landmark += bs[i];
        }
#else
        for (size_t n = 0; n < edges.size(); ++n) {
            auto &edge = edges[n];
            // 若曾经solve problem, 则无需再次计算
            // edge->compute_residual();
            // edge->compute_jacobians();
            auto &&jacobians = edge->jacobians();
            auto &&vertices = edge->vertices();

            assert(jacobians.size() == vertices.size());

            // 计算edge的鲁棒权重
            double drho;
            MatXX robust_information;
            VecX robust_residual;
            edge->robust_information(drho, robust_information, robust_residual);

            for (size_t i = 0; i < vertices.size(); ++i) {
                auto v_i = vertices[i];
                auto jacobian_i = jacobians[i];
                ulong index_i = v_i->ordering_id();
                ulong dim_i = v_i->local_dimension();

                MatXX JtW = jacobian_i.transpose() * robust_information;
                for (size_t j = i; j < vertices.size(); ++j) {
                    auto &&v_j = vertices[j];
                    if (v_j->is_fixed()) continue;

                    auto &&jacobian_j = jacobians[j];
                    ulong index_j = v_j->ordering_id();
                    ulong dim_j = v_j->local_dimension();

                    MatXX hessian = JtW * jacobian_j;

                    assert(hessian.rows() == v_i->local_dimension() && hessian.cols() == v_j->local_dimension());
                    // 所有的信息矩阵叠加起来
                    h_state_landmark.block(index_i, index_j, dim_i, dim_j) += hessian;
                    if (j != i) {
                        // 对称的下三角
                        h_state_landmark.block(index_j, index_i, dim_j, dim_i) += hessian.transpose();
                    }
                }
                b_state_landmark.segment(index_i, dim_i) -= jacobian_i.transpose() * robust_residual;
            }
        }
#endif

        // marginalize与边连接的landmark
        MatXX h_state_schur;
        VecX b_state_schur;
        if (landmark_size > 0) {
            // Hll
            VecX Hll = h_state_landmark.block(state_dim, state_dim, landmark_size, landmark_size).diagonal();
            // 由于叠加了eps, 所以能够保证Hll可逆
            VecX Hll_inv = Hll.array().inverse();

            // Hll^-1 * Hsl^T
            MatXX temp_H = Hll_inv.asDiagonal() * h_state_landmark.block(state_dim, 0, landmark_size, state_dim);  
            // Hll^-1 * bl
            VecX temp_b = Hll_inv.cwiseProduct(b_state_landmark.tail(landmark_size));   

            // (Hss - Hsl * Hll^-1 * Hls) * xs = bs - Hsl * Hll^-1 * bl
#ifdef USE_OPENMP
            // Hss - Hsl * Hll^-1 * Hls
            h_state_schur = MatXX::Zero(state_dim, state_dim);
#pragma omp parallel for num_threads(NUM_THREADS)
            for (ulong i = 0; i < state_dim; ++i) {
                h_state_schur(i, i) = -temp_H.col(i).dot(h_state_landmark.block(state_dim, 0, landmark_size, state_dim).col(i));
                for (ulong j = i + 1; j < state_dim; ++j) {
                    h_state_schur(i, j) = -temp_H.col(j).dot(h_state_landmark.block(state_dim, 0, landmark_size, state_dim).col(i));
                    h_state_schur(j, i) = h_state_schur(i, j);
                }
            }
            h_state_schur += h_state_landmark.block(0, 0, state_dim, state_dim);
#else
            // Hss - Hsl * Hll^-1 * Hls
            h_state_schur = h_state_landmark.block(0, 0, state_dim, state_dim) - h_state_landmark.block(0, state_dim, state_dim, landmark_size) * temp_H;
#endif
            // bs - Hsl * Hll^-1 * bl
            b_state_schur = b_state_landmark.head(state_dim) - temp_H.transpose() * b_state_landmark.tail(landmark_size);
        } else {
            h_state_schur = h_state_landmark;
            b_state_schur = b_state_landmark;
        }

        // 叠加之前的先验
        if(_h_prior.rows() > 0) {
            h_state_schur += _h_prior;
            b_state_schur += _b_prior;
        } else {
            _h_prior = MatXX::Zero(state_dim, state_dim);
            _b_prior = VecX::Zero(state_dim, 1);
        }

        // Marginalize掉pose和motion
        ulong marginalized_size = vertex_pose->local_dimension() + vertex_motion->local_dimension();
        ulong reserve_size = state_dim - marginalized_size;
        MatXX Hmr = h_state_schur.block(reserve_size, 0, marginalized_size, reserve_size);

        MatXX temp_H(MatXX::Zero(marginalized_size, reserve_size));  // Hmm^-1 * Hrm^T
        VecX temp_b(VecX::Zero(marginalized_size, 1));   // Hmm^-1 * bm

        if (marginalized_size == 1) {
            unsigned index = reserve_size;
            if (h_state_schur(index, index) > 1e-12) {
                temp_H = Hmr / h_state_schur(index, index);
                temp_b = b_state_schur.tail(marginalized_size) / h_state_schur(index, index);
            } else {
                temp_H.setZero();
                temp_b.setZero();
            }
            
        } else {
            auto Hmm_ldlt = h_state_schur.block(reserve_size, reserve_size, marginalized_size, marginalized_size).ldlt();
            if (Hmm_ldlt.info() == Eigen::Success) {
                temp_H = Hmm_ldlt.solve(Hmr);
                temp_b = Hmm_ldlt.solve(b_state_schur.tail(marginalized_size));
            } else {
                temp_H.setZero();
                temp_b.setZero();
            }
        }

        // (Hrr - Hrm * Hmm^-1 * Hmr) * dxp = br - Hrm * Hmm^-1 * bm
#ifdef USE_OPENMP
        // Hrr - Hrm * Hmm^-1 * Hmr
        MatXX h_state_schur_block = h_state_schur.block(0, 0, reserve_size, reserve_size);
        h_state_schur = MatXX::Zero(reserve_size, reserve_size);
#pragma omp parallel for num_threads(NUM_THREADS)
        for (ulong i = 0; i < reserve_size; ++i) {
            h_state_schur(i, i) = -Hmr.col(i).dot(temp_H.col(i));
            for (ulong j = i + 1; j < reserve_size; ++j) {
                h_state_schur(i, j) = -Hmr.col(i).dot(temp_H.col(j));
                h_state_schur(j, i) = h_state_schur(i, j);
            }
        }
        h_state_schur += h_state_schur_block;
#else

        // Hrr - Hrm * Hmm^-1 * Hmr
        h_state_schur = h_state_schur.block(0, 0, reserve_size, reserve_size) - h_state_schur.block(0, reserve_size, reserve_size, marginalized_size) * temp_H;

#endif
        // br - Hrm * Hmm^-1 * bm
        b_state_schur = b_state_schur.head(reserve_size) - temp_H.transpose() * b_state_schur.tail(marginalized_size);

        _h_prior.topLeftCorner(reserve_size, reserve_size) = h_state_schur;
        _b_prior.topRows(reserve_size) = b_state_schur;

        return true;
    }

    bool TinySLAMProblem::solve_linear_system(VecX &delta_x) {
        if (delta_x.rows() != (_ordering_states + _ordering_landmarks)) {
            delta_x.resize(_ordering_states + _ordering_landmarks, 1);
        }

        /*
        * [Hpp Hpl][xp] = [bp]
        * [Hlp Hll][xl] = [bl]
        *
        * (Hpp - Hpl * Hll^-1 * Hlp) * xp = bp - Hpl * Hll^-1 * bl
        * Hll * xl = bl - Hlp * xp
        * */
        ulong reserve_size = _ordering_states;
        ulong marg_size = _ordering_landmarks;

        // 由于叠加了lambda, 所以能够保证Hll可逆
        VecX Hll = _hessian.block(reserve_size, reserve_size, marg_size, marg_size).diagonal();
        // Hll + λ
        Hll += _diag_lambda.tail(marg_size);
        VecX Hll_inv = Hll.array().inverse();

        // Hll^-1 * Hpl^T
        MatXX temp_H = Hll_inv.asDiagonal() * _hessian.block(reserve_size, 0, marg_size, reserve_size);  
        // Hll^-1 * bl
        VecX temp_b = Hll_inv.cwiseProduct(_b.tail(marg_size));   

        // (Hpp - Hpl * Hll^-1 * Hlp) * dxp = bp - Hpl * Hll^-1 * bl
        // 这里即使叠加了lambda, 也有可能因为数值精度的问题而导致 _h_pp_schur 不可逆
        
#ifdef USE_OPENMP
        _h_pp_schur = MatXX::Zero(reserve_size, reserve_size);
#pragma omp parallel for num_threads(NUM_THREADS)
        for (ulong i = 0; i < reserve_size; ++i) {
            _h_pp_schur(i, i) = -temp_H.col(i).dot(_hessian.block(reserve_size, 0, marg_size, reserve_size).col(i));
            for (ulong j = i + 1; j < reserve_size; ++j) {
                _h_pp_schur(i, j) = -temp_H.col(j).dot(_hessian.block(reserve_size, 0, marg_size, reserve_size).col(i));
                _h_pp_schur(j, i) = _h_pp_schur(i, j);
            }
        }
        _h_pp_schur += _hessian.block(0, 0, reserve_size, reserve_size);
#else
        // Hpp - Hpl * Hll^-1 * Hlp
        _h_pp_schur = _hessian.block(0, 0, reserve_size, reserve_size) - _hessian.block(0, reserve_size, reserve_size, marg_size) * temp_H;
#endif
        // Hpp - Hpl * Hll^-1 * Hlp + λ
        _h_pp_schur += _diag_lambda.head(reserve_size).asDiagonal();

        // bp - Hpl * Hll^-1 * bl
        _b_pp_schur = _b.head(reserve_size) - temp_H.transpose() * _b.tail(marg_size);

        // Solve: Hpp * xp = bp
#ifdef USE_PCG_SOLVER
        auto n_pcg = _h_pp_schur.rows();                       // 迭代次数
        delta_x.head(reserve_size) = PCG_solver(_h_pp_schur, _b_pp_schur, n_pcg);
#else
        auto &&H_pp_schur_ldlt = _h_pp_schur.ldlt();
        if (H_pp_schur_ldlt.info() != Eigen::Success) {
            return false;   // H_pp_schur不是正定矩阵
        }
        delta_x.head(reserve_size) =  H_pp_schur_ldlt.solve(_b_pp_schur);
#endif

        // Hll * xl = bl - Hlp * xp
        delta_x.tail(marg_size) = temp_b - temp_H * delta_x.head(reserve_size);

        return true;
    }

    void TinySLAMProblem::add_prior_to_hessian() {
        if(_h_prior.rows() > 0) {
            MatXX H_prior_tmp = _h_prior;
            VecX b_prior_tmp = _b_prior;

            // 只有没有被fix的pose存在先验, landmark没有先验
#ifdef USE_OPENMP
#pragma omp parallel for num_threads(NUM_THREADS)
            for (size_t n = 0; n < _num_state_vertices; ++n) {
                auto &&vertex = _vertices[n];
                if (vertex->is_fixed() ) {
                    ulong idx = vertex->ordering_id();
                    ulong dim = vertex->local_dimension();
                    H_prior_tmp.block(idx,0, dim, H_prior_tmp.cols()).setZero();
                    H_prior_tmp.block(0,idx, H_prior_tmp.rows(), dim).setZero();
                    b_prior_tmp.segment(idx,dim).setZero();
//                std::cout << " fixed prior, set the Hprior and bprior part to zero, idx: "<<idx <<" dim: "<<dim<<std::endl;
                }
            }
#else
            for (size_t n = 0; n < _num_state_vertices; ++n) {
                auto &&vertex = _vertices[n];
                if (vertex->is_fixed() ) {
                    ulong idx = vertex->ordering_id();
                    ulong dim = vertex->local_dimension();
                    H_prior_tmp.block(idx,0, dim, H_prior_tmp.cols()).setZero();
                    H_prior_tmp.block(0,idx, H_prior_tmp.rows(), dim).setZero();
                    b_prior_tmp.segment(idx,dim).setZero();
//                std::cout << " fixed prior, set the Hprior and bprior part to zero, idx: "<<idx <<" dim: "<<dim<<std::endl;
                }
            }
#endif
            _hessian.topLeftCorner(_ordering_states, _ordering_states) += H_prior_tmp;
            _b.head(_ordering_states) += b_prior_tmp;
        }
    }

    void TinySLAMProblem::initialize() {
        TinyProblem::initialize();
        _h_prior = MatXX::Zero(_ordering_states, _ordering_states);
        _b_prior = VecX::Zero(_ordering_states, 1);
        _b_prior_bp = VecX::Zero(_ordering_states, 1);
    }

    void TinySLAMProblem::update_prior(const VecX &delta_x) {
        if (_b_prior.rows() > 0) {
            _b_prior_bp = _b_prior;
            _b_prior -= _h_prior * delta_x.head(_ordering_states);
        }
    }
}