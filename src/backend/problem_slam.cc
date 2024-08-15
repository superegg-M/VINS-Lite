//
// Created by Cain on 2024/3/7.
//

#include <iostream>
#include "utility/utility.h"
#include "../thirdparty/Sophus/sophus/se3.hpp"
#include "../../include/parameters.h"
#include <omp.h>

#include "backend/problem_slam.h"

// #define USE_PCG_SOLVER


namespace graph_optimization {
    using namespace std;

    /*
    * marginalize 所有和 frame 相连的 edge: imu factor, projection factor
    * 假设vertex_pose和vertex_motion, 分别排序在vertex的倒数第二与倒数第一位
    * */
    bool ProblemSLAM::marginalize(const std::shared_ptr<Vertex>& vertex_pose, 
                                  const std::shared_ptr<Vertex>& vertex_motion,
                                  const std::vector<std::shared_ptr<Vertex>> &vertices_landmark,
                                  const std::vector<std::shared_ptr<Edge>> &edges) {                 

        // 计算所需marginalize的edge的hessian
        ulong state_dim = _ordering_poses;
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

#pragma omp parallel for num_threads(NUM_THREADS) default(none) shared(Hs, bs, edges)
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

                if (jacobian_i.rows() == 0) {
                    continue;
                }

                MatXX JtW = jacobian_i.transpose() * robust_information;
                for (size_t j = i; j < vertices.size(); ++j) {
                    auto &&v_j = vertices[j];
                    if (v_j->is_fixed()) continue;

                    auto &&jacobian_j = jacobians[j];
                    ulong index_j = v_j->ordering_id();
                    ulong dim_j = v_j->local_dimension();

                    if (jacobian_j.rows() == 0) {
                        continue;
                    }

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
                if (v_i->is_fixed()) continue;

                auto &&jacobian_i = jacobians[i];
                ulong index_i = v_i->ordering_id();
                ulong dim_i = v_i->local_dimension();
// std::cout << "name: " << edge->type_info() << std::endl;
// std::cout << "v_i: " << v_i->type_info() << ", " << v_i->ordering_id() << std::endl;
// std::cout << "jacobian_i.transpose(): " << jacobian_i.transpose().rows() << ", " << jacobian_i.transpose().cols() << ", robust_information: " << robust_information.rows() << ", " << robust_information.cols() << std::endl;
                if (jacobian_i.rows() == 0) {
                    continue;
                }

                MatXX JtW = jacobian_i.transpose() * robust_information;
                for (size_t j = i; j < vertices.size(); ++j) {
                    auto &&v_j = vertices[j];
                    if (v_j->is_fixed()) continue;

                    auto &&jacobian_j = jacobians[j];
                    ulong index_j = v_j->ordering_id();
                    ulong dim_j = v_j->local_dimension();
// std::cout << "JtW: " << JtW.rows() << ", " << JtW.cols() << ", jacobian_j: " << jacobian_j.rows() << ", " << jacobian_j.cols() << std::endl;
                    if (jacobian_j.rows() == 0) {
                        continue;
                    }

                    MatXX hessian = JtW * jacobian_j;

                    assert(hessian.rows() == v_i->local_dimension() && hessian.cols() == v_j->local_dimension());
                    // 所有的信息矩阵叠加起来
                    h_state_landmark.block(index_i, index_j, dim_i, dim_j) += hessian;
                    if (j != i) {
                        // 对称的下三角
                        h_state_landmark.block(index_j, index_i, dim_j, dim_i) += hessian.transpose();
                    }
                }
// std::cout << "jacobian_i.transpose(): " << jacobian_i.transpose().rows() << ", " << jacobian_i.transpose().cols() << ", robust_residual: " << robust_residual.rows() << ", " << robust_residual.cols() << std::endl;

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
#pragma omp parallel for num_threads(NUM_THREADS) default(none) shared(h_state_schur, h_state_landmark, temp_H, state_dim, landmark_size)
            for (ulong i = 0; i < state_dim; ++i) {
                h_state_schur(i, i) = -temp_H.col(i).dot(h_state_landmark.col(i).tail(landmark_size));
                for (ulong j = i + 1; j < state_dim; ++j) {
                    h_state_schur(i, j) = -temp_H.col(j).dot(h_state_landmark.col(i).tail(landmark_size));
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
//        VecX temp_b(VecX::Zero(marginalized_size, 1));   // Hmm^-1 * bm

        if (marginalized_size == 1) {
            unsigned index = reserve_size;
            if (h_state_schur(index, index) > 1e-12) {
                temp_H = Hmr / h_state_schur(index, index);
//                temp_b = b_state_schur.tail(marginalized_size) / h_state_schur(index, index);
            } else {
                temp_H.setZero();
//                temp_b.setZero();
            }
            
        } else {
            auto Hmm_ldlt = h_state_schur.block(reserve_size, reserve_size, marginalized_size, marginalized_size).ldlt();
            if (Hmm_ldlt.info() == Eigen::Success) {
                temp_H = Hmm_ldlt.solve(Hmr);
//                temp_b = Hmm_ldlt.solve(b_state_schur.tail(marginalized_size));
            } else {
                temp_H.setZero();
//                temp_b.setZero();
            }
        }

        // (Hrr - Hrm * Hmm^-1 * Hmr) * dxp = br - Hrm * Hmm^-1 * bm
#ifdef USE_OPENMP
        // Hrr - Hrm * Hmm^-1 * Hmr
        MatXX h_state_schur_block = h_state_schur.block(0, 0, reserve_size, reserve_size);
        h_state_schur = MatXX::Zero(reserve_size, reserve_size);
#pragma omp parallel for num_threads(NUM_THREADS) default(none) shared(h_state_schur, Hmr, temp_H, reserve_size)
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



    /*
    * marginalize 所有和 frame 相连的 edge: imu factor, projection factor
    * */
    bool ProblemSLAM::marginalize(const std::shared_ptr<Vertex>& vertex_pose, const std::shared_ptr<Vertex>& vertex_motion) {
        // 重新计算一篇ordering
        // initialize_ordering();
        ulong state_dim = _ordering_poses;

        // 所需被marginalize的edge
        auto &&marginalized_edges = get_connected_edges(vertex_pose);

        // 所需被marginalize的landmark
        ulong marginalized_landmark_size = 0;
        std::unordered_map<unsigned long, shared_ptr<Vertex>> marginalized_landmark;  // O(1)查找
        for (auto &edge : marginalized_edges) {
            auto vertices_edge = edge->vertices();
            for (auto &vertex : vertices_edge) {
                if (is_landmark_vertex(vertex)
                    && marginalized_landmark.find(vertex->id()) == marginalized_landmark.end()) {
                    // 修改landmark的ordering_id, 方便hessian的计算
                    vertex->set_ordering_id(state_dim + marginalized_landmark_size);
                    marginalized_landmark.insert(make_pair(vertex->id(), vertex));
                    marginalized_landmark_size += vertex->local_dimension();
                }
            }
        }

#ifdef USE_OPENMP
        std::vector<std::pair<unsigned long, shared_ptr<Vertex>>> marginalized_landmark_vector;
        marginalized_landmark_vector.reserve(marginalized_landmark.size());
        for (auto &landmark : marginalized_landmark) {
            marginalized_landmark_vector.emplace_back(landmark);
        }
#endif

        // 计算所需marginalize的edge的hessian
        ulong cols = state_dim + marginalized_landmark_size;
        MatXX h_state_landmark = VecX::Constant(cols, 1, 1e-6).asDiagonal();
        VecX b_state_landmark(VecX::Zero(cols));

#ifdef USE_OPENMP
        MatXX Hs[NUM_THREADS];       ///< Hessian矩阵
        VecX bs[NUM_THREADS];       ///< 负梯度
        for (unsigned int i = 0; i < NUM_THREADS; ++i) {
            Hs[i] = MatXX::Zero(cols, cols);
            bs[i] = VecX::Zero(cols);
        }

#pragma omp parallel for num_threads(NUM_THREADS) default(none) shared(marginalized_edges, Hs, bs)
        for (size_t n = 0; n < marginalized_edges.size(); ++n) {
            unsigned int index = omp_get_thread_num();

            auto &edge = marginalized_edges[n];
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
                if (jacobian_i.rows() == 0) continue;

                ulong index_i = v_i->ordering_id();
                ulong dim_i = v_i->local_dimension();

                MatXX JtW = jacobian_i.transpose() * robust_information;
                for (size_t j = i; j < vertices.size(); ++j) {
                    auto &&v_j = vertices[j];
                    if (v_j->is_fixed()) continue;

                    auto &&jacobian_j = jacobians[j];
                    if (jacobian_j.rows() == 0) continue;

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
        for (size_t n = 0; n < marginalized_edges.size(); ++n) {
            auto &edge = marginalized_edges[n];
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
                auto &&v_i = vertices[i];
                if (v_i->is_fixed()) continue;

                auto &&jacobian_i = jacobians[i];
                if (jacobian_i.rows() == 0) continue;

                ulong index_i = v_i->ordering_id();
                ulong dim_i = v_i->local_dimension();

                MatXX JtW = jacobian_i.transpose() * robust_information;
                for (size_t j = i; j < vertices.size(); ++j) {
                    auto &&v_j = vertices[j];
                    if (v_j->is_fixed()) continue;

                    auto &&jacobian_j = jacobians[j];
                    if (jacobian_j.rows() == 0) continue;

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
        if (marginalized_landmark_size > 0) {
//            MatXX Hss = h_state_landmark.block(0, 0, state_dim, state_dim);
            MatXX Hll = h_state_landmark.block(state_dim, state_dim, marginalized_landmark_size, marginalized_landmark_size);
            MatXX Hsl = h_state_landmark.block(0, state_dim, state_dim, marginalized_landmark_size);
//            MatXX Hlp = h_state_landmark.block(state_dim, 0, marginalized_landmark_size, state_dim);
            VecX bss = b_state_landmark.segment(0, state_dim);
            VecX bll = b_state_landmark.segment(state_dim, marginalized_landmark_size);

            MatXX temp_H(MatXX::Zero(marginalized_landmark_size, state_dim));  // Hll^-1 * Hsl^T
            VecX temp_b(VecX::Zero(marginalized_landmark_size, 1));   // Hll^-1 * bl
#ifdef USE_OPENMP
#pragma omp parallel for num_threads(NUM_THREADS) default(none) shared(marginalized_landmark_vector, Hll, Hsl, bll, temp_H, temp_b, state_dim)
            for (size_t n = 0; n < marginalized_landmark_vector.size(); ++n) {
                auto &&landmark_vertex = marginalized_landmark_vector[n];
                ulong idx = landmark_vertex.second->ordering_id() - state_dim;
                ulong size = landmark_vertex.second->local_dimension();
                if (size == 1) {
                    if (Hll(idx, idx) > 1e-12) {
                        temp_H.row(idx).noalias() = Hsl.col(idx) / Hll(idx, idx);
                        temp_b(idx) = bll(idx) / Hll(idx, idx);
                    } else {
                        temp_H.row(idx).setZero();
                        temp_b(idx) = 0.;
                    }
                } else {
                    auto Hmm_ldlt = Hll.block(idx, idx, size, size).ldlt();
                    if (Hmm_ldlt.info() == Eigen::Success) {
                        temp_H.block(idx, 0, size, state_dim).noalias() = Hmm_ldlt.solve(Hsl.block(0, idx, state_dim, size).transpose());
                        temp_b.segment(idx, size).noalias() = Hmm_ldlt.solve(bll.segment(idx, size));
                    } else {
                        temp_H.block(idx, 0, size, state_dim).setZero();
                        temp_b.segment(idx, size).setZero();
                    }
                }
            }
#else
            for (const auto& landmark_vertex : marginalized_landmark) {
                ulong idx = landmark_vertex.second->ordering_id() - state_dim;
                ulong size = landmark_vertex.second->local_dimension();
                if (size == 1) {
                    if (Hll(idx, idx) > 1e-12) {
                        temp_H.row(idx) = Hsl.col(idx) / Hll(idx, idx);
                        temp_b(idx) = bll(idx) / Hll(idx, idx);
                    } else {
                        temp_H.row(idx).setZero();
                        temp_b(idx) = 0.;
                    }
                } else {
                    auto Hmm_ldlt = Hll.block(idx, idx, size, size).ldlt();
                    if (Hmm_ldlt.info() == Eigen::Success) {
                        temp_H.block(idx, 0, size, state_dim) = Hmm_ldlt.solve(Hsl.block(0, idx, state_dim, size).transpose());
                        temp_b.segment(idx, size) = Hmm_ldlt.solve(bll.segment(idx, size));
                    } else {
                        temp_H.block(idx, 0, size, state_dim).setZero();
                        temp_b.segment(idx, size).setZero();
                    }
                }
            }
#endif

            // (Hpp - Hsl * Hll^-1 * Hlp) * dxp = bp - Hsl * Hll^-1 * bl
#ifdef USE_OPENMP
            h_state_schur = MatXX::Zero(state_dim, state_dim);
#pragma omp parallel for num_threads(NUM_THREADS) default(none) shared(h_state_schur, Hsl, temp_H, state_dim)
            for (ulong i = 0; i < state_dim; ++i) {
                h_state_schur(i, i) = -Hsl.row(i).dot(temp_H.col(i));
                for (ulong j = i + 1; j < state_dim; ++j) {
                    h_state_schur(i, j) = -Hsl.row(i).dot(temp_H.col(j));
                    h_state_schur(j, i) = h_state_schur(i, j);
                }
            }
            h_state_schur += h_state_landmark.block(0, 0, state_dim, state_dim);
#else
            h_state_schur = h_state_landmark.block(0, 0, state_dim, state_dim) - Hsl * temp_H;
#endif
            b_state_schur = bss - Hsl * temp_b;
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

        // 把需要marginalize的pose和motion的vertices移动到最下面
        ulong marginalized_state_dim = 0;
        auto move_vertex_to_bottom = [&](const std::shared_ptr<Vertex>& vertex) {
            ulong idx = vertex->ordering_id();
            ulong dim = vertex->local_dimension();
            marginalized_state_dim += dim;

            // 将 row i 移动矩阵最下面
            Eigen::MatrixXd temp_rows = h_state_schur.block(idx, 0, dim, state_dim);
            Eigen::MatrixXd temp_bot_rows = h_state_schur.block(idx + dim, 0, state_dim - idx - dim, state_dim);
            h_state_schur.block(idx, 0, state_dim - idx - dim, state_dim).noalias() = temp_bot_rows;
            h_state_schur.block(state_dim - dim, 0, dim, state_dim).noalias() = temp_rows;

            // 将 col i 移动矩阵最右边
            Eigen::MatrixXd temp_cols = h_state_schur.block(0, idx, state_dim, dim);
            Eigen::MatrixXd temp_right_cols = h_state_schur.block(0, idx + dim, state_dim, state_dim - idx - dim);
            h_state_schur.block(0, idx, state_dim, state_dim - idx - dim).noalias() = temp_right_cols;
            h_state_schur.block(0, state_dim - dim, state_dim, dim).noalias() = temp_cols;

            Eigen::VectorXd temp_b = b_state_schur.segment(idx, dim);
            Eigen::VectorXd temp_b_tail = b_state_schur.segment(idx + dim, state_dim - idx - dim);
            b_state_schur.segment(idx, state_dim - idx - dim).noalias() = temp_b_tail;
            b_state_schur.segment(state_dim - dim, dim).noalias() = temp_b;
        };
        if (vertex_motion) {
            move_vertex_to_bottom(vertex_motion);
        }
        move_vertex_to_bottom(vertex_pose);

        // marginalize与边相连的所有pose和motion顶点
        auto marginalize_bottom_vertex = [&](const std::shared_ptr<Vertex> &vertex) {
            ulong marginalized_size = vertex->local_dimension();
            ulong reserve_size = state_dim - marginalized_size;
//            MatXX Hrr = h_state_schur.block(0, 0, reserve_size, reserve_size);
            MatXX Hmm = h_state_schur.block(reserve_size, reserve_size, marginalized_size, marginalized_size);
            MatXX Hrm = h_state_schur.block(0, reserve_size, reserve_size, marginalized_size);
//            MatXX Hmr = h_state_schur.block(reserve_size, 0, marginalized_size, reserve_size);
            VecX brr = b_state_schur.segment(0, reserve_size);
            VecX bmm = b_state_schur.segment(reserve_size, marginalized_size);

            MatXX temp_H(MatXX::Zero(marginalized_size, reserve_size));  // Hmm^-1 * Hrm^T
            VecX temp_b(VecX::Zero(marginalized_size, 1));   // Hmm^-1 * bm
            ulong size = vertex->local_dimension();
            if (size == 1) {
                if (Hmm(0, 0) > 1e-12) {
                    temp_H = Hrm.transpose() / Hmm(0, 0);
                    temp_b = bmm / Hmm(0, 0);
                } else {
                    temp_H.setZero();
                    temp_b.setZero();
                }
                
            } else {
                auto Hmm_ldlt = Hmm.ldlt();
                if (Hmm_ldlt.info() == Eigen::Success) {
                    temp_H = Hmm_ldlt.solve(Hrm.transpose());
                    temp_b = Hmm_ldlt.solve(bmm);
                } else {
                    temp_H.setZero();
                    temp_b.setZero();
                }
            }

            // (Hrr - Hrm * Hmm^-1 * Hmr) * dxp = br - Hrm * Hmm^-1 * bm
#ifdef USE_OPENMP
            MatXX h_state_schur_block = h_state_schur.block(0, 0, reserve_size, reserve_size);
            h_state_schur = MatXX::Zero(reserve_size, reserve_size);
#pragma omp parallel for num_threads(NUM_THREADS) default(none) shared(h_state_schur, Hrm, temp_H, reserve_size)
            for (ulong i = 0; i < reserve_size; ++i) {
                h_state_schur(i, i) = -Hrm.row(i).dot(temp_H.col(i));
                for (ulong j = i + 1; j < reserve_size; ++j) {
                    h_state_schur(i, j) = -Hrm.row(i).dot(temp_H.col(j));
                    h_state_schur(j, i) = h_state_schur(i, j);
                }
            }
            h_state_schur += h_state_schur_block;
#else

            // (Hrr - Hrm * Hmm^-1 * Hmr) * dxp = br - Hrm * Hmm^-1 * bm
            h_state_schur = h_state_schur.block(0, 0, reserve_size, reserve_size) - Hrm * temp_H;
            // for (ulong i = 0; i < reserve_size; ++i) {
            //     h_state_schur(i, i) -= Hrm.row(i).dot(temp_H.col(i));
            //     for (ulong j = i + 1; j < reserve_size; ++j) {
            //         h_state_schur(i, j) -= Hrm.row(i).dot(temp_H.col(j));
            //         h_state_schur(j, i) = h_state_schur(i, j);
            //     }
            // }
#endif
            b_state_schur = brr - Hrm * temp_b;

            state_dim = reserve_size;
        };
        marginalize_bottom_vertex(vertex_pose);
        if (vertex_motion) {
            marginalize_bottom_vertex(vertex_motion);
        }

        _h_prior.topLeftCorner(state_dim, state_dim) = h_state_schur;
        _b_prior.topRows(state_dim) = b_state_schur;

        return true;
    }

    VecX ProblemSLAM::multiply_hessian(const VecX &x) {
        VecX v(VecX::Zero(x.rows(), x.cols()));
        for (unsigned long i = 0; i < _ordering_poses; i++) {
            v(i) += _hessian(i, i) * x(i);  // 计算对角线部分
            for (unsigned long j = i + 1; j < _ordering_generic; j++) { // 计算非对角线部分
                v(i) += _hessian(i, j) * x(j);  // 上三角部分
                v(j) += _hessian(i, j) * x(i);  // 下三角部分
            }
        }
        for (unsigned long i = _ordering_poses; i < _ordering_generic; ++i) {
            v(i) += _hessian(i, i) * x(i);
        }
        return v;
    }

    bool ProblemSLAM::solve_linear_system(VecX &delta_x) {
        if (delta_x.rows() != _ordering_generic) {
            delta_x.resize(_ordering_generic, 1);
        }

        /*
        * [Hpp Hpl][xp] = [bp]
        * [Hlp Hll][xl] = [bl]
        *
        * (Hpp - Hpl * Hll^-1 * Hlp) * xp = bp - Hpl * Hll^-1 * bl
        * Hll * xl = bl - Hlp * xp
        * */
        ulong reserve_size = _ordering_poses;
        ulong marg_size = _ordering_landmarks;

        // Hll
        VecX Hll = _hessian.block(reserve_size, reserve_size, marg_size, marg_size).diagonal();
        // Hll + λ
        Hll += _diag_lambda.tail(marg_size);
        // 由于叠加了lambda, 所以能够保证Hll可逆
        VecX Hll_inv = Hll.array().inverse();

        // Hll^-1 * Hpl^T
        MatXX temp_H = Hll_inv.asDiagonal() * _hessian.block(reserve_size, 0, marg_size, reserve_size);  
        // Hll^-1 * bl
        VecX temp_b = Hll_inv.cwiseProduct(_b.tail(marg_size));   

        // (Hpp - Hpl * Hll^-1 * Hlp) * dxp = bp - Hpl * Hll^-1 * bl
        // 这里即使叠加了lambda, 也有可能因为数值精度的问题而导致 _h_pp_schur 不可逆
#ifdef USE_OPENMP
        _h_pp_schur = MatXX::Zero(reserve_size, reserve_size);
#pragma omp parallel for num_threads(NUM_THREADS) default(none) shared(temp_H, reserve_size, marg_size)
        for (ulong i = 0; i < reserve_size; ++i) {
            _h_pp_schur(i, i) = -temp_H.col(i).dot(_hessian.col(i).tail(marg_size));
            for (ulong j = i + 1; j < reserve_size; ++j) {
                _h_pp_schur(i, j) = -temp_H.col(j).dot(_hessian.col(i).tail(marg_size));
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

    void ProblemSLAM::add_prior_to_hessian() {
        if(_h_prior.rows() > 0) {
            MatXX H_prior_tmp = _h_prior;
            VecX b_prior_tmp = _b_prior;

            // 只有没有被fix的pose存在先验, landmark没有先验
//#ifdef USE_OPENMP
//#pragma omp parallel for num_threads(NUM_THREADS) default(none) shared(H_prior_tmp, b_prior_tmp)
//            for (size_t n = 0; n < _pose_vertices.size(); ++n) {
//                auto &&vertex = _pose_vertices[n];
//                if (vertex->is_fixed() ) {
//                    ulong idx = vertex->ordering_id();
//                    ulong dim = vertex->local_dimension();
//                    H_prior_tmp.block(idx,0, dim, H_prior_tmp.cols()).setZero();
//                    H_prior_tmp.block(0,idx, H_prior_tmp.rows(), dim).setZero();
//                    b_prior_tmp.segment(idx,dim).setZero();
//                }
//            }
//#else
//            for (const auto& vertex: _idx_pose_vertices) {
//                if (is_pose_vertex(vertex.second) && vertex.second->is_fixed() ) {
//                    ulong idx = vertex.second->ordering_id();
//                    ulong dim = vertex.second->local_dimension();
//                    H_prior_tmp.block(idx,0, dim, H_prior_tmp.cols()).setZero();
//                    H_prior_tmp.block(0,idx, H_prior_tmp.rows(), dim).setZero();
//                    b_prior_tmp.segment(idx,dim).setZero();
////                std::cout << " fixed prior, set the Hprior and bprior part to zero, idx: "<<idx <<" dim: "<<dim<<std::endl;
//                }
//            }
//#endif
            // 应该没有必要用多线程
            for (auto & vertex : _pose_vertices) {
                if (vertex->is_fixed() ) {
                    ulong idx = vertex->ordering_id();
                    ulong dim = vertex->local_dimension();
                    H_prior_tmp.block(idx,0, dim, H_prior_tmp.cols()).setZero();
                    H_prior_tmp.block(0,idx, H_prior_tmp.rows(), dim).setZero();
                    b_prior_tmp.segment(idx,dim).setZero();
                }
            }
            _hessian.topLeftCorner(_ordering_poses, _ordering_poses) += H_prior_tmp;
            _b.head(_ordering_poses) += b_prior_tmp;
        }
    }

    void ProblemSLAM::initialize_ordering() {
        // std::cout << "new: " << _ordering_poses << ", " << _ordering_landmarks << std::endl;

        // _ordering_generic = 0;

        // // 分配pose的维度
        // _ordering_poses = 0;
        // _idx_pose_vertices.clear();
        // for (auto &vertex: _vertices) {
        //     if (is_pose_vertex(vertex.second)) {
        //         vertex.second->set_ordering_id(_ordering_poses);
        //         _idx_pose_vertices.emplace_back(vertex.second->ordering_id(), vertex.second);
        //         // _idx_pose_vertices.insert(pair<ulong, std::shared_ptr<Vertex>>(vertex.second->id(), vertex.second));
        //         _ordering_poses += vertex.second->local_dimension();
        //     }
        // }

        // // 分配landmark的维度
        // _ordering_landmarks = 0;
        // _idx_landmark_vertices.clear();
        // for (auto &vertex: _vertices) {
        //     if (is_landmark_vertex(vertex.second)) {
        //         vertex.second->set_ordering_id(_ordering_landmarks + _ordering_poses);
        //         _idx_landmark_vertices.emplace_back(vertex.second->ordering_id(), vertex.second);
        //         // _idx_landmark_vertices.insert(pair<ulong, std::shared_ptr<Vertex>>(vertex.second->id(), vertex.second));
        //         _ordering_landmarks += vertex.second->local_dimension();
        //     }
        // }

        _ordering_generic = _ordering_poses + _ordering_landmarks;


        // std::cout << "old: " << _ordering_poses << ", " << _ordering_landmarks << std::endl;
    }

    bool ProblemSLAM::add_state_vertex(const std::shared_ptr<Vertex>& vertex) {
        if (Problem::add_vertex(vertex)) {
            vertex->set_ordering_id(_ordering_poses);
            _ordering_poses += vertex->local_dimension();
            _pose_vertices.emplace_back(vertex);
            return true;
        }
        return false;
    }

    bool ProblemSLAM::add_landmark_vertex(const std::shared_ptr<Vertex>& vertex) {
        if (Problem::add_vertex(vertex)) {
            vertex->set_ordering_id(_ordering_landmarks + _ordering_poses);
            _ordering_landmarks += vertex->local_dimension();
            _landmark_vertices.emplace_back(vertex);
            return true;
        }
        return false;
    }

    bool ProblemSLAM::is_pose_vertex(const std::shared_ptr<Vertex>& v) {
        string type = v->type_info();
        return type == string("VertexPose") || type == string("VertexMotion");
    }

    bool ProblemSLAM::is_landmark_vertex(const std::shared_ptr<Vertex>& v) {
        string type = v->type_info();
        return type == string("VertexPointXYZ") || type == string("VertexInverseDepth");
    }

    void ProblemSLAM::update_prior(const VecX &delta_x) {
        if (_b_prior.rows() > 0) {
            _b_prior_bp = _b_prior;
            _b_prior -= _h_prior * delta_x.head(_ordering_poses);
        }
    }

    bool ProblemSLAM::check_ordering() {
        unsigned long current_ordering = 0;
        for (const auto& v: _pose_vertices) {
            if (v->ordering_id() != current_ordering) {
                return false;
            }
            current_ordering += v->local_dimension();
        }

        for (const auto& v: _landmark_vertices) {
            if (v->ordering_id() != current_ordering) {
                return false;
            }
            current_ordering += v->local_dimension();
        }

        return true;
    }
}