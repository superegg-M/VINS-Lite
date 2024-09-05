//
// Created by Cain on 2024/3/7.
//

#include <iostream>
#include "utility/utility.h"
#include "../thirdparty/Sophus/sophus/se3.hpp"
#include "../../include/parameters.h"
#include <omp.h>
#include <Eigen/Sparse>
#include <valarray>
#include <atomic>

#include "backend/problem_slam.h"

// #define USE_PCG_SOLVER


namespace graph_optimization {
    using namespace std;

    /*
    * marginalize 所有和 frame 相连的 edge: imu factor, projection factor
    * vertex排序必须满足: extra, pose[0], motion[0], pose[1], motion[1], ..., pose[WINDOWS_SIZE], motion[WINDOWS_SIZE]
    * 若不满足, 则需要外部调整_H_prior和_b_prior的顺序, 以适应vertex的排序
    * */
    bool ProblemSLAM::marginalize(const std::shared_ptr<Vertex>& vertex_pose, 
                                  const std::shared_ptr<Vertex>& vertex_motion,
                                  const std::vector<std::shared_ptr<Vertex>> &marginalized_landmarks,
                                  const std::vector<std::shared_ptr<Edge>> &marginalized_edges) {
        // 重新计算一篇ordering
        // initialize_ordering();
        ulong state_dim = _ordering_poses;
        ulong marginalized_landmarks_size = marginalized_landmarks.size();
        ulong cols = state_dim + marginalized_landmarks_size;

        // 为H加上eps, 避免出现对角线元素为0的情况
        MatXX h_state_landmark = VecX::Constant(cols, 1, 1e-6).asDiagonal();
        VecX b_state_landmark(VecX::Zero(cols));

        // 修改landmark的ordering_id, 方便hessian的计算
        for (unsigned i = 0; i < marginalized_landmarks_size; ++i) {
            marginalized_landmarks[i]->set_ordering_id(state_dim + i);
        }

#ifdef USE_OPENMP
        auto H = h_state_landmark.data();   ///< Hessian矩阵
        auto b = b_state_landmark.data();   ///< 负梯度
#pragma omp parallel for num_threads(NUM_THREADS) default(none) shared(marginalized_edges, H, b, cols)
        for (size_t n = 0; n < marginalized_edges.size(); ++n) {
            unsigned int index = omp_get_thread_num();

            auto &edge = marginalized_edges[n];
            auto &&jacobians = edge->jacobians();
            auto &&verticies = edge->vertices();
            // assert(jacobians.size() == vertices.size());

            // 计算edge的鲁棒权重
            // double drho;
            // MatXX robust_information;
            // VecX robust_residual;
            // edge->robust_information(drho, robust_information, robust_residual);
            auto &&robust_information = edge->get_robust_info();
            auto &&robust_residual = edge->get_robust_res();

            for (size_t i = 0; i < verticies.size(); ++i) {
                auto &&v_i = verticies[i];
                if (v_i->is_fixed()) continue;    // Hessian 里不需要添加它的信息，也就是它的雅克比为 0

                auto &&jacobian_i = jacobians[i];
                if (jacobian_i.rows() == 0) continue;

                ulong index_i = v_i->ordering_id();
                ulong dim_i = v_i->local_dimension();

                MatXX JtW = jacobian_i.transpose() * robust_information.selfadjointView<Eigen::Upper>();
                for (size_t j = i; j < verticies.size(); ++j) {
                    auto &&v_j = verticies[j];
                    if (v_j->is_fixed()) continue;

                    auto &&jacobian_j = jacobians[j];
                    if (jacobian_j.rows() == 0) continue;

                    ulong index_j = v_j->ordering_id();
                    ulong dim_j = v_j->local_dimension();

                    // if (index_i < index_j) {
                    //     Hs[index].block(index_i, index_j, dim_i, dim_j).noalias() += JtW * jacobian_j;
                    // } else if (index_i > index_j) {
                    //     Hs[index].block(index_j, index_i, dim_j, dim_i).noalias() += (JtW * jacobian_j).transpose();
                    // } else {
                    //     Hs[index].block(index_i, index_i, dim_i, dim_i).triangularView<Eigen::Upper>() += JtW * jacobian_j;
                    // }

//                    assert(v_j->ordering_id() != -1);
                    MatXX hessian = JtW * jacobian_j;   // TODO: 这里能继续优化, 因为J'*W*J也是对称矩阵
                    auto h_pt = hessian.data();
                    // 所有的信息矩阵叠加起来
                    for (ulong c = 0; c < dim_j; ++c) {
                        for (ulong r = 0; r < dim_i; ++r) {
                            #pragma omp atomic
                            H[cols * (index_j + c) + index_i + r] += *(h_pt + dim_i * c + r);
                        }
                    }
                    // Hs[index].block(index_i, index_j, dim_i, dim_j).noalias() += hessian;
                    if (j != i) {
                        // 对称的下三角
                        for (ulong c = 0; c < dim_j; ++c) {
                            for (ulong r = 0; r < dim_i; ++r) {
                                #pragma omp atomic
                                H[cols * (index_i + r) + index_j + c] += *(h_pt + dim_i * c + r);
                            } 
                        }
                        // Hs[index].block(index_j, index_i, dim_j, dim_i).noalias() += hessian.transpose();
                    }
                }
                VecX g = jacobian_i.transpose() * robust_residual;
                auto g_pt = g.data();
                for (ulong r = 0; r < dim_i; ++r) {
                    #pragma omp atomic
                    b[index_i + r] -= *(g_pt + r);
                }
                // bs[index].segment(index_i, dim_i).noalias() -= jacobian_i.transpose() * robust_residual;
            }
        }
#else
        for (size_t n = 0; n < marginalized_edges.size(); ++n) {
            auto &edge = marginalized_edges[n];
            // 若曾经solve problem, 则无需再次计算
            // edge->compute_residual();
            // edge->compute_jacobians();
            auto &&jacobians = edge->jacobians();
            auto &&vertices = edge->vertices();

            // assert(jacobians.size() == vertices.size());

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

                    // assert(hessian.rows() == v_i->local_dimension() && hessian.cols() == v_j->local_dimension());
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
        if (marginalized_landmarks_size > 0) {
            // Hll
            VecX Hll = h_state_landmark.diagonal().segment(state_dim, marginalized_landmarks_size);
            // 由于叠加了eps, 所以能够保证Hll可逆
            VecX Hll_inv = Hll.array().inverse();

            // Hll^-1 * Hsl^T
            MatXX temp_H = Hll_inv.asDiagonal() * h_state_landmark.block(state_dim, 0, marginalized_landmarks_size, state_dim);
            // Hll^-1 * bl
//            VecX temp_b = Hll_inv.cwiseProduct(b_state_landmark.tail(marginalized_landmark_size));

            // (Hss - Hsl * Hll^-1 * Hls) * xs = bs - Hsl * Hll^-1 * bl
#ifdef USE_OPENMP
            static std::vector<std::pair<ulong, ulong>> coord;
            if (coord.size() != ((state_dim + 1) * state_dim) / 2) {
                coord.clear();
                coord.reserve(((state_dim + 1) * state_dim) / 2);
                for (ulong i = 0; i < state_dim; ++i) {
                    for (ulong j = i; j < state_dim; ++j) {
                        coord.emplace_back(i, j);
                    }
                }
            }
            h_state_schur = MatXX::Zero(state_dim, state_dim);
#pragma omp parallel for num_threads(NUM_THREADS) default(none) shared(coord, h_state_landmark, h_state_schur, temp_H, state_dim, marginalized_landmarks_size)
            for (size_t n = 0; n < coord.size(); ++n) {
                ulong i = coord[n].first;
                ulong j = coord[n].second;
                h_state_schur(i, j) = -temp_H.col(j).dot(h_state_landmark.col(i).tail(marginalized_landmarks_size));
                h_state_schur(j, i) = h_state_schur(i, j);
            }
            // for (ulong i = 0; i < state_dim; ++i) {
            //     h_state_schur(i, i) = -temp_H.col(i).dot(h_state_landmark.col(i).tail(marginalized_landmarks_size));
            //     for (ulong j = i + 1; j < state_dim; ++j) {
            //         h_state_schur(i, j) = -temp_H.col(j).dot(h_state_landmark.col(i).tail(marginalized_landmarks_size));
            //         h_state_schur(j, i) = h_state_schur(i, j);
            //     }
            // }
            h_state_schur += h_state_landmark.block(0, 0, state_dim, state_dim);
#else
            h_state_schur = h_state_landmark.block(0, 0, state_dim, state_dim) - h_state_landmark.block(0, state_dim, state_dim, marginalized_landmarks_size) * temp_H;
#endif
            // h_state_schur.triangularView<Eigen::Upper>() = h_state_landmark.block(0, 0, state_dim, state_dim) - h_state_landmark.block(0, state_dim, state_dim, marginalized_landmarks_size) * temp_H;
            b_state_schur = b_state_landmark.head(state_dim) - temp_H.transpose() * b_state_landmark.tail(marginalized_landmarks_size);
        } else {
            h_state_schur = h_state_landmark;
            // h_state_schur.triangularView<Eigen::Upper>() = h_state_landmark;
            b_state_schur = b_state_landmark;
        }

        // 叠加之前的先验
        if(_h_prior.rows() > 0) {
            h_state_schur += _h_prior;
            // h_state_schur.triangularView<Eigen::Upper>() += _h_prior;
            b_state_schur += _b_prior;
        } else {
            _h_prior = MatXX::Zero(state_dim, state_dim);
            _b_prior = VecX::Zero(state_dim, 1);
        }
        // h_state_schur = h_state_schur.selfadjointView<Eigen::Upper>();

#ifdef DIRECT_MARGINALIZE
        // 边缘化
        ulong marg_index = vertex_pose->ordering_id();
        ulong marg_dim = vertex_pose->local_dimension() + vertex_motion->local_dimension();
        ulong res_index = marg_index + marg_dim;
        ulong res_dim = state_dim - res_index;

        auto &&h_11_ldlt = h_state_schur.block(marg_index, marg_index, marg_dim, marg_dim).ldlt();
        MatXX h_10_hat = h_11_ldlt.solve(h_state_schur.block(marg_index, 0, marg_dim, marg_index));
        MatXX h_12_hat = h_11_ldlt.solve(h_state_schur.block(marg_index, res_index, marg_dim, res_dim));

        _h_prior.block(0, 0, marg_index, marg_index).triangularView<Eigen::Upper>() = h_state_schur.block(0, 0, marg_index, marg_index)
                                                                                      - h_state_schur.block(0, marg_index, marg_index, marg_dim) * h_10_hat;
        _h_prior.block(0, marg_index, marg_index, res_dim).noalias() = h_state_schur.block(0, res_index, marg_index, res_dim)
                                                                       - h_state_schur.block(0, marg_index, marg_index, marg_dim) * h_12_hat;
        _h_prior.block(marg_index, marg_index, res_dim, res_dim).triangularView<Eigen::Upper>() = h_state_schur.block(res_index, res_index, res_dim, res_dim)
                                                                                                  - h_state_schur.block(res_index, marg_index, res_dim, marg_dim) * h_12_hat;
        _h_prior = _h_prior.selfadjointView<Eigen::Upper>();

        _b_prior.head(marg_index) = b_state_schur.head(marg_index) - h_10_hat.transpose() * b_state_schur.segment(marg_index, marg_dim);
        _b_prior.segment(marg_index, res_dim) = b_state_schur.tail(res_dim) - h_12_hat.transpose() * b_state_schur.segment(marg_index, marg_dim);
#else
        /* 先移动再边缘化 */
        // 把需要marginalize的pose和motion的vertices移动到最下面
        ulong idx = vertex_pose->ordering_id();
        ulong dim = vertex_pose->local_dimension() + (vertex_motion ? vertex_motion->local_dimension() : 0);

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

        // marginalize与边相连的所有pose和motion顶点
        ulong marginalized_size = dim; //vertex_pose->local_dimension() + vertex_motion->local_dimension();
        ulong reserve_size = state_dim - marginalized_size;

        // Hmm^-1 * Hrm^T
        MatXX temp_H(MatXX::Zero(marginalized_size, reserve_size));

        // Hmm^-1 * bm
//            VecX temp_b(VecX::Zero(marginalized_size, 1));

        if (marginalized_size == 1) {
            if (h_state_schur(reserve_size, reserve_size) > 1e-12) {
                temp_H = h_state_schur.block(reserve_size, 0, marginalized_size, reserve_size) / h_state_schur(reserve_size, reserve_size);
//                    temp_b = bmm / h_state_schur(reserve_size, reserve_size);
            } else {
                temp_H.setZero();
//                    temp_b.setZero();
            }

        } else {
            auto &&Hmm_ldlt = h_state_schur.block(reserve_size, reserve_size, marginalized_size, marginalized_size).ldlt();
            if (Hmm_ldlt.info() == Eigen::Success) {
                temp_H = Hmm_ldlt.solve(h_state_schur.block(reserve_size, 0, marginalized_size, reserve_size));
//                    temp_b = Hmm_ldlt.solve(bmm);
            } else {
                temp_H.setZero();
//                    temp_b.setZero();
            }
        }

        // (Hrr - Hrm * Hmm^-1 * Hmr) * dxp = br - Hrm * Hmm^-1 * bm
#ifdef USE_OPENMP
        _h_prior.topLeftCorner(reserve_size, reserve_size) = h_state_schur.topLeftCorner(reserve_size, reserve_size);
#pragma omp parallel for num_threads(NUM_THREADS) default(none) shared(h_state_schur, temp_H, reserve_size, marginalized_size)
        for (ulong i = 0; i < reserve_size; ++i) {
            _h_prior(i, i) -= temp_H.col(i).dot(h_state_schur.col(i).tail(marginalized_size));
            for (ulong j = i + 1; j < reserve_size; ++j) {
                _h_prior(i, j) -= temp_H.col(j).dot(h_state_schur.col(i).tail(marginalized_size));
                _h_prior(j, i) = _h_prior(i, j);
            }
        }
#else
        // (Hrr - Hrm * Hmm^-1 * Hmr) * dxp = br - Hrm * Hmm^-1 * bm
        _h_prior.topLeftCorner(reserve_size, reserve_size) = h_state_schur.topLeftCorner(reserve_size, reserve_size) - h_state_schur.block(0, reserve_size, reserve_size, marginalized_size) * temp_H;
#endif
        _b_prior.head(reserve_size) = b_state_schur.head(reserve_size) - temp_H.transpose() * b_state_schur.tail(marginalized_size);
#endif
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

// #ifdef USE_OPENMP
//         std::vector<std::pair<unsigned long, shared_ptr<Vertex>>> marginalized_landmark_vector;
//         marginalized_landmark_vector.reserve(marginalized_landmark.size());
//         for (auto &landmark : marginalized_landmark) {
//             marginalized_landmark_vector.emplace_back(landmark);
//         }
// #endif

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
            // assert(jacobians.size() == vertices.size());

            // 计算edge的鲁棒权重
            // double drho;
            // MatXX robust_information;
            // VecX robust_residual;
            // edge->robust_information(drho, robust_information, robust_residual);
            auto &&robust_information = edge->get_robust_info();
            auto &&robust_residual = edge->get_robust_res();

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

            // assert(jacobians.size() == vertices.size());

            // 计算edge的鲁棒权重
            // double drho;
            // MatXX robust_information;
            // VecX robust_residual;
            // edge->robust_information(drho, robust_information, robust_residual);
            auto &&robust_information = edge->get_robust_info();
            auto &&robust_residual = edge->get_robust_res();

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

                    // assert(hessian.rows() == v_i->local_dimension() && hessian.cols() == v_j->local_dimension());
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
            // Hll
            VecX Hll = h_state_landmark.block(state_dim, state_dim, marginalized_landmark_size, marginalized_landmark_size).diagonal();
            // 由于叠加了eps, 所以能够保证Hll可逆
            VecX Hll_inv = Hll.array().inverse();

            // Hll^-1 * Hsl^T
            MatXX temp_H = Hll_inv.asDiagonal() * h_state_landmark.block(state_dim, 0, marginalized_landmark_size, state_dim);
            // Hll^-1 * bl
//            VecX temp_b = Hll_inv.cwiseProduct(b_state_landmark.tail(marginalized_landmark_size));

            // (Hss - Hsl * Hll^-1 * Hlp) * dxp = bp - Hsl * Hll^-1 * bl
#ifdef USE_OPENMP
            h_state_schur = MatXX::Zero(state_dim, state_dim);
#pragma omp parallel for num_threads(NUM_THREADS) default(none) shared(h_state_landmark, h_state_schur, temp_H, state_dim, marginalized_landmark_size)
            for (ulong i = 0; i < state_dim; ++i) {
                h_state_schur(i, i) = -temp_H.col(i).dot(h_state_landmark.col(i).tail(marginalized_landmark_size));
                for (ulong j = i + 1; j < state_dim; ++j) {
                    h_state_schur(i, j) = -temp_H.col(j).dot(h_state_landmark.col(i).tail(marginalized_landmark_size));
                    h_state_schur(j, i) = h_state_schur(i, j);
                }
            }
            h_state_schur += h_state_landmark.block(0, 0, state_dim, state_dim);
#else
            h_state_schur = h_state_landmark.block(0, 0, state_dim, state_dim) - h_state_landmark.block(0, state_dim, state_dim, marginalized_landmark_size) * temp_H;
#endif
            b_state_schur = b_state_landmark.head(state_dim) - temp_H.transpose() * b_state_landmark.tail(marginalized_landmark_size);
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

            if (idx + dim == state_dim) {
                return;
            }

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

            // Hmm^-1 * Hrm^T
            MatXX temp_H(MatXX::Zero(marginalized_size, reserve_size));

            // Hmm^-1 * bm
//            VecX temp_b(VecX::Zero(marginalized_size, 1));

            ulong size = vertex->local_dimension();
            if (size == 1) {
                if (h_state_schur(reserve_size, reserve_size) > 1e-12) {
                    temp_H = h_state_schur.block(reserve_size, 0, marginalized_size, reserve_size) / h_state_schur(reserve_size, reserve_size);
//                    temp_b = bmm / h_state_schur(reserve_size, reserve_size);
                } else {
                    temp_H.setZero();
//                    temp_b.setZero();
                }
                
            } else {
                auto &&Hmm_ldlt = h_state_schur.block(reserve_size, reserve_size, marginalized_size, marginalized_size).ldlt();
                if (Hmm_ldlt.info() == Eigen::Success) {
                    temp_H = Hmm_ldlt.solve(h_state_schur.block(reserve_size, 0, marginalized_size, reserve_size));
//                    temp_b = Hmm_ldlt.solve(bmm);
                } else {
                    temp_H.setZero();
//                    temp_b.setZero();
                }
            }

            // (Hrr - Hrm * Hmm^-1 * Hmr) * dxp = br - Hrm * Hmm^-1 * bm
#ifdef USE_OPENMP
            MatXX h_state_schur_bp = h_state_schur;
            h_state_schur = MatXX::Zero(reserve_size, reserve_size);
#pragma omp parallel for num_threads(NUM_THREADS) default(none) shared(h_state_schur, h_state_schur_bp, temp_H, reserve_size, marginalized_size)
            for (ulong i = 0; i < reserve_size; ++i) {
                h_state_schur(i, i) = -temp_H.col(i).dot(h_state_schur_bp.col(i).tail(marginalized_size));
                for (ulong j = i + 1; j < reserve_size; ++j) {
                    h_state_schur(i, j) = -temp_H.col(j).dot(h_state_schur_bp.col(i).tail(marginalized_size));
                    h_state_schur(j, i) = h_state_schur(i, j);
                }
            }
            h_state_schur += h_state_schur_bp.block(0, 0, reserve_size, reserve_size);
#else

            // (Hrr - Hrm * Hmm^-1 * Hmr) * dxp = br - Hrm * Hmm^-1 * bm
            h_state_schur = h_state_schur.block(0, 0, reserve_size, reserve_size) - h_state_schur.block(0, reserve_size, reserve_size, marginalized_size) * temp_H;
#endif
            b_state_schur = b_state_schur.head(reserve_size) - temp_H.transpose() * b_state_schur.tail(marginalized_size);

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
        TicToc t_linear_system;
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
        if (_h_pp_schur.rows() != reserve_size) {
            _h_pp_schur = MatXX::Zero(reserve_size, reserve_size);
        } else {
            _h_pp_schur.setZero();
        }
        static std::vector<std::pair<ulong, ulong>> coord;
        if (coord.size() != ((reserve_size + 1) * reserve_size) / 2) {
            coord.clear();
            coord.reserve(((reserve_size + 1) * reserve_size) / 2);
            for (ulong i = 0; i < reserve_size; ++i) {
                for (ulong j = i; j < reserve_size; ++j) {
                    coord.emplace_back(i, j);
                }
            }
        }
#pragma omp parallel for num_threads(NUM_THREADS) default(none) shared(temp_H, reserve_size, marg_size, coord)
        for (size_t n = 0; n < coord.size(); ++n) {
                ulong i = coord[n].first;
                ulong j = coord[n].second;
                _h_pp_schur(i, j) = -temp_H.col(j).dot(_hessian.col(i).tail(marg_size));
                _h_pp_schur(j, i) = _h_pp_schur(i, j);
        }
        // for (ulong i = 0; i < reserve_size; ++i) {
        //     _h_pp_schur(i, i) = -temp_H.col(i).dot(_hessian.col(i).tail(marg_size));
        //     for (ulong j = i + 1; j < reserve_size; ++j) {
        //         _h_pp_schur(i, j) = -temp_H.col(j).dot(_hessian.col(i).tail(marg_size));
        //         _h_pp_schur(j, i) = _h_pp_schur(i, j);
        //     }
        // }
        _h_pp_schur += _hessian.block(0, 0, reserve_size, reserve_size);
#else
        // Hpp - Hpl * Hll^-1 * Hlp
        _h_pp_schur = _hessian.block(0, 0, reserve_size, reserve_size) - _hessian.block(0, reserve_size, reserve_size, marg_size) * temp_H;
#endif
        // Hpp - Hpl * Hll^-1 * Hlp + λ
        _h_pp_schur += _diag_lambda.head(reserve_size).asDiagonal();

        // bp - Hpl * Hll^-1 * bl
        _b_pp_schur = _b.head(reserve_size) - temp_H.transpose() * _b.tail(marg_size);

        _t_linear_schur_cost += t_linear_system.toc();
        t_linear_system.tic();

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

        _t_linear_ldlt_cost += t_linear_system.toc();
        _t_linear_system_cost = _t_linear_schur_cost + _t_linear_ldlt_cost;

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

    bool ProblemSLAM::add_reproj_edge(const std::shared_ptr<Edge> &edge) {
        _edges.emplace_back(edge);
        _reproj_edges.emplace_back(edge);
        return true;
        
        // if (Problem::add_edge(edge)) {
        //     _reproj_edges.emplace_back(edge);
        //     return true;
        // }
        // return false;
    }

    bool ProblemSLAM::add_imu_edge(const std::shared_ptr<Edge> &edge) {
        _edges.emplace_back(edge);
        _imu_edges.emplace_back(edge);
        return true;

        // if (Problem::add_edge(edge)) {
        //     _imu_edges.emplace_back(edge);
        //     return true;
        // }
        // return false;
    }

    void ProblemSLAM::update_hessian() {
        TicToc t_h, t_cost;

        ulong size = _ordering_generic;
        if (_hessian.rows() != size) {
            _hessian = MatXX::Zero(size, size); ///< Hessian矩阵
            _b = VecX::Zero(size);  ///< 负梯度 
        } else {
            _hessian.setZero();
            _b.setZero();
        }

#ifdef USE_OPENMP
        auto H = _hessian.data();
        auto b = _b.data();
        // std::atomic<std::valarray> h[size];
        // for (auto &it : h) {
        //     h.resize(size);
        // }
        
        // 遍历每个残差，并计算他们的雅克比，得到最后的 H = J^T * J
//        omp_set_num_threads(NUM_THREADS);
#pragma omp parallel for num_threads(NUM_THREADS) default(none) shared(H, b, size)
        for (size_t n = 0; n < _reproj_edges.size(); ++n) {
            unsigned int index = omp_get_thread_num();

            auto &&edge = _reproj_edges[n];
            auto &&jacobians = edge->jacobians();
            auto &&verticies = edge->vertices();
            // assert(jacobians.size() == verticies.size());

            // 计算edge的鲁棒权重
            // double drho;
            // MatXX robust_information;
            // VecX robust_residual;
            // edge->robust_information(drho, robust_information, robust_residual);
            edge->compute_robust();
            auto &&robust_information = edge->get_robust_info();
            auto &&robust_residual = edge->get_robust_res();
            for (size_t i = 0; i < verticies.size(); ++i) {
                auto &&v_i = verticies[i];
                if (v_i->is_fixed()) continue;    // Hessian 里不需要添加它的信息，也就是它的雅克比为 0

                auto &&jacobian_i = jacobians[i];
                ulong index_i = v_i->ordering_id();
                ulong dim_i = v_i->local_dimension();

                MatXX JtW = jacobian_i.transpose() * robust_information.selfadjointView<Eigen::Upper>();
                for (size_t j = i; j < verticies.size(); ++j) {
                    auto &&v_j = verticies[j];
                    if (v_j->is_fixed()) continue;

                    auto &&jacobian_j = jacobians[j];
                    ulong index_j = v_j->ordering_id();
                    ulong dim_j = v_j->local_dimension();

                    // if (index_i < index_j) {
                    //     Hs[index].block(index_i, index_j, dim_i, dim_j).noalias() += JtW * jacobian_j;
                    // } else if (index_i > index_j) {
                    //     Hs[index].block(index_j, index_i, dim_j, dim_i).noalias() += (JtW * jacobian_j).transpose();
                    // } else {
                    //     Hs[index].block(index_i, index_i, dim_i, dim_i).triangularView<Eigen::Upper>() += JtW * jacobian_j;
                    // }

//                    assert(v_j->ordering_id() != -1);
                    MatXX hessian = JtW * jacobian_j;   // TODO: 这里能继续优化, 因为J'*W*J也是对称矩阵
                    auto h_pt = hessian.data();
                    // 所有的信息矩阵叠加起来
                    for (ulong c = 0; c < dim_j; ++c) {
                        for (ulong r = 0; r < dim_i; ++r) {
                            #pragma omp atomic
                            H[size * (index_j + c) + index_i + r] += *(h_pt + dim_i * c + r);
                        }
                    }
                    // Hs[index].block(index_i, index_j, dim_i, dim_j).noalias() += hessian;
                    if (j != i) {
                        // 对称的下三角
                        for (ulong c = 0; c < dim_j; ++c) {
                            for (ulong r = 0; r < dim_i; ++r) {
                                #pragma omp atomic
                                H[size * (index_i + r) + index_j + c] += *(h_pt + dim_i * c + r);
                            } 
                        }
                        // Hs[index].block(index_j, index_i, dim_j, dim_i).noalias() += hessian.transpose();
                    }
                }
                VecX g = jacobian_i.transpose() * robust_residual;
                auto g_pt = g.data();
                for (ulong r = 0; r < dim_i; ++r) {
                    #pragma omp atomic
                    b[index_i + r] -= *(g_pt + r);
                }
                // bs[index].segment(index_i, dim_i).noalias() -= jacobian_i.transpose() * robust_residual;
            }
        }

#pragma omp parallel for num_threads(NUM_THREADS) default(none) shared(H, b, size)
        for (size_t n = 0; n < _imu_edges.size(); ++n) {
            unsigned int index = omp_get_thread_num();

            auto &&edge = _imu_edges[n];
            auto &&jacobians = edge->jacobians();
            auto &&verticies = edge->vertices();
            // assert(jacobians.size() == verticies.size());

            // 计算edge的鲁棒权重
            // double drho;
            // MatXX robust_information;
            // VecX robust_residual;
            // edge->robust_information(drho, robust_information, robust_residual);
            edge->compute_robust();
            auto &&robust_information = edge->get_robust_info();
            auto &&robust_residual = edge->get_robust_res();
            for (size_t i = 0; i < verticies.size(); ++i) {
                auto &&v_i = verticies[i];
                if (v_i->is_fixed()) continue;    // Hessian 里不需要添加它的信息，也就是它的雅克比为 0

                auto &&jacobian_i = jacobians[i];
                ulong index_i = v_i->ordering_id();
                ulong dim_i = v_i->local_dimension();

                MatXX JtW = jacobian_i.transpose() * robust_information.selfadjointView<Eigen::Upper>();
                for (size_t j = i; j < verticies.size(); ++j) {
                    auto &&v_j = verticies[j];
                    if (v_j->is_fixed()) continue;

                    auto &&jacobian_j = jacobians[j];
                    ulong index_j = v_j->ordering_id();
                    ulong dim_j = v_j->local_dimension();

                    // if (index_i < index_j) {
                    //     Hs[index].block(index_i, index_j, dim_i, dim_j).noalias() += JtW * jacobian_j;
                    // } else if (index_i > index_j) {
                    //     Hs[index].block(index_j, index_i, dim_j, dim_i).noalias() += (JtW * jacobian_j).transpose();
                    // } else {
                    //     Hs[index].block(index_i, index_i, dim_i, dim_i).triangularView<Eigen::Upper>() += JtW * jacobian_j;
                    // }

//                    assert(v_j->ordering_id() != -1);
                    MatXX hessian = JtW * jacobian_j;   // TODO: 这里能继续优化, 因为J'*W*J也是对称矩阵
                    auto h_pt = hessian.data();
                    // 所有的信息矩阵叠加起来
                    for (ulong c = 0; c < dim_j; ++c) {
                        for (ulong r = 0; r < dim_i; ++r) {
                            #pragma omp atomic
                            H[size * (index_j + c) + index_i + r] += *(h_pt + dim_i * c + r);
                        }
                    }
                    // Hs[index].block(index_i, index_j, dim_i, dim_j).noalias() += hessian;
                    if (j != i) {
                        // 对称的下三角
                        for (ulong c = 0; c < dim_j; ++c) {
                            for (ulong r = 0; r < dim_i; ++r) {
                                #pragma omp atomic
                                H[size * (index_i + r) + index_j + c] += *(h_pt + dim_i * c + r);
                            } 
                        }
                        // Hs[index].block(index_j, index_i, dim_j, dim_i).noalias() += hessian.transpose();
                    }
                }
                VecX g = jacobian_i.transpose() * robust_residual;
                auto g_pt = g.data();
                for (ulong r = 0; r < dim_i; ++r) {
                    #pragma omp atomic
                    b[index_i + r] -= *(g_pt + r);
                }
                // bs[index].segment(index_i, dim_i).noalias() -= jacobian_i.transpose() * robust_residual;
            }
        }
#else
        // 遍历每个残差，并计算他们的雅克比，得到最后的 H = J^T * J
        for (auto &edge: edges) {
            auto &&jacobians = edge->jacobians();
            auto &&verticies = edge->vertices();
            // assert(jacobians.size() == verticies.size());

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

                    // assert(v_j->ordering_id() != -1);
                    MatXX hessian = JtW * jacobian_j;   // TODO: 这里能继续优化, 因为J'*W*J也是对称矩阵
                    // 所有的信息矩阵叠加起来
                    _hessian.block(index_i, index_j, dim_i, dim_j).noalias() += hessian;
                    if (j != i) {
                        // 对称的下三角
                        _hessian.block(index_j, index_i, dim_j, dim_i).noalias() += hessian.transpose();
                    }
                }
                _b.segment(index_i, dim_i).noalias() -= jacobian_i.transpose() * robust_residual;
            }

        }
#endif

        // 叠加先验
        add_prior_to_hessian();

//        _delta_x = VecX::Zero(size);  // initial delta_x = 0_n;

        _t_hessian_cost += t_h.toc();
    }

    void ProblemSLAM::clear() {
        Problem::clear();
        _pose_vertices.clear();
        _landmark_vertices.clear();
        _reproj_edges.clear();
        _imu_edges.clear();
        _ordering_poses = 0;
        _ordering_landmarks = 0;
    }
}