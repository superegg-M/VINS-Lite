#include <iostream>

#include "backend/problem.h"

namespace graph_optimization {
    bool Problem::Solve(unsigned long iterations) {
        double chi2 = 0.;
        double stop_threshold_lm = 0.;
        auto ComputeLambdaInitLM = [&]() {
            _ni = 2.;
            _current_lambda = -1.;
            chi2 = 0.0;

            for (auto &edge: _edges) {
                chi2 += edge.second->get_robust_chi2();
            }
            chi2 *= 0.5;

            stop_threshold_lm = 1e-8 * chi2;          // 迭代条件为 误差下降 1e-6 倍

            double max_diagonal = 0;
            ulong size = _hessian.cols();
            assert(_hessian.rows() == _hessian.cols() && "Hessian is not square");
            for (ulong i = 0; i < size; ++i) {
                max_diagonal = std::max(fabs(_hessian(i, i)), max_diagonal);
            }

            max_diagonal = std::min(5e10, max_diagonal);
            double tau = 1e-5;  // 1e-5
            _current_lambda = tau * max_diagonal;

            _diag_lambda = tau * _hessian.diagonal();
            for (int i = 0; i < _hessian.rows(); ++i) {
                _diag_lambda(i) = std::min(std::max(_diag_lambda(i), 1e-6), 1e6);
            }
        };

        auto IsGoodStepInLM = [&]() -> bool {
            double scale = 0.;
            scale = 0.5 * _delta_x.transpose() * (VecX(_diag_lambda.array() * _delta_x.array()) + _b);
            scale += 1e-6;    // make sure it's non-zero :)

            // recompute residuals after update state
            double new_chi2 = 0.0;
            for (auto &edge: _edges) {
                edge.second->compute_residual();
                edge.second->compute_chi2();
                new_chi2 += edge.second->get_robust_chi2();
            }
            new_chi2 *= 0.5; 

            double rho = (chi2 - new_chi2) / scale;
            if (rho > 0 && std::isfinite(new_chi2))   // last step was good, 误差在下降
            {
                double alpha = 1. - pow((2 * rho - 1), 3);
                alpha = std::min(alpha, 2. / 3.);
                double scale_factor = (std::max)(1. / 3., alpha);
                _current_lambda *= scale_factor;
                _diag_lambda *= scale_factor;
                _ni = 2.;
                chi2 = new_chi2;
                return true;
            } else {
                _current_lambda *= _ni;
                _diag_lambda *= _ni;
                _ni *= 2;
                return false;
            }
        };

        auto MakeHessian = [&]() {
            TicToc t_h;
            // 直接构造大的 H 矩阵
            ulong size = _ordering_generic;
            MatXX H(MatXX::Zero(size, size));
            VecX b(VecX::Zero(size));

            for (auto &edge: _edges) {
                edge.second->compute_residual();
                edge.second->compute_chi2();
                edge.second->compute_jacobians();

                // TODO:: robust cost
                auto jacobians = edge.second->jacobians();
                auto verticies = edge.second->vertices();
                assert(jacobians.size() == verticies.size());
                for (size_t i = 0; i < verticies.size(); ++i) {
                    auto v_i = verticies[i];
                    if (v_i->is_fixed()) continue;    // Hessian 里不需要添加它的信息，也就是它的雅克比为 0

                    auto jacobian_i = jacobians[i];
                    ulong index_i = v_i->ordering_id();
                    ulong dim_i = v_i->local_dimension();

                    // 鲁棒核函数会修改残差和信息矩阵，如果没有设置 robust cost function，就会返回原来的
                    double drho;
                    MatXX robustInfo(edge.second->information().rows(),edge.second->information().cols());
                    edge.second->robust_information(drho,robustInfo);

                    MatXX JtW = jacobian_i.transpose() * robustInfo;
                    for (size_t j = i; j < verticies.size(); ++j) {
                        auto v_j = verticies[j];

                        if (v_j->is_fixed()) continue;

                        auto jacobian_j = jacobians[j];
                        ulong index_j = v_j->ordering_id();
                        ulong dim_j = v_j->local_dimension();

                        assert(v_j->ordering_id() != -1);
                        MatXX hessian = JtW * jacobian_j;

                        // 所有的信息矩阵叠加起来
                        H.block(index_i, index_j, dim_i, dim_j).noalias() += hessian;
                        if (j != i) {
                            // 对称的下三角
                            H.block(index_j, index_i, dim_j, dim_i).noalias() += hessian.transpose();

                        }
                    }
                    b.segment(index_i, dim_i).noalias() -= drho * jacobian_i.transpose()* edge.second->information() * edge.second->residual();
                }
            }
            _hessian = H;
            _b = b;
            _t_hessian_cost += t_h.toc();

            add_prior_to_hessian();

        //     if(_h_prior.rows() > 0)
        //     {
        //         MatXX H_prior_tmp = _h_prior;
        //         VecX b_prior_tmp = _b_prior;

        //         /// 遍历所有 POSE 顶点，然后设置相应的先验维度为 0 .  fix 外参数, SET PRIOR TO ZERO
        //         /// landmark 没有先验
        //         for (auto &vertex: _vertices) {
        //             if (is_pose_vertex(vertex.second) && vertex.second->is_fixed() ) {
        //                 int idx = vertex.second->ordering_id();
        //                 int dim = vertex.second->local_dimension();
        //                 H_prior_tmp.block(idx,0, dim, H_prior_tmp.cols()).setZero();
        //                 H_prior_tmp.block(0,idx, H_prior_tmp.rows(), dim).setZero();
        //                 b_prior_tmp.segment(idx,dim).setZero();
        // //                std::cout << " fixed prior, set the Hprior and bprior part to zero, idx: "<<idx <<" dim: "<<dim<<std::endl;
        //             }
        //         }
        //         _hessian.topLeftCorner(ordering_poses_, ordering_poses_) += H_prior_tmp;
        //         _b.head(ordering_poses_) += b_prior_tmp;
        //     }

            _delta_x = VecX::Zero(size);  // initial delta_x = 0_n;
        };

        // // 遍历edge, 构建 H 矩阵
        MakeHessian();
        // update_residual();
        // update_chi2();
        // update_jacobian();
        // update_hessian();
        // LM 初始化
        ComputeLambdaInitLM();
        // LM 算法迭代求解
        bool stop = false;
        int iter = 0;
        double last_chi2 = chi2;
        while (!stop && (iter < iterations)) {
            std::cout << "iter: " << iter << " , chi= " << chi2 << " , Lambda= " << _current_lambda << std::endl;
            bool one_step_success = false;
            int false_cnt = 0;
            while (!one_step_success && false_cnt < 10) {
                // setLambda
                if (!solve_linear_system(_delta_x)) {
                    false_cnt ++;
                    one_step_success = false;
                    _current_lambda *= _ni;
                    _diag_lambda *= _ni;
                    _ni *= 2.;
                    std::cout << "!!!!!!!!!!!!!! Solver unstable !!!!!!!!!!!" << std::endl;
                    continue;
                }

                // 更新状态量
                update_states(_delta_x);
                // update_residual();
                // update_chi2();

                // 判断当前步是否可行以及 LM 的 lambda 怎么更新, chi2 也计算一下
                one_step_success = IsGoodStepInLM();
                // 后续处理，
                if (one_step_success) {
                    // // 在新线性化点 构建 hessian
                    MakeHessian();

                    // update_jacobian();
                    // update_hessian();

                    false_cnt = 0;
                } else {
                    false_cnt ++;
                    rollback_states(_delta_x);   // 误差没下降，回滚
                }
            }
            iter++;

            if (fabs(last_chi2 - chi2) < 1e-3 * last_chi2 || chi2 < stop_threshold_lm) {
                std::cout << "fabs(last_chi_ - currentChi_) < 1e-2 * last_chi_ || currentChi_ < stopThresholdLM_" << std::endl;
                stop = true;
            }
            last_chi2 = chi2;
        }

        return true;
    }
}