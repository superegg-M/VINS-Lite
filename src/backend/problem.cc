#include <iostream>
#include <fstream>
#include <eigen3/Eigen/Dense>
#include <iomanip>
#include "backend/problem.h"
#include "utility/tic_toc.h"
#include <opencv2/core.hpp>

#ifdef USE_OPENMP

#include <omp.h>

#endif

using namespace std;

// define the format you want, you only need one instance of this...
const static Eigen::IOFormat CSVFormat(Eigen::StreamPrecision, Eigen::DontAlignCols, ", ", "\n");

void writeToCSVfile(std::string name, Eigen::MatrixXd matrix) {
    std::ofstream f(name.c_str());
    f << matrix.format(CSVFormat);
}

namespace myslam {
namespace backend {
void Problem::LogoutVectorSize() {
    // LOG(INFO) <<
    //           "1 problem::LogoutVectorSize verticies_:" << verticies_.size() <<
    //           " edges:" << edges_.size();
}

Problem::Problem(ProblemType problemType) :
    problemType_(problemType) {
    LogoutVectorSize();
    verticies_marg_.clear();
}

Problem::~Problem() {
//    std::cout << "Problem IS Deleted"<<std::endl;
    global_vertex_id = 0;
}

bool Problem::AddVertex(const std::shared_ptr<Vertex>& vertex) {
    if (verticies_.find(vertex->Id()) != verticies_.end()) {
        // LOG(WARNING) << "Vertex " << vertex->Id() << " has been added before";
        return false;
    } else {
        verticies_.insert(pair<unsigned long, shared_ptr<Vertex>>(vertex->Id(), vertex));
    }

    if (problemType_ == ProblemType::SLAM_PROBLEM) {
        if (IsPoseVertex(vertex)) {
            ResizePoseHessiansWhenAddingPose(vertex);
        }
    }
    return true;
}

void Problem::AddOrderingSLAM(const std::shared_ptr<myslam::backend::Vertex>& v) {
    if (IsPoseVertex(v)) {
        v->SetOrderingId(ordering_poses_);
        idx_pose_vertices_.insert(pair<ulong, std::shared_ptr<Vertex>>(v->Id(), v));
        ordering_poses_ += v->LocalDimension();
    } else if (IsLandmarkVertex(v)) {
        v->SetOrderingId(ordering_landmarks_);
        ordering_landmarks_ += v->LocalDimension();
        idx_landmark_vertices_.insert(pair<ulong, std::shared_ptr<Vertex>>(v->Id(), v));
    }
}

void Problem::ResizePoseHessiansWhenAddingPose(const shared_ptr<Vertex>& v) {

    int size = H_prior_.rows() + v->LocalDimension();
    H_prior_.conservativeResize(size, size);
    b_prior_.conservativeResize(size);

    b_prior_.tail(v->LocalDimension()).setZero();
    H_prior_.rightCols(v->LocalDimension()).setZero();
    H_prior_.bottomRows(v->LocalDimension()).setZero();

}
void Problem::ExtendHessiansPriorSize(int dim)
{
    int size = H_prior_.rows() + dim;
    H_prior_.conservativeResize(size, size);
    b_prior_.conservativeResize(size);

    b_prior_.tail(dim).setZero();
    H_prior_.rightCols(dim).setZero();
    H_prior_.bottomRows(dim).setZero();
}

bool Problem::IsPoseVertex(std::shared_ptr<myslam::backend::Vertex> v) {
    string type = v->TypeInfo();
    return type == string("VertexPose") ||
            type == string("VertexSpeedBias");
}

bool Problem::IsLandmarkVertex(std::shared_ptr<myslam::backend::Vertex> v) {
    string type = v->TypeInfo();
    return type == string("VertexPointXYZ") ||
           type == string("VertexInverseDepth");
}

bool Problem::AddEdge(const shared_ptr<Edge>& edge) {
    if (edges_.find(edge->Id()) == edges_.end()) {
        edges_.insert(pair<ulong, std::shared_ptr<Edge>>(edge->Id(), edge));
    } else {
        // LOG(WARNING) << "Edge " << edge->Id() << " has been added before!";
        return false;
    }

    for (auto &vertex: edge->Verticies()) {
        vertexToEdge_.insert(pair<ulong, shared_ptr<Edge>>(vertex->Id(), edge));
    }
    return true;
}

vector<shared_ptr<Edge>> Problem::GetConnectedEdges(const std::shared_ptr<Vertex>& vertex) {
    vector<shared_ptr<Edge>> edges;
    auto range = vertexToEdge_.equal_range(vertex->Id());
    for (auto iter = range.first; iter != range.second; ++iter) {

        // 并且这个edge还需要存在，而不是已经被remove了
        if (edges_.find(iter->second->Id()) == edges_.end())
            continue;

        edges.emplace_back(iter->second);
    }
    return edges;
}

bool Problem::RemoveVertex(const std::shared_ptr<Vertex>& vertex) {
    //check if the vertex is in map_verticies_
    if (verticies_.find(vertex->Id()) == verticies_.end()) {
        // LOG(WARNING) << "The vertex " << vertex->Id() << " is not in the problem!" << endl;
        return false;
    }

    // 这里要 remove 该顶点对应的 edge.
    vector<shared_ptr<Edge>> remove_edges = GetConnectedEdges(vertex);
    for (const auto & remove_edge : remove_edges) {
        RemoveEdge(remove_edge);
    }

    if (IsPoseVertex(vertex))
        idx_pose_vertices_.erase(vertex->Id());
    else
        idx_landmark_vertices_.erase(vertex->Id());

    vertex->SetOrderingId(-1);      // used to debug
    verticies_.erase(vertex->Id());
    vertexToEdge_.erase(vertex->Id());

    return true;
}

bool Problem::RemoveEdge(const std::shared_ptr<Edge>& edge) {
    //check if the edge is in map_edges_
    if (edges_.find(edge->Id()) == edges_.end()) {
        // LOG(WARNING) << "The edge " << edge->Id() << " is not in the problem!" << endl;
        return false;
    }

    edges_.erase(edge->Id());
    return true;
}

bool Problem::Solve(int iterations) {


    if (edges_.empty() || verticies_.empty()) {
        std::cerr << "\nCannot solve problem without edges or verticies" << std::endl;
        return false;
    }

    TicToc t_solve;
    // 统计优化变量的维数，为构建 H 矩阵做准备
    SetOrdering();
    // 遍历edge, 构建 H 矩阵
    MakeHessian();
    // LM 初始化
    ComputeLambdaInitLM();
    // LM 算法迭代求解
    bool stop = false;
    int iter = 0;
    double last_chi_ = currentChi_;
    while (!stop && (iter < iterations)) {
        std::cout << "iter: " << iter << " , chi= " << currentChi_ << " , Lambda= " << currentLambda_ << std::endl;
        bool oneStepSuccess = false;
        int false_cnt = 0;
        while (!oneStepSuccess && false_cnt < 10)  // 不断尝试 Lambda, 直到成功迭代一步
        {
            // setLambda
//            AddLambdatoHessianLM();
            // 第四步，解线性方程
            if (!SolveLinearSystem()) {
                false_cnt ++;
                oneStepSuccess = false;
                currentLambda_ *= ni_;
                _diag_lambda *= ni_;
                ni_ *= 2.;
                std::cout << "!!!!!!!!!!!!!! Solver unstable !!!!!!!!!!!" << std::endl;
                continue;
            }
            //
//            RemoveLambdaHessianLM();

            // 优化退出条件1： delta_x_ 很小则退出
//            if (delta_x_.squaredNorm() <= 1e-6 || false_cnt > 10)
            // TODO:: 退出条件还是有问题, 好多次误差都没变化了，还在迭代计算，应该搞一个误差不变了就中止
//            if ( false_cnt > 10)
//            {
//                stop = true;
//                break;
//            }

            // 更新状态量
            UpdateStates();
            // 判断当前步是否可行以及 LM 的 lambda 怎么更新, chi2 也计算一下
            oneStepSuccess = IsGoodStepInLM();
            // 后续处理，
            if (oneStepSuccess) {
//                std::cout << " get one step success\n";

                // 在新线性化点 构建 hessian
                MakeHessian();
                // TODO:: 这个判断条件可以丢掉，条件 b_max <= 1e-12 很难达到，这里的阈值条件不应该用绝对值，而是相对值
//                double b_max = 0.0;
//                for (int i = 0; i < b_.size(); ++i) {
//                    b_max = max(fabs(b_(i)), b_max);
//                }
//                // 优化退出条件2： 如果残差 b_max 已经很小了，那就退出
//                stop = (b_max <= 1e-12);
                false_cnt = 0;
            } else {
                false_cnt ++;
                RollbackStates();   // 误差没下降，回滚
            }
        }
        iter++;

        // 优化退出条件3： currentChi_ 跟第一次的 chi2 相比，下降了 1e6 倍则退出
        // TODO:: 应该改成前后两次的误差已经不再变化
//        if (sqrt(currentChi_) <= stopThresholdLM_)
//        if (sqrt(currentChi_) < 1e-15)
        // if(last_chi_ - currentChi_ < 1e-5)
        // {
        //     std::cout << "sqrt(currentChi_) <= stopThresholdLM_" << std::endl;
        //     stop = true;
        // }
        if (fabs(last_chi_ - currentChi_) < 1e-3 * last_chi_ || currentChi_ < stopThresholdLM_) {
            std::cout << "fabs(last_chi_ - currentChi_) < 1e-2 * last_chi_ || currentChi_ < stopThresholdLM_" << std::endl;
            stop = true;
        }
        last_chi_ = currentChi_;
    }
    std::cout << "problem solve cost: " << t_solve.toc() << " ms" << std::endl;
    std::cout << "   makeHessian cost: " << t_hessian_cost_ << " ms" << std::endl;
    t_hessian_cost_ = 0.;
    return true;
}

bool Problem::SolveGenericProblem(int iterations) {
    return true;
}

void Problem::SetOrdering() {

    // 每次重新计数
    ordering_poses_ = 0;
    ordering_generic_ = 0;
    ordering_landmarks_ = 0;

    // Note:: verticies_ 是 map 类型的, 顺序是按照 id 号排序的
    for (auto &vertex: verticies_) {
        ordering_generic_ += vertex.second->LocalDimension();  // 所有的优化变量总维数

        if (problemType_ == ProblemType::SLAM_PROBLEM)    // 如果是 slam 问题，还要分别统计 pose 和 landmark 的维数，后面会对他们进行排序
        {
            AddOrderingSLAM(vertex.second);
        }

    }

    if (problemType_ == ProblemType::SLAM_PROBLEM) {
        // 这里要把 landmark 的 ordering 加上 pose 的数量，就保持了 landmark 在后,而 pose 在前
        ulong all_pose_dimension = ordering_poses_;
        for (auto &landmarkVertex : idx_landmark_vertices_) {
            landmarkVertex.second->SetOrderingId(
                landmarkVertex.second->OrderingId() + all_pose_dimension
            );
        }
    }

//    CHECK_EQ(CheckOrdering(), true);
}

bool Problem::CheckOrdering() {
    if (problemType_ == ProblemType::SLAM_PROBLEM) {
        int current_ordering = 0;
        for (auto &v: idx_pose_vertices_) {
            assert(v.second->OrderingId() == current_ordering);
            current_ordering += v.second->LocalDimension();
        }

        for (auto &v: idx_landmark_vertices_) {
            assert(v.second->OrderingId() == current_ordering);
            current_ordering += v.second->LocalDimension();
        }
    }
    return true;
}

void Problem::MakeHessian() {
    TicToc t_h;
    // 直接构造大的 H 矩阵
    ulong size = ordering_generic_;
    MatXX H(MatXX::Zero(size, size));
    VecX b(VecX::Zero(size));

    // TODO:: accelate, accelate, accelate
//#ifdef USE_OPENMP
//#pragma omp parallel for
//#endif
//     int vec_index = 0;
//     vector<std::shared_ptr<Edge>> edges(edges_.size());
//     for (auto &edge: edges_) {
//         edges[vec_index++] = edge.second;
//     }
// #pragma omp parallel for
//     for (auto &edge : edges) {
//         edge->ComputeResidual();
//         edge->ComputeJacobians();

//         // TODO:: robust cost
//         auto jacobians = edge->Jacobians();
//         auto verticies = edge->Verticies();
//         assert(jacobians.size() == verticies.size());
//         for (size_t i = 0; i < verticies.size(); ++i) {
//             auto v_i = verticies[i];
//             if (v_i->IsFixed()) continue;    // Hessian 里不需要添加它的信息，也就是它的雅克比为 0

//             auto jacobian_i = jacobians[i];
//             ulong index_i = v_i->OrderingId();
//             ulong dim_i = v_i->LocalDimension();

//             // 鲁棒核函数会修改残差和信息矩阵，如果没有设置 robust cost function，就会返回原来的
//             double drho;
//             MatXX robustInfo(edge->Information().rows(),edge->Information().cols());
//             edge->RobustInfo(drho,robustInfo);

//             MatXX JtW = jacobian_i.transpose() * robustInfo;
//             for (size_t j = i; j < verticies.size(); ++j) {
//                 auto v_j = verticies[j];

//                 if (v_j->IsFixed()) continue;

//                 auto jacobian_j = jacobians[j];
//                 ulong index_j = v_j->OrderingId();
//                 ulong dim_j = v_j->LocalDimension();

//                 assert(v_j->OrderingId() != -1);
//                 MatXX hessian = JtW * jacobian_j;

//                 // 所有的信息矩阵叠加起来
//                 H.block(index_i, index_j, dim_i, dim_j).noalias() += hessian;
//                 if (j != i) {
//                     // 对称的下三角
//                     H.block(index_j, index_i, dim_j, dim_i).noalias() += hessian.transpose();

//                 }
//             }
//             b.segment(index_i, dim_i).noalias() -= drho * jacobian_i.transpose()* edge->Information() * edge->Residual();
//         }
//     }
    // auto parallel_task = [&](const cv::Range& range) {
    //     for (int k = range.start; k < range.end; ++k) {
    //         edges[k]->ComputeResidual();
    //         edges[k]->ComputeJacobians();

    //         // TODO:: robust cost
    //         auto jacobians = edges[k]->Jacobians();
    //         auto verticies = edges[k]->Verticies();
    //         assert(jacobians.size() == verticies.size());
    //         for (size_t i = 0; i < verticies.size(); ++i) {
    //             auto v_i = verticies[i];
    //             if (v_i->IsFixed()) continue;    // Hessian 里不需要添加它的信息，也就是它的雅克比为 0

    //             auto jacobian_i = jacobians[i];
    //             ulong index_i = v_i->OrderingId();
    //             ulong dim_i = v_i->LocalDimension();

    //             // 鲁棒核函数会修改残差和信息矩阵，如果没有设置 robust cost function，就会返回原来的
    //             double drho;
    //             MatXX robustInfo(edges[k]->Information().rows(),edges[k]->Information().cols());
    //             edges[k]->RobustInfo(drho,robustInfo);

    //             MatXX JtW = jacobian_i.transpose() * robustInfo;
    //             for (size_t j = i; j < verticies.size(); ++j) {
    //                 auto v_j = verticies[j];

    //                 if (v_j->IsFixed()) continue;

    //                 auto jacobian_j = jacobians[j];
    //                 ulong index_j = v_j->OrderingId();
    //                 ulong dim_j = v_j->LocalDimension();

    //                 assert(v_j->OrderingId() != -1);
    //                 MatXX hessian = JtW * jacobian_j;

    //                 // 所有的信息矩阵叠加起来
    //                 H.block(index_i, index_j, dim_i, dim_j).noalias() += hessian;
    //                 if (j != i) {
    //                     // 对称的下三角
    //                     H.block(index_j, index_i, dim_j, dim_i).noalias() += hessian.transpose();

    //                 }
    //             }
    //             b.segment(index_i, dim_i).noalias() -= drho * jacobian_i.transpose()* edges[k]->Information() * edges[k]->Residual();
    //         }
    //     }
    // };
    // cv::parallel_for_(cv::Range(0, edges.size()), parallel_task);
    for (auto &edge: edges_) {
        edge.second->ComputeResidual();
        edge.second->ComputeJacobians();

        // TODO:: robust cost
        auto jacobians = edge.second->Jacobians();
        auto verticies = edge.second->Verticies();
        assert(jacobians.size() == verticies.size());
        for (size_t i = 0; i < verticies.size(); ++i) {
            auto v_i = verticies[i];
            if (v_i->IsFixed()) continue;    // Hessian 里不需要添加它的信息，也就是它的雅克比为 0

            auto jacobian_i = jacobians[i];
            ulong index_i = v_i->OrderingId();
            ulong dim_i = v_i->LocalDimension();

            // 鲁棒核函数会修改残差和信息矩阵，如果没有设置 robust cost function，就会返回原来的
            double drho;
            MatXX robustInfo(edge.second->Information().rows(),edge.second->Information().cols());
            edge.second->RobustInfo(drho,robustInfo);

            MatXX JtW = jacobian_i.transpose() * robustInfo;
            for (size_t j = i; j < verticies.size(); ++j) {
                auto v_j = verticies[j];

                if (v_j->IsFixed()) continue;

                auto jacobian_j = jacobians[j];
                ulong index_j = v_j->OrderingId();
                ulong dim_j = v_j->LocalDimension();

                assert(v_j->OrderingId() != -1);
                MatXX hessian = JtW * jacobian_j;

                // 所有的信息矩阵叠加起来
                H.block(index_i, index_j, dim_i, dim_j).noalias() += hessian;
                if (j != i) {
                    // 对称的下三角
                    H.block(index_j, index_i, dim_j, dim_i).noalias() += hessian.transpose();

                }
            }
            b.segment(index_i, dim_i).noalias() -= drho * jacobian_i.transpose()* edge.second->Information() * edge.second->Residual();
        }
    }
    Hessian_ = H;
    b_ = b;
    t_hessian_cost_ += t_h.toc();

    if(H_prior_.rows() > 0)
    {
        MatXX H_prior_tmp = H_prior_;
        VecX b_prior_tmp = b_prior_;

        /// 遍历所有 POSE 顶点，然后设置相应的先验维度为 0 .  fix 外参数, SET PRIOR TO ZERO
        /// landmark 没有先验
        for (auto &vertex: verticies_) {
            if (IsPoseVertex(vertex.second) && vertex.second->IsFixed() ) {
                int idx = vertex.second->OrderingId();
                int dim = vertex.second->LocalDimension();
                H_prior_tmp.block(idx,0, dim, H_prior_tmp.cols()).setZero();
                H_prior_tmp.block(0,idx, H_prior_tmp.rows(), dim).setZero();
                b_prior_tmp.segment(idx,dim).setZero();
//                std::cout << " fixed prior, set the Hprior and bprior part to zero, idx: "<<idx <<" dim: "<<dim<<std::endl;
            }
        }
        Hessian_.topLeftCorner(ordering_poses_, ordering_poses_) += H_prior_tmp;
        b_.head(ordering_poses_) += b_prior_tmp;
    }

    delta_x_ = VecX::Zero(size);  // initial delta_x = 0_n;
}

/*
 * Solve Hx = b, we can use PCG iterative method or use sparse Cholesky
 */
bool Problem::SolveLinearSystem() {


    if (problemType_ == ProblemType::GENERIC_PROBLEM) {
        // PCG solver
        MatXX H = Hessian_;
        for (size_t i = 0; i < Hessian_.cols(); ++i) {
            H(i, i) += currentLambda_;
        }
        // delta_x_ = PCGSolver(H, b_, H.rows() * 2);
        delta_x_ = H.ldlt().solve(b_);

    } else {

//        TicToc t_Hmminv;
        // step1: schur marginalization --> Hpp, bpp
        int reserve_size = ordering_poses_;
        int marg_size = ordering_landmarks_;
        MatXX Hmm = Hessian_.block(reserve_size, reserve_size, marg_size, marg_size);
        for (int i = 0; i < marg_size; ++i) {
            // Hmm(i, i) += currentLambda_;    // LM Method
            Hmm(i, i) += _diag_lambda(i + reserve_size);
        }
        MatXX Hmp = Hessian_.block(reserve_size, 0, marg_size, reserve_size);
        VecX bmm = b_.segment(reserve_size, marg_size);
        MatXX temp_H(MatXX::Zero(marg_size, reserve_size));
        VecX temp_b(VecX::Zero(marg_size, 1));

//         int vec_index = 0;
//         vector<std::shared_ptr<Vertex>> landmarks(idx_landmark_vertices_.size());
//         for (auto &landmark_vertex : idx_landmark_vertices_) {
//             landmarks[vec_index++] = landmark_vertex.second;
//         }
// #pragma omp parallel for
//         for (auto &landmark : landmarks) {
//             int idx = landmark->OrderingId() - reserve_size;
//             int size = landmark->LocalDimension();
//             if (size == 1) {
//                 temp_H.row(idx) = Hmp.row(idx) / Hmm(idx, idx);
//                 temp_b(idx) = bmm(idx) / Hmm(idx, idx);
//             } else {
//                 auto &&Hmm_ldlt = Hmm.block(idx, idx, size, size).ldlt();
//                 temp_H.block(idx, 0, size, reserve_size) = Hmm_ldlt.solve(Hmp.block(idx, 0, size, reserve_size));
//                 temp_b.segment(idx, size) = Hmm_ldlt.solve(bmm.segment(idx, size));
//             }
//         }
        // auto parallel_task = [&](const cv::Range& range) {
        //     for (int k = range.start; k < range.end; ++k) {
        //         int idx = landmarks[k]->OrderingId() - reserve_size;
        //         int size = landmarks[k]->LocalDimension();
        //         if (size == 1) {
        //             temp_H.row(idx) = Hmp.row(idx) / Hmm(idx, idx);
        //             temp_b(idx) = bmm(idx) / Hmm(idx, idx);
        //         } else {
        //             auto &&Hmm_ldlt = Hmm.block(idx, idx, size, size).ldlt();
        //             temp_H.block(idx, 0, size, reserve_size) = Hmm_ldlt.solve(Hmp.block(idx, 0, size, reserve_size));
        //             temp_b.segment(idx, size) = Hmm_ldlt.solve(bmm.segment(idx, size));
        //         }
        //     }
        // };
        // cv::parallel_for_(cv::Range(0, landmarks.size()), parallel_task);
        for (const auto &landmark_vertex : idx_landmark_vertices_) {
            int idx = landmark_vertex.second->OrderingId() - reserve_size;
            int size = landmark_vertex.second->LocalDimension();
            if (size == 1) {
                temp_H.row(idx) = Hmp.row(idx) / Hmm(idx, idx);
                temp_b(idx) = bmm(idx) / Hmm(idx, idx);
            } else {
                auto &&Hmm_ldlt = Hmm.block(idx, idx, size, size).ldlt();
                temp_H.block(idx, 0, size, reserve_size) = Hmm_ldlt.solve(Hmp.block(idx, 0, size, reserve_size));
                temp_b.segment(idx, size) = Hmm_ldlt.solve(bmm.segment(idx, size));
            }
        }
        H_pp_schur_ = Hessian_.block(0, 0, reserve_size, reserve_size) - Hmp.transpose() * temp_H;
        b_pp_schur_ = b_.segment(0, reserve_size) - Hmp.transpose() * temp_b;
        for (ulong i = 0; i < ordering_poses_; ++i) {
            // H_pp_schur_(i, i) += currentLambda_;    // LM Method
            H_pp_schur_(i, i) += _diag_lambda(i); 
        }

        // step2: solve Hpp * delta_x = bpp        
        // TicToc t_linearsolver;
        auto &&H_pp_schur_ldlt = H_pp_schur_.ldlt();
        if (H_pp_schur_ldlt.info() != Eigen::Success) {
            return false;   // H_pp_schur不是正定矩阵
        }
        VecX delta_x_pp =  H_pp_schur_ldlt.solve(b_pp_schur_);
        // auto &&delta_x_pp = PCGSolver(H_pp_schur_, b_pp_schur_, H_pp_schur_.rows());
        delta_x_.head(reserve_size) = delta_x_pp;
        // std::cout << " Linear Solver Time Cost: " << t_linearsolver.toc() << std::endl;

        // step3: solve Hmm * delta_x = bmm - Hmp * delta_x_pp;
        VecX delta_x_ll(marg_size);
        delta_x_ll = temp_b - temp_H * delta_x_pp;
        delta_x_.tail(marg_size) = delta_x_ll;

//        std::cout << "schur time cost: "<< t_Hmminv.toc()<<std::endl;

        return true;
    }

}

void Problem::UpdateStates() {

    // update vertex
    for (auto &vertex: verticies_) {
        vertex.second->BackUpParameters();    // 保存上次的估计值

        ulong idx = vertex.second->OrderingId();
        ulong dim = vertex.second->LocalDimension();
        VecX delta = delta_x_.segment(idx, dim);
        vertex.second->Plus(delta);
    }

    // update prior
    if (b_prior_.rows() > 0) {
        // BACK UP b_prior_
        b_prior_backup_ = b_prior_;
        // err_prior_backup_ = err_prior_;

        /// update with first order Taylor, b' = b + \frac{\delta b}{\delta x} * \delta x
        /// \delta x = Computes the linearized deviation from the references (linearization points)
        b_prior_ -= H_prior_ * delta_x_.head(ordering_poses_);       // update the error_prior
        // err_prior_ = -Jt_prior_inv_ * b_prior_.head(ordering_poses_ - 15);

//        std::cout << "                : "<< b_prior_.norm()<<" " <<err_prior_.norm()<< std::endl;
//        std::cout << "     delta_x_ ex: "<< delta_x_.head(6).norm() << std::endl;
    }

}

void Problem::RollbackStates() {

    // update vertex
    for (auto &vertex: verticies_) {
        vertex.second->RollBackParameters();
    }

    // Roll back prior_
    if (b_prior_.rows() > 0) {
        b_prior_ = b_prior_backup_;
        // err_prior_ = err_prior_backup_;
    }
}

/// LM
void Problem::ComputeLambdaInitLM() {
    ni_ = 2.;
    currentLambda_ = -1.;
    currentChi_ = 0.0;

    for (auto &edge: edges_) {
        currentChi_ += edge.second->RobustChi2();
    }
    // if (err_prior_.rows() > 0)
    //     currentChi_ += err_prior_.norm();
    currentChi_ *= 0.5;

    stopThresholdLM_ = 1e-8 * currentChi_;          // 迭代条件为 误差下降 1e-6 倍

    double maxDiagonal = 0;
    ulong size = Hessian_.cols();
    assert(Hessian_.rows() == Hessian_.cols() && "Hessian is not square");
    for (ulong i = 0; i < size; ++i) {
        maxDiagonal = std::max(fabs(Hessian_(i, i)), maxDiagonal);
    }

    maxDiagonal = std::min(5e10, maxDiagonal);
    double tau = 1e-5;  // 1e-5
    currentLambda_ = tau * maxDiagonal;
//        std::cout << "currentLamba_: "<<maxDiagonal<<" "<<currentLambda_<<std::endl;

    _diag_lambda = tau * Hessian_.diagonal();
    for (int i = 0; i < Hessian_.rows(); ++i) {
        _diag_lambda(i) = std::min(std::max(_diag_lambda(i), 1e-6), 1e6);
    }
}

void Problem::AddLambdatoHessianLM() {
    ulong size = Hessian_.cols();
    assert(Hessian_.rows() == Hessian_.cols() && "Hessian is not square");
    for (ulong i = 0; i < size; ++i) {
        Hessian_(i, i) += currentLambda_;
    }
}

void Problem::RemoveLambdaHessianLM() {
    ulong size = Hessian_.cols();
    assert(Hessian_.rows() == Hessian_.cols() && "Hessian is not square");
    // TODO:: 这里不应该减去一个，数值的反复加减容易造成数值精度出问题？而应该保存叠加lambda前的值，在这里直接赋值
    for (ulong i = 0; i < size; ++i) {
        Hessian_(i, i) -= currentLambda_;
    }
}

bool Problem::IsGoodStepInLM() {
    double scale = 0;
//    scale = 0.5 * delta_x_.transpose() * (currentLambda_ * delta_x_ + b_);
//    scale += 1e-3;    // make sure it's non-zero :)
    // scale = 0.5* delta_x_.transpose() * (currentLambda_ * delta_x_ + b_);
    scale = 0.5 * delta_x_.transpose() * (VecX(_diag_lambda.array() * delta_x_.array()) + b_);
    scale += 1e-6;    // make sure it's non-zero :)

    // recompute residuals after update state
    double tempChi = 0.0;
    for (auto &edge: edges_) {
        edge.second->ComputeResidual();
        tempChi += edge.second->RobustChi2();
    }
    // if (err_prior_.size() > 0)
    //     tempChi += err_prior_.norm();
    tempChi *= 0.5;          // 1/2 * err^2

    double rho = (currentChi_ - tempChi) / scale;
    if (rho > 0 && isfinite(tempChi))   // last step was good, 误差在下降
    {
        double alpha = 1. - pow((2 * rho - 1), 3);
        alpha = std::min(alpha, 2. / 3.);
        double scaleFactor = (std::max)(1. / 3., alpha);
        currentLambda_ *= scaleFactor;
        _diag_lambda *= scaleFactor;
        ni_ = 2;
        currentChi_ = tempChi;
        return true;
    } else {
        currentLambda_ *= ni_;
        _diag_lambda *= ni_;
        ni_ *= 2;
        return false;
    }
}

/** @brief conjugate gradient with perconditioning
 *
 *  the jacobi PCG method
 *
 */
VecX Problem::PCGSolver(const MatXX &A, const VecX &b, int maxIter = -1) {
    assert(A.rows() == A.cols() && "PCG solver ERROR: A is not a square matrix");
    int rows = b.rows();
    int n = maxIter < 0 ? rows : maxIter;
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

    int i = 0;
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
    return x;
}

// VecX Problem::MultiPCGSolver(const MatXX &A, const MatXX &B, int maxIter = -1) {
//     assert(A.rows() == A.cols() && "PCG solver ERROR: A is not a square matrix");
//     int rows = B.rows();
//     int n = maxIter < 0 ? rows : maxIter;
//     double threshold = 1e-6 * b.norm();

//     MatXX x(MatXX::Zero(rows, B.cols()));
//     VecX M = A.diagonal();
//     MatXX r(-B);
//     MatXX y(MatXX::Zero(rows, B.cols()));
//     for (int i = 0; i < y.rows(); ++i) {
//         y.row(i) = r.row(i) / M(i);
//     }
//     MatXX p(-y);
//     MatXX Ap = A * p;
//     VecX ry(VecX::Zero(B.cols(), 1));
//     for (int i = 0; i < B.cols(); ++i) {
//         ry(i) = r.col(i).dot(y.col(i));
//     }
//     VecX ry_prev = ry;
//     VecX alpha(VecX::Zero(B.cols(), 1));
//     for (int i = 0; i < B.cols(); ++i) {
//         alpha(i) = ry(i) / p.col(i).dot(Ap.col(i));
//     }
//     VecX beta;
//     x += alpha * p;
//     r += alpha * Ap;

//     int i = 0;
//     while (r.norm() > threshold && ++i < n) {
//         y = r.array() / M.array();
//         ry = r.dot(y);
//         beta = ry / ry_prev;
//         ry_prev = ry;
//         p = -y + beta * p;

//         Ap = A * p;
//         alpha = ry / p.dot(Ap);
//         x += alpha * p;
//         r += alpha * Ap;
//     }
//     return x;
// }

/*
 *  marg 所有和 frame 相连的 edge: imu factor, projection factor
 *  如果某个landmark和该frame相连，但是又不想加入marg, 那就把改edge先去掉
 *
 */
bool Problem::Marginalize(const std::vector<std::shared_ptr<Vertex> > margVertexs, int pose_dim, double prior_lambda) {

    SetOrdering();
    /// 找到需要 marg 的 edge, margVertexs[0] is frame, its edge contained pre-intergration
    std::vector<shared_ptr<Edge>> marg_edges = GetConnectedEdges(margVertexs[0]);

    std::unordered_map<int, shared_ptr<Vertex>> margLandmark;
    // 构建 Hessian 的时候 pose 的顺序不变，landmark的顺序要重新设定
    int marg_landmark_size = 0;
//    std::cout << "\n marg edge 1st id: "<< marg_edges.front()->Id() << " end id: "<<marg_edges.back()->Id()<<std::endl;
// #pragma omp parallel for
    for (auto &marg_edge : marg_edges) {
//        std::cout << "marg edge id: "<< marg_edges[i]->Id() <<std::endl;
        auto &&verticies = marg_edge->Verticies();
        for (auto &iter : verticies) {
            if (IsLandmarkVertex(iter) && margLandmark.find(iter->Id()) == margLandmark.end()) {
                iter->SetOrderingId(pose_dim + marg_landmark_size);
                margLandmark.insert(make_pair(iter->Id(), iter));
                marg_landmark_size += iter->LocalDimension();
            }
        }
    }
//    std::cout << "pose dim: " << pose_dim <<std::endl;
    int cols = pose_dim + marg_landmark_size;
    /// 构建误差 H 矩阵 H = H_marg + H_pp_prior
    MatXX H_marg(MatXX::Zero(cols, cols));
    VecX b_marg(VecX::Zero(cols));
    int ii = 0;
// #pragma omp parallel for    
    for (auto &edge: marg_edges) {
        edge->ComputeResidual();
        edge->ComputeJacobians();
        auto jacobians = edge->Jacobians();
        auto verticies = edge->Verticies();
        ii++;

        assert(jacobians.size() == verticies.size());
        for (size_t i = 0; i < verticies.size(); ++i) {
            auto v_i = verticies[i];
            auto jacobian_i = jacobians[i];
            ulong index_i = v_i->OrderingId();
            ulong dim_i = v_i->LocalDimension();

            double drho;
            MatXX robustInfo(edge->Information().rows(),edge->Information().cols());
            edge->RobustInfo(drho,robustInfo);

            for (size_t j = i; j < verticies.size(); ++j) {
                auto v_j = verticies[j];
                auto jacobian_j = jacobians[j];
                ulong index_j = v_j->OrderingId();
                ulong dim_j = v_j->LocalDimension();

                MatXX hessian = jacobian_i.transpose() * robustInfo * jacobian_j;

                assert(hessian.rows() == v_i->LocalDimension() && hessian.cols() == v_j->LocalDimension());
                // 所有的信息矩阵叠加起来
                H_marg.block(index_i, index_j, dim_i, dim_j) += hessian;
                if (j != i) {
                    // 对称的下三角
                    H_marg.block(index_j, index_i, dim_j, dim_i) += hessian.transpose();
                }
            }
            b_marg.segment(index_i, dim_i) -= drho * jacobian_i.transpose() * edge->Information() * edge->Residual();
        }
    }
    std::cout << "edge factor cnt: " << ii <<std::endl;

    /// marg landmark
    // int vec_index = 0;
    // vector<std::shared_ptr<Vertex>> landmarks(idx_landmark_vertices_.size());
    // for (auto &landmark_vertex : margLandmark) {
    //     landmarks[vec_index++] = landmark_vertex.second;
    // }    
    int reserve_size = pose_dim;
    if (marg_landmark_size > 0) {
        int marg_size = marg_landmark_size;
        MatXX Hmm = H_marg.block(reserve_size, reserve_size, marg_size, marg_size);
        MatXX Hmp = H_marg.block(reserve_size, 0, marg_size, reserve_size);
        VecX bpp = b_marg.segment(0, reserve_size);
        VecX bmm = b_marg.segment(reserve_size, marg_size);

        MatXX temp_H(MatXX::Zero(marg_size, reserve_size));
        VecX temp_b(VecX::Zero(marg_size, 1));

        // int vec_index = 0;
        // vector<std::shared_ptr<Vertex>> landmarks(idx_landmark_vertices_.size());
        // for (auto &landmark_vertex : margLandmark) {
        //     landmarks[vec_index++] = landmark_vertex.second;
        // }
// #pragma omp parallel for
//         for (auto &landmark : landmarks) {
//             int idx = landmark->OrderingId() - reserve_size;
//             int size = landmark->LocalDimension();
//             if (size == 1) {
//                 double mu = std::min(std::max(fabs(Hmm(idx, idx)), 1e-6), 1e6);
//                 while (Hmm(idx, idx) < 1e-6 && mu < 1e6) {
//                     Hmm(idx, idx) += mu;
//                     mu *= 10.;
//                 }
//                 if (Hmm(idx, idx) > 1e-6) {
//                     temp_H.row(idx) = Hmp.row(idx) / Hmm(idx, idx);
//                     temp_b(idx) = bmm(idx) / Hmm(idx, idx);
//                 }
//                 // temp_H.row(idx) = Hmp.row(idx) / Hmm(idx, idx);
//                 // temp_b(idx) = bmm(idx) / Hmm(idx, idx);
//             } else {
//                 double mu = std::min(std::max(prior_lambda * 1e-5, 1e-6), 1e6);
//                 auto Hmm_ldlt = Hmm.block(idx, idx, size, size).ldlt();
//                 while (Hmm_ldlt.info() != Eigen::Success && mu < 1e6) {
//                     for (int i = 0; i < size; ++i) {
//                         Hmm(i + size, i + size) += mu;
//                     }
//                     mu *= 10.;
//                     Hmm_ldlt = Hmm.block(idx, idx, size, size).ldlt();
//                 }
//                 if (Hmm_ldlt.info() == Eigen::Success) {
//                     temp_H.block(idx, 0, size, reserve_size) = Hmm_ldlt.solve(Hmp.block(idx, 0, size, reserve_size));
//                     temp_b.segment(idx, size) = Hmm_ldlt.solve(bmm.segment(idx, size));
//                 }
//                 // auto Hmm_ldlt = Hmm.block(idx, idx, size, size).ldlt();
//                 // temp_H.block(idx, 0, size, reserve_size) = Hmm_ldlt.solve(Hmp.block(idx, 0, size, reserve_size));
//                 // temp_b.segment(idx, size) = Hmm_ldlt.solve(bmm.segment(idx, size));
//             }
//         }

        for (auto &iter : margLandmark) {
            int idx = iter.second->OrderingId() - reserve_size;
            int size = iter.second->LocalDimension();
            if (size == 1) {
                double mu = std::min(std::max(fabs(Hmm(idx, idx)), 1e-6), 1e6);
                while (Hmm(idx, idx) < 1e-6 && mu < 1e6) {
                    Hmm(idx, idx) += mu;
                    mu *= 10.;
                }
                if (Hmm(idx, idx) > 1e-6) {
                    temp_H.row(idx) = Hmp.row(idx) / Hmm(idx, idx);
                    temp_b(idx) = bmm(idx) / Hmm(idx, idx);
                }
                // temp_H.row(idx) = Hmp.row(idx) / Hmm(idx, idx);
                // temp_b(idx) = bmm(idx) / Hmm(idx, idx);
            } else {
                double mu = std::min(std::max(prior_lambda * 1e-5, 1e-6), 1e6);
                auto Hmm_ldlt = Hmm.block(idx, idx, size, size).ldlt();
                while (Hmm_ldlt.info() != Eigen::Success && mu < 1e6) {
                    for (int i = 0; i < size; ++i) {
                        Hmm(i + size, i + size) += mu;
                    }
                    mu *= 10.;
                    Hmm_ldlt = Hmm.block(idx, idx, size, size).ldlt();
                }
                if (Hmm_ldlt.info() == Eigen::Success) {
                    temp_H.block(idx, 0, size, reserve_size) = Hmm_ldlt.solve(Hmp.block(idx, 0, size, reserve_size));
                    temp_b.segment(idx, size) = Hmm_ldlt.solve(bmm.segment(idx, size));
                }
                // auto Hmm_ldlt = Hmm.block(idx, idx, size, size).ldlt();
                // temp_H.block(idx, 0, size, reserve_size) = Hmm_ldlt.solve(Hmp.block(idx, 0, size, reserve_size));
                // temp_b.segment(idx, size) = Hmm_ldlt.solve(bmm.segment(idx, size));
            }
        }
        H_marg = H_marg.block(0, 0, reserve_size, reserve_size) - Hmp.transpose() * temp_H;
        b_marg = bpp - Hmp.transpose() * temp_b;
    }

    VecX b_prior_before = b_prior_;
    if(H_prior_.rows() > 0)
    {
        H_marg += H_prior_;
        b_marg += b_prior_;
    }

    /// marg frame and speedbias
    int marg_dim = 0;

    // index 大的先移动
    for (int k = margVertexs.size() -1 ; k >= 0; --k)
    {
        int idx = margVertexs[k]->OrderingId();
        int dim = margVertexs[k]->LocalDimension();
//        std::cout << k << " "<<idx << std::endl;
        marg_dim += dim;
        // move the marg pose to the Hmm bottown right
        // 将 row i 移动矩阵最下面
        Eigen::MatrixXd temp_rows = H_marg.block(idx, 0, dim, reserve_size);
        Eigen::MatrixXd temp_botRows = H_marg.block(idx + dim, 0, reserve_size - idx - dim, reserve_size);
        H_marg.block(idx, 0, reserve_size - idx - dim, reserve_size) = temp_botRows;
        H_marg.block(reserve_size - dim, 0, dim, reserve_size) = temp_rows;

        // 将 col i 移动矩阵最右边
        Eigen::MatrixXd temp_cols = H_marg.block(0, idx, reserve_size, dim);
        Eigen::MatrixXd temp_rightCols = H_marg.block(0, idx + dim, reserve_size, reserve_size - idx - dim);
        H_marg.block(0, idx, reserve_size, reserve_size - idx - dim) = temp_rightCols;
        H_marg.block(0, reserve_size - dim, reserve_size, dim) = temp_cols;

        Eigen::VectorXd temp_b = b_marg.segment(idx, dim);
        Eigen::VectorXd temp_btail = b_marg.segment(idx + dim, reserve_size - idx - dim);
        b_marg.segment(idx, reserve_size - idx - dim) = temp_btail;
        b_marg.segment(reserve_size - dim, dim) = temp_b;
    }

    double eps = 1e-8;
    int m2 = marg_dim;
    int n2 = reserve_size - marg_dim;   // marg pose
    Eigen::MatrixXd Amm = H_marg.block(n2, n2, m2, m2);
    // Eigen::MatrixXd Amm = 0.5 * (H_marg.block(n2, n2, m2, m2) + H_marg.block(n2, n2, m2, m2).transpose());
    Eigen::MatrixXd Amr = H_marg.block(n2, 0, m2, n2);
    Eigen::VectorXd brr = b_marg.segment(0, n2);
    // for (int i = 0; i < Amm.rows(); ++i) {
    //     Amm(i, i) += prior_lambda;
    // }
    // for (int i = 0; i < Amm.cols(); ++i) {
    //     if (Amm(i, i) < eps) {
    //         Amm(i, i) = eps;
    //     }
    // }
    // for (int i = 0; i < Amm.cols(); ++i) {
    //     for (int j = i + 1; j < Amm.cols(); ++j) {
    //         double tmp = Amm(i, i) * Amm(j, i);
    //         if (Amm(i, j) * Amm(i, j) > tmp) {
    //             if (Amm(i, j) > 0.) {
    //                 Amm(i, j) = sqrt(tmp);
    //             } else {
    //                 Amm(i, j) = -sqrt(tmp);
    //             }
    //         }
    //     }
    // }

    MatXX temp_H(MatXX::Zero(m2, n2));
    VecX temp_b(VecX::Zero(m2, 1));
    int size = marg_dim;
    if (size == 1) {
        double mu = std::min(std::max(prior_lambda * 1e-5, 1e-6), 1e6);
        while (Amm(0, 0) < 1e-6 && mu < 1e6) {
            Amm(0, 0) += mu;
            mu *= 10.;
        }
        if (Amm(0, 0) > 1e-6) {
            temp_H = Amr / Amm(0, 0);
            temp_b = b_marg.segment(n2, m2) / Amm(0, 0);
        }
        // temp_H = Amr / Amm(0, 0);
        // temp_b = b_marg.segment(n2, m2) / Amm(0, 0);
    } else {
        double mu = std::min(std::max(prior_lambda * 1e-5, 1e-6), 1e6);
        auto Amm_ldlt = Amm.ldlt();
        while (Amm_ldlt.info() != Eigen::Success && mu < 1e6) {
            for (int i = 0; i < Amm.rows(); ++i) {
                Amm(i, i) += mu;
                mu *= 10.;
            }
            Amm_ldlt = Amm.ldlt();
        }
        if (Amm_ldlt.info() == Eigen::Success) {
            temp_H = Amm_ldlt.solve(Amr);
            temp_b = Amm_ldlt.solve(b_marg.segment(n2, m2));
        } 
        // auto Amm_ldlt = Amm.ldlt();
        // temp_H = Amm_ldlt.solve(Amr);
        // temp_b = Amm_ldlt.solve(b_marg.segment(n2, m2));
        // for (int i = 0; i < Amr.cols(); ++i) {
        //     temp_H.col(i) = PCGSolver(Amm, Amr.col(i));
        // }
        // temp_b = PCGSolver(Amm, b_marg.segment(n2, m2));
    }
    H_prior_ = H_marg.block(0, 0, n2, n2) - Amr.transpose() * temp_H;
    b_prior_ = b_marg.segment(0, n2) - Amr.transpose() * temp_b;

    H_prior_ = 0.5 * (H_prior_ + H_prior_.transpose());

    // Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> saes2(H_prior_);
    // Eigen::VectorXd S = Eigen::VectorXd((saes2.eigenvalues().array() > eps).select(saes2.eigenvalues().array(), 0));
    // Eigen::VectorXd S_inv = Eigen::VectorXd(
    //         (saes2.eigenvalues().array() > eps).select(saes2.eigenvalues().array().inverse(), 0));

    // Eigen::VectorXd S_sqrt = S.cwiseSqrt();
    // Eigen::VectorXd S_inv_sqrt = S_inv.cwiseSqrt();
    // Jt_prior_inv_ = S_inv_sqrt.asDiagonal() * saes2.eigenvectors().transpose();
    // err_prior_ = -Jt_prior_inv_ * b_prior_;

    // MatXX J = S_sqrt.asDiagonal() * saes2.eigenvectors().transpose();
    // H_prior_ = J.transpose() * J;
    // MatXX tmp_h = MatXX( (H_prior_.array().abs() > 1e-9).select(H_prior_.array(),0) );
    // H_prior_ = tmp_h;

    // std::cout << "my marg b prior: " <<b_prior_.rows()<<" norm: "<< b_prior_.norm() << std::endl;
    // std::cout << "    error prior: " <<err_prior_.norm() << std::endl;

    // remove vertex and remove edge
// #pragma omp parallel for
    for (size_t k = 0; k < margVertexs.size(); ++k) {
        RemoveVertex(margVertexs[k]);
    }

// #pragma omp parallel for
//     for (auto &landmark: landmarks) {
//         RemoveVertex(landmark);
//     }
    for (auto &landmark: margLandmark) {
        RemoveVertex(landmark.second);
    }

    return true;

}

}
}






