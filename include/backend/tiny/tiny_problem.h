//
// Created by Cain on 2024/8/13.
//

#ifndef GRAPH_OPTIMIZATION_TINY_PROBLEM_H
#define GRAPH_OPTIMIZATION_TINY_PROBLEM_H

#include <vectory>
#include <memory>

#include "../../utility/tic_toc.h"
#include "../eigen_types.h"
#include "../vertex.h"
#include "../edge.h"
#include <omp.h>

#define USE_OPENMP

namespace graph_optimization {
    class TinyProblem {
    public:
        enum class SolverType {
            STEEPEST_DESCENT,
            GAUSS_NEWTON,
            LEVENBERG_MARQUARDT,
            DOG_LEG,
            TMP
        };
        typedef unsigned long ulong;

        EIGEN_MAKE_ALIGNED_OPERATOR_NEW;

    public:
        explicit TinyProblem();
        explicit TinyProblem(size_t num_vertices, size_t num_edges) {
            _vertices.reserve(num_vertices);
            _edges.reserve(num_edges);
        }
        virtual ~TinyProblem() = default;

        virtual void add_vertex(const std::shared_ptr<Vertex>& vertex) { 
            vertex->set_ordering_id(_ordering_generic);
            _ordering_generic += vertex->local_dimension();
            _vertices.emplace_back(vertex); 
        }
        virtual void add_edge(const std::shared_ptr<Edge>& edge) { _edges.emplace_back(edge); }
        void extend_prior_hessian_size(ulong dim);
        bool solve(unsigned long iterations);

    public:
        void get_outlier_edges(std::vector<std::shared_ptr<Edge>> &outlier_edges);  ///< 取得在优化中被判断为outlier部分的边，方便前端去除outlier
        MatXX get_h_prior() const { return _h_prior; }
        VecX get_b_prior() const { return _b_prior; }
        double get_chi2() const { return _chi2; }

    public:
        void set_solver_type(SolverType type) { _solver_type = type; }
        void set_h_prior(const MatXX &h_prior) { _h_prior = h_prior; }
        void set_b_prior(const VecX &b_prior) { _b_prior = b_prior; }
    public:
        MatXX &h_prior() { return _h_prior; }
        VecX &b_prior() { return _b_prior; }

    protected:
        virtual void initialize();    ///< 设置各顶点的ordering_index
        void update_states(const VecX &delta_x);    ///< x_bp = x, x = x + Δx, b_prior_bp = b_prior, b_prior = b_prior - H_prior * Δx
        void rollback_states(const VecX &delta_x);  ///< x = x_bp, b_prior = b_prior_bp
        virtual void update_prior(const VecX &delta_x); ///< b_prior_bp = b_prior, b_prior = b_prior - H_prior * Δx; 在update_states()中运行
        void update_residual(); ///< 计算每条边的残差;      运行顺序必须遵循: update_state -> update_residual
        void update_jacobian(); ///< 计算每条边的残差;      运行顺序必须遵循: update_residual -> update_jacobian
        void update_chi2(); ///< 计算综合的chi2;           运行顺序必须遵循: update_residual -> update_chi2
        void update_hessian();    ///< 计算H, b, J, f;    运行顺序必须遵循: update_jacobian -> update_hessian
        virtual void add_prior_to_hessian();    ///< H = H + H_prior;   在make_hessian()时会运行
        void initialize_lambda();   ///< 计算λ;   运行顺序必须遵循: update_hessian -> initialize_lambda
        virtual VecX multiply_hessian(const VecX &x); ///< 计算: Hx
        virtual double calculate_hessian_norm_square(const VecX &x);    ///< 计算: x'Hx
        virtual bool solve_linear_system(VecX &delta_x);    ///< 解: (H+λ)Δx = b
        bool one_step_steepest_descent(VecX &delta_x);  ///< 计算: h_sd = alpha*g
        bool one_step_gauss_newton(VecX &delta_x);  ///< 计算: h_gn = (H+λ)/g;   运行顺序必须遵循: update_hessian -> one_step_gauss_newton
        bool calculate_steepest_descent(VecX &delta_x, unsigned long iterations=10);
        bool calculate_gauss_newton(VecX &delta_x, unsigned long iterations=10);
        bool calculate_levenberg_marquardt(VecX &delta_x, unsigned long iterations=10);
        bool calculate_dog_leg(VecX &delta_x, unsigned long iterations=10);

        /// 计算并更新Prior部分
        void compute_prior();

        void logout_vector_size();

        void save_hessian_diagonal_elements();
        void load_hessian_diagonal_elements();

        /// PCG 迭代线性求解器
        VecX PCG_solver(const MatXX &A, const VecX &b, unsigned long max_iter=0);

        bool Solve(unsigned long iterations=10);

    protected:
        bool _debug = false;
        SolverType _solver_type {SolverType::DOG_LEG};

        double _t_residual_cost = 0.;
        double _t_chi2_cost = 0.;
        double _t_jacobian_cost = 0.;
        double _t_hessian_cost = 0.;
        double _t_PCG_solve_cost = 0.;

        ulong _ordering_generic = 0;
        double _chi2 {0.};

        MatXX _hessian;
        VecX _b;
        VecX _hessian_diag;
        VecX _delta_x;
        VecX _delta_x_sd;
        VecX _delta_x_gn;
        VecX _delta_x_lm;
        VecX _delta_x_dl;

        MatXX _h_prior;
        VecX _b_prior;
        VecX _b_prior_bp;

        std::vector<std::shared_ptr<Vertex>> _vertices;   ///< 所有的顶点
        std::vector<std::shared_ptr<Edge>> _edges;    ///< 所有的边

        // Gauss-Newton or Levenberg-Marquardt
        double _ni {2.};                 //控制 lambda 缩放大小
        VecX _diag_lambda;
        double _current_lambda {0.};
        double _lambda_min {1e-6};
        double _lambda_max {1e6};

        // Dog leg
        double _delta {1.};
        double _delta_min {1e-6};
        double _delta_max {1e6};
    };
}

#endif //GRAPH_OPTIMIZATION_TINY_PROBLEM_H
