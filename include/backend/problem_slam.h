//
// Created by Cain on 2024/3/7.
//

#ifndef GRAPH_OPTIMIZATION_PROBLEM_SLAM_H
#define GRAPH_OPTIMIZATION_PROBLEM_SLAM_H

#include "problem.h"

namespace graph_optimization {
    class ProblemSLAM : public Problem {
    public:
        ProblemSLAM() = default;
        void clear() override;

        bool add_state_vertex(const std::shared_ptr<Vertex>& vertex);
        bool add_landmark_vertex(const std::shared_ptr<Vertex>& vertex);
        bool add_reproj_edge(const std::shared_ptr<Edge> &edge);
        bool add_imu_edge(const std::shared_ptr<Edge> &edge);

        bool marginalize(const std::shared_ptr<Vertex>& vertex_pose, const std::shared_ptr<Vertex>& vertex_motion); ///< 边缘化pose和与之相关的edge
        bool marginalize(const std::shared_ptr<Vertex>& vertex_pose, 
                         const std::shared_ptr<Vertex>& vertex_motion,
                         const std::vector<std::shared_ptr<Vertex>> &vertices_landmark,
                         const std::vector<std::shared_ptr<Edge>> &edges);
        bool marginalize(const std::shared_ptr<Vertex>& vertex_pose, 
                         const std::shared_ptr<Vertex>& vertex_motion,
                         const std::vector<std::shared_ptr<Vertex>> &marginalized_landmarks,
                         const std::vector<std::shared_ptr<Edge>> &marginalized_edges,
                         bool marg_oldest);

    public:
        static bool is_pose_vertex(const std::shared_ptr<Vertex>& v);   ///< 判断一个顶点是否为Pose顶点
        static bool is_landmark_vertex(const std::shared_ptr<Vertex>& v);   ///< 判断一个顶点是否为landmark顶点

    protected:
        void initialize_ordering() override;
        bool check_ordering() override;
        void add_prior_to_hessian() override;
        VecX multiply_hessian(const VecX &x) override;
        bool solve_linear_system(VecX &delta_x) override;
        void update_prior(const VecX &delta_x) override;
        void update_hessian() override;

    public:
        ulong _ordering_poses = 0;
        ulong _ordering_landmarks = 0;
        std::vector<std::shared_ptr<Vertex>> _pose_vertices;
        std::vector<std::shared_ptr<Vertex>> _landmark_vertices;
        std::vector<std::shared_ptr<Edge>> _reproj_edges;
        std::vector<std::shared_ptr<Edge>> _imu_edges;

        // 使用schur补求解线性方程组时的过程量
        MatXX _h_pp_schur;
        VecX _b_pp_schur;
//            MatXX _h_pp;
//            VecX _b_pp;
//            MatXX _h_ll;
//            VecX _b_ll;
    };
}

#endif //GRAPH_OPTIMIZATION_PROBLEM_SLAM_H
