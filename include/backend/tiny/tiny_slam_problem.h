//
// Created by Cain on 2024/8/13.
//

#ifndef GRAPH_OPTIMIZATION_TINY_PROBLEM_SLAM_H
#define GRAPH_OPTIMIZATION_TINY_PROBLEM_SLAM_H

#include "tiny_problem.h"

namespace graph_optimization {
    class TinySLAMProblem : public TinyProblem {
    public:
        TinySLAMProblem() = default;

        void add_state_vertex(const std::shared_ptr<Vertex>& vertex) {
            ++_num_state_vertices;
            _ordering_states += vertex->local_dimension();
            TinyProblem::add_vertex(vertex);
        }
        void add_landmark_vertex(const std::shared_ptr<Vertex>& vertex) {
            ++_num_landmark_vertices;
            _ordering_landmarks += vertex->local_dimension();
            TinyProblem::add_vertex(vertex);
        }
        bool marginalize(const std::shared_ptr<Vertex>& vertex_pose, 
                         const std::shared_ptr<Vertex>& vertex_motion,
                         const std::vector<std::shared_ptr<Vertex>> &vertices_landmark,
                         const std::vector<std::shared_ptr<Edge>> &edges);

    public:
        static bool is_pose_vertex(const std::shared_ptr<Vertex>& v) {
            string type = v->type_info();
            return type == string("VertexPose") || type == string("VertexMotion");
        }
        static bool is_landmark_vertex(const std::shared_ptr<Vertex>& v) {
            string type = v->type_info();
            return type == string("VertexPointXYZ") || type == string("VertexInverseDepth");
        }

    protected:
        void initialize() override;
        void add_prior_to_hessian() override;
        bool solve_linear_system(VecX &delta_x) override;
        void update_prior(const VecX &delta_x) override;

    public:
        ulong _num_state_vertices = 0;
        ulong _num_landmark_vertices = 0;
        ulong _ordering_states = 0;
        ulong _ordering_landmarks = 0;

        // 使用schur补求解线性方程组时的过程量
        MatXX _h_pp_schur;
        VecX _b_pp_schur;
    };
}

#endif //GRAPH_OPTIMIZATION_TINY_PROBLEM_SLAM_H