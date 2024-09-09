//
// Created by Cain on 2024/1/11.
//

#ifndef GRAPH_OPTIMIZATION_ESTIMATOR_H
#define GRAPH_OPTIMIZATION_ESTIMATOR_H

#include "parameters.h"

#include "graph_optimization/eigen_types.h"
#include "graph_optimization/problem.h"
#include "graph_optimization/problem_slam.h"

#include "utility/tic_toc.h"
#include "utility/utility.h"
#include "sophus/so3.hpp"

#include <unordered_map>
#include <queue>
//#include <opencv2/core/eigen.hpp>

#include "imu_integration.h"
#include "edge/edge_imu.h"
#include "edge/edge_reprojection.h"
#include "edge/edge_pnp.h"
#include "edge/edge_pnp_sim3.h"
#include "edge/edge_epipolar.h"
#include "vertex/vertex_inverse_depth.h"
#include "vertex/vertex_pose.h"
#include "vertex/vertex_motion.h"
#include "vertex/vertex_scale.h"
#include "vertex/vertex_quaternion.h"
#include "vertex/vertex_spherical.h"
#include "data_structure/feature.h"
#include "data_structure/frame.h"
#include "data_structure/landmark.h"
//#include "data_structure/map.h"

#define INITIAL_ONLY_SLIDING_WINDOW_IS_FULL

namespace vins {
    using namespace graph_optimization;

    class Estimator {
    public:
        enum class SolverFlag {
            INITIAL,
            OPTIMIZATION
        };

        enum class MarginalizationFlag {
            NONE,
            MARGIN_OLD,
            MARGIN_SECOND_NEW
        };

    public:
        Estimator();
        virtual ~Estimator();

        void set_ext_param(unsigned int index, const Eigen::Vector3d &t_ic, const Eigen::Quaterniond &q_ic);
        void set_estimate_ext_param(bool is_estimate) { _is_estimate_ext_param = is_estimate; }

        // interface
        void process_imu(double t, const Vec3 &linear_acceleration, const Vec3 &angular_velocity);

        void process_image(const std::vector<std::pair<unsigned long, std::vector<Vec7>>> &image, uint64_t time_stamp);

        // 后端
        void backend();
        void prepare_landmarks();
        bool initialize();
        void reset_initialization_flags();
        void optimization();
        void slide_window();

        // visual initialize
        bool search_relative_pose(Mat33 &r, Vec3 &t, unsigned long &imu_index);
        bool structure_from_motion();
        bool stereo_visual_initialize(Qd *q_wi_init=nullptr, Vec3 *t_wi_init=nullptr);

        // inertial initialize
        bool align_visual_to_imu();

        // 2D-2D
        /**
         * 以 frame_i 的 imu 为建立惯性系, 计算frame_i到frame_j的位姿,
         * @param R R_ij
         * @param t t_ij_i
         * @param frame_i 基准
         * @param frame_j 目标
         * @param is_init_landmark 是否同时初始化 frame_i 到 frame_j 中的landmark
         * @param max_iters 最大 RANSAC 次数, 默认为30
         * @return 返回本质矩阵计算是否成功
         */
        bool compute_essential_matrix(Mat33 &R, Vec3 &t, const std::shared_ptr<Frame> &frame_i, const std::shared_ptr<Frame> &frame_j, bool is_init_landmark=true, unsigned int max_iters=1000);
        bool compute_homography_matrix(Mat33 &R, Vec3 &t, const std::shared_ptr<Frame> &frame_i, const std::shared_ptr<Frame> &frame_j, bool is_init_landmark=true, unsigned int max_iters=1000);
        // Eigen::Matrix<double, 3, 3> solve_essential_5pt(const Eigen::Matrix<double, 5, 9> &D);
        // Eigen::Matrix<double, 3, 3> solve_essential_8pt(const Eigen::Matrix<double, 8, 9> &D);

        // 2D-3D
        unsigned long global_triangulate_with(const std::shared_ptr<Frame> &frame_i, bool enforce=false);
        unsigned long global_triangulate_with(const std::shared_ptr<Frame> &frame_i, const std::shared_ptr<Frame> &frame_j, bool enforce=false);
        bool global_triangulate_feature(const std::shared_ptr<Landmark> &landmark, bool enforce=false);
//        void global_triangulate_between(unsigned long i, unsigned long j, bool enforce=false);
        unsigned long local_triangulate_with(const std::shared_ptr<Frame> &frame_i, bool enforce=false);
        unsigned long local_triangulate_with(const std::shared_ptr<Frame> &frame_i, const std::shared_ptr<Frame> &frame_j, bool enforce=false);
        bool local_triangulate_feature(const std::shared_ptr<Landmark> &landmark, bool enforce=false);
//        void local_triangulate_between(unsigned long i, unsigned long j, bool enforce=false);

        // 3D-2D
        bool pnp(const std::shared_ptr<Frame> &frame_i, Qd *q_wi_init=nullptr, Vec3 *t_wi_init=nullptr, unsigned int num_iters=5);
        bool pnp_local(const std::shared_ptr<Frame> &frame_i, Qd *q_wi_init=nullptr, Vec3 *t_wi_init=nullptr, unsigned int num_iters=5);
        bool epnp(const std::shared_ptr<Frame> &frame_i);
        bool mlpnp(const std::shared_ptr<Frame> &frame_i, unsigned int batch_size=36, unsigned int num_batches=30);
        bool dltpnp(const std::shared_ptr<Frame> &frame_i, unsigned int batch_size=36, unsigned int num_batches=30);
        bool iter_pnp(const std::shared_ptr<Frame> &frame_i, Qd *q_wi_init=nullptr, Vec3 *t_wi_init=nullptr, unsigned int num_iters=5);
        bool iter_pnp_local(const std::shared_ptr<Frame> &frame_i, Qd *q_wi_init=nullptr, Vec3 *t_wi_init=nullptr, unsigned int num_iters=5);

        // bundle adjustment
        void global_bundle_adjustment(std::vector<std::shared_ptr<VertexPose>> *fixed_poses=nullptr);
        void local_bundle_adjustment(std::vector<std::shared_ptr<VertexPose>> *fixed_poses=nullptr);

        bool is_landmark_suitable(const std::shared_ptr<Landmark> &landmark) {
            return landmark->observations.size() >= 2 && _sliding_window.size() >= 2 &&
                   landmark->observations.front().first->time_us < _sliding_window[_sliding_window.size() - 2].first->time_us;
        }
        bool is_sliding_window_full() const { return _sliding_window.size() >= WINDOW_SIZE; }
        bool is_data_enough() const {
#ifdef INITIAL_ONLY_SLIDING_WINDOW_IS_FULL
            return _sliding_window.size() >= WINDOW_SIZE;
#else
            return _stream.size() >= WINDOW_SIZE;
#endif
        }

        void search_outlier_landmarks(unsigned int iteration=5);
        bool remove_outlier_landmarks(bool lazy=false);
        bool remove_untriangulated_landmarks(bool lazy=false);

        double get_td() const { return _td; }
        Frame get_frame() const { return *_frame; }
        std::vector<Eigen::Vector3d> get_positions() const;

        bool _is_estimate_ext_param {false};
        bool _is_visual_initialized {false};
        bool _is_visual_aligned_to_imu {false};

        Eigen::Matrix2d _project_sqrt_info;
        SolverFlag solver_flag {SolverFlag::INITIAL};
        MarginalizationFlag  marginalization_flag {MarginalizationFlag::NONE};

    public:
        Vec3 _g {0., 0., -9.8};
        Vec3 _acc_latest = Vec3::Zero();
        Vec3 _gyro_latest = Vec3::Zero();

        double _td {0.};
        std::vector<double *> _ext_params;
        std::vector<double *> _ext_params_bp;
        std::vector<Eigen::Map<Eigen::Vector3d>> _t_ic;
        std::vector<Eigen::Map<Eigen::Quaterniond>> _q_ic;

        std::shared_ptr<Frame> _frame {std::make_shared<Frame>()};
        std::shared_ptr<IMUIntegration> _pre_integral_stream {std::make_shared<IMUIntegration>()};
        std::shared_ptr<IMUIntegration> _pre_integral_window {std::make_shared<IMUIntegration>()};

        graph_optimization::ProblemSLAM _problem;
        std::vector<VertexPose> _vertex_ext_vec;
        std::vector<VertexPose> _vertex_pose_vec;
        std::vector<VertexMotion> _vertex_motion_vec;
        std::vector<EdgeImu> _edge_imu;

        std::vector<size_t> _num_landmarks;
        std::vector<std::vector<VertexInverseDepth>> _vertex_landmarks_vec;

        std::vector<size_t> _num_edges_12;
        std::vector<size_t> _num_edges_21;
        std::vector<size_t> _num_edges_22;
        std::vector<std::vector<EdgeReprojectionOneImuTwoCameras>> _edges_12_vec;
        std::vector<std::vector<EdgeReprojectionTwoImuOneCameras>> _edges_21_vec;
        std::vector<std::vector<EdgeReprojectionTwoImuTwoCameras>> _edges_22_vec;

        std::vector<std::vector<Vertex*>> _marg_landmarks_vec;
        std::vector<std::vector<Edge*>> _marg_edges_vec;
        std::vector<Vertex*> _marg_landmarks;
        std::vector<Edge*> _marg_edges;

        std::vector<vins::IMUIntegration*> _pre_integral_vec;

        std::list<std::pair<std::shared_ptr<Frame>, std::shared_ptr<IMUIntegration>>> _stream;
        std::deque<std::pair<std::shared_ptr<Frame>, std::shared_ptr<IMUIntegration>>> _sliding_window;

        std::unordered_map<unsigned long, std::shared_ptr<Landmark>> _landmarks;
        std::vector<std::shared_ptr<Landmark>> _suitable_landmarks;
        std::vector<std::shared_ptr<Landmark>> _landmarks_vector;

        std::vector<std::vector<unsigned long>> _landmark_erase_id_vec;
    protected:
//        Map &_map;
    };
}

#endif //GRAPH_OPTIMIZATION_ESTIMATOR_H
