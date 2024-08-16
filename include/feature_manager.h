#ifndef FEATURE_MANAGER_H
#define FEATURE_MANAGER_H

#include <list>
#include <algorithm>
#include <vector>
#include <numeric>
#include <map>
#include <unordered_map>
using namespace std;

#include <eigen3/Eigen/Dense>
using namespace Eigen;

// #include <ros/console.h>
// #include <ros/assert.h>

#include "parameters.h"
#include "backend/vertex_inverse_depth.h"
#include<memory>

#define USE_OPENMP

class FeaturePerFrame {
public:
    FeaturePerFrame(const Eigen::Matrix<double, 7, 1> &data, double td) {
        point.x() = data(0);
        point.y() = data(1);
        point.z() = data(2);
        uv.x() = data(3);
        uv.y() = data(4);
        velocity.x() = data(5);
        velocity.y() = data(6);
        cur_td = td;
    }
    double cur_td;
    Vector3d point;
    Vector2d uv;
    Vector2d velocity;
    double z{};
    bool is_used{};
    double parallax{};
    MatrixXd A;
    VectorXd b;
    double dep_gradient{};
};

class FeaturePerId {
public:
    const unsigned int feature_id;
    unsigned int start_frame_id;
    vector<FeaturePerFrame> feature_per_frame;

    bool is_outlier {false};
    bool is_margin {false};
    double estimated_depth {-1.};
    int solve_flag {0}; // 0 haven't solve yet; 1 solve succ; 2 solve fail;

    Vector3d gt_p;
    shared_ptr<graph_optimization::VertexInverseDepth> vertex_landmark {new graph_optimization::VertexInverseDepth};

    FeaturePerId(unsigned int feature_idx, unsigned int start_frame_idx)
    : feature_id(feature_idx), start_frame_id(start_frame_idx) {

    }

    /*!
         * 判断该特征点是否能够用于计算重投影误差
         * 由于WINDOW的最后一帧(WINDOW_SIZE - 1)不一定是key_frame,
         * 所以最后一帧的key_frame为WINDOW中的倒数第二帧(WINDOW_SIZE - 2).
         * 而要用与计算重投影误差至少需要被2个key_frame观测到,
         * 所以start_frame < WINDOW_SIZE - 2
         * @return
         */
    bool is_suitable_to_reprojection() const { return get_used_num() >= 2 && start_frame_id + 2 < WINDOW_SIZE; }

    unsigned int get_start_frame_id() const { return start_frame_id; }
    unsigned int get_end_frame_id() const { return start_frame_id + feature_per_frame.size() - 1; }
    unsigned int get_used_num() const { return feature_per_frame.size(); }
};

class FeatureManager {
public:
    explicit FeatureManager(Matrix3d r_wi[]);

    void set_r_ic(Matrix3d r_ic[]);

    void clear_state();

    unsigned int get_suitable_feature_count();

    bool add_feature_and_check_latest_frame_parallax(unsigned int frame_count, const map<int, vector<pair<int, Eigen::Matrix<double, 7, 1>>>> &image, double td);
    void debugShow();
    vector<pair<Vector3d, Vector3d>> get_corresponding(unsigned int frame_count_l, unsigned int frame_count_r);

    //void updateDepth(const VectorXd &x);
    void set_depth(const VectorXd &x);
    void remove_failures();
    void clear_depth(const VectorXd &x);
    VectorXd get_depth_vector();
    void triangulate(Vector3d p_imu[], Vector3d t_ic[], Matrix3d r_ic[]);

    /// @brief 滑窗中移除了最老帧，更新frame index，并将持有特征的深度信息转移给次老帧
    void remove_back_shift_depth(const Eigen::Matrix3d &marg_R, const Eigen::Vector3d &marg_P, const Eigen::Matrix3d &new_R, const Eigen::Vector3d &new_P);
    void remove_back();
    void remove_front(unsigned int frame_count);
    void remove_outlier();

#ifdef USE_OPENMP
    void update_features_vector();
#endif

//    list<FeaturePerId> feature;
    unordered_map<unsigned long, FeaturePerId> features_map;
#ifdef USE_OPENMP
    // constexpr static unsigned int NUM_THREADS = 8;
    vector<pair<unsigned long, FeaturePerId *>> features_vector;
#endif
    vector<unsigned long> feature_id_erase;
    unsigned int last_track_num {0};

    unsigned long activate_features_num = 0;

private:
    double compensated_parallax2(const FeaturePerId &it_per_id, unsigned int frame_count);
    const Matrix3d *_r_wi;
    Matrix3d _r_ic[NUM_OF_CAM];
};

#endif