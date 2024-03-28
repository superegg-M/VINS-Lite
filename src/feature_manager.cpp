#include <iostream>

#include "feature_manager.h"


FeatureManager::FeatureManager(Matrix3d r_wi[]) : _r_wi(r_wi) {
    for (auto & r : _r_ic) {
        r.setIdentity();
    }
}

void FeatureManager::set_r_ic(Matrix3d r_ic[]) {
    for (unsigned int i = 0; i < NUM_OF_CAM; ++i) {
        _r_ic[i] = r_ic[i];
    }
}

void FeatureManager::clear_state() {
    feature.clear();
}

unsigned int FeatureManager::get_feature_count() {
    unsigned int cnt = 0;
    for (auto &it : feature) {
        if (it.is_suitable_to_reprojection()) {
            ++cnt;
        }
    }
    return cnt;
}

// 无论视差是否够, 都为把frame关联到其所观测到的landmark上
// <int, vector<pair<int, Eigen::Matrix<double, 7, 1>>>: (feature_id, [(camera_id, camera_param), ...])
bool FeatureManager::add_feature_and_check_latest_frame_parallax(unsigned int frame_count, const map<int, vector<pair<int, Eigen::Matrix<double, 7, 1>>>> &image, double td) {
    // TODO: 应该把新的frame加入到problem的camera顶点中
    //ROS_DEBUG("input feature: %d", (int)image.size());
    //ROS_DEBUG("num of feature: %d", get_feature_count());
    double parallax_sum = 0;
    int parallax_num = 0;
    last_track_num = 0;
    for (auto &id_pts : image)  // 遍历image中的所有landmarks
    {
        FeaturePerFrame f_per_fra(id_pts.second[0].second, td); // 对于所有的landmark, 这个参数应该都是一样的

        int feature_id = id_pts.first;
        auto it = find_if(feature.begin(), feature.end(), [feature_id](const FeaturePerId &it)
                          {
            return it.feature_id == feature_id;
                          });

        if (it == feature.end())    // 新的landmark, 由于新landmark只有一个关联的frame, 所以其没有edge
        {
            feature.emplace_back(feature_id, frame_count);  // 把新的landmark保存在feature列表中
            feature.back().feature_per_frame.push_back(f_per_fra);  // 把frame保存在新的landmark的帧队列中
            // TODO: 应该在这里给problem加入landmark顶点
        }
        else if (it->feature_id == feature_id)  // frame所观测到的landmark已经存在, 则需要把frame加入到landmark对应的edge的vertex中
        {
            it->feature_per_frame.push_back(f_per_fra);
            last_track_num++;
            // TODO: 应该在这里给problem加入edge
        }
    }

    if (frame_count < 2 || last_track_num < 20)
        return true;

    for (auto &it_per_id : feature)
    {
        if (it_per_id.start_frame_id <= frame_count - 2 &&
            it_per_id.start_frame_id + int(it_per_id.feature_per_frame.size()) - 1 >= frame_count - 1)
        {
            parallax_sum += compensated_parallax2(it_per_id, frame_count);
            parallax_num++;
            // std::cout << "Case 0: frame_count = " << frame_count << ", start_frame_id = " << it_per_id.start_frame_id << ", end_frame = " << (it_per_id.start_frame_id + int(it_per_id.feature_per_frame.size()) - 1) << std::endl;

        }
    }

    if (parallax_num == 0)
    {
        return true;
    }
    else
    {
        //ROS_DEBUG("parallax_sum: %lf, parallax_num: %d", parallax_sum, parallax_num);
        //ROS_DEBUG("current parallax: %lf", parallax_sum / parallax_num * FOCAL_LENGTH);
        return parallax_sum / parallax_num >= MIN_PARALLAX;
    }
}

void FeatureManager::debugShow()
{
    //ROS_DEBUG("debug show");
    for (auto &it : feature)
    {
        assert(!it.feature_per_frame.empty());
        assert(it.start_frame_id >= 0);
        assert(it.get_used_num() >= 0);

        //ROS_DEBUG("%d,%d,%d ", it.feature_id, it.used_num, it.start_frame_id);
        int sum = 0;
        for (auto &j : it.feature_per_frame)
        {
            //ROS_DEBUG("%d,", int(j.is_used));
            sum += j.is_used;
            printf("(%lf,%lf) ",j.point(0), j.point(1));
        }
        assert(it.get_used_num() == sum);
    }
}

vector<pair<Vector3d, Vector3d>> FeatureManager::get_corresponding(unsigned int frame_count_l, unsigned int frame_count_r)
{
    vector<pair<Vector3d, Vector3d>> corres;
    for (auto &it : feature) {
        if (it.start_frame_id <= frame_count_l && it.get_end_frame_id() >= frame_count_r) {
            Vector3d a = Vector3d::Zero(), b = Vector3d::Zero();
            unsigned int idx_l = frame_count_l - it.start_frame_id;
            unsigned int idx_r = frame_count_r - it.start_frame_id;

            a = it.feature_per_frame[idx_l].point;
            b = it.feature_per_frame[idx_r].point;
            
            corres.emplace_back(a, b);
        }
    }
    return corres;
}

void FeatureManager::set_depth(const VectorXd &x) {
    unsigned int feature_index = 0;
    for (auto &it_per_id : feature) {
        if (it_per_id.is_suitable_to_reprojection()) {
            it_per_id.estimated_depth = 1.0 / x(feature_index++);
            //ROS_INFO("feature id %d , start_frame_id %d, depth %f ", it_per_id->feature_id, it_per_id-> start_frame_id, it_per_id->estimated_depth);
            if (it_per_id.estimated_depth < 0) {
                it_per_id.solve_flag = 2;
            }
            else {
                it_per_id.solve_flag = 1;
            }
        }

    }
}

void FeatureManager::remove_failures() {
    for (auto it = feature.begin(), it_next = feature.begin(); it != feature.end(); it = it_next) {
        it_next++;
        if (it->solve_flag == 2) {
            feature.erase(it);
        }
    }
}

void FeatureManager::clear_depth(const VectorXd &x) {
    unsigned int feature_index = 0;
    for (auto &it_per_id : feature) {
        if (it_per_id.is_suitable_to_reprojection()) {
            it_per_id.estimated_depth = 1.0 / x(feature_index++);
        }

    }
}

VectorXd FeatureManager::get_depth_vector() {
    VectorXd dep_vec(get_feature_count());
    unsigned int feature_index = 0;
    for (auto &it_per_id : feature) {
        if (it_per_id.is_suitable_to_reprojection()) {
#if 1
            dep_vec(feature_index++) = 1. / it_per_id.estimated_depth;
#else
            dep_vec(++feature_index) = it_per_id->estimated_depth;
#endif
        }
    }
    return dep_vec;
}

void FeatureManager::triangulate(Vector3d p_imu[], Vector3d t_ic[], Matrix3d r_ic[]) {
    for (auto &it_per_id : feature) {
        if (it_per_id.is_suitable_to_reprojection()) {
            if (it_per_id.estimated_depth > 0.)
                continue;

            unsigned int imu_i = it_per_id.start_frame_id;

            assert(NUM_OF_CAM == 1);
            Eigen::MatrixXd svd_A(2 * it_per_id.feature_per_frame.size(), 4);
            Eigen::Matrix<double, 3, 4> P;
            Eigen::Vector3d f;

            Eigen::Vector3d t_wci_w = p_imu[imu_i] + _r_wi[imu_i] * t_ic[0];
            Eigen::Matrix3d r_wci = _r_wi[imu_i] * r_ic[0];

            P.leftCols<3>() = Eigen::Matrix3d::Identity();
            P.rightCols<1>() = Eigen::Vector3d::Zero();

            f = it_per_id.feature_per_frame[0].point.normalized();
            svd_A.row(0) = f[0] * P.row(2) - f[2] * P.row(0);
            svd_A.row(1) = f[1] * P.row(2) - f[2] * P.row(1);

            for (unsigned int j = 1; j < it_per_id.feature_per_frame.size(); ++j) {
                unsigned int imu_j = imu_i + j;

                Eigen::Vector3d t_wcj_w = p_imu[imu_j] + _r_wi[imu_j] * t_ic[0];
                Eigen::Matrix3d r_wcj = _r_wi[imu_j] * r_ic[0];
                Eigen::Vector3d t_cicj_ci = r_wci.transpose() * (t_wcj_w - t_wci_w);
                Eigen::Matrix3d r_cicj = r_wci.transpose() * r_wcj;

                P.leftCols<3>() = r_cicj.transpose();
                P.rightCols<1>() = -r_cicj.transpose() * t_cicj_ci;

                f = it_per_id.feature_per_frame[j].point.normalized();
                svd_A.row(2 * j) = f[0] * P.row(2) - f[2] * P.row(0);
                svd_A.row(2 * j + 1) = f[1] * P.row(2) - f[2] * P.row(1);
            }

            Eigen::Vector4d svd_V = Eigen::JacobiSVD<Eigen::MatrixXd>(svd_A, Eigen::ComputeThinV).matrixV().rightCols<1>();
            double depth = svd_V[2] / svd_V[3];
            it_per_id.estimated_depth = depth;

            if (it_per_id.estimated_depth < 0.1) {
                it_per_id.estimated_depth = INIT_DEPTH;
            }
        }
    }
}

void FeatureManager::remove_outlier()
{
    // ROS_BREAK();
    return;
    int i = -1;
    for (auto it = feature.begin(), it_next = feature.begin();
         it != feature.end(); it = it_next)
    {
        it_next++;
        i += it->get_used_num() != 0;
        if (it->get_used_num() != 0 && it->is_outlier == true)
        {
            feature.erase(it);
        }
    }
}

void FeatureManager::remove_back_shift_depth(const Eigen::Matrix3d &marg_R, const Eigen::Vector3d &marg_P, const Eigen::Matrix3d &new_R, const Eigen::Vector3d &new_P)
{
    for (auto it = feature.begin(), it_next = feature.begin(); it != feature.end(); it = it_next) {
        it_next++;

        if (it->start_frame_id != 0)
            it->start_frame_id--;
        else
        {
            Eigen::Vector3d uv_i = it->feature_per_frame[0].point;  
            it->feature_per_frame.erase(it->feature_per_frame.begin());
            if (it->feature_per_frame.size() < 2)
            {
                feature.erase(it);
                continue;
            }
            else
            {
                Eigen::Vector3d pts_i = uv_i * it->estimated_depth;
                Eigen::Vector3d w_pts_i = marg_R * pts_i + marg_P;
                Eigen::Vector3d pts_j = new_R.transpose() * (w_pts_i - new_P);
                double dep_j = pts_j(2);
                if (dep_j > 0)
                    it->estimated_depth = dep_j;
                else
                    it->estimated_depth = INIT_DEPTH;
            }
        }
        // remove tracking-lost feature after marginalize
        /*
        if (it->get_end_frame_id() < WINDOW_SIZE - 1)
        {
            feature.erase(it);
        }
        */
    }
}

void FeatureManager::remove_back()
{
    for (auto it = feature.begin(), it_next = feature.begin(); it != feature.end(); it = it_next) {
        it_next++;

        if (it->start_frame_id != 0)
            it->start_frame_id--;
        else
        {
            it->feature_per_frame.erase(it->feature_per_frame.begin());
            if (it->feature_per_frame.empty())
                feature.erase(it);
        }
    }
}

void FeatureManager::remove_front(unsigned int frame_count)
{
    for (auto it = feature.begin(), it_next = feature.begin(); it != feature.end(); it = it_next) {
        it_next++;

        if (it->start_frame_id == frame_count)
        {
            it->start_frame_id--;
            // std::cout << "Case 1: frame_count = " << frame_count << ", start_frame_id = " << it->start_frame_id << ", end_frame = " << it->get_end_frame_id() << std::endl;
        }
        else
        {
            unsigned int j = WINDOW_SIZE - 1 - it->start_frame_id;
            if (it->get_end_frame_id() < frame_count - 1)
                continue;
            it->feature_per_frame.erase(it->feature_per_frame.begin() + j);
            if (it->feature_per_frame.empty())
                feature.erase(it);
            // std::cout << "Case 2: frame_count = " << frame_count << ", start_frame_id = " << it->start_frame_id << ", end_frame = " << it->get_end_frame_id() << std::endl;
        }
    }
}

double FeatureManager::compensated_parallax2(const FeaturePerId &it_per_id, unsigned int frame_count)
{
    //check the second last frame is keyframe or not
    //parallax betwwen seconde last frame and third last frame
    const FeaturePerFrame &frame_i = it_per_id.feature_per_frame[frame_count - 2 - it_per_id.start_frame_id];
    const FeaturePerFrame &frame_j = it_per_id.feature_per_frame[frame_count - 1 - it_per_id.start_frame_id];

    double ans = 0;
    Vector3d p_j = frame_j.point;

    double u_j = p_j(0);
    double v_j = p_j(1);

    Vector3d p_i = frame_i.point;
    Vector3d p_i_comp;

    //int r_i = frame_count - 2;
    //int r_j = frame_count - 1;
    //p_i_comp = _r_ic[camera_id_j].transpose() * _r_wi[r_j].transpose() * _r_wi[r_i] * _r_ic[camera_id_i] * p_i;
    p_i_comp = p_i;
    double dep_i = p_i(2);
    double u_i = p_i(0) / dep_i;
    double v_i = p_i(1) / dep_i;
    double du = u_i - u_j, dv = v_i - v_j;

    double dep_i_comp = p_i_comp(2);
    double u_i_comp = p_i_comp(0) / dep_i_comp;
    double v_i_comp = p_i_comp(1) / dep_i_comp;
    double du_comp = u_i_comp - u_j, dv_comp = v_i_comp - v_j;

    ans = max(ans, sqrt(min(du * du + dv * dv, du_comp * du_comp + dv_comp * dv_comp)));

    return ans;
}