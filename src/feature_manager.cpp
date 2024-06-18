#include <iostream>
#include <omp.h>
#include "feature_manager.h"

FeatureManager::FeatureManager(Matrix3d r_wi[]) : _r_wi(r_wi), feature_id_erase(NUM_OF_F) {
    for (auto & r : _r_ic) {
        r.setIdentity();
    }
    feature_id_erase.clear();
}

void FeatureManager::set_r_ic(Matrix3d r_ic[]) {
    for (unsigned int i = 0; i < NUM_OF_CAM; ++i) {
        _r_ic[i] = r_ic[i];
    }
}

void FeatureManager::clear_state() {
    features_map.clear();
    feature_id_erase.clear();
}

unsigned int FeatureManager::get_suitable_feature_count() {
    unsigned int cnt = 0;
    for (auto &it : features_map) {
        if (it.second.is_suitable_to_reprojection()) {
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
    //ROS_DEBUG("num of feature: %d", get_suitable_feature_count());
    double parallax_sum = 0;
    int parallax_num = 0;
    last_track_num = 0;
    for (auto &id_pts : image) {    // 遍历image中的所有landmarks
        FeaturePerFrame f_per_fra(id_pts.second[0].second, td); // 对于所有的landmark, 这个参数应该都是一样的
        unsigned long feature_id = id_pts.first;
        auto it = features_map.find(feature_id);

        /*
             * 1. 如果该feature是新的feature, 则把feature加入到feature_map中
             * 2. 如果是已经出现过的feature, 则说明该特征点已经在被跟踪着, 同时该特征点一定在newest frame被观测到
             * */
        if (it == features_map.end()) { // 新的landmark, 由于新landmark只有一个关联的frame, 所以其没有edge
            features_map.emplace(pair<unsigned long, FeaturePerId>(feature_id, FeaturePerId(feature_id, frame_count)));  // 把新的landmark保存在feature列表中
            it = features_map.find(feature_id);
            // TODO: 应该在这里给problem加入landmark顶点
        }
        else {  // frame所观测到的landmark已经存在, 则需要把frame加入到landmark对应的edge的vertex中
            last_track_num++;
            // TODO: 应该在这里给problem加入edge
        }
        it->second.feature_per_frame.emplace_back(f_per_fra);  // 把frame保存在新的landmark的帧队列中
    }

    /*
     * 1. 只有1帧, 说明WINDOW中目前并没有任何frame, 所以此时需要无条件把该frame加入到WINDOW中
     * 2. image中只有20个特征点处于被追踪的状态, 则说明camera frame已经与WINDOW中的frame差异很大, 所以要去掉WINDOW中的old frame
     * */
    if (frame_count < 2 || last_track_num < 20)
        return true;

    /*
     * 如果一个特征start_frame_id <= newest_frame_id - 1 && end_frame_id >= newest_frame_id
     * 则该特征一定在newest_frame中被观测到且处于被追踪的状态
     * */
    for (auto &it_per_id : features_map) {
        if (it_per_id.second.start_frame_id + 2 <= frame_count &&
            it_per_id.second.start_frame_id + it_per_id.second.get_end_frame_id() + 1 >= frame_count) {
            parallax_sum += compensated_parallax2(it_per_id.second, frame_count);
            ++parallax_num;
            // std::cout << "Case 0: frame_count = " << frame_count << ", start_frame_id = " << it_per_id.start_frame_id << ", end_frame = " << (it_per_id.start_frame_id + int(it_per_id.feature_per_frame.size()) - 1) << std::endl;

        }
    }

    /*
     * 如果parallax_num == 0, 则说明对于所有feature, 都不满足start_frame_id <= newest_frame_id - 1 && end_frame_id >= newest_frame_id
     * 但是对于条件 end_frame_id >= newest_frame_id 必定有feature满足, 比如说当前image中所观测到的feature
     * 所以说明对于所有feature均不满足的条件为start_frame_id <= newest_frame_id - 1,
     * 即: start_frame_id > newest_frame_id - 1 <=> start_frame_id >= newest_frame_id
     * 则说明WINDOW中比newest_frame旧的frame, 都已经没有任何feature与之相连, 所以要去掉WINDOW中的old frame
     * */
    if (parallax_num == 0) {
        return true;
    } else {
        //ROS_DEBUG("parallax_sum: %lf, parallax_num: %d", parallax_sum, parallax_num);
        //ROS_DEBUG("current parallax: %lf", parallax_sum / parallax_num * FOCAL_LENGTH);
        return parallax_sum / parallax_num >= MIN_PARALLAX;
    }
}

void FeatureManager::debugShow() {
    //ROS_DEBUG("debug show");
    for (auto &it : features_map) {
        assert(!it.second.feature_per_frame.empty());
        assert(it.second.start_frame_id >= 0);
        assert(it.second.get_used_num() >= 0);

        //ROS_DEBUG("%d,%d,%d ", it.feature_id, it.used_num, it.start_frame_id);
        int sum = 0;
        for (auto &j : it.second.feature_per_frame) {
            //ROS_DEBUG("%d,", int(j.is_used));
            sum += j.is_used;
            printf("(%lf,%lf) ",j.point(0), j.point(1));
        }
        assert(it.second.get_used_num() == sum);
    }
}

vector<pair<Vector3d, Vector3d>> FeatureManager::get_corresponding(unsigned int frame_count_l, unsigned int frame_count_r) {
    vector<pair<Vector3d, Vector3d>> corres;
    for (auto &it : features_map) {
        if (it.second.start_frame_id <= frame_count_l && it.second.get_end_frame_id() >= frame_count_r) {
            Vector3d a = Vector3d::Zero(), b = Vector3d::Zero();
            unsigned int idx_l = frame_count_l - it.second.start_frame_id;
            unsigned int idx_r = frame_count_r - it.second.start_frame_id;

            a = it.second.feature_per_frame[idx_l].point;
            b = it.second.feature_per_frame[idx_r].point;
            
            corres.emplace_back(a, b);
        }
    }
    return corres;
}

void FeatureManager::set_depth(const VectorXd &x) {
    unsigned int feature_index = 0;
    for (auto &it_per_id : features_map) {
        if (it_per_id.second.is_suitable_to_reprojection()) {
            it_per_id.second.estimated_depth = 1.0 / x(feature_index++);
            //ROS_INFO("feature id %d , start_frame_id %d, depth %f ", it_per_id->feature_id, it_per_id-> start_frame_id, it_per_id->estimated_depth);
            if (it_per_id.second.estimated_depth < 0) {
                it_per_id.second.solve_flag = 2;
            }
            else {
                it_per_id.second.solve_flag = 1;
            }
        }
    }
}

void FeatureManager::remove_failures() {
    feature_id_erase.clear();
    for (auto &it : features_map) {
        if (it.second.solve_flag == 2) {
            feature_id_erase.emplace_back(it.first);
        }
    }
    for (auto &id : feature_id_erase) {
        features_map.erase(id);
    }
}

void FeatureManager::clear_depth(const VectorXd &x) {
    unsigned int feature_index = 0;
    for (auto &it_per_id : features_map) {
        if (it_per_id.second.is_suitable_to_reprojection()) {
            it_per_id.second.estimated_depth = 1.0 / x(feature_index++);
        }
    }
}

VectorXd FeatureManager::get_depth_vector() {
    VectorXd dep_vec(get_suitable_feature_count());
    unsigned int feature_index = 0;
    for (auto &it_per_id : features_map) {
        if (it_per_id.second.is_suitable_to_reprojection()) {
#if 1
            dep_vec(feature_index++) = 1. / it_per_id.second.estimated_depth;
#else
            dep_vec(++feature_index) = it_per_id->estimated_depth;
#endif
        }
    }
    return dep_vec;
}

void FeatureManager::triangulate(Vector3d p_imu[], Vector3d t_ic[], Matrix3d r_ic[]) {
    for (auto &it_per_id : features_map) {
        if (it_per_id.second.is_suitable_to_reprojection()) {
            if (it_per_id.second.estimated_depth > 0.)
                continue;

            unsigned int imu_i = it_per_id.second.start_frame_id;

            assert(NUM_OF_CAM == 1);
            Eigen::MatrixXd svd_A(2 * it_per_id.second.feature_per_frame.size(), 4);
            Eigen::Matrix<double, 3, 4> P;
            Eigen::Vector3d f;

            Eigen::Vector3d t_wci_w = p_imu[imu_i] + _r_wi[imu_i] * t_ic[0];
            Eigen::Matrix3d r_wci = _r_wi[imu_i] * r_ic[0];

            P.leftCols<3>() = Eigen::Matrix3d::Identity();
            P.rightCols<1>() = Eigen::Vector3d::Zero();

            f = it_per_id.second.feature_per_frame[0].point.normalized();
            svd_A.row(0) = f[0] * P.row(2) - f[2] * P.row(0);
            svd_A.row(1) = f[1] * P.row(2) - f[2] * P.row(1);

            for (unsigned int j = 1; j < it_per_id.second.feature_per_frame.size(); ++j) {
                unsigned int imu_j = imu_i + j;

                Eigen::Vector3d t_wcj_w = p_imu[imu_j] + _r_wi[imu_j] * t_ic[0];
                Eigen::Matrix3d r_wcj = _r_wi[imu_j] * r_ic[0];
                Eigen::Vector3d t_cicj_ci = r_wci.transpose() * (t_wcj_w - t_wci_w);
                Eigen::Matrix3d r_cicj = r_wci.transpose() * r_wcj;

                P.leftCols<3>() = r_cicj.transpose();
                P.rightCols<1>() = -r_cicj.transpose() * t_cicj_ci;

                f = it_per_id.second.feature_per_frame[j].point.normalized();
                svd_A.row(2 * j) = f[0] * P.row(2) - f[2] * P.row(0);
                svd_A.row(2 * j + 1) = f[1] * P.row(2) - f[2] * P.row(1);
            }

            Eigen::Vector4d svd_V = Eigen::JacobiSVD<Eigen::MatrixXd>(svd_A, Eigen::ComputeThinV).matrixV().rightCols<1>();
            double depth = svd_V[2] / svd_V[3];
            it_per_id.second.estimated_depth = depth;

            if (it_per_id.second.estimated_depth < 0.1) {
                it_per_id.second.estimated_depth = INIT_DEPTH;
            }
        }
    }
}

void FeatureManager::remove_outlier() {
    // ROS_BREAK();
    return;
//    int i = -1;
//    for (auto it = feature.begin(), it_next = feature.begin();
//         it != feature.end(); it = it_next)
//    {
//        it_next++;
//        i += it->get_used_num() != 0;
//        if (it->get_used_num() != 0 && it->is_outlier == true)
//        {
//            feature.erase(it);
//        }
//    }
}

/*!
 * 删除WINDOW中的oldest帧, 并重新计算feature的深度
 * @param marg_R
 * @param marg_P
 * @param new_R
 * @param new_P
 */
void FeatureManager::remove_back_shift_depth(const Eigen::Matrix3d &marg_R, const Eigen::Vector3d &marg_P, const Eigen::Matrix3d &new_R, const Eigen::Vector3d &new_P) {
#ifdef USE_OPENMP
    static vector<unsigned long> ids_erase[NUM_THREADS];
    for (unsigned int i = 0; i < NUM_THREADS; ++i) {
        ids_erase[i].clear();
    }
#pragma omp parallel for num_threads(NUM_THREADS)
    for (size_t n = 0; n < features_vector.size(); ++n) {
        unsigned int index = omp_get_thread_num();

        auto &&feature = features_vector[n];
        if (feature.second->start_frame_id != 0) {   // 由于所有帧都左移1个frame, 所以feature中的start_frame_id需要减1
            --feature.second->start_frame_id;
        } else {    // 对于start_frame_id=0的feature, 则需要把frame从feature中删除并且重新以新的start_frame计算深度
            Eigen::Vector3d uv_i = feature.second->feature_per_frame[0].point;
            feature.second->feature_per_frame.erase(feature.second->feature_per_frame.begin());   // 把frame从feature中删除
            if (feature.second->feature_per_frame.size() < 2) {    // 如果删除frame后, frame数小于2, 则该feature已无法计算重投影误差, 所以直接删除
                ids_erase[index].emplace_back(feature.first);
                continue;
            }
            // 以新的start_frame计算feature的深度
            Eigen::Vector3d p_feature_c = uv_i * feature.second->estimated_depth;
            Eigen::Vector3d p_feature_w = marg_R * p_feature_c + marg_P;
            Eigen::Vector3d p_feature_c_new = new_R.transpose() * (p_feature_w - new_P);
            double depth_new = p_feature_c_new[2];
            if (depth_new > 0) {
                feature.second->estimated_depth = depth_new;
            } else {
                feature.second->estimated_depth = INIT_DEPTH;
            }
        }
        // remove tracking-lost feature after marginalize
        /*
        if (it->endFrame() < WINDOW_SIZE - 1)
        {
            feature.erase(it);
        }
        */
    }

    // 删除frame小于2的feature
    for (auto &ids : ids_erase) {
        for (auto &id : ids) {
            features_map.erase(id);
        }
    }
#else
    feature_id_erase.clear();

    for (auto &feature : features_map) {
        if (feature.second.start_frame_id != 0) {   // 由于所有帧都左移1个frame, 所以feature中的start_frame_id需要减1
            --feature.second.start_frame_id;
        } else {    // 对于start_frame_id=0的feature, 则需要把frame从feature中删除并且重新以新的start_frame计算深度
            Eigen::Vector3d uv_i = feature.second.feature_per_frame[0].point;
            feature.second.feature_per_frame.erase(feature.second.feature_per_frame.begin());   // 把frame从feature中删除
            if (feature.second.feature_per_frame.size() < 2) {    // 如果删除frame后, frame数小于2, 则该feature已无法计算重投影误差, 所以直接删除
                feature_id_erase.emplace_back(feature.first);
                continue;
            }
            // 以新的start_frame计算feature的深度
            Eigen::Vector3d p_feature_c = uv_i * feature.second.estimated_depth;
            Eigen::Vector3d p_feature_w = marg_R * p_feature_c + marg_P;
            Eigen::Vector3d p_feature_c_new = new_R.transpose() * (p_feature_w - new_P);
            double depth_new = p_feature_c_new[2];
            if (depth_new > 0) {
                feature.second.estimated_depth = depth_new;
            } else {
                feature.second.estimated_depth = INIT_DEPTH;
            }
        }
        // remove tracking-lost feature after marginalize
        /*
        if (it->endFrame() < WINDOW_SIZE - 1)
        {
            feature.erase(it);
        }
        */
    }

    // 删除frame小于2的feature
    for (auto &id : feature_id_erase) {
        features_map.erase(id);
    }
#endif
}

/*!
 * 删除WINDOW中的oldest帧
 */
void FeatureManager::remove_back() {
#ifdef USE_OPENMP
    static vector<unsigned long> ids_erase[NUM_THREADS];
    for (unsigned int i = 0; i < NUM_THREADS; ++i) {
        ids_erase[i].clear();
    }

#pragma omp parallel for num_threads(NUM_THREADS)
    for (size_t n = 0; n < features_vector.size(); ++n) {
        unsigned int index = omp_get_thread_num();

        auto &&feature = features_vector[n];
        if (feature.second->start_frame_id != 0) {
            --feature.second->start_frame_id;
        }
        else {
            feature.second->feature_per_frame.erase(feature.second->feature_per_frame.begin());
            if (feature.second->feature_per_frame.empty()) {
                ids_erase[index].emplace_back(feature.first);
            }
        }
    }

    // 删除frame小于2的feature
    for (auto &ids : ids_erase) {
        for (auto &id : ids) {
            features_map.erase(id);
        }
    }
#else
    feature_id_erase.clear();

    for (auto &feature : features_map) {
        if (feature.second.start_frame_id != 0) {
            --feature.second.start_frame_id;
        }
        else {
            feature.second.feature_per_frame.erase(feature.second.feature_per_frame.begin());
            if (feature.second.feature_per_frame.empty()) {
                feature_id_erase.emplace_back(feature.first);
            }
        }
    }

    // 删除frame小于2的feature
    for (auto &id : feature_id_erase) {
        features_map.erase(id);
    }
#endif
}

/*!
 * 删除WINDOW中的newest帧
 * @param frame_count frame的总个数
 */
void FeatureManager::remove_front(unsigned int frame_count) {
    /*
     * WINDOW的大小为frame_count
     * Camera最新观测到的camera_frame的id为frame_count
     * */
#ifdef USE_OPENMP
    static vector<unsigned long> ids_erase[NUM_THREADS];
    for (unsigned int i = 0; i < NUM_THREADS; ++i) {
        ids_erase[i].clear();
    }

#pragma omp parallel for num_threads(NUM_THREADS)
    for (size_t n = 0; n < features_vector.size(); ++n) {
        unsigned int index = omp_get_thread_num();

        auto &&feature = features_vector[n];
        /*
         * 1. 若start_frame为camera_frame, 则start_frame需要减1, 因为WINDOW中的newest_frame会被删除, 其id为frame_count - 1,
         *    而frame_camera则会移东到WINDOW中的newest_frame中。
         *
         * 2. 若end_frame为newest_frame(id为frame_count-1), 则需要把该end_frame从feature中删除, 因为newest_frame会被删除
         * */
        if (feature.second->start_frame_id == frame_count) {
            --feature.second->start_frame_id;
        } else if (feature.second->get_end_frame_id() + 1 >= frame_count) {
            // feature.second.feature_local_infos.begin() + j 对应newest_frame
            unsigned long j = frame_count - 1 - feature.second->start_frame_id;
            feature.second->feature_per_frame.erase(feature.second->feature_per_frame.begin() + j);
            if (feature.second->feature_per_frame.empty()) {
                ids_erase[index].emplace_back(feature.first);
            }
        }
    }

    for (auto &ids : ids_erase) {
        for (auto &id : ids) {
            features_map.erase(id);
        }
    }
#else
    feature_id_erase.clear();

    for (auto &feature : features_map) {
        /*
         * 1. 若start_frame为camera_frame, 则start_frame需要减1, 因为WINDOW中的newest_frame会被删除, 其id为frame_count - 1,
         *    而frame_camera则会移东到WINDOW中的newest_frame中。
         *
         * 2. 若end_frame为newest_frame(id为frame_count-1), 则需要把该end_frame从feature中删除, 因为newest_frame会被删除
         * */
        if (feature.second.start_frame_id == frame_count) {
            --feature.second.start_frame_id;
        } else if (feature.second.get_end_frame_id() + 1 >= frame_count) {
            // feature.second.feature_local_infos.begin() + j 对应newest_frame
            unsigned long j = frame_count - 1 - feature.second.start_frame_id;
            feature.second.feature_per_frame.erase(feature.second.feature_per_frame.begin() + j);
            if (feature.second.feature_per_frame.empty()) {
                feature_id_erase.emplace_back(feature.first);
            }
        }
    }

    // 删除frame数为0的feature
    for (auto &id : feature_id_erase) {
        features_map.erase(id);
    }
#endif

}


double FeatureManager::compensated_parallax2(const FeaturePerId &it_per_id, unsigned int frame_count) {
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

#ifdef USE_OPENMP
void FeatureManager::update_features_vector() {
    features_vector.clear();
    for (auto &feature : features_map) {
        features_vector.emplace_back(feature.first, &feature.second);
    }
}
#endif