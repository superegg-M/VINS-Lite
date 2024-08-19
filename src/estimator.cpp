#include "estimator.h"

#include "backend/vertex_inverse_depth.h"
#include "backend/vertex_pose.h"
#include "backend/vertex_motion.h"
#include "backend/edge_reprojection.h"
#include "backend/edge_imu.h"

#include <ostream>
#include <fstream>


Estimator::Estimator() : f_manager{Rs}, _problem() {
    // ROS_INFO("init begins");

    for (size_t i = 0; i < WINDOW_SIZE + 1; i++) {
        pre_integrations[i] = nullptr;
    }
    for(auto &it: all_image_frame)
    {
        it.second.pre_integration = nullptr;
    }
    tmp_pre_integration = nullptr;
    
    clearState();
}

void Estimator::setParameter() {
    for (int i = 0; i < NUM_OF_CAM; i++) {
        tic[i] = TIC[i];
        ric[i] = RIC[i];
        // cout << "1 Estimator::setParameter tic: " << tic[i].transpose()
        //     << " _r_ic: " << _r_ic[i] << endl;
    }
    cout << "1 Estimator::setParameter FOCAL_LENGTH: " << FOCAL_LENGTH << endl;
    f_manager.set_r_ic(ric);
    project_sqrt_info_ = FOCAL_LENGTH / 1.5 * Matrix2d::Identity();
    td = TD;
}

void Estimator::clearState() {
    for (int i = 0; i < WINDOW_SIZE + 1; i++) {
        Rs[i].setIdentity();
        Ps[i].setZero();
        Vs[i].setZero();
        Bas[i].setZero();
        Bgs[i].setZero();
        dt_buf[i].clear();
        linear_acceleration_buf[i].clear();
        angular_velocity_buf[i].clear();

        if (pre_integrations[i] != nullptr)
            delete pre_integrations[i];
        pre_integrations[i] = nullptr;
    }

    for (int i = 0; i < NUM_OF_CAM; i++) {
        tic[i] = Vector3d::Zero();
        ric[i] = Matrix3d::Identity();
    }

    for (auto &it : all_image_frame) {
        if (it.second.pre_integration != nullptr) {
            delete it.second.pre_integration;
            it.second.pre_integration = nullptr;
        }
    }

    solver_flag = INITIAL;
    first_imu = false,
    sum_of_back = 0;
    sum_of_front = 0;
    frame_count = 0;
    solver_flag = INITIAL;
    initial_timestamp = 0;
    all_image_frame.clear();
    td = TD;

    if (tmp_pre_integration != nullptr)
        delete tmp_pre_integration;
    
    tmp_pre_integration = nullptr;
    
    last_marginalization_parameter_blocks.clear();

    f_manager.clear_state();

    failure_occur = false;
    relocalization_info = false;

    drift_correct_r = Matrix3d::Identity();
    drift_correct_t = Vector3d::Zero();
}

void Estimator::processIMU(double dt, const Vector3d &linear_acceleration, const Vector3d &angular_velocity) {
    if (!first_imu) {
        first_imu = true;
        acc_0 = linear_acceleration;
        gyr_0 = angular_velocity;
    }

    if (!pre_integrations[frame_count]) {
#ifdef CAIN_IMU_INTEGRATION
    pre_integrations[frame_count] = new vins::IMUIntegration{acc_0, gyr_0, Bas[frame_count], Bgs[frame_count]};
#else
    pre_integrations[frame_count] = new IntegrationBase{acc_0, gyr_0, Bas[frame_count], Bgs[frame_count]};
#endif
    }
    if (frame_count != 0) {
        pre_integrations[frame_count]->push_back(dt, linear_acceleration, angular_velocity);
        //if(solver_flag != NON_LINEAR)
        tmp_pre_integration->push_back(dt, linear_acceleration, angular_velocity);

        dt_buf[frame_count].push_back(dt);
        linear_acceleration_buf[frame_count].push_back(linear_acceleration);
        angular_velocity_buf[frame_count].push_back(angular_velocity);

        // _r_wi[j], Ps[j], Vs[j]为地球坐标系下的旋转矩阵，位移，速度
        int j = frame_count;
        Vector3d un_acc_0 = Rs[j] * (acc_0 - Bas[j]) - g;   
        Vector3d un_gyr = 0.5 * (gyr_0 + angular_velocity) - Bgs[j];
        Rs[j] *= Utility::deltaQ(un_gyr * dt).toRotationMatrix();
        Vector3d un_acc_1 = Rs[j] * (linear_acceleration - Bas[j]) - g;
        Vector3d un_acc = 0.5 * (un_acc_0 + un_acc_1);
        Ps[j] += dt * Vs[j] + 0.5 * dt * dt * un_acc;
        Vs[j] += dt * un_acc;
    }
    acc_0 = linear_acceleration;
    gyr_0 = angular_velocity;
}

// TODO: 在运行processImage时向_problem加入vertex和edge, 而不在problemSolve中加
void Estimator::processImage(const map<int, vector<pair<int, Eigen::Matrix<double, 7, 1>>>> &image, double header) {
    //ROS_DEBUG("new image coming ------------------------------------------");
    // cout << "Adding feature points: " << image.size()<<endl;
    // 若视差大于阈值, 则认为image为关键帧。但无论是否为key frame, 都会把这次观测到的frame关联到landmark上
    if (f_manager.add_feature_and_check_latest_frame_parallax(frame_count, image, td))
        marginalization_flag = MARGIN_OLD;
    else
        marginalization_flag = MARGIN_SECOND_NEW;

    //ROS_DEBUG("this frame is--------------------%s", marginalization_flag ? "reject" : "accept");
    //ROS_DEBUG("%s", marginalization_flag ? "Non-keyframe" : "Keyframe");
    //ROS_DEBUG("Solving %d", frame_count);
    // cout << "number of feature: " << f_manager.get_suitable_feature_count()<<endl;
    Headers[frame_count] = header;

    ImageFrame imageframe(image, header);
    imageframe.pre_integration = tmp_pre_integration;
    all_image_frame.insert(make_pair(header, imageframe));

// #ifdef PRINT_INFO    
//     std::cout << "numbers of all_image_frame: " << all_image_frame.size() << std::endl;
// #endif

#ifdef CAIN_IMU_INTEGRATION
    tmp_pre_integration = new vins::IMUIntegration{acc_0, gyr_0, Bas[frame_count], Bgs[frame_count]};
#else
    tmp_pre_integration = new IntegrationBase{acc_0, gyr_0, Bas[frame_count], Bgs[frame_count]};
#endif    

    if (ESTIMATE_EXTRINSIC == 2) {
        cout << "calibrating extrinsic param, rotation movement is needed" << endl;
        if (frame_count != 0) {
            vector<pair<Vector3d, Vector3d>> corres = f_manager.get_corresponding(frame_count - 1, frame_count);
            Matrix3d calib_ric;
            if (initial_ex_rotation.CalibrationExRotation(corres, pre_integrations[frame_count]->get_delta_q(), calib_ric)) {
                // ROS_WARN("initial extrinsic rotation calib success");
                // ROS_WARN_STREAM("initial extrinsic rotation: " << endl
                                                            //    << calib_ric);
                ric[0] = calib_ric;
                RIC[0] = calib_ric;
                ESTIMATE_EXTRINSIC = 1;
            }
        }
    }

#ifdef USE_OPENMP
    f_manager.update_features_vector();
#endif

    if (solver_flag == INITIAL) {
        if (frame_count == WINDOW_SIZE) {
            bool result = false;
            if (ESTIMATE_EXTRINSIC != 2 && (header - initial_timestamp) > 0.1) {
                // cout << "1 initialStructure" << endl;
                result = initialStructure();
                initial_timestamp = header;
            }
            if (result) {
                solver_flag = NON_LINEAR;
                // solveOdometry();
                slideWindow();
                f_manager.remove_failures();
                cout << "Initialization finish!" << endl;
                last_R = Rs[WINDOW_SIZE];
                last_P = Ps[WINDOW_SIZE];
                last_R0 = Rs[0];
                last_P0 = Ps[0];
            }
            else
                slideWindow();
        }
        else
            frame_count++;
    } else {
        TicToc t_solve;
        solveOdometry();
        //ROS_DEBUG("solver costs: %fms", t_solve.toc());

        if (failureDetection()) {
            // ROS_WARN("failure detection!");
            failure_occur = true;
            clearState();
            setParameter();
            // ROS_WARN("system reboot!");
            return;
        }

        TicToc t_margin;
        slideWindow();
        f_manager.remove_failures();
        //ROS_DEBUG("marginalization costs: %fms", t_margin.toc());
        // prepare output of VINS
        key_poses.clear();
        for (int i = 0; i <= WINDOW_SIZE; i++)
            key_poses.push_back(Ps[i]);

        last_R = Rs[WINDOW_SIZE];
        last_P = Ps[WINDOW_SIZE];
        last_R0 = Rs[0];
        last_P0 = Ps[0];
    }
}
bool Estimator::initialStructure() {
    TicToc t_sfm;
    //check imu observibility
    {
        map<double, ImageFrame>::iterator frame_it;
        Vector3d sum_g;
        for (frame_it = all_image_frame.begin(), frame_it++; frame_it != all_image_frame.end(); frame_it++) {
            double dt = frame_it->second.pre_integration->get_sum_dt();
            Vector3d tmp_g = frame_it->second.pre_integration->get_delta_v() / dt;
            sum_g += tmp_g;
        }
        Vector3d aver_g;
        aver_g = sum_g * 1.0 / ((int)all_image_frame.size() - 1);
        double var = 0;
        for (frame_it = all_image_frame.begin(), frame_it++; frame_it != all_image_frame.end(); frame_it++) {
            double dt = frame_it->second.pre_integration->get_sum_dt();
            Vector3d tmp_g = frame_it->second.pre_integration->get_delta_v() / dt;
            var += (tmp_g - aver_g).transpose() * (tmp_g - aver_g);
            //cout << "frame g " << tmp_g.transpose() << endl;
        }
        var = sqrt(var / ((int)all_image_frame.size() - 1));
        //ROS_WARN("IMU variation %f!", var);
        if (var < 0.25) {
            // ROS_INFO("IMU excitation not enouth!");
            //return false;
        }
    }
    // global sfm
    Quaterniond Q[frame_count + 1];
    Vector3d T[frame_count + 1];
    map<int, Vector3d> sfm_tracked_points;
    vector<SFMFeature> sfm_f;
    for (auto &it_per_id : f_manager.features_map) {
        unsigned int imu_j = it_per_id.second.start_frame_id;
        SFMFeature tmp_feature;
        tmp_feature.state = false;
        tmp_feature.id = it_per_id.second.feature_id;
        for (auto &it_per_frame : it_per_id.second.feature_per_frame) {
            Vector3d pts_j = it_per_frame.point;
            tmp_feature.observation.emplace_back(imu_j++, Eigen::Vector2d{pts_j.x(), pts_j.y()});
        }
        sfm_f.push_back(tmp_feature);
    }
    Matrix3d relative_R;
    Vector3d relative_T;
    int l;
    if (!relativePose(relative_R, relative_T, l)) {
        cout << "Not enough features or parallax; Move device around" << endl;
        return false;
    }
    GlobalSFM sfm;
    if (!sfm.construct(frame_count + 1, Q, T, l,
                       relative_R, relative_T,
                       sfm_f, sfm_tracked_points)) {
        cout << "global SFM failed!" << endl;
        marginalization_flag = MARGIN_OLD;
        return false;
    }

    //solve pnp for all frame
    map<double, ImageFrame>::iterator frame_it;
    map<int, Vector3d>::iterator it;
    frame_it = all_image_frame.begin();
    for (int i = 0; frame_it != all_image_frame.end(); frame_it++) {
        // provide initial guess
        cv::Mat r, rvec, t, D, tmp_r;
        if ((frame_it->first) == Headers[i]) {
            frame_it->second.is_key_frame = true;
            frame_it->second.R = Q[i].toRotationMatrix() * RIC[0].transpose();
            frame_it->second.T = T[i];
            i++;
            continue;
        }
        if ((frame_it->first) > Headers[i]) {
            i++;
        }
        Matrix3d R_inital = (Q[i].inverse()).toRotationMatrix();
        Vector3d P_inital = -R_inital * T[i];
        cv::eigen2cv(R_inital, tmp_r);
        cv::Rodrigues(tmp_r, rvec);
        cv::eigen2cv(P_inital, t);

        frame_it->second.is_key_frame = false;
        vector<cv::Point3f> pts_3_vector;
        vector<cv::Point2f> pts_2_vector;
        for (auto &id_pts : frame_it->second.points) {
            int feature_id = id_pts.first;
            for (auto &i_p : id_pts.second) {
                it = sfm_tracked_points.find(feature_id);
                if (it != sfm_tracked_points.end()) {
                    Vector3d world_pts = it->second;
                    cv::Point3f pts_3(world_pts(0), world_pts(1), world_pts(2));
                    pts_3_vector.push_back(pts_3);
                    Vector2d img_pts = i_p.second.head<2>();
                    cv::Point2f pts_2(img_pts(0), img_pts(1));
                    pts_2_vector.push_back(pts_2);
                }
            }
        }
        cv::Mat K = (cv::Mat_<double>(3, 3) << 1, 0, 0, 0, 1, 0, 0, 0, 1);
        if (pts_3_vector.size() < 6) {
            cout << "Not enough points for solve pnp pts_3_vector size " << pts_3_vector.size() << endl;
            return false;
        }
        if (!cv::solvePnP(pts_3_vector, pts_2_vector, K, D, rvec, t, 1)) {
            cout << " solve pnp fail!" << endl;
            return false;
        }
        cv::Rodrigues(rvec, r);
        MatrixXd R_pnp, tmp_R_pnp;
        cv::cv2eigen(r, tmp_R_pnp);
        R_pnp = tmp_R_pnp.transpose();
        MatrixXd T_pnp;
        cv::cv2eigen(t, T_pnp);
        T_pnp = R_pnp * (-T_pnp);
        frame_it->second.R = R_pnp * RIC[0].transpose();
        frame_it->second.T = T_pnp;
    }
    if (visualInitialAlign()) {
        return true;
    } else {
        cout << "misalign visual structure with IMU" << endl;
        return false;
    }
}

bool Estimator::visualInitialAlign() {
    TicToc t_g;
    VectorXd x;
    //solve scale

    bool result = VisualIMUAlignment(all_image_frame, Bgs, g, x);
    if (!result) {
        //ROS_DEBUG("solve g failed!");
        return false;
    }

    // change state
    for (int i = 0; i <= frame_count; i++) {
        Matrix3d Ri = all_image_frame[Headers[i]].R;
        Vector3d Pi = all_image_frame[Headers[i]].T;
        Ps[i] = Pi;
        Rs[i] = Ri;
        all_image_frame[Headers[i]].is_key_frame = true;
    }

    VectorXd dep = f_manager.get_depth_vector();
    for (int i = 0; i < dep.size(); i++)
        dep[i] = -1;
    f_manager.clear_depth(dep);

    //triangulat on cam pose , no tic
    Vector3d TIC_TMP[NUM_OF_CAM];
    for (int i = 0; i < NUM_OF_CAM; i++)
        TIC_TMP[i].setZero();
    ric[0] = RIC[0];
    f_manager.set_r_ic(ric);
    f_manager.triangulate(Ps, &(TIC_TMP[0]), &(RIC[0]));

    double s = (x.tail<1>())(0);
    for (int i = 0; i <= WINDOW_SIZE; i++) {
        pre_integrations[i]->repropagate(Vector3d::Zero(), Bgs[i]);
    }
    for (int i = frame_count; i >= 0; i--)
        Ps[i] = s * Ps[i] - Rs[i] * TIC[0] - (s * Ps[0] - Rs[0] * TIC[0]);
    int kv = -1;
    map<double, ImageFrame>::iterator frame_i;
    for (frame_i = all_image_frame.begin(); frame_i != all_image_frame.end(); frame_i++) {
        if (frame_i->second.is_key_frame) {
            kv++;
            Vs[kv] = frame_i->second.R * x.segment<3>(kv * 3);
        }
    }
    for (auto &it_per_id : f_manager.features_map) {
        if (it_per_id.second.is_suitable_to_reprojection()) {
            it_per_id.second.estimated_depth *= s;
        }
    }

    Matrix3d R0 = Utility::g2R(g);
    double yaw = Utility::R2ypr(R0 * Rs[0]).x();
    R0 = Utility::ypr2R(Eigen::Vector3d{-yaw, 0, 0}) * R0;
    g = R0 * g;
    //Matrix3d rot_diff = R0 * _r_wi[0].transpose();
    Matrix3d rot_diff = R0;
    for (int i = 0; i <= frame_count; i++) {
        Ps[i] = rot_diff * Ps[i];
        Rs[i] = rot_diff * Rs[i];
        Vs[i] = rot_diff * Vs[i];
    }
    //ROS_DEBUG_STREAM("g0     " << g.transpose());
    //ROS_DEBUG_STREAM("my R0  " << Utility::R2ypr(_r_wi[0]).transpose());

    return true;
}

bool Estimator::relativePose(Matrix3d &relative_R, Vector3d &relative_T, int &l) {
    // find previous frame which contians enough correspondance and parallex with newest frame
    for (int i = 0; i < WINDOW_SIZE; i++) {
        vector<pair<Vector3d, Vector3d>> corres;
        corres = f_manager.get_corresponding(i, WINDOW_SIZE);
        if (corres.size() > 20) {
            double sum_parallax = 0;
            double average_parallax;
            for (int j = 0; j < int(corres.size()); j++) {
                Vector2d pts_0(corres[j].first(0), corres[j].first(1));
                Vector2d pts_1(corres[j].second(0), corres[j].second(1));
                double parallax = (pts_0 - pts_1).norm();
                sum_parallax = sum_parallax + parallax;
            }
            average_parallax = 1.0 * sum_parallax / int(corres.size());
            if (average_parallax * 460 > 30 && m_estimator.solveRelativeRT(corres, relative_R, relative_T)) {
                l = i;
                //ROS_DEBUG("average_parallax %f choose l %d and newest frame to triangulate the whole structure", average_parallax * 460, l);
                return true;
            }
        }
    }
    return false;
}

void Estimator::solveOdometry() {
    if (frame_count < WINDOW_SIZE)
        return;
    if (solver_flag == NON_LINEAR) {
        TicToc t_tri;
        f_manager.triangulate(Ps, tic, ric);
        //cout << "triangulation costs : " << t_tri.toc() << endl;        
        backendOptimization();
    }
}

void Estimator::vector2double() {
    for (int i = 0; i <= WINDOW_SIZE; i++) {
        para_Pose[i][0] = Ps[i].x();
        para_Pose[i][1] = Ps[i].y();
        para_Pose[i][2] = Ps[i].z();
        Quaterniond q{Rs[i]};
        para_Pose[i][3] = q.x();
        para_Pose[i][4] = q.y();
        para_Pose[i][5] = q.z();
        para_Pose[i][6] = q.w();

        para_SpeedBias[i][0] = Vs[i].x();
        para_SpeedBias[i][1] = Vs[i].y();
        para_SpeedBias[i][2] = Vs[i].z();

        para_SpeedBias[i][3] = Bas[i].x();
        para_SpeedBias[i][4] = Bas[i].y();
        para_SpeedBias[i][5] = Bas[i].z();

        para_SpeedBias[i][6] = Bgs[i].x();
        para_SpeedBias[i][7] = Bgs[i].y();
        para_SpeedBias[i][8] = Bgs[i].z();
    }
    for (int i = 0; i < NUM_OF_CAM; i++) {
        para_Ex_Pose[i][0] = tic[i].x();
        para_Ex_Pose[i][1] = tic[i].y();
        para_Ex_Pose[i][2] = tic[i].z();
        Quaterniond q{ric[i]};
        para_Ex_Pose[i][3] = q.x();
        para_Ex_Pose[i][4] = q.y();
        para_Ex_Pose[i][5] = q.z();
        para_Ex_Pose[i][6] = q.w();
    }

    VectorXd dep = f_manager.get_depth_vector();
    for (int i = 0; i < f_manager.get_suitable_feature_count(); i++)
        para_Feature[i][0] = dep(i);
    if (ESTIMATE_TD)
        para_Td[0][0] = td;
}

void Estimator::double2vector()
{
    Vector3d origin_R0 = Utility::R2ypr(Rs[0]);
    Vector3d origin_P0 = Ps[0];

    if (failure_occur) {
        origin_R0 = Utility::R2ypr(last_R0);
        origin_P0 = last_P0;
        failure_occur = false;
    }
    Vector3d origin_R00 = Utility::R2ypr(Quaterniond(para_Pose[0][6],
                                                     para_Pose[0][3],
                                                     para_Pose[0][4],
                                                     para_Pose[0][5])
                                             .toRotationMatrix());
    double y_diff = origin_R0.x() - origin_R00.x();
    //TODO
    Matrix3d rot_diff = Utility::ypr2R(Vector3d(y_diff, 0, 0));
    if (abs(abs(origin_R0.y()) - 90) < 1.0 || abs(abs(origin_R00.y()) - 90) < 1.0) {
        //ROS_DEBUG("euler singular point!");
        rot_diff = Rs[0] * Quaterniond(para_Pose[0][6],
                                       para_Pose[0][3],
                                       para_Pose[0][4],
                                       para_Pose[0][5])
                               .toRotationMatrix()
                               .transpose();
    }

    for (int i = 0; i <= WINDOW_SIZE; i++) {

        Rs[i] = rot_diff * Quaterniond(para_Pose[i][6], para_Pose[i][3], para_Pose[i][4], para_Pose[i][5]).normalized().toRotationMatrix();

        Ps[i] = rot_diff * Vector3d(para_Pose[i][0] - para_Pose[0][0],
                                    para_Pose[i][1] - para_Pose[0][1],
                                    para_Pose[i][2] - para_Pose[0][2]) +
                origin_P0;

        Vs[i] = rot_diff * Vector3d(para_SpeedBias[i][0],
                                    para_SpeedBias[i][1],
                                    para_SpeedBias[i][2]);

        Bas[i] = Vector3d(para_SpeedBias[i][3],
                          para_SpeedBias[i][4],
                          para_SpeedBias[i][5]);

        Bgs[i] = Vector3d(para_SpeedBias[i][6],
                          para_SpeedBias[i][7],
                          para_SpeedBias[i][8]);
    }

    for (int i = 0; i < NUM_OF_CAM; i++) {
        tic[i] = Vector3d(para_Ex_Pose[i][0],
                          para_Ex_Pose[i][1],
                          para_Ex_Pose[i][2]);
        ric[i] = Quaterniond(para_Ex_Pose[i][6],
                             para_Ex_Pose[i][3],
                             para_Ex_Pose[i][4],
                             para_Ex_Pose[i][5])
                     .toRotationMatrix();
    }

    VectorXd dep = f_manager.get_depth_vector();
    for (int i = 0; i < f_manager.get_suitable_feature_count(); i++)
        dep(i) = para_Feature[i][0];
    f_manager.set_depth(dep);
    if (ESTIMATE_TD)
        td = para_Td[0][0];

    // relative info between two loop frame
    if (relocalization_info) {
        Matrix3d relo_r;
        Vector3d relo_t;
        relo_r = rot_diff * Quaterniond(relo_Pose[6], relo_Pose[3], relo_Pose[4], relo_Pose[5]).normalized().toRotationMatrix();
        relo_t = rot_diff * Vector3d(relo_Pose[0] - para_Pose[0][0],
                                     relo_Pose[1] - para_Pose[0][1],
                                     relo_Pose[2] - para_Pose[0][2]) +
                 origin_P0;
        double drift_correct_yaw;
        drift_correct_yaw = Utility::R2ypr(prev_relo_r).x() - Utility::R2ypr(relo_r).x();
        drift_correct_r = Utility::ypr2R(Vector3d(drift_correct_yaw, 0, 0));
        drift_correct_t = prev_relo_t - drift_correct_r * relo_t;
        relo_relative_t = relo_r.transpose() * (Ps[relo_frame_local_index] - relo_t);
        relo_relative_q = relo_r.transpose() * Rs[relo_frame_local_index];
        relo_relative_yaw = Utility::normalizeAngle(Utility::R2ypr(Rs[relo_frame_local_index]).x() - Utility::R2ypr(relo_r).x());
        //cout << "vins relo " << endl;
        //cout << "vins relative_t " << relo_relative_t.transpose() << endl;
        //cout << "vins relative_yaw " <<relo_relative_yaw << endl;
        relocalization_info = 0;
    }
}

bool Estimator::failureDetection() {
    if (f_manager.last_track_num < 2) {
        //ROS_INFO(" little feature %d", f_manager.last_track_num);
        //return true;
    }
    if (Bas[WINDOW_SIZE].norm() > 2.5) {
        //ROS_INFO(" big IMU acc bias estimation %f", Bas[WINDOW_SIZE].norm());
        return true;
    }
    if (Bgs[WINDOW_SIZE].norm() > 1.0) {
        //ROS_INFO(" big IMU gyr bias estimation %f", Bgs[WINDOW_SIZE].norm());
        return true;
    }
    /*
    if (tic(0) > 1)
    {
        //ROS_INFO(" big extri param estimation %d", tic(0) > 1);
        return true;
    }
    */
    Vector3d tmp_P = Ps[WINDOW_SIZE];
    if ((tmp_P - last_P).norm() > 5) {
        //ROS_INFO(" big translation");
        return true;
    }
    if (abs(tmp_P.z() - last_P.z()) > 1) {
        //ROS_INFO(" big z translation");
        return true;
    }
    Matrix3d tmp_R = Rs[WINDOW_SIZE];
    Matrix3d delta_R = tmp_R.transpose() * last_R;
    Quaterniond delta_Q(delta_R);
    double delta_angle;
    delta_angle = acos(delta_Q.w()) * 2.0 / 3.14 * 180.0;
    if (delta_angle > 50) {
        //ROS_INFO(" big delta_angle ");
        //return true;
    }
    return false;
}

void Estimator::MargOldFrame() {
    // backend::LossFunction *lossfunction;
    // lossfunction = new backend::CauchyLoss(1.0);

    auto &problem = _problem;
    auto &vertexCams_vec = _vertex_pose_vec;
    auto &vertexVB_vec = _vertex_motion_vec;
//    auto &pose_dim = _state_dim;

    problem.marginalize(vertexCams_vec[0], vertexVB_vec[0]);
    // problem.marginalize(vertexCams_vec[0], vertexVB_vec[0], _marg_landmarks, _marg_edges);
    // problem.marginalize(vertexCams_vec[0], vertexVB_vec[0], _marg_landmarks, _marg_edges, true);
    Hprior_ = problem.get_h_prior();
    bprior_ = problem.get_b_prior();
    // errprior_ = problem.get_err_prior();
    // Jprior_inv_ = problem.get_Jt_prior();
}
void Estimator::MargNewFrame() {
    auto &problem = _problem;
    auto &vertexCams_vec = _vertex_pose_vec;
    auto &vertexVB_vec = _vertex_motion_vec;
//    auto &pose_dim = _state_dim;

    problem.marginalize(vertexCams_vec[WINDOW_SIZE - 1], vertexVB_vec[WINDOW_SIZE - 1]);
    // problem.marginalize(vertexCams_vec[WINDOW_SIZE - 1], vertexVB_vec[WINDOW_SIZE - 1], _marg_landmarks, _marg_edges);
    // problem.marginalize(vertexCams_vec[WINDOW_SIZE - 1], vertexVB_vec[WINDOW_SIZE - 1], _marg_landmarks, _marg_edges, false);
    Hprior_ = problem.get_h_prior();
    bprior_ = problem.get_b_prior();
    // errprior_ = problem.get_err_prior();
    // Jprior_inv_ = problem.get_Jt_prior();
}
void Estimator::problemSolve() {
    TicToc t_new_problem;

    // _problem = graph_optimization::ProblemSLAM();
    _problem.clear();
    _vertex_pose_vec.clear();
    _vertex_motion_vec.clear();
    _state_dim = 0;
    auto &problem = _problem;
    auto &vertexCams_vec = _vertex_pose_vec;
    auto &vertexVB_vec = _vertex_motion_vec;
    auto &pose_dim = _state_dim;

    _marg_landmarks.clear();
    _marg_edges.clear();

    // 先把 外参数 节点加入图优化，这个节点在以后一直会被用到，所以我们把他放在第一个
    shared_ptr<graph_optimization::VertexPose> vertexExt(new graph_optimization::VertexPose());
    {
        Eigen::VectorXd pose(7);
        pose << para_Ex_Pose[0][0], para_Ex_Pose[0][1], para_Ex_Pose[0][2], para_Ex_Pose[0][3], para_Ex_Pose[0][4], para_Ex_Pose[0][5], para_Ex_Pose[0][6];
        vertexExt->set_parameters(pose);

        if (!ESTIMATE_EXTRINSIC) {
            //ROS_DEBUG("fix extinsic param");
            // TODO:: set Hessian prior to zero
            vertexExt->set_fixed();
        } else {
            //ROS_DEBUG("estimate extinsic param");
        }
        problem.add_state_vertex(vertexExt);
        pose_dim += vertexExt->local_dimension();
    }

    // 相机的顶点(pose和motion), WINDOW_SIZE + 1 个
    for (int i = 0; i < WINDOW_SIZE + 1; i++) {      
        shared_ptr<graph_optimization::VertexPose> vertexCam(new graph_optimization::VertexPose());
        Eigen::VectorXd pose(7);
        pose << para_Pose[i][0], para_Pose[i][1], para_Pose[i][2], para_Pose[i][3], para_Pose[i][4], para_Pose[i][5], para_Pose[i][6];
        vertexCam->set_parameters(pose);
        vertexCams_vec.emplace_back(vertexCam);
        problem.add_state_vertex(vertexCam);
        pose_dim += vertexCam->local_dimension();

        shared_ptr<graph_optimization::VertexMotion> vertexVB(new graph_optimization::VertexMotion());
        Eigen::VectorXd vb(9);
        vb << para_SpeedBias[i][0], para_SpeedBias[i][1], para_SpeedBias[i][2],
            para_SpeedBias[i][3], para_SpeedBias[i][4], para_SpeedBias[i][5],
            para_SpeedBias[i][6], para_SpeedBias[i][7], para_SpeedBias[i][8];
        vertexVB->set_parameters(vb);
        vertexVB_vec.emplace_back(vertexVB);
        problem.add_state_vertex(vertexVB);
        pose_dim += vertexVB->local_dimension();
    }

    // 记录被marg的imu
    int imu_marg = (marginalization_flag == MARGIN_OLD) ? 0 : WINDOW_SIZE - 1; 

    // 边: IMU预积分误差, WINDOW_SIZE个
    for (int i = 0; i < WINDOW_SIZE; i++) {
        int j = i + 1;
        if (pre_integrations[j]->get_sum_dt() > 10.0)     // 间隔太长的不考虑
            continue;

        std::shared_ptr<graph_optimization::EdgeImu> imu_edge(new graph_optimization::EdgeImu(pre_integrations[j]));
        std::vector<std::shared_ptr<graph_optimization::Vertex>> edge_vertex;
        edge_vertex.push_back(vertexCams_vec[i]);
        edge_vertex.push_back(vertexVB_vec[i]);
        edge_vertex.push_back(vertexCams_vec[j]);
        edge_vertex.push_back(vertexVB_vec[j]);
        imu_edge->set_vertices(edge_vertex);
        // problem.add_edge(imu_edge);
        problem.add_imu_edge(imu_edge);

        if (i == imu_marg || j == imu_marg) {
            _marg_edges.emplace_back(imu_edge);
        }
    }

    // 边: 重投影误差
    // 重投影误差的边所包含的顶点是会发生变化的, 随着old key frame被marginalize和new key frame被加入WINDOWS中
    vector<shared_ptr<graph_optimization::VertexInverseDepth>> vertexPt_vec;
    MatXX information = project_sqrt_info_.transpose() * project_sqrt_info_;
    {
        unsigned int feature_index = 0;
        // 遍历每一个特征
#ifdef USE_OPENMP
        static vector<shared_ptr<graph_optimization::VertexInverseDepth>> vertex_vec[NUM_THREADS];
        static vector<shared_ptr<graph_optimization::EdgeReprojection>> edge_vec[NUM_THREADS];
        static vector<shared_ptr<graph_optimization::Vertex>> marg_landmarks_vec[NUM_THREADS];
        static vector<shared_ptr<graph_optimization::Edge>> marg_edges_vec[NUM_THREADS];
        for (auto &vertices : vertex_vec) {
            vertices.clear();
        }
        for (auto &edges : edge_vec) {
            edges.clear();
        }
        for (auto &landmarks : marg_landmarks_vec) {
            landmarks.clear();
        }
        for (auto &edges : marg_edges_vec) {
            edges.clear();
        }
#pragma omp parallel for num_threads(NUM_THREADS)
        for (size_t n = 0; n < f_manager.suitable_features.size(); ++n) {
            // 由于第WINDOW_SIZE帧是new frame, 其是否为key frame是不定的, 而第WINDOW_SIZE - 1帧是否为key frame也是不定的
            // 第WINDOW_SIZE - 2帧才必定是key frame, 而landmark至少需被2个key frame观测到
            // 所以landmark的start frame必须小于WINDOW_SIZE - 2才有意义
            unsigned int i = omp_get_thread_num();
            auto &&feature_pt = f_manager.suitable_features[n];
            // if (feature_pt->is_suitable_to_reprojection()) {
            Vec1 inv_d(1. / feature_pt->estimated_depth);
            feature_pt->vertex_landmark->set_parameters(inv_d);
            vertex_vec[i].emplace_back(feature_pt->vertex_landmark);

            // 遍历所有的观测 (landmark所关联的frame), 计算视觉重投影误差
            unsigned int imu_i = feature_pt->start_frame_id;
            const Vector3d &pts_i = feature_pt->feature_per_frame[0].point;
            for (unsigned int index = 1; index < feature_pt->feature_per_frame.size(); ++index) {
                unsigned int imu_j = imu_i + index;
                const Vector3d &pts_j = feature_pt->feature_per_frame[index].point;

                std::shared_ptr<graph_optimization::EdgeReprojection> edge(new graph_optimization::EdgeReprojection(pts_i, pts_j));
                std::vector<std::shared_ptr<graph_optimization::Vertex>> edge_vertex;
                edge->add_vertex(feature_pt->vertex_landmark);
                edge->add_vertex(vertexCams_vec[imu_i]);
                edge->add_vertex(vertexCams_vec[imu_j]);
                edge->add_vertex(vertexExt);

                edge->set_information(information);
                // edge->set_loss_function(lossfunction);

                edge_vec[i].emplace_back(edge);
                if (imu_j == imu_marg) {
                    marg_edges_vec[i].emplace_back(edge);
                    marg_landmarks_vec[i].emplace_back(feature_pt->vertex_landmark);
                }
                if (imu_i == imu_marg) {
                    marg_edges_vec[i].emplace_back(edge);
                }
            }
            if (imu_i == imu_marg) {
                marg_landmarks_vec[i].emplace_back(feature_pt->vertex_landmark);
            }
            // }
        }
        // std::cout << "imu_marg = " << imu_marg << std::endl;

        for (auto &vertices : vertex_vec) {
            for (auto &vertex : vertices) {
                problem.add_landmark_vertex(vertex);
            }
        }
        for (auto &edges : edge_vec) {
            for (auto &edge : edges) {
                // problem.add_edge(edge);
                problem.add_reproj_edge(edge);
            }
        }
        for (auto &landmarks : marg_landmarks_vec) {
            for (auto &landmark : landmarks) {
                _marg_landmarks.emplace_back(landmark);
            }
        }
        for (auto &edges : marg_edges_vec) {
            for (auto &edge : edges) {
                _marg_edges.emplace_back(edge);
            }
        }
#else
        for (auto &it_per_id : f_manager.features_map) { // 遍历每个landmark
            // 由于第WINDOW_SIZE帧是new frame, 其是否为key frame是不定的, 而第WINDOW_SIZE - 1帧是否为key frame也是不定的
            // 第WINDOW_SIZE - 2帧才必定是key frame, 而landmark至少需被2个key frame观测到
            // 所以landmark的start frame必须小于WINDOW_SIZE - 2才有意义
            if (it_per_id.second.is_suitable_to_reprojection()) {
                // shared_ptr<graph_optimization::VertexInverseDepth> vertexPoint(new graph_optimization::VertexInverseDepth());
                Vec1 inv_d(para_Feature[feature_index++][0]);
                it_per_id.second.vertex_landmark->set_parameters(inv_d);
                problem.add_landmark_vertex(it_per_id.second.vertex_landmark);
                vertexPt_vec.push_back(it_per_id.second.vertex_landmark);

                // 遍历所有的观测 (landmark所关联的frame), 计算视觉重投影误差
                unsigned int imu_i = it_per_id.second.start_frame_id;
                const Vector3d &pts_i = it_per_id.second.feature_per_frame[0].point;
                for (unsigned int index = 1; index < it_per_id.second.feature_per_frame.size(); ++index) {
                    unsigned int imu_j = imu_i + index;
                    const Vector3d &pts_j = it_per_id.second.feature_per_frame[index].point;

                    std::shared_ptr<graph_optimization::EdgeReprojection> edge(new graph_optimization::EdgeReprojection(pts_i, pts_j));
                    std::vector<std::shared_ptr<graph_optimization::Vertex>> edge_vertex;
                    edge_vertex.push_back(it_per_id.second.vertex_landmark);
                    edge_vertex.push_back(vertexCams_vec[imu_i]);
                    edge_vertex.push_back(vertexCams_vec[imu_j]);
                    edge_vertex.push_back(vertexExt);

                    edge->set_vertices(edge_vertex);
                    edge->set_information(information);

                    // edge->set_loss_function(lossfunction);
                    problem.add_reproj_edge(edge);

                    if (imu_j == imu_marg) {
                        _marg_edges.emplace_back(edge);
                        _marg_landmarks.emplace_back(it_per_id.second.vertex_landmark);
                    }
                    if (imu_i == imu_marg) {
                        _marg_edges.emplace_back(edge);
                    }
                }
                if (imu_i == imu_marg) {
                    _marg_landmarks.emplace_back(it_per_id.second.vertex_landmark);
                }
            }
        }
#endif
    }

    // 先验
    {
        // 已经有 Prior 了
        if (Hprior_.rows() > 0) {
            // 外参数先验设置为 0. TODO:: 这个应该放到 solver 里去弄
            //            Hprior_.block(0,0,6,Hprior_.cols()).setZero();
            //            Hprior_.block(0,0,Hprior_.rows(),6).setZero();

//            problem.set_h_prior(Hprior_); // 告诉这个 problem
//            problem.set_b_prior(bprior_);
//            // problem.set_err_prior(errprior_);
//            // problem.set_Jt_prior(Jprior_inv_);
//            problem.extend_prior_hessian_size(15); // 但是这个 prior 还是之前的维度，需要扩展下装新的pose

            problem.set_h_prior(Hprior_);
            problem.set_b_prior(bprior_);
        }
    }

#ifdef PRINT_INFO  
    std::cout << "t_new_problem = " << t_new_problem.toc() << std::endl;
#endif  

    problem.solve(5);
    // _lambda_last = std::max(problem.get_current_lambda(), 0.);
    // std::cout << "_ordering_landmarks: " << problem._ordering_landmarks << std::endl;

    // update bprior_,  Hprior_ do not need update
    if (Hprior_.rows() > 0) {
        bprior_ = problem.get_b_prior();
// #ifdef PRINT_INFO        
//         std::cout << "----------- update bprior -------------\n";
//         std::cout << "             before: " << bprior_.norm() << std::endl;
//         std::cout << "                     " << errprior_.norm() << std::endl;
        
//         // errprior_ = problem.get_err_prior();
//         std::cout << "             after: " << bprior_.norm() << std::endl;
//         std::cout << "                    " << errprior_.norm() << std::endl;
// #endif        
    }

    // update parameter
    for (int i = 0; i < WINDOW_SIZE + 1; i++) {
        VecX p = vertexCams_vec[i]->get_parameters();
        for (int j = 0; j < 7; ++j) {
            para_Pose[i][j] = p[j];
        }

        VecX vb = vertexVB_vec[i]->get_parameters();
        for (int j = 0; j < 9; ++j) {
            para_SpeedBias[i][j] = vb[j];
        }
    }

    // 遍历每一个特征
#ifdef USE_OPENMP
    unsigned int feature_index = 0;
    // for (size_t n = 0; n < f_manager.features_vector.size(); ++n) {
    //     auto &&it_per_id = f_manager.features_vector[n];
    //     if (it_per_id.second->is_suitable_to_reprojection()) {
    //         para_Feature[feature_index++][0] = it_per_id.second->vertex_landmark->get_parameters()[0];
    //     }
    // }
    for (size_t n = 0; n < f_manager.suitable_features.size(); ++n) {
        para_Feature[feature_index++][0] = f_manager.suitable_features[n]->vertex_landmark->get_parameters()[0];
    }
#else    
    for (int i = 0; i < vertexPt_vec.size(); ++i) {
        VecX f = vertexPt_vec[i]->get_parameters();
        para_Feature[i][0] = f[0];
    }
#endif
}

void Estimator::backendOptimization() {
    TicToc t_solver;
    // 借助 vins 框架，维护变量
    vector2double();
    // 构建求解器
    problemSolve();
    // 优化后的变量处理下自由度
    double2vector();
    //ROS_INFO("whole time for solver: %f", t_solver.toc());

    // 维护 marg
    TicToc t_whole_marginalization;
    if (marginalization_flag == MARGIN_OLD) {
        vector2double();

        MargOldFrame();
    } else {
        if (Hprior_.rows() > 0) {

            vector2double();

            MargNewFrame();
        }
    }
#ifdef PRINT_INFO    
    std::cout << "t_whole_marginalization = " << t_whole_marginalization.toc() << std::endl;
#endif    
}


void Estimator::slideWindow() {
    TicToc t_margin;

    if (marginalization_flag == MARGIN_OLD) {
        double t_0 = Headers[0];
        back_R0 = Rs[0];
        back_P0 = Ps[0];
        if (frame_count == WINDOW_SIZE) {
            for (int i = 0; i < WINDOW_SIZE; i++) {
                Rs[i].swap(Rs[i + 1]);

                std::swap(pre_integrations[i], pre_integrations[i + 1]);

                dt_buf[i].swap(dt_buf[i + 1]);
                linear_acceleration_buf[i].swap(linear_acceleration_buf[i + 1]);
                angular_velocity_buf[i].swap(angular_velocity_buf[i + 1]);

                Headers[i] = Headers[i + 1];
                Ps[i].swap(Ps[i + 1]);
                Vs[i].swap(Vs[i + 1]);
                Bas[i].swap(Bas[i + 1]);
                Bgs[i].swap(Bgs[i + 1]);
            }
            Headers[WINDOW_SIZE] = Headers[WINDOW_SIZE - 1];
            Ps[WINDOW_SIZE] = Ps[WINDOW_SIZE - 1];
            Vs[WINDOW_SIZE] = Vs[WINDOW_SIZE - 1];
            Rs[WINDOW_SIZE] = Rs[WINDOW_SIZE - 1];
            Bas[WINDOW_SIZE] = Bas[WINDOW_SIZE - 1];
            Bgs[WINDOW_SIZE] = Bgs[WINDOW_SIZE - 1];

            delete pre_integrations[WINDOW_SIZE];
#ifdef CAIN_IMU_INTEGRATION
    pre_integrations[WINDOW_SIZE] = new vins::IMUIntegration{acc_0, gyr_0, Bas[WINDOW_SIZE], Bgs[WINDOW_SIZE]};
#else
    pre_integrations[WINDOW_SIZE] = new IntegrationBase{acc_0, gyr_0, Bas[WINDOW_SIZE], Bgs[WINDOW_SIZE]};
#endif            
            dt_buf[WINDOW_SIZE].clear();
            linear_acceleration_buf[WINDOW_SIZE].clear();
            angular_velocity_buf[WINDOW_SIZE].clear();

            if (true || solver_flag == INITIAL) {
                map<double, ImageFrame>::iterator it_0;
                it_0 = all_image_frame.find(t_0);
                delete it_0->second.pre_integration;
                it_0->second.pre_integration = nullptr;

                for (map<double, ImageFrame>::iterator it = all_image_frame.begin(); it != it_0; ++it) {
                    if (it->second.pre_integration)
                        delete it->second.pre_integration;
                    it->second.pre_integration = nullptr;
                }

                all_image_frame.erase(all_image_frame.begin(), it_0);
                all_image_frame.erase(t_0);
            }
            slideWindowOld();
        }
    }
    else {
        if (frame_count == WINDOW_SIZE) {
            double t_2nd = Headers[frame_count - 1];

            for (unsigned int i = 0; i < dt_buf[frame_count].size(); i++) {
                double tmp_dt = dt_buf[frame_count][i];
                Vector3d tmp_linear_acceleration = linear_acceleration_buf[frame_count][i];
                Vector3d tmp_angular_velocity = angular_velocity_buf[frame_count][i];

                pre_integrations[frame_count - 1]->push_back(tmp_dt, tmp_linear_acceleration, tmp_angular_velocity);

                dt_buf[frame_count - 1].push_back(tmp_dt);
                linear_acceleration_buf[frame_count - 1].push_back(tmp_linear_acceleration);
                angular_velocity_buf[frame_count - 1].push_back(tmp_angular_velocity);
            }

            Headers[frame_count - 1] = Headers[frame_count];
            Ps[frame_count - 1] = Ps[frame_count];
            Vs[frame_count - 1] = Vs[frame_count];
            Rs[frame_count - 1] = Rs[frame_count];
            Bas[frame_count - 1] = Bas[frame_count];
            Bgs[frame_count - 1] = Bgs[frame_count];

            delete pre_integrations[WINDOW_SIZE];
#ifdef CAIN_IMU_INTEGRATION
    pre_integrations[WINDOW_SIZE] = new vins::IMUIntegration{acc_0, gyr_0, Bas[WINDOW_SIZE], Bgs[WINDOW_SIZE]};
#else
    pre_integrations[WINDOW_SIZE] = new IntegrationBase{acc_0, gyr_0, Bas[WINDOW_SIZE], Bgs[WINDOW_SIZE]};
#endif            

            dt_buf[WINDOW_SIZE].clear();
            linear_acceleration_buf[WINDOW_SIZE].clear();
            angular_velocity_buf[WINDOW_SIZE].clear();

            // TODO: 是否应该删除次新帧？
            map<double, ImageFrame>::iterator it_2nd;
            it_2nd = all_image_frame.find(t_2nd);
            all_image_frame.erase(it_2nd);

            slideWindowNew();
        }
    }
#ifdef PRINT_INFO
    std::cout << "t_margin = " << t_margin.toc() << std::endl;
#endif    
}

// real marginalization is removed in solve_ceres()
void Estimator::slideWindowNew() {
    sum_of_front++;
    f_manager.remove_front(frame_count);
}
// real marginalization is removed in solve_ceres()
void Estimator::slideWindowOld() {
    sum_of_back++;

    bool shift_depth = solver_flag == NON_LINEAR ? true : false;
    if (shift_depth)
    {
        Matrix3d R0, R1;
        Vector3d P0, P1;
        R0 = back_R0 * ric[0];
        R1 = Rs[0] * ric[0];
        P0 = back_P0 + back_R0 * tic[0];
        P1 = Ps[0] + Rs[0] * tic[0];
        f_manager.remove_back_shift_depth(R0, P0, R1, P1);
    }
    else
        f_manager.remove_back();
}
