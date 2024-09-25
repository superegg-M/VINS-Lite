#include "feature_tracker.h"

int FeatureTracker::n_id = 0;

bool inBorder(const cv::Point2f &pt)
{
    const int BORDER_SIZE = 1;
    int img_x = cvRound(pt.x);
    int img_y = cvRound(pt.y);
    return BORDER_SIZE <= img_x && img_x < COL - BORDER_SIZE && BORDER_SIZE <= img_y && img_y < ROW - BORDER_SIZE;
}

// 只有保留status大于0的元素 
void reduceVector(vector<cv::Point2f> &v, vector<uchar> status)
{
    int j = 0;
    for (int i = 0; i < int(v.size()); i++)
        if (status[i])
            v[j++] = v[i];
    v.resize(j);
}

// 只有保留status大于0的元素 
void reduceVector(vector<int> &v, vector<uchar> status)
{
    int j = 0;
    for (int i = 0; i < int(v.size()); i++)
        if (status[i])
            v[j++] = v[i];
    v.resize(j);
}


FeatureTracker::FeatureTracker()
{
}

void FeatureTracker::setMask()
{
    if(FISHEYE)
        mask = fisheye_mask.clone();
    else
        mask = cv::UMat(ROW, COL, CV_8UC1, cv::Scalar(255));
    

    // prefer to keep features that are tracked for long time
    vector<pair<int, pair<cv::Point2f, int>>> cnt_pts_id;

    for (unsigned int i = 0; i < forw_pts.size(); i++)
        cnt_pts_id.push_back(make_pair(track_cnt[i], make_pair(forw_pts[i], ids[i])));

    // 根据跟踪次数进行排序
    sort(cnt_pts_id.begin(), cnt_pts_id.end(), [](const pair<int, pair<cv::Point2f, int>> &a, const pair<int, pair<cv::Point2f, int>> &b)
         {
            return a.first > b.first;
         });

    forw_pts.clear();
    ids.clear();
    track_cnt.clear();

    cv::Mat mask_cpu;
    mask.copyTo(mask_cpu);
    for (auto &it : cnt_pts_id)
    {
        if (mask_cpu.at<uchar>(it.second.first) == 255)     // 如果该区域没被mask, 则进行mask
        {
            forw_pts.push_back(it.second.first);
            ids.push_back(it.second.second);
            track_cnt.push_back(it.first);
            cv::circle(mask, it.second.first, MIN_DIST, 0, -1);     // 0代表赋0, 而-1表带实心圆
        }
    }
}

// 把特征点检测出的点n_pts加入光流点forw_pts中
void FeatureTracker::addPoints()
{
    for (auto &p : n_pts)
    {
        forw_pts.push_back(p);
        ids.push_back(-1);
        track_cnt.push_back(1);
    }
}

void FeatureTracker::readImage(const cv::UMat &_img, double _cur_time)
{
    cv::UMat img;
    TicToc t_r;
    cur_time = _cur_time;

    if (EQUALIZE)
    {
        cv::Ptr<cv::CLAHE> clahe = cv::createCLAHE(3.0, cv::Size(8, 8));
        TicToc t_c;
        clahe->apply(_img, img);
        //ROS_DEBUG("CLAHE costs: %fms", t_c.toc());
    }
    else
        img = _img;

    if (forw_img.empty())
    {
        prev_img = cur_img = forw_img = img;
    }
    else
    {
        forw_img = img;
    }

    forw_pts.clear();
    match_pts1.clear();
    match_pts2.clear();

    for (auto &n : track_cnt)   // 每个点的跟踪次数加一
        n++;

    if (PUB_THIS_FRAME)
    {
        // 初始化SIFT检测器
        cv::Ptr<cv::xfeatures2d::SIFT> sift = cv::xfeatures2d::SIFT::create(MAX_CNT,4,0.06,20);
        // cv::Ptr<cv::ORB> orb=cv::ORB::create(MAX_CNT);
        int minHessian = 5000;
        cv::Ptr<cv::xfeatures2d::SURF> surf = cv::xfeatures2d::SURF::create(minHessian);

        // 检测特征点和计算描述符
        // if(!keypoints1.size())
        // surf->detectAndCompute(cur_img, cv::noArray(), keypoints1, descriptors1);

        surf->detectAndCompute(forw_img, cv::noArray(), keypoints2, descriptors2);

        // 设置BF参数
        // cv::BFMatcher matcher(cv::NORM_L2);
        // cv::FlannBasedMatcher matcher();

        // 匹配描述符
        // std::vector<cv::DMatch> matches,good_matches;
        // matcher.match(descriptors1, descriptors2, matches);

        cv::Ptr<cv::DescriptorMatcher> matcher = cv::DescriptorMatcher::create(cv::DescriptorMatcher::FLANNBASED);
        std::vector< std::vector<cv::DMatch> > knn_matches;
        matcher->knnMatch( descriptors1, descriptors2, knn_matches, 2 );
        //-- 使用洛氏比率测试筛选匹配项
        const float ratio_thresh = 0.7f;
        std::vector<cv::DMatch> good_matches;
        for (size_t i = 0; i < knn_matches.size(); i++)
        {
            if (knn_matches[i][0].distance < ratio_thresh * knn_matches[i][1].distance)
            {
                good_matches.push_back(knn_matches[i][0]);
            }
        }

        // // 计算最大最小距离
        // auto min_max=minmax_element(matches.begin(),matches.end(),
        //     [](const cv::DMatch &m1,const cv::DMatch &m2){return m1.distance<m2.distance;});
        
        // double min_dist = min_max.first->distance;
        // double max_dist = min_max.second->distance;
        // cout<<"!!!!!!!!!!!!!!!!!!!min_dist:"<<min_dist<<endl;

        // // 筛选好的匹配
        // for (const auto& m : matches) {
        //     if (m.distance <= max(2*min_dist,10.0)) {
        //         good_matches.push_back(m);
        //     }
        // }

        // std::vector<cv::DMatch> good_matches=matches;

        for(int i=0;i<(int)good_matches.size();i++){
            match_pts1.push_back(keypoints1[good_matches[i].queryIdx].pt);
            match_pts2.push_back(keypoints2[good_matches[i].trainIdx].pt);
        }

        // 添加新跟踪点
        for(auto &p:match_pts1){
            if(find(cur_pts.begin(),cur_pts.end(),p)==cur_pts.end()){
                cur_pts.push_back(p);
                ids.push_back(-1);
                track_cnt.push_back(1);
            }
        }
        // 删除未跟踪点
        int j=0;
        for(int i=0;i<(int)cur_pts.size();i++){
            if(find(match_pts1.begin(),match_pts1.end(),cur_pts[i])!=match_pts1.end()){
                cur_pts[j]=cur_pts[i];
                ids[j]=ids[i];
                track_cnt[j]=track_cnt[i];
                if(cur_un_pts.size()>0){
                    cur_un_pts[j]=cur_un_pts[i];
                }
                forw_pts.push_back(match_pts2[i]);
                j++;
            }
        }
        cur_pts.resize(j);
        ids.resize(j);            
        track_cnt.resize(j);
        if(cur_un_pts.size()>0){
            cur_un_pts.resize(j);
        }

        cout<<"match size:"<<good_matches.size()<<endl;
        cout<<"tracker size: "<<cur_pts.size()<<endl;

#ifdef DISPLAY_TRAJ
        // 绘制匹配结果
        cv::UMat img_matches;
        cv::drawMatches(cur_img, keypoints1, forw_img, keypoints2, good_matches, img_matches);
        // 显示图像
        cv::imshow("Matches", img_matches);
        cv::waitKey(1);
#endif
        
        // rejectWithF();  // 根据基础矩阵过滤掉一部分outliner的点
        //ROS_DEBUG("set mask begins");
        // TicToc t_m;
        // setMask();      // 计算mask, 并且mask掉不需要的点
        //ROS_DEBUG("set mask costs %fms", t_m.toc());

        //ROS_DEBUG("detect feature begins");
        // TicToc t_t;
        // int n_max_cnt = MAX_CNT - static_cast<int>(forw_pts.size());
        // if (n_max_cnt > 0)      // 由于对点进行了过滤和mask, 所以会存在点不够的情况
        // {
        // if(mask.empty())
        //     cout << "mask is empty " << endl;
        // if (mask.type() != CV_8UC1)
        //     cout << "mask type wrong " << endl;
        // if (mask.size() != forw_img.size())
        //     cout << "wrong size " << endl;

        // // 角点提取(默认为shi-tomasi, 传入true则为Harris)    
        // cv::goodFeaturesToTrack(forw_img, n_pts, MAX_CNT - forw_pts.size(), 0.01, MIN_DIST, mask);
        

        // }
        // else
        //     n_pts.clear();
        //ROS_DEBUG("detect feature costs: %fms", t_t.toc());

        //ROS_DEBUG("add feature begins");
        // TicToc t_a;
        // addPoints();    // 把特征点n_pts加入光流点forw_pts中
        //ROS_DEBUG("selectFeature costs: %fms", t_a.toc());
    }

    // if (cur_pts.size() > 0)
    // {
    //     TicToc t_o;
    //     vector<uchar> status;
    //     vector<float> err;
    //     // cv::calcOpticalFlowPyrLK(cur_img, forw_img, cur_pts, forw_pts, status, err, cv::Size(21, 21), 3);


    //     // for (int i = 0; i < int(forw_pts.size()); i++)
    //     //     if (status[i] && !inBorder(forw_pts[i]))    // 若超过边界, 依然会把该点的跟踪设为失败
    //     //         status[i] = 0;
    //     // // TODO: 使用list代替vector也能实现更快的删除, 但是cur_pts和forw_pts必须为数组        
    //     // reduceVector(prev_pts, status);
    //     // reduceVector(cur_pts, status);
    //     // reduceVector(forw_pts, status);
    //     // reduceVector(ids, status);
    //     // reduceVector(cur_un_pts, status);
    //     // reduceVector(track_cnt, status);
    //     // //ROS_DEBUG("temporal optical flow costs: %fms", t_o.toc());
    // }

    prev_img = cur_img;
    prev_pts = cur_pts;
    prev_un_pts = cur_un_pts;
    cur_img = forw_img;
    cur_pts = forw_pts;
    undistortedPoints();    // 计算光流点的速度
    prev_time = cur_time;
    keypoints1=keypoints2;
    descriptors1=descriptors2;
}

void FeatureTracker::rejectWithF()
{
    if (forw_pts.size() >= 8)
    {
        //ROS_DEBUG("FM ransac begins");
        TicToc t_f;
        vector<cv::Point2f> un_cur_pts(cur_pts.size()), un_forw_pts(forw_pts.size());
        for (unsigned int i = 0; i < cur_pts.size(); i++)
        {
            Eigen::Vector3d tmp_p;
            m_camera->liftProjective(Eigen::Vector2d(cur_pts[i].x, cur_pts[i].y), tmp_p);   // 将像素坐标转化为无畸变的归一化坐标
            tmp_p.x() = FOCAL_LENGTH * tmp_p.x() / tmp_p.z() + COL / 2.0;
            tmp_p.y() = FOCAL_LENGTH * tmp_p.y() / tmp_p.z() + ROW / 2.0;
            un_cur_pts[i] = cv::Point2f(tmp_p.x(), tmp_p.y());

            m_camera->liftProjective(Eigen::Vector2d(forw_pts[i].x, forw_pts[i].y), tmp_p); // 将像素坐标转化为无畸变的归一化坐标
            tmp_p.x() = FOCAL_LENGTH * tmp_p.x() / tmp_p.z() + COL / 2.0;
            tmp_p.y() = FOCAL_LENGTH * tmp_p.y() / tmp_p.z() + ROW / 2.0;
            un_forw_pts[i] = cv::Point2f(tmp_p.x(), tmp_p.y());
        }

        vector<uchar> status;   // 标记在计算基础矩阵时, outliner的点
        cv::findFundamentalMat(un_cur_pts, un_forw_pts, cv::FM_RANSAC, F_THRESHOLD, 0.99, status);
        int size_a = cur_pts.size();
        reduceVector(prev_pts, status);
        reduceVector(cur_pts, status);
        reduceVector(forw_pts, status);
        reduceVector(cur_un_pts, status);
        reduceVector(ids, status);
        reduceVector(track_cnt, status);
        //ROS_DEBUG("FM ransac: %d -> %lu: %f", size_a, forw_pts.size(), 1.0 * forw_pts.size() / size_a);
        //ROS_DEBUG("FM ransac costs: %fms", t_f.toc());
    }
}

bool FeatureTracker::updateID(unsigned int i)
{
    if (i < ids.size())
    {
        if (ids[i] == -1)
            ids[i] = n_id++;
        return true;
    }
    else
        return false;
}

void FeatureTracker::readIntrinsicParameter(const string &calib_file)
{
    cout << "reading paramerter of camera " << calib_file << endl;
    m_camera = CameraFactory::instance()->generateCameraFromYamlFile(calib_file);
}

void FeatureTracker::showUndistortion(const string &name)
{
    cv::Mat cur_img_cpu;
    cur_img.copyTo(cur_img_cpu);
    cv::Mat undistortedImg(ROW + 600, COL + 600, CV_8UC1, cv::Scalar(0));
    vector<Eigen::Vector2d> distortedp, undistortedp;
    for (int i = 0; i < COL; i++)
        for (int j = 0; j < ROW; j++)
        {
            Eigen::Vector2d a(i, j);
            Eigen::Vector3d b;
            m_camera->liftProjective(a, b);     // 将像素坐标转化为无畸变的归一化坐标
            distortedp.push_back(a);
            undistortedp.push_back(Eigen::Vector2d(b.x() / b.z(), b.y() / b.z()));
            //printf("%f,%f->%f,%f,%f\n)\n", a.x(), a.y(), b.x(), b.y(), b.z());
        }
    for (int i = 0; i < int(undistortedp.size()); i++)
    {
        cv::Mat pp(3, 1, CV_32FC1);
        pp.at<float>(0, 0) = undistortedp[i].x() * FOCAL_LENGTH + COL / 2;
        pp.at<float>(1, 0) = undistortedp[i].y() * FOCAL_LENGTH + ROW / 2;
        pp.at<float>(2, 0) = 1.0;
        //cout << trackerData[0].K << endl;
        //printf("%lf %lf\n", p.at<float>(1, 0), p.at<float>(0, 0));
        //printf("%lf %lf\n", pp.at<float>(1, 0), pp.at<float>(0, 0));
        if (pp.at<float>(1, 0) + 300 >= 0 && pp.at<float>(1, 0) + 300 < ROW + 600 && pp.at<float>(0, 0) + 300 >= 0 && pp.at<float>(0, 0) + 300 < COL + 600)
        {
            undistortedImg.at<uchar>(pp.at<float>(1, 0) + 300, pp.at<float>(0, 0) + 300) = cur_img_cpu.at<uchar>(distortedp[i].y(), distortedp[i].x());
        }
        else
        {
            //ROS_ERROR("(%f %f) -> (%f %f)", distortedp[i].y, distortedp[i].x, pp.at<float>(1, 0), pp.at<float>(0, 0));
        }
    }
    cv::imshow(name, undistortedImg);
    cv::waitKey(0);
}

void FeatureTracker::undistortedPoints()
{
    cur_un_pts.clear();
    cur_un_pts_map.clear();
    //cv::undistortPoints(cur_pts, un_pts, K, cv::UMat());
    for (unsigned int i = 0; i < cur_pts.size(); i++)
    {
        Eigen::Vector2d a(cur_pts[i].x, cur_pts[i].y);
        Eigen::Vector3d b;
        m_camera->liftProjective(a, b);     // 将像素坐标转化为无畸变的归一化坐标

        cur_un_pts.push_back(cv::Point2f(b.x() / b.z(), b.y() / b.z()));
        cur_un_pts_map.insert(make_pair(ids[i], cv::Point2f(b.x() / b.z(), b.y() / b.z())));
        // printf("cur pts id %d %f %f\n", ids[i], cur_un_pts[i].x, cur_un_pts[i].y);
    }
    // caculate points velocity
    if (!prev_un_pts_map.empty())
    {
        double dt = cur_time - prev_time;
        pts_velocity.clear();
        for (unsigned int i = 0; i < cur_un_pts.size(); i++)
        {
            if (ids[i] != -1)   // 不是新加的点
            {
                std::map<int, cv::Point2f>::iterator it;
                it = prev_un_pts_map.find(ids[i]);
                if (it != prev_un_pts_map.end())    // 以前就存在的点
                {
                    double v_x = (cur_un_pts[i].x - it->second.x) / dt;
                    double v_y = (cur_un_pts[i].y - it->second.y) / dt;
                    pts_velocity.push_back(cv::Point2f(v_x, v_y));
                }
                else
                    pts_velocity.push_back(cv::Point2f(0, 0));
            }
            else    // 新点的速度置为0
            {
                pts_velocity.push_back(cv::Point2f(0, 0));
            }
        }
    }
    else
    {
        for (unsigned int i = 0; i < cur_pts.size(); i++)
        {
            pts_velocity.push_back(cv::Point2f(0, 0));  // 初始时刻全部点的速度置为0
        }
    }
    prev_un_pts_map = cur_un_pts_map;
}
