//
// Created by Cain on 2024/6/12.
//

#ifndef GRAPH_OPTIMIZATION_MAP_H
#define GRAPH_OPTIMIZATION_MAP_H

#include <iostream>
#include <unordered_map>
#include <map>
#include <memory>
#include "backend/eigen_types.h"
#include "thirdparty/Sophus/sophus/so3.hpp"

#include "frame.h"
#include "map_point.h"

namespace vins {
    /**
     * @brief 地图
     * 和地图的交互：前端调用InsertKeyframe和InsertMapPoint插入新帧和地图点，后端维护地图的结构，判定outlier/剔除等等
     */
    class Map {
    public:
        EIGEN_MAKE_ALIGNED_OPERATOR_NEW;
        typedef std::shared_ptr<Map> Ptr;
        typedef std::unordered_map<unsigned long, MapPoint::Ptr> LandmarksType;
        typedef std::unordered_map<unsigned long, Frame::Ptr> KeyframesType;

        Map() {}

        /// 增加一个关键帧
        void insert_key_frame(Frame::Ptr frame);
        /// 增加一个地图顶点
        void insert_map_point(MapPoint::Ptr map_point);

        /// 获取所有地图点
        const LandmarksType &get_all_map_points() const {
//            std::unique_lock<std::mutex> lck(data_mutex_);
            return _landmarks;
        }
        /// 获取所有关键帧
        const KeyframesType &get_all_key_frames() const {
//            std::unique_lock<std::mutex> lck(data_mutex_);
            return _keyframes;
        }

        /// 获取激活地图点
        const LandmarksType &get_active_map_points() const {
//            std::unique_lock<std::mutex> lck(data_mutex_);
            return _active_landmarks;
        }

        /// 获取激活关键帧
        const KeyframesType &GetActiveKeyFrames() const {
//            std::unique_lock<std::mutex> lck(data_mutex_);
            return _active_keyframes;
        }

        /// 清理map中观测数量为零的点
        void clean_map();

    private:
        // 将旧的关键帧置为不活跃状态
        void remove_old_key_frame();

//        std::mutex data_mutex_;
        LandmarksType _landmarks;         // all landmarks
        LandmarksType _active_landmarks;  // active landmarks
        KeyframesType _keyframes;         // all key-frames
        KeyframesType _active_keyframes;  // active key-frames

        Frame::Ptr _current_frame = nullptr;

        // settings
        unsigned int _num_active_keyframes = 10;  // 激活的关键帧数量
    };
}

#endif //GRAPH_OPTIMIZATION_MAP_H
