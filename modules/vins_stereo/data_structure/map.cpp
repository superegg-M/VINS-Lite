//
// Created by Cain on 2024/6/12.
//

#include "map.h"
#include "feature.h"

namespace vins {
    void Map::insert_key_frame(Frame::Ptr frame) {
        _current_frame = frame;
        if (_keyframes.find(frame->keyframe_id) == _keyframes.end()) {
            _keyframes.insert(make_pair(frame->keyframe_id, frame));
            _active_keyframes.insert(make_pair(frame->keyframe_id, frame));
        } else {
            _keyframes[frame->keyframe_id] = frame;
            _active_keyframes[frame->keyframe_id] = frame;
        }

        if (_active_keyframes.size() > _num_active_keyframes) {
            remove_old_key_frame();
        }
    }

    void Map::insert_map_point(MapPoint::Ptr map_point) {
        if (_landmarks.find(map_point->id) == _landmarks.end()) {
            _landmarks.insert(make_pair(map_point->id, map_point));
            _active_landmarks.insert(make_pair(map_point->id, map_point));
        } else {
            _landmarks[map_point->id] = map_point;
            _active_landmarks[map_point->id] = map_point;
        }
    }

    void Map::remove_old_key_frame() {
        if (_current_frame == nullptr) return;
        // 寻找与当前帧最近与最远的两个关键帧
        double max_dis = 0, min_dis = 9999;
        double max_kf_id = 0, min_kf_id = 0;
        const auto &state = _current_frame->state;
        for (auto& kf : _active_keyframes) {
            if (kf.second == _current_frame) continue;
//            auto dis = (kf.second->Pose() * Twc).log().norm();
            double dis = 1.;
            if (dis > max_dis) {
                max_dis = dis;
                max_kf_id = kf.first;
            }
            if (dis < min_dis) {
                min_dis = dis;
                min_kf_id = kf.first;
            }
        }

        const double min_dis_th = 0.2;  // 最近阈值
        Frame::Ptr frame_to_remove = nullptr;
        if (min_dis < min_dis_th) {
            // 如果存在很近的帧，优先删掉最近的
            frame_to_remove = _keyframes.at(min_kf_id);
        } else {
            // 删掉最远的
            frame_to_remove = _keyframes.at(max_kf_id);
        }

        std::cout << "remove keyframe " << frame_to_remove->keyframe_id << std::endl;
        // remove keyframe and landmark observation
        _active_keyframes.erase(frame_to_remove->keyframe_id);
        for (auto &feat : frame_to_remove->features_left) {
            auto mp = feat->map_point.lock();
            if (mp) {
                mp->remove_observation(feat);
            }
        }
        for (auto &feat : frame_to_remove->features_right) {
            if (feat == nullptr) continue;
            auto mp = feat->map_point.lock();
            if (mp) {
                mp->remove_observation(feat);
            }
        }

        clean_map();
    }

    void Map::clean_map() {
        unsigned int cnt_landmark_removed = 0;
        for (auto iter = _active_landmarks.begin(); iter != _active_landmarks.end();) {
            if (iter->second->observed_times == 0) {
                iter = _active_landmarks.erase(iter);
                cnt_landmark_removed++;
            } else {
                ++iter;
            }
        }
        std::cout << "Removed " << cnt_landmark_removed << " active landmarks" << std::endl;
    }
}