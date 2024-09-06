//
// Created by Cain on 2024/4/28.
//

#include "../estimator.h"

#include <array>
#include <memory>
#include <random>
#include <iostream>
#include <ostream>
#include <fstream>

namespace vins {
    using namespace graph_optimization;
    using namespace std;

    void Estimator::process_image(const vector<pair<unsigned long, vector<Vec7>>> &image, uint64_t time_stamp) {
        TicToc t_process;
        // 为frame赋时间
        _frame->time_us = time_stamp;
//        pnp_local(_frame, _frame->q(), _frame->p());

        // 把frame加入到stream中
        _stream.emplace_back(_frame, _pre_integral_stream);

        // TODO: 是否应该使用多线程来遍历 image ?
        // 把frame中的feature加入到landmark中
        if (_sliding_window.empty()) {
            for (auto &image_it : image) {
#if NUM_OF_CAM > 1
                if (image_it.second.size() < 2) {
                    continue;
                }
#endif
                unsigned long landmark_id = image_it.first;
                std::shared_ptr<Landmark> landmark;
                auto landmark_it = _landmarks.find(landmark_id);

                // 若landmark不在landmark map中，则需要新建
                if (landmark_it == _landmarks.end()) {
                    landmark = std::make_shared<Landmark>(landmark_id);
                    _landmarks.emplace(landmark_id, landmark);
                } else {
                    landmark = landmark_it->second;
                }

                // 把feature加入到frame中
                std::shared_ptr<Feature> feature = std::make_shared<Feature>();
                feature->frame = _frame;
                feature->landmark = landmark;
                for (size_t i = 0; i < std::min(image_it.second.size(), _ext_params.size()); ++i) {
                    feature->points.emplace_back(image_it.second[i].head<3>());
//                    feature->points.back() /= feature->points.back().z();
                }
                _frame->features.emplace(landmark_id, feature);

                // 把观测信息记录进landmark中
                landmark->observations.emplace_back(_frame, feature);
            }
            // 把第一次定义为keyframe
            _frame->is_key_frame = true;
        } else {
            // 遍历image中的每个feature
            double parallax_sum = 0.;
            unsigned int parallax_num = 0;
            unsigned int last_track_num = 0;
            unsigned int long_track_num = 0;
            unsigned int new_landmark_num = 0;
#ifdef USE_OPENMP
            static std::vector<std::pair<unsigned long, std::shared_ptr<Landmark>>> landmarks_vec[NUM_THREADS];
            static std::vector<std::pair<unsigned long, std::shared_ptr<Feature>>> features_vec[NUM_THREADS];
            for (auto &landmarks : landmarks_vec) {
                landmarks.clear();
            }
            for (auto &features : features_vec) {
                features.clear();
            }
#pragma omp parallel for num_threads(NUM_THREADS) reduction(+:parallax_sum, parallax_num, last_track_num, long_track_num, new_landmark_num)
            for (size_t n = 0; n < image.size(); ++n) {
#if NNUM_OF_CAM > 1
                if (image[n].second.size() < 2) {
                    continue;
                }
#endif
                unsigned int index = omp_get_thread_num();
                unsigned long landmark_id = image[n].first;
                std::shared_ptr<Landmark> landmark;
                auto landmark_it = _landmarks.find(landmark_id);

                // 若landmark不在landmark map中，则需要新建
                if (landmark_it == _landmarks.end()) {
                    landmark = std::make_shared<Landmark>(landmark_id);
                    landmarks_vec[index].emplace_back(landmark_id, landmark);
                    ++new_landmark_num;
                } else {
                    landmark = landmark_it->second;
                    ++last_track_num;   // 记录有几个特征点被跟踪了
                    if (landmark->observations.size() > WINDOW_SIZE / 3 + 1) {
                        ++long_track_num;
                    }
                }

                // 把feature加入到frame中
                std::shared_ptr<Feature> feature = std::make_shared<Feature>();
                feature->frame = _frame;
                feature->landmark = landmark;
                for (size_t i = 0; i < std::min(image[n].second.size(), _ext_params.size()); ++i) {
                    feature->points.emplace_back(image[n].second[i].head<3>());
//                    feature->points.back() /= feature->points.back().z();
                }
                features_vec[index].emplace_back(landmark_id, feature);

                // 把观测信息记录进landmark中
                landmark->observations.emplace_back(_frame, feature);

                // TODO: 这里假设只有一个双目，后续需要考虑多个双目的情况

                /*
                 * 计算每个特征点的视差，用于判断是否为keyframe:
                 * 若windows中的newest frame是key frame, 则和newest frame计算视差
                 * 否则, 和2nd newest frame计算视差，因为2nd newest必为key frame
                 * */
                std::shared_ptr<Frame> frame_ref;
                if (_sliding_window.size() == 1 || _sliding_window.back().first->is_key_frame) {
                    frame_ref = _sliding_window.back().first;
                } else {
                    frame_ref = _sliding_window[_sliding_window.size() - 2].first;
                }
                auto it = frame_ref->features.find(landmark_id);
                if (it != frame_ref->features.end()) {
                    double u_i = it->second->points[0].x();
                    double v_i = it->second->points[0].y();

                    double u_j = feature->points[0].x();
                    double v_j = feature->points[0].y();

                    double du = u_j - u_i;
                    double dv = v_j - v_i;

                    parallax_sum += std::max(abs(du), abs(dv));
                    ++parallax_num;
                }
            }
            for (auto &landmarks : landmarks_vec) {
                for (auto &landmark_it : landmarks) {
                    _landmarks.emplace(landmark_it.first, landmark_it.second);
                }
            }
            for (auto &features : features_vec) {
                for (auto &feature_it : features) {
                    _frame->features.emplace(feature_it.first, feature_it.second);
                }
            }
#else
            for (auto &image_it : image) {
#if NUM_OF_CAM > 1
                if (image_it.second.size() < 2) {
                    continue;
                }
#endif
                unsigned long landmark_id = image_it.first;
                std::shared_ptr<Landmark> landmark;
                auto landmark_it = _landmarks.find(landmark_id);

                // 若landmark不在landmark map中，则需要新建
                if (landmark_it == _landmarks.end()) {
                    landmark = std::make_shared<Landmark>(landmark_id);
                    _landmarks.emplace(landmark_id, landmark);
                    ++new_landmark_num;
                } else {
                    landmark = landmark_it->second;
                    ++last_track_num;   // 记录有几个特征点被跟踪了
                    if (landmark->observations.size() > WINDOW_SIZE / 3) {
                        ++long_track_num;
                    }
                }

                // 把feature加入到frame中
                std::shared_ptr<Feature> feature = std::make_shared<Feature>();
                feature->frame = _frame;
                feature->landmark = landmark;
                for (size_t i = 0; i < std::min(image_it.second.size(), _ext_params.size()); ++i) {
                    feature->points.emplace_back(image_it.second[i].head<3>());
//                    feature->points.back() /= feature->points.back().z();
                }
                _frame->features.emplace(landmark_id, feature);

                // 把观测信息记录进landmark中
                landmark->observations.emplace_back(_frame, feature);

                // TODO: 这里假设只有一个双目，后续需要考虑多个双目的情况

                /*
                 * 计算每个特征点的视差，用于判断是否为keyframe:
                 * 若windows中的newest frame是key frame, 则和newest frame计算视差
                 * 否则, 和2nd newest frame计算视差，因为2nd newest必为key frame
                 * */
                std::shared_ptr<Frame> frame_ref;
                if (_sliding_window.size() == 1 || _sliding_window.back().first->is_key_frame) {
                    frame_ref = _sliding_window.back().first;
                } else {
                    frame_ref = _sliding_window[_sliding_window.size() - 2].first;
                }
                auto it = frame_ref->features.find(landmark_id);
                if (it != frame_ref->features.end()) {
                    double u_i = it->second->points[0].x();
                    double v_i = it->second->points[0].y();

                    double u_j = feature->points[0].x();
                    double v_j = feature->points[0].y();

                    double du = u_j - u_i;
                    double dv = v_j - v_i;

                    parallax_sum += std::max(abs(du), abs(dv));
                    ++parallax_num;
                }
            }
#endif
            /*
             * 1. 若没有任何一个特征点在上个key frame中出现，则说明当前帧与上个key frame差异很大，所以一定是key frame
             * 2. 若当前帧中大部分的特征点都是新出现的，说明当前帧与历史所有帧差异都很大，所以一定是key frame
             * 其余情况则需通过平均视差值来判断是否为key frame
             * */
            if (parallax_num == 0 || last_track_num < 20 || long_track_num < 40 || last_track_num < 2 * new_landmark_num) {
                _frame->is_key_frame = true;
            } else {
                _frame->is_key_frame = parallax_sum / parallax_num >= MIN_PARALLAX;  // 若视差大于一定值，则认为是key frame
            }
        }

        // 若windows中的最新一帧是key frame, 则需要在滑窗满时marg最老帧
        if (_sliding_window.empty()) {
            marginalization_flag = MarginalizationFlag::NONE;
        } else {
            if (_sliding_window.back().first->is_key_frame) {
                marginalization_flag = is_sliding_window_full() ? MarginalizationFlag::MARGIN_OLD : MarginalizationFlag::NONE;
            } else {
                marginalization_flag = MarginalizationFlag::MARGIN_SECOND_NEW;
            }
        }

        auto process_cost = t_process.toc();
#ifdef PRINT_INFO
        std::cout << "process_cost = " << process_cost << " ms" << std::endl;
#endif
//        std::cout << "5" << std::endl;
//        std::cout << "_windows.size() = " << _windows.size() << std::endl;
//        marginalization_flag = MARGIN_OLD;
//        slide_window();
//        std::cout << "6" << std::endl;
        std::cout << "_sliding_window.size() = " << _sliding_window.size() << std::endl;
        std::cout << "_landmarks.size() = " << _landmarks.size() << std::endl;
        if (marginalization_flag == MarginalizationFlag::MARGIN_OLD) {
            std::cout << "marg_flag = old" << std::endl;
        } else if (marginalization_flag == MarginalizationFlag::MARGIN_SECOND_NEW) {
            std::cout << "marg_flag = new" << std::endl;
        } else {
            std::cout << "marg_flag = none" << std::endl;
        }
        std::cout << "is key frame = " << _frame->is_key_frame << std::endl;

        backend();

        // 下一时刻的值
        _frame = std::make_shared<Frame>(_frame->state);
        _pre_integral_stream = std::make_shared<vins::IMUIntegration>(_acc_latest, _gyro_latest, _frame->ba(), _frame->bg());
    }
}