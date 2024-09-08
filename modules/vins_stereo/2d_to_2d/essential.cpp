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

#define FIVE_POINT_ALGORITHM

namespace vins {
    using namespace graph_optimization;
    using namespace std;

#ifdef FIVE_POINT_ALGORITHM
#define N_POINTS 5

    class Polynomial {
    public:
        // clang-format off
        enum GRevLexMonomials {
            XXX = 0, XXY = 1, XYY = 2, YYY = 3, XXZ = 4, XYZ = 5, YYZ = 6, XZZ = 7, YZZ = 8, ZZZ = 9,
            XX = 10, XY = 11, YY = 12, XZ = 13, YZ = 14, ZZ = 15, X = 16, Y = 17, Z = 18, I = 19
        };
        // clang-format on

        Eigen::Matrix<double, 20, 1> v;

        Polynomial(const Eigen::Matrix<double, 20, 1> &coeffcients) :
            v(coeffcients) {
        }

    public:
        Polynomial() :
            Polynomial(Eigen::Matrix<double, 20, 1>::Zero()) {
        }

        Polynomial(double w) {
            v.setZero();
            v[I] = w;
        }

        Polynomial(double x, double y, double z, double w) {
            v.setZero();
            v[X] = x;
            v[Y] = y;
            v[Z] = z;
            v[I] = w;
        }

        void set_xyzw(double x, double y, double z, double w) {
            v.setZero();
            v[X] = x;
            v[Y] = y;
            v[Z] = z;
            v[I] = w;
        }

        Polynomial operator-() const {
            return Polynomial(-v);
        }

        Polynomial operator+(const Polynomial &b) const {
            return Polynomial(v + b.v);
        }

        Polynomial operator-(const Polynomial &b) const {
            return Polynomial(v - b.v);
        }

        Polynomial operator*(const Polynomial &b) const {
            Polynomial r;

            r.v[I] = v[I] * b.v[I];

            r.v[Z] = v[I] * b.v[Z] + v[Z] * b.v[I];
            r.v[Y] = v[I] * b.v[Y] + v[Y] * b.v[I];
            r.v[X] = v[I] * b.v[X] + v[X] * b.v[I];

            r.v[ZZ] = v[I] * b.v[ZZ] + v[Z] * b.v[Z] + v[ZZ] * b.v[I];
            r.v[YZ] = v[I] * b.v[YZ] + v[Z] * b.v[Y] + v[Y] * b.v[Z] + v[YZ] * b.v[I];
            r.v[XZ] = v[I] * b.v[XZ] + v[Z] * b.v[X] + v[X] * b.v[Z] + v[XZ] * b.v[I];
            r.v[YY] = v[I] * b.v[YY] + v[Y] * b.v[Y] + v[YY] * b.v[I];
            r.v[XY] = v[I] * b.v[XY] + v[Y] * b.v[X] + v[X] * b.v[Y] + v[XY] * b.v[I];
            r.v[XX] = v[I] * b.v[XX] + v[X] * b.v[X] + v[XX] * b.v[I];

            r.v[ZZZ] = v[I] * b.v[ZZZ] + v[Z] * b.v[ZZ] + v[ZZ] * b.v[Z] + v[ZZZ] * b.v[I];
            r.v[YZZ] = v[I] * b.v[YZZ] + v[Z] * b.v[YZ] + v[Y] * b.v[ZZ] + v[ZZ] * b.v[Y] + v[YZ] * b.v[Z] + v[YZZ] * b.v[I];
            r.v[XZZ] = v[I] * b.v[XZZ] + v[Z] * b.v[XZ] + v[X] * b.v[ZZ] + v[ZZ] * b.v[X] + v[XZ] * b.v[Z] + v[XZZ] * b.v[I];
            r.v[YYZ] = v[I] * b.v[YYZ] + v[Z] * b.v[YY] + v[Y] * b.v[YZ] + v[YZ] * b.v[Y] + v[YY] * b.v[Z] + v[YYZ] * b.v[I];
            r.v[XYZ] = v[I] * b.v[XYZ] + v[Z] * b.v[XY] + v[Y] * b.v[XZ] + v[X] * b.v[YZ] + v[YZ] * b.v[X] + v[XZ] * b.v[Y] + v[XY] * b.v[Z] + v[XYZ] * b.v[I];
            r.v[XXZ] = v[I] * b.v[XXZ] + v[Z] * b.v[XX] + v[X] * b.v[XZ] + v[XZ] * b.v[X] + v[XX] * b.v[Z] + v[XXZ] * b.v[I];
            r.v[YYY] = v[I] * b.v[YYY] + v[Y] * b.v[YY] + v[YY] * b.v[Y] + v[YYY] * b.v[I];
            r.v[XYY] = v[I] * b.v[XYY] + v[Y] * b.v[XY] + v[X] * b.v[YY] + v[YY] * b.v[X] + v[XY] * b.v[Y] + v[XYY] * b.v[I];
            r.v[XXY] = v[I] * b.v[XXY] + v[Y] * b.v[XX] + v[X] * b.v[XY] + v[XY] * b.v[X] + v[XX] * b.v[Y] + v[XXY] * b.v[I];
            r.v[XXX] = v[I] * b.v[XXX] + v[X] * b.v[XX] + v[XX] * b.v[X] + v[XXX] * b.v[I];

            return r;
        }

        const Eigen::Matrix<double, 20, 1> &coeffcients() const {
            return v;
        }
    };

    Polynomial operator*(const double &scale, const Polynomial &poly) {
        return Polynomial(scale * poly.coeffcients());
    }

    std::vector<Eigen::Matrix<double, 3, 3>> solve_essential(const Eigen::Matrix<double, N_POINTS, 9> &D) {
        auto to_matrix = [](const Eigen::Matrix<double, 9, 1> &vec) -> Eigen::Matrix<double, 3, 3> {
            return (Eigen::Matrix<double, 3, 3>() << vec.segment<3>(0), vec.segment<3>(3), vec.segment<3>(6)).finished();
        };

        // 计算 D*vec(E) = 0 的 nullspace
        Eigen::Matrix<double, 9, 4> nullspace = D.jacobiSvd(Eigen::ComputeFullV).matrixV().block<9, 4>(0, 5);
        auto Ex = to_matrix(nullspace.col(0));
        auto Ey = to_matrix(nullspace.col(1));
        auto Ez = to_matrix(nullspace.col(2));
        auto Ew = to_matrix(nullspace.col(3));

        // E = x*Ex + y*Ey + z*Ez + w*Ew, w = 1
        Eigen::Matrix<Polynomial, 3, 3> Epoly;
        Epoly << Polynomial(Ex(0, 0), Ey(0, 0), Ez(0, 0), Ew(0, 0)), Polynomial(Ex(0, 1), Ey(0, 1), Ez(0, 1), Ew(0, 1)), Polynomial(Ex(0, 2), Ey(0, 2), Ez(0, 2), Ew(0, 2)),
                 Polynomial(Ex(1, 0), Ey(1, 0), Ez(1, 0), Ew(1, 0)), Polynomial(Ex(1, 1), Ey(1, 1), Ez(1, 1), Ew(1, 1)), Polynomial(Ex(1, 2), Ey(1, 2), Ez(1, 2), Ew(1, 2)),
                 Polynomial(Ex(2, 0), Ey(2, 0), Ez(2, 0), Ew(2, 0)), Polynomial(Ex(2, 1), Ey(2, 1), Ez(2, 1), Ew(2, 1)), Polynomial(Ex(2, 2), Ey(2, 2), Ez(2, 2), Ew(2, 2));

        // 计算约束矩阵
        Eigen::Matrix<double, 10, 20> polynomials;   

        // E*E^T*E - 0.5*trace(E*E^T)*E = 0
        Eigen::Matrix<Polynomial, 3, 3> EEt = Epoly * Epoly.transpose();  
        Eigen::Matrix<Polynomial, 3, 3> singular_value_constraints = (EEt * Epoly) - (0.5 * EEt.trace()) * Epoly;
        for (size_t i = 0; i < 3; ++i) {
            for (size_t j = 0; j < 3; ++j) {
                polynomials.row(i * 3 + j) = singular_value_constraints(i, j).coeffcients();
            }
        }

        // det(E) = 0
        Polynomial detE = Epoly.determinant();
        polynomials.row(9) = detE.coeffcients();

        // 对约束矩阵进行高斯消元
        std::array<size_t, 10> perm;
        for (size_t i = 0; i < 10; ++i) {
            perm[i] = i;
        }
        for (size_t i = 0; i < 10; ++i) {
            for (size_t j = i + 1; j < 10; ++j) {
                if (abs(polynomials(perm[i], i)) < abs(polynomials(perm[j], i))) {
                    std::swap(perm[i], perm[j]);
                }
            }
            if (polynomials(perm[i], i) == 0) continue;
            polynomials.row(perm[i]) /= polynomials(perm[i], i);
            for (size_t j = i + 1; j < 10; ++j) {
                polynomials.row(perm[j]) -= polynomials.row(perm[i]) * polynomials(perm[j], i);
            }
        }
        for (size_t i = 9; i > 0; --i) {
            for (size_t j = 0; j < i; ++j) {
                polynomials.row(perm[j]) -= polynomials.row(perm[i]) * polynomials(perm[j], i);
            }
        }

        Eigen::Matrix<double, 10, 10> action;
        action.row(0) = -polynomials.block<1, 10>(perm[Polynomial::XXX], Polynomial::XX);
        action.row(1) = -polynomials.block<1, 10>(perm[Polynomial::XXY], Polynomial::XX);
        action.row(2) = -polynomials.block<1, 10>(perm[Polynomial::XYY], Polynomial::XX);
        action.row(3) = -polynomials.block<1, 10>(perm[Polynomial::XXZ], Polynomial::XX);
        action.row(4) = -polynomials.block<1, 10>(perm[Polynomial::XYZ], Polynomial::XX);
        action.row(5) = -polynomials.block<1, 10>(perm[Polynomial::XZZ], Polynomial::XX);
        action.row(6) = Eigen::Matrix<double, 10, 1>::Unit(Polynomial::XX - Polynomial::XX).transpose();
        action.row(7) = Eigen::Matrix<double, 10, 1>::Unit(Polynomial::XY - Polynomial::XX).transpose();
        action.row(8) = Eigen::Matrix<double, 10, 1>::Unit(Polynomial::XZ - Polynomial::XX).transpose();
        action.row(9) = Eigen::Matrix<double, 10, 1>::Unit(Polynomial::X - Polynomial::XX).transpose();

        // 特征分解
        Eigen::EigenSolver<Eigen::Matrix<double, 10, 10>> eigen(action, true);
        Eigen::Matrix<std::complex<double>, 10, 1> xs = eigen.eigenvalues();

        // Essential矩阵计算
        std::vector<Eigen::Matrix<double, 3, 3>> results;
        results.reserve(10);
        for (size_t i = 0; i < 10; ++i) {
            if (abs(xs[i].imag()) < 1.0e-10) {
                Eigen::Matrix<double, 10, 1> h = eigen.eigenvectors().col(i).real();
                double xw = h(Polynomial::X - Polynomial::XX);
                double yw = h(Polynomial::Y - Polynomial::XX);
                double zw = h(Polynomial::Z - Polynomial::XX);
                double w = h(Polynomial::I - Polynomial::XX);
                results.emplace_back(to_matrix((nullspace * Eigen::Vector4d(xw / w, yw / w, zw / w, 1.)).normalized()));
            }
        }
        return results;
    }
#else
#define N_POINTS 8

    std::vector<Eigen::Matrix<double, 3, 3>> solve_essential(const Eigen::Matrix<double, N_POINTS, 9> &D) {
        Eigen::JacobiSVD<Eigen::Matrix<double, 8, 9>> D_svd(D, Eigen::ComputeFullU | Eigen::ComputeFullV);
        Eigen::Matrix<double, 9, 1> e = D_svd.matrixV().col(8);
        Eigen::Matrix<double, 3, 3> E_raw;
        E_raw << e(0), e(1), e(2),
                 e(3), e(4), e(5),
                 e(6), e(7), e(8);
        Eigen::JacobiSVD<Eigen::Matrix<double, 3, 3>> E_svd(E_raw, Eigen::ComputeFullU | Eigen::ComputeFullV);
        Eigen::Matrix<double, 3, 1> s = E_svd.singularValues();
        s(0) = 0.5 * (s(0) + s(1));
        s(1) = s(0);
        s(2) = 0.;

        std::vector<Eigen::Matrix<double, 3, 3>> results(1);
        results[0] = E_svd.matrixU() * s.asDiagonal() * E_svd.matrixV().transpose();
        return results;
    }
#endif


    bool Estimator::compute_essential_matrix(Mat33 &R, Vec3 &t, const std::shared_ptr<Frame> &frame_i, const std::shared_ptr<Frame> &frame_j,
                                             bool is_init_landmark, unsigned int max_iters) {
        constexpr static double th_e2 = 0.3841 / FOCAL_LENGTH;
        constexpr static double th_score = 0.5991 / FOCAL_LENGTH;
        constexpr static unsigned long th_count = 20;

        unsigned long max_num_points = max(frame_i->features.size(), frame_j->features.size());
        vector<pair<const Vec3*, const Vec3*>> match_pairs;
        match_pairs.reserve(max_num_points);
        vector<unsigned long> landmark_ids;
        landmark_ids.reserve(max_num_points);

        // 找出匹配对
        for (auto &feature_i : frame_i->features) {
            unsigned long landmark_id = feature_i.first;
            auto &&feature_j = frame_j->features.find(landmark_id);
            if (feature_j == frame_j->features.end()) {
                continue;
            }
            match_pairs.emplace_back(&feature_i.second->points[0], &feature_j->second->points[0]);
            landmark_ids.emplace_back(landmark_id);
        }

        // 匹配对必须大于一定数量
        unsigned long num_points = match_pairs.size();
        if (num_points < th_count) {
            return false;
        }

        // 计算平均视差
        double average_parallax = 0.;
        for (auto &match_pair : match_pairs) {
            double du = match_pair.first->x() - match_pair.second->x();
            double dv = match_pair.first->y() - match_pair.second->y();
            average_parallax += max(abs(du), abs(dv));
        }
        average_parallax /= double(num_points);
#ifdef PRINT_INFO
        std::cout << "average_parallax = " << average_parallax << std::endl;
#endif

        // 平均视差必须大于一定值
        if (average_parallax * 460. < 60.) {
            return false;
        }

        // 归一化变换参数
        Mat33 Ti, Tj;
        double meas_x_i = 0., meas_y_i = 0.;
        double dev_x_i = 0., dev_y_i = 0.;
        double meas_x_j = 0., meas_y_j = 0.;
        double dev_x_j = 0., dev_y_j = 0.;

        // 计算均值
        for (auto &match_pair : match_pairs) {
            meas_x_i += match_pair.first->x();
            meas_y_i += match_pair.first->y();
            meas_x_j += match_pair.second->x();
            meas_y_j += match_pair.second->y();
        }
        meas_x_i /= double(num_points);
        meas_y_i /= double(num_points);
        meas_x_j /= double(num_points);
        meas_y_j /= double(num_points);

        // 计算Dev
        for (auto &match_pair : match_pairs) {
            dev_x_i += abs(match_pair.first->x() - meas_x_i);
            dev_y_i += abs(match_pair.first->y() - meas_y_i);
            dev_x_j += abs(match_pair.second->x() - meas_x_j);
            dev_y_j += abs(match_pair.second->y() - meas_y_j);
        }
        dev_x_i /= double(num_points);
        dev_y_i /= double(num_points);
        dev_x_j /= double(num_points);
        dev_y_j /= double(num_points);

        Ti << 1. / dev_x_i, 0., -meas_x_i / dev_x_i,
                0., 1. / dev_y_i, -meas_y_i / dev_y_i,
                0., 0., 1.;
        Tj << 1. / dev_x_j, 0., -meas_x_j / dev_x_j,
                0., 1. / dev_y_j, -meas_y_j / dev_y_j,
                0., 0., 1.;

        // 归一化后的点
        vector<pair<Vec2, Vec2>> normal_match_pairs(num_points);
        for (unsigned long k = 0; k < num_points; ++k) {
            normal_match_pairs[k].first.x() = (match_pairs[k].first->x() - meas_x_i) / dev_x_i;
            normal_match_pairs[k].first.y() = (match_pairs[k].first->y() - meas_y_i) / dev_y_i;
            normal_match_pairs[k].second.x() = (match_pairs[k].second->x() - meas_x_j) / dev_x_j;
            normal_match_pairs[k].second.y() = (match_pairs[k].second->y() - meas_y_j) / dev_y_j;
        }

        // 构造随机index batch
        std::random_device rd;
        std::mt19937 gen(rd());
        vector<array<unsigned long, N_POINTS>> point_indices_set(max_iters);    // TODO: 设为静态变量
        array<unsigned long, N_POINTS> local_index_map {};
        
        vector<unsigned long> global_index_map(num_points);
        for (unsigned long k = 0; k < num_points; ++k) {
            global_index_map[k] = k;
        }

        for (unsigned int n = 0; n < max_iters; ++n) {
            for (unsigned int k = 0; k < N_POINTS; ++k) {
                std::uniform_int_distribution<unsigned int> dist(0, global_index_map.size() - 1);
                unsigned int rand_i = dist(gen);
                auto index = global_index_map[rand_i];
                point_indices_set[n][k] = index;
                local_index_map[k] = index;

                global_index_map[rand_i] = global_index_map.back();
                global_index_map.pop_back();
            }

            for (unsigned int k = 0; k < N_POINTS; ++k) {
                global_index_map.emplace_back(local_index_map[k]);
            }
        }

        // TODO: 使用多线程
        // RANSAC: 计算本质矩阵
        Mat33 best_E;
        double best_score = 0.;
        unsigned int best_index = 0;
        unsigned int curr_index = 1;
        unsigned long num_outliers[2] = {num_points, num_points};
        vector<bool> is_outliers[2] {vector<bool>(num_points, true), vector<bool>(num_points, true)};
        for (unsigned int n = 0; n < max_iters; ++n) {
            // 极线误差矩阵
            Eigen::Matrix<double, N_POINTS, 9> D;
            for (unsigned int k = 0; k < N_POINTS; ++k) {
                unsigned int index = point_indices_set[n][k];
                double u1 = normal_match_pairs[index].first.x();
                double v1 = normal_match_pairs[index].first.y();
                double u2 = normal_match_pairs[index].second.x();
                double v2 = normal_match_pairs[index].second.y();
                D(k, 0) = u1 * u2;
                D(k, 1) = u1 * v2;
                D(k, 2) = u1;
                D(k, 3) = v1 * u2;
                D(k, 4) = v1 * v2;
                D(k, 5) = v1;
                D(k, 6) = u2;
                D(k, 7) = v2;
                D(k, 8) = 1.;
            }

            // 求解本质矩阵
            auto &&results = solve_essential(D);
            for (auto &E : results) {
                double e00 = E(0, 0), e01 = E(0, 1), e02 = E(0, 2),
                       e10 = E(1, 0), e11 = E(1, 1), e12 = E(1, 2),
                       e20 = E(2, 0), e21 = E(2, 1), e22 = E(2, 2);

                // 计算分数
                double score = 0.;
                num_outliers[curr_index] = 0;
                for (unsigned long k = 0; k < num_points; ++k) {
                    bool is_outlier = false;

                    double u1 = normal_match_pairs[k].first.x();
                    double v1 = normal_match_pairs[k].first.y();
                    double u2 = normal_match_pairs[k].second.x();
                    double v2 = normal_match_pairs[k].second.y();

                    double a = u1 * e00 + v1 * e10 + e20;
                    double b = u1 * e01 + v1 * e11 + e21;
                    double c = u1 * e02 + v1 * e12 + e22;
                    double num = a * u2 + b * v2 + c;
                    double e2 = num * num / (a * a + b * b);
                    if (e2 > th_e2) {
                        is_outlier = true;
                    } else {
                        score += th_score - e2;
                    }

                    a = u2 * e00 + v2 * e01 + e02;
                    b = u2 * e10 + v2 * e11 + e12;
                    c = u2 * e20 + v2 * e21 + e22;
                    num = u1 * a + v1 * b + c;
                    e2 = num * num / (a * a + b * b);
                    if (e2 > th_e2) {
                        is_outlier = true;
                    } else {
                        score += th_score - e2;
                    }

                    is_outliers[curr_index][k] = is_outlier;
                    if (is_outlier) {
                        ++num_outliers[curr_index];
                    }
                }

                if (score > best_score) {
                    best_score = score;
                    best_E = E;
                    std::swap(curr_index, best_index);
                }
            } 
        }

        // outlier的点过多
        if (10 * num_outliers[best_index] > 2 * num_points) {
#ifdef PRINT_INFO
            std::cout << "10 * num_outliers > 5 * num_points" << std::endl;
#endif
            return false;
        }

        // 从E中还原出R, t
        best_E = Ti.transpose() * best_E * Tj;
        Eigen::JacobiSVD<Mat33> E_svd(best_E, Eigen::ComputeFullU | Eigen::ComputeFullV);
        Mat33 V = E_svd.matrixV();
        Mat33 U1 = E_svd.matrixU();

        if (V.determinant() < 0.) {
            V *= -1.;
        }
        if (U1.determinant() < 0.) {
            U1 *= -1.;
        }

        Vec3 t1 = U1.col(2);
        t1 = t1 / t1.norm();
        Vec3 t2 = -t1;

        U1.col(0).swap(U1.col(1));
        Mat33 U2 = U1;
        U1.col(1) *= -1.;
        U2.col(0) *= -1.;

        Mat33 R1 = U1 * V.transpose();
        Mat33 R2 = U2 * V.transpose();

        // 进行三角化，通过深度筛选出正确的R, t
        auto tri = [&](const Vec3 *point_i, const Vec3 *point_j, const Mat33 &R, const Vec3 &t, Vec3 &p) -> bool {
            Vec3 RTt = R.transpose() * t;
            Mat43 A;
            A.row(0) << 1., 0., -point_i->x();
            A.row(1) << 0., 1., -point_i->y();
            A.row(2) = R.col(0).transpose() - point_j->x() * R.col(2).transpose();
            A.row(3) = R.col(1).transpose() - point_j->y() * R.col(2).transpose();
            Vec4 b;
            b << 0., 0., RTt[0] - RTt[2] * point_j->x(), RTt[1] - RTt[2] * point_j->y();

//            Mat33 ATA = A.transpose() * A;
//            Vec3 ATb = A.transpose() * b;
//            auto &&ATA_ldlt = ATA.ldlt();
//            if (ATA_ldlt.info() == Eigen::Success) {
//                p = ATA_ldlt.solve(ATb);
//                return true;
//            } else {
//                return false;
//            }

            // 使用QR分解求解最小二乘问题，这样数值精度更加稳定
            p = A.fullPivHouseholderQr().solve(b);
            return true;
        };

        auto tri_all_points = [&](const Mat33 &R, const Vec3 &t, vector<pair<bool, Vec3>> &points) -> unsigned long {
            unsigned long succeed_count = 0;
            for (unsigned long k = 0; k < num_points; ++k) {
                if (is_outliers[best_index][k]) {
                    continue;
                }

                points[k].first = tri(match_pairs[k].first, match_pairs[k].second, R, t, points[k].second);
                if (!points[k].first) {
                    continue;
                }

                points[k].first = points[k].second[2] > 0.;
//                std::cout << "points[k].second[2] = " << points[k].second[2] << std::endl;
                if (!points[k].first) {
                    continue;
                }

                Vec3 pj = R.transpose() * (points[k].second - t);
                points[k].first = pj[2] > 0.;
//                std::cout << "pj[2] = " << pj[2] << std::endl;
                if (!points[k].first) {
                    continue;
                }

                ++succeed_count;
            }
            return succeed_count;
        };

        vector<pair<bool, Vec3>> points_w[4];
        unsigned long succeed_points[4];
        points_w[0].resize(num_points);
        points_w[1].resize(num_points);
        points_w[2].resize(num_points);
        points_w[3].resize(num_points);

        succeed_points[0] = tri_all_points(R1, t1, points_w[0]);
        succeed_points[1] = tri_all_points(R1, t2, points_w[1]);
        succeed_points[2] = tri_all_points(R2, t1, points_w[2]);
        succeed_points[3] = tri_all_points(R2, t2, points_w[3]);

        unsigned long max_succeed_points = max(succeed_points[0], max(succeed_points[1], max(succeed_points[2], succeed_points[3])));
        unsigned long min_succeed_points = 9 * (num_points - num_outliers[best_index]) / 10; // 至少要超过90%的点成功被三角化

        if (max_succeed_points < min_succeed_points) {
#ifdef PRINT_INFO
            std::cout << "max_succeed_points = " << max_succeed_points << ", min_succeed_points = " << min_succeed_points << std::endl;
            std::cout << "succeed_points[0] = " << succeed_points[0] << std::endl;
            std::cout << "succeed_points[1] = " << succeed_points[1] << std::endl;
            std::cout << "succeed_points[2] = " << succeed_points[2] << std::endl;
            std::cout << "succeed_points[3] = " << succeed_points[3] << std::endl;
#endif
            return false;
        }

        unsigned long lim_succeed_points = 7 * max_succeed_points / 10;
        unsigned long num_similar = 0;  // 记录有多少组解使得70%的点都能三角化
        if (succeed_points[0] > lim_succeed_points) {
            ++num_similar;
        }
        if (succeed_points[1] > lim_succeed_points) {
            ++num_similar;
        }
        if (succeed_points[2] > lim_succeed_points) {
            ++num_similar;
        }
        if (succeed_points[3] > lim_succeed_points) {
            ++num_similar;
        }
        if (num_similar > 1) {  // 不允许超过1组解使得70%的点都能三角化
#ifdef PRINT_INFO
            std::cout << "num_similar > 1" << std::endl;
#endif
            return false;
        }

        unsigned int which_case;
        if (succeed_points[0] == max_succeed_points) {
            which_case = 0;
            R = R1;
            t = t1;
        } else if (succeed_points[1] == max_succeed_points) {
            which_case = 1;
            R = R1;
            t = t2;
        } else if (succeed_points[2] == max_succeed_points) {
            which_case = 2;
            R = R2;
            t = t1;
        } else {
            which_case = 3;
            R = R2;
            t = t2;
        }

        // 转到imu坐标系
        Qd q12;
        Vec3 t12;
        q12 = _q_ic[0] * R * _q_ic[0].inverse();
        t12 = _q_ic[0] * t - q12 * _t_ic[0] + _t_ic[0];

        R = q12;
        t = t12;

        // 把三角化的结果赋值给landmark
        if (is_init_landmark) {
            for (unsigned long k = 0; k < num_points; ++k) {
                auto &&landmark = _landmarks.find(landmark_ids[k]);
                if (landmark == _landmarks.end()) {
                    continue;
                }

                // 没有outliers以及深度为正的点才会进行赋值
                if (!is_outliers[best_index][k] && points_w[which_case][k].first) {
                    // 转到frame_i的imu系
                    landmark->second->position = _q_ic[0] * points_w[which_case][k].second + _t_ic[0];

                    landmark->second->is_triangulated = true;
                    landmark->second->is_outlier = false;
                } else {
                    landmark->second->is_triangulated = false;
                    landmark->second->is_outlier = true;
                }
            }
        }

        return true;
    }
}