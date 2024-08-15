#include "../eigen_types.h"
#include "../thirdparty/Sophus/sophus/se3.hpp"
#include <g2o/core/base_vertex.h>

namespace g2o {
    class VertexPose : public g2o::BaseVertex<6, Sophus::SE3> {
    public:
        EIGEN_MAKE_ALIGNED_OPERATOR_NEW;
        VertexPose() = default;

        virtual void setToOriginImpl() override { _estimate = Sophus::SE3(); }

        virtual void oplusImpl(const double *update) override {
            _estimate.so3() = _estimate.so3() * Sophus::SO3::exp(Eigen::Map<const Eigen::Vector3d>(&update[0]));
            _estimate.translation() += Eigen::Map<const Eigen::Vector3d>(&update[3]);
        }

        virtual bool read(std::istream &in) override { return true; }

        virtual bool write(std::ostream &out) const override { return true; }
    };


    class VertexVector3 : public g2o::BaseVertex<3, Eigen::Vector3d> {
    public:
        EIGEN_MAKE_ALIGNED_OPERATOR_NEW;
        VertexVector3() = default;

        virtual void setToOriginImpl() override { _estimate.setZero(); }

        virtual void oplusImpl(const double *update) override {
            _estimate += Eigen::Map<const Eigen::Vector3d>(update);
        }

        virtual bool read(std::istream &in) override { return true; }

        virtual bool write(std::ostream &out) const override { return true; }
    }
}