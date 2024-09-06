//
// Created by Cain on 2023/2/20.
//

#ifndef GRAPH_OPTIMIZATION_VERTEX_H
#define GRAPH_OPTIMIZATION_VERTEX_H

#include "eigen_types.h"

namespace graph_optimization {
    class Vertex {
        EIGEN_MAKE_ALIGNED_OPERATOR_NEW;

    public:
        /*!
         * 构造函数
         * @param data - 顶点数据
         * @param num_dimension - 顶点自身维度
         * @param local_dimension - 本地参数化维度, 为0时认为与本身维度一样
         */
        explicit Vertex(double *data, unsigned long num_dimension, unsigned long local_dimension=0);

        virtual ~Vertex();

        /// 返回变量维度
        unsigned long dimension() const { return  _dimension; };

        /// 返回变量本地维度
        unsigned long local_dimension() const { return _local_dimension; };

        /// 该顶点的id
        unsigned long id() const { return _id; }

        /// 返回参数的指针
        const double *get_parameters() const { return _parameters; }

        /// 返回参数的指针
        const double *parameters() const { return _parameters; }

        /// 返回参数的指针
        double *&parameters() { return _parameters; }

        /// 设置参数值
        void set_parameters(const VecX &params) { std::memcpy(_parameters, params.data(), _dimension * sizeof(double)); }

        /// 设置参数值
        void set_parameters(const double *params) { std::memcpy(_parameters, params, _dimension * sizeof(double)); }

        void save_parameters();
        bool load_parameters();

        /// 加法，可重定义
        /// 默认是向量加
        virtual void plus(const VecX &delta) = 0;
        virtual void plus(double *delta) = 0;

        /// 返回顶点的名称，在子类中实现
        virtual std::string type_info() const = 0;

        unsigned long ordering_id() const { return _ordering_id; }

        void set_ordering_id(unsigned long id) { _ordering_id = id; };

        /// 固定该点的估计值
        void set_fixed(bool fixed = true) {
            _fixed = fixed;
        }

        /// 测试该点是否被固定
        bool is_fixed() const { return _fixed; }

    protected:
        bool saved_parameters {false};
        double *_parameters;   ///< 实际存储的变量值
        double *_parameters_backup; ///< 每次迭代优化中对参数进行备份，用于回退
        unsigned long _dimension;   ///< 参数本身的维度
        unsigned long _local_dimension;   ///< 局部参数化维度
        unsigned long _id;  ///< 顶点的id，自动生成

        /// ordering id是在problem中排序后的id，用于寻找雅可比对应块
        /// ordering id带有维度信息，例如ordering_id=6则对应Hessian中的第6列
        /// 从零开始
        unsigned long _ordering_id = 0;

        bool _fixed = false;    ///< 是否固定

    private:
        static unsigned long _global_vertex_id;
    };
}

#endif //GRAPH_OPTIMIZATION_VERTEX_H
