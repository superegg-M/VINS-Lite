find_package(Eigen3 REQUIRED)
find_package(OpenMP REQUIRED)
find_package(OpenCV REQUIRED)

add_library(vins_stereo SHARED
	vertex/vertex_pose.cpp
        vertex/vertex_motion.cpp
        vertex/vertex_point3d.cpp
        vertex/vertex_inverse_depth.cpp
        vertex/vertex_scale.cpp
        vertex/vertex_quaternion.cpp
        vertex/vertex_spherical.cpp
        vertex/vertex_position.h
        vertex/vertex_velocity.h
        vertex/vertex_acc_bias.h
        vertex/vertex_gyro_bias.h
        vertex/vertex_vector.h

        edge/edge_imu.cpp
        edge/edge_reprojection.cpp
        edge/edge_pnp.cpp
        edge/edge_pnp_sim3.cpp
        edge/edge_epipolar.cpp
        edge/edge_align.cpp
        edge/edge_align_linear.cpp

        2d_to_2d/essential.cpp
        2d_to_2d/homography.cpp

        2d_to_3d/local_triangulate.cpp
        2d_to_3d/global_triangulate.cpp

        3d_to_2d/pnp.cpp
        3d_to_2d/dlt_pnp.cpp
        3d_to_2d/ml_pnp.cpp
        3d_to_2d/e_pnp.cpp
        3d_to_2d/iter_pnp.cpp

        bundle_adjustment/global_bundle_adjustment.cpp
        bundle_adjustment/local_bundle_adjustment.cpp

        initialize/visual_initialize.cpp
        initialize/inertial_initialize.cpp

        data_process/image_process.cpp
        data_process/imu_process.cpp

        data_structure/feature.cpp
        data_structure/frame.cpp
        data_structure/landmark.cpp
#        data_structure/map.h

        imu_integration.cpp
        estimator.cpp
)

target_include_directories(vins_stereo PUBLIC
  ${CMAKE_SOURCE_DIR}/lib/
  ${CMAKE_SOURCE_DIR}/include/
  ${CMAKE_SOURCE_DIR}/thirdparty/Sophus/
)

target_link_libraries(vins_stereo PUBLIC graph_optimization OpenMP::OpenMP_CXX ${OpenCV_LIBS})

target_compile_definitions(vins_stereo PRIVATE -DUSE_OPENMP)
target_compile_definitions(vins_stereo PRIVATE -DNUM_THREADS=4)
# target_compile_definitions(vins_stereo PRIVATE -DPRINT_INFO)
# target_compile_definitions(vins_stereo PRIVATE -DREDUCE_MOTION)
target_compile_definitions(vins_stereo PRIVATE -DNUM_OF_CAM=1)
target_compile_definitions(vins_stereo PRIVATE -DUSE_IMU)

#find_library(BACKEND_LIB graph_optimization
#        ../../lib/graph_optimization
#)

#target_link_libraries(vo_test ${BACKEND_LIB})
