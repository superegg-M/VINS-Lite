find_package(Eigen3 REQUIRED)
find_package (OpenMP REQUIRED)

add_library(graph_optimization SHARED
        vertex.cpp
        edge.cpp
        problem.cpp
        steepest_descent.cpp
        gauss_newton.cpp
        levenberg_marquardt.cpp
        dog_leg.cpp
        lbfgs.cpp
        loss_function.cpp
        problem_slam.cpp)

target_include_directories(graph_optimization PUBLIC
  ${CMAKE_SOURCE_DIR}/include/
  ${CMAKE_SOURCE_DIR}/thirdparty/Sophus/
)

set_target_properties(graph_optimization PROPERTIES LINKER_LANGUAGE CXX)
target_compile_definitions(graph_optimization PRIVATE -DUSE_OPENMP)
target_compile_definitions(graph_optimization PRIVATE -DNUM_THREADS=8)

#set_target_properties(graph_optimization PROPERTIES
#        LIBRARY_OUTPUT_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
#)
#set_target_properties(graph_optimization PROPERTIES
#        LIBRARY_OUTPUT_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/
#)

#find_library(TIC_TOC_LIB tic_toc
#        ${CMAKE_BINARY_DIR}/lib/tic_toc
#)
#find_library(SO_PLUS_LIB so_plus
#        ${CMAKE_BINARY_DIR}/lib/thirdparty/Sophus/sophus
#)
#target_link_libraries(graph_optimization ${TIC_TOC_LIB} ${SO_PLUS_LIB})
target_link_libraries(graph_optimization OpenMP::OpenMP_CXX)

#target_include_directories(graph_optimization PUBLIC ${GRAPH_OPTIMIZATION_SOURCE_DIR})

#target_link_libraries(graph_optimization PUBLIC eigen3)
#target_include_directories(graph_optimization PUBLIC ${LM_SOURCE_DIR})
