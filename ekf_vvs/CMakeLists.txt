cmake_minimum_required(VERSION 2.8.3)
project(ekf_vvs)

## Compile as C++11, supported in ROS Kinetic and newer
 set( CMAKE_BUILD_TYPE "Debug" )
 add_compile_options(-std=c++11)

## Find catkin macros and libraries
## if COMPONENTS list like find_package(catkin REQUIRED COMPONENTS xyz)
## is used, also find other catkin packages

find_package(catkin REQUIRED COMPONENTS
  cv_bridge
  image_transport
  roscpp
  rospy
  sensor_msgs
  std_msgs
  tf
  kdl_parser
  control_blocks_msgs
  camera_pixel_msg
)

find_package(orocos_kdl REQUIRED VERSION 1.4)
MESSAGE(STATUS "kdl: ${orocos_kdl_LIBRARIES}")

find_package(PkgConfig REQUIRED)
pkg_check_modules(pinocchio REQUIRED pinocchio)

#find_package(orocos_kdl REQUIRED)
#MESSAGE(STATUS "kdl: ${orocos_kdl_LIBRARIES}")

#set(orocos_kdl_LIBRARIES ${orocos_kdl_LIBRARY})

find_package(Eigen3 REQUIRED)
MESSAGE(STATUS "Eigen: ${EIGEN3_INCLUDE_DIR}")

find_package( VISP REQUIRED )
MESSAGE(STATUS "visp: ${VISP_INCLUDE_DIRS}")

catkin_package(
  INCLUDE_DIRS
        include

  CATKIN_DEPENDS
        cv_bridge
        image_transport
        roscpp
        rospy
        sensor_msgs
        std_msgs
        tf
        kdl_parser
        control_blocks_msgs
  DEPENDS
        VISP
        EIGEN3
        orocos_kdl
)

include_directories(
  include
  ${catkin_INCLUDE_DIRS}
  ${EIGEN3_INCLUDE_DIR}
  ${orocos_kdl_LIBRARIES}
  ${VISP_INCLUDE_DIRS}
  ${pinocchio_INCLUDE_DIRS}
)
link_directories(${pinocchio_LIBRARY_DIRS})


add_library(${PROJECT_NAME}  src/ellipseSrc/ellipse_detection.cpp
        include/ellipse_detection/ellipse_detection.h
        src/ellipseSrc/EllipseDetectorYaed.cpp src/ellipseSrc/common.cpp
        include/ellipse_detection/EllipseDetectorYaed.h include//ellipse_detection/common.h
        )

FILE(GLOB_RECURSE LibFiles "include/ekf_vvs/*.hpp" "include/ekf_vvs/*.hxx"
                           "robot_model/*.urdf" "config/*.yaml" "launch/*.launch"
                           "scripts/*.py")
add_custom_target(headers SOURCES ${LibFiles})



add_executable(VS_cam_vel_input_sim_node src/VS_cam_vel_input_sim.cpp)
target_link_libraries(VS_cam_vel_input_sim_node  ${VISP_LIBRARIES} ${catkin_LIBRARIES})

add_executable(VS_est_depth_in_control_camVel_node src/VS_est_depth_in_control_camVel.cpp)
target_link_libraries(VS_est_depth_in_control_camVel_node ${VISP_LIBRARIES} ${catkin_LIBRARIES} )


add_executable(ellipse_tracking_sim_node  src/ellipse_tracking.cpp)
target_link_libraries(ellipse_tracking_sim_node  ${catkin_LIBRARIES}  ${VISP_LIBRARIES} ${PROJECT_NAME})


add_executable(VS_with_ekf_cam_frame_node  src/VS_with_ekf_cam_frame.cpp)
target_link_libraries(VS_with_ekf_cam_frame_node  ${catkin_LIBRARIES}  ${VISP_LIBRARIES} -lorocos-kdl)
target_compile_options(VS_with_ekf_cam_frame_node PUBLIC ${pinocchio_CFLAGS_OTHER})

add_executable(VS_sensor_fusion_node  src/VS_sensor_fusion.cpp)
target_link_libraries(VS_sensor_fusion_node  ${catkin_LIBRARIES}  ${VISP_LIBRARIES} -lorocos-kdl)
target_compile_options(VS_sensor_fusion_node PUBLIC ${pinocchio_CFLAGS_OTHER})




#${VISP_LIBRARIES} ${orocos_kdl_LIBRARIES} 93
#${VISP_LIBRARIES} 101
#${VISP_LIBRARIES} ${orocos_kdl_LIBRARIES}#90
#
#add_executable(inverse src/inverse.cpp)


