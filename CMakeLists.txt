cmake_minimum_required(VERSION 2.8.3)
project(target_icp_registration)

find_package(catkin REQUIRED COMPONENTS
  pcl_conversions
  pcl_ros
  roscpp
  sensor_msgs
  tf_conversions
)

find_package(PCL 1.7 REQUIRED)

catkin_package()

include_directories(${catkin_INCLUDE_DIRS} ${PCL_INCLUDE_DIRS})
link_directories(${PCL_LIBRARY_DIRS})
add_definitions(${PCL_DEFINITIONS})

# Executables
add_executable(icp_registration src/icp_registration.cpp)
target_link_libraries (icp_registration ${PCL_LIBRARIES} ${catkin_LIBRARIES})
