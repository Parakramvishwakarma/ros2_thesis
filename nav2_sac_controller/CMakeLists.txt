cmake_minimum_required(VERSION 3.8)
project(nav2_sac_controller)

# Add compile options for warnings
if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

# Find dependencies
find_package(ament_cmake REQUIRED)
find_package(ament_cmake_ros REQUIRED)
find_package(rclcpp REQUIRED)
find_package(rclcpp_action REQUIRED)
find_package(rclcpp_lifecycle REQUIRED)
find_package(nav2_common REQUIRED)
find_package(std_msgs REQUIRED)
find_package(visualization_msgs REQUIRED)
find_package(nav2_util REQUIRED)
find_package(nav2_core REQUIRED)
find_package(nav2_msgs REQUIRED)
find_package(example_interfaces REQUIRED)
find_package(nav_msgs REQUIRED)
find_package(geometry_msgs REQUIRED)
find_package(builtin_interfaces REQUIRED)
find_package(tf2_ros REQUIRED)
find_package(nav2_costmap_2d REQUIRED)
find_package(pluginlib REQUIRED)
find_package(tf2 REQUIRED)
find_package(rosidl_default_generators REQUIRED)
find_package(custom_interfaces REQUIRED)


# add_executable(nav2_sac_controller src/nav2_sac_controller.cpp)


# Export plugin description file for pluginlib
pluginlib_export_plugin_description_file(nav2_core nav2_sac_controller.xml)

# Define the library target
add_library(nav2_sac_controller SHARED src/sac_controller.cpp)

# Set C++ standards and include directories
target_compile_features(nav2_sac_controller PUBLIC cxx_std_17)
target_include_directories(nav2_sac_controller PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
  $<INSTALL_INTERFACE:include>
)

# Link dependencies to the target
ament_target_dependencies(
  nav2_sac_controller
  "rclcpp"
  "rclcpp_action"
  "rclcpp_lifecycle"
  "nav2_common"
  "std_msgs"
  "example_interfaces"
  "visualization_msgs"
  "nav2_util"
  "nav2_core"
  "nav2_msgs"
  "nav_msgs"
  "geometry_msgs"
  "builtin_interfaces"
  "tf2_ros"
  "nav2_costmap_2d"
  "pluginlib"
  "tf2"
  "custom_interfaces"
)

# Set compile definitions
target_compile_definitions(nav2_sac_controller PRIVATE "NAV2_SAC_CONTROLLER_BUILDING_LIBRARY")

# Install include directory and library
install(
  DIRECTORY include/
  DESTINATION include
)
install(
  TARGETS nav2_sac_controller
  EXPORT export_${PROJECT_NAME}
  ARCHIVE DESTINATION lib
  LIBRARY DESTINATION lib
  RUNTIME DESTINATION bin
)

# Install launch and config directories
install(DIRECTORY
  launch
  config
  DESTINATION share/${PROJECT_NAME}/
)

# Setup testing if enabled
if(BUILD_TESTING)
  find_package(ament_lint_auto REQUIRED)
  set(ament_cmake_copyright_FOUND TRUE) # Adjust these settings if copyright checks are needed
  set(ament_cmake_cpplint_FOUND TRUE)
  ament_lint_auto_find_test_dependencies()
endif()

# Export package information
ament_export_include_directories(include)
ament_export_libraries(nav2_sac_controller)
ament_export_targets(export_${PROJECT_NAME})

ament_package()
