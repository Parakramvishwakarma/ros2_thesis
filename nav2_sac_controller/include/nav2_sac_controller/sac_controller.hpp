/*
 * SPDX-License-Identifier: BSD-3-Clause
 *
 *  Author(s): Shrijit Singh <shrijitsingh99@gmail.com>
 *
 */

#ifndef NAV2_PURE_PURSUIT_CONTROLLER__PURE_PURSUIT_CONTROLLER_HPP_
#define NAV2_PURE_PURSUIT_CONTROLLER__PURE_PURSUIT_CONTROLLER_HPP_

#include <string>
#include <vector>
#include <memory>
#include <chrono>
#include <functional>

#include "nav2_core/controller.hpp"
#include "rclcpp/rclcpp.hpp"
#include "pluginlib/class_loader.hpp"
#include "pluginlib/class_list_macros.hpp"
#include "geometry_msgs/msg/twist_stamped.hpp"
#include "geometry_msgs/msg/pose_stamped.hpp"
#include "std_msgs/msg/float64.hpp"
#include "example_interfaces/msg/float64_multi_array.hpp"
#include "nav_msgs/msg/path.hpp"
#include "nav_msgs/msg/odometry.hpp"
#include "tf2_ros/buffer.h"
#include "nav2_costmap_2d/costmap_2d_ros.hpp"
#include "custom_interfaces/msg/observations.hpp"     // CHANGE



namespace nav2_sac_controller
{

class SACController : public nav2_core::Controller
{
public:
   SACController();
   ~SACController();

  void configure(
  const rclcpp_lifecycle::LifecycleNode::WeakPtr & parent,
  std::string name, std::shared_ptr<tf2_ros::Buffer> tf,
  std::shared_ptr<nav2_costmap_2d::Costmap2DROS> costmap_ros) override;

  void cleanup() override;
  void activate() override;
  void deactivate() override;
  void setSpeedLimit(const double & speed_limit, const bool & percentage) override;

  geometry_msgs::msg::TwistStamped computeVelocityCommands(
    const geometry_msgs::msg::PoseStamped & pose,
    const geometry_msgs::msg::Twist & velocity,
    nav2_core::GoalChecker * goal_checker) override;

  void setPlan(const nav_msgs::msg::Path & path) override;

protected:
  nav_msgs::msg::Path transformGlobalPlan(
    const geometry_msgs::msg::PoseStamped & pose,
     geometry_msgs::msg::PoseStamped & robot_pose);

  bool transformPose(
    const std::shared_ptr<tf2_ros::Buffer> tf,
    const std::string frame,
    const geometry_msgs::msg::PoseStamped & in_pose,
    geometry_msgs::msg::PoseStamped & out_pose,
    const rclcpp::Duration & transform_tolerance
  ) const;

//goal pose will just be the end of the global_plan
// hence in this function we will convert the odom frame pose to global plan and file last pose and the distance to it
  bool eucledianDistanceToGoal(
    const geometry_msgs::msg::PoseStamped & in_pose,
    const geometry_msgs::msg::PoseStamped & goal_pose,
    float & distance
  );

  bool findAngle(
    const geometry_msgs::msg::PoseStamped & in_pose,
    const geometry_msgs::msg::PoseStamped & ref_pose,
    float & angle
  );

  void actionCallback(const geometry_msgs::msg::Twist::SharedPtr msg); // Callback for the action subscriber
  void odomCallback(const nav_msgs::msg::Odometry::SharedPtr msg);





  // float getspeed();

  rclcpp_lifecycle::LifecycleNode::WeakPtr node_;
  std::shared_ptr<tf2_ros::Buffer> tf_;

  std::string plugin_name_;
  std::shared_ptr<nav2_costmap_2d::Costmap2DROS> costmap_ros_;
  rclcpp::Logger logger_ {rclcpp::get_logger("SACController")};
  rclcpp::Clock::SharedPtr clock_;

  double desired_linear_vel_;
  double lookahead_dist_;
  double max_angular_vel_;
  int offset_from_furtherest_;
  example_interfaces::msg::Float64MultiArray obeservationArray_;

  rclcpp::Duration transform_tolerance_ {0, 0};
  nav_msgs::msg::Path global_plan_;
  std::shared_ptr<rclcpp_lifecycle::LifecyclePublisher<nav_msgs::msg::Path>> global_pub_;
  rclcpp::Publisher<custom_interfaces::msg::Observations>::SharedPtr publisher_;
  rclcpp::Subscription<geometry_msgs::msg::Twist>::SharedPtr action_subscriber_; // Subscriber for actions
  float current_distance_to_target_;
  float last_distance_to_target_;
  geometry_msgs::msg::Twist latest_action_; // Store the most recent action
  rclcpp::Subscription<nav_msgs::msg::Odometry>::SharedPtr odom_subscriber_;
  geometry_msgs::msg::Twist current_velocity_;  // To store the latest velocity from /odom

};

}  // namespace nav2_SAC_controller

#endif  // NAV2_SAC_CONTROLLER__SAC_CONTROLLER_HPP_
