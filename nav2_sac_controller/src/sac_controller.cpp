/*
 * SPDX-License-Identifier: BSD-3-Clause
 *
 *  Author(s): Shrijit Singh <shrijitsingh99@gmail.com>
 *  Contributor: Pham Cong Trang <phamcongtranghd@gmail.com>
 *  Contributor: Mitchell Sayer <mitchell4408@gmail.com>
 */

#include <algorithm>
#include <string>
#include <memory>

#include "nav2_core/exceptions.hpp"
#include "nav2_util/node_utils.hpp"
#include "nav2_sac_controller/sac_controller.hpp"
#include "nav2_sac_controller/utils.hpp"
#include "nav2_util/geometry_utils.hpp"

using std::hypot;
using std::min;
using std::max;
using std::abs;
using nav2_util::declare_parameter_if_not_declared;
using nav2_util::geometry_utils::euclidean_distance;


namespace nav2_sac_controller
{

SACController::SACController()
// tf_(nullptr), costmap_(nullptr)
{
  tf_ = nullptr;
  latest_action_ = geometry_msgs::msg::Twist();  // Initialize the latest action

}

SACController::~SACController()
{
  RCLCPP_INFO(
    logger_, "Destroying plugin of type SAC Controller"
  );
}
/**
 * Find element in iterator with the minimum calculated value
 */
template<typename Iter, typename Getter>
Iter min_by(Iter begin, Iter end, Getter getCompareVal)
{
  if (begin == end) {
    return end;
  }
  auto lowest = getCompareVal(*begin);
  Iter lowest_it = begin;
  for (Iter it = ++begin; it != end; ++it) {
    auto comp = getCompareVal(*it);
    if (comp < lowest) {
      lowest = comp;
      lowest_it = it;
    }
  }
  return lowest_it;
}

void SACController::configure(
  const rclcpp_lifecycle::LifecycleNode::WeakPtr & parent,
  std::string name, std::shared_ptr<tf2_ros::Buffer> tf,
  std::shared_ptr<nav2_costmap_2d::Costmap2DROS> costmap_ros)
{
  node_ = parent;
  auto node = node_.lock();
  double transform_tolerance;

  costmap_ros_ = costmap_ros;
  tf_ = tf;
  plugin_name_ = name;
  logger_ = node->get_logger();
  clock_ = node->get_clock();

  declare_parameter_if_not_declared(
    node, plugin_name_ + ".desired_linear_vel", rclcpp::ParameterValue(
      0.2));
  declare_parameter_if_not_declared(
    node, plugin_name_ + ".lookahead_dist",
    rclcpp::ParameterValue(0.4));
  declare_parameter_if_not_declared(
    node, plugin_name_ + ".max_angular_vel", rclcpp::ParameterValue(
      1.0));
  declare_parameter_if_not_declared(
    node, plugin_name_ + ".transform_tolerance", rclcpp::ParameterValue(
      0.1));
  declare_parameter_if_not_declared(
    node, plugin_name_ + ".offset_from_furthest",
    rclcpp::ParameterValue(40));

  node->get_parameter(plugin_name_ + ".desired_linear_vel", desired_linear_vel_);
  node->get_parameter(plugin_name_ + ".offset_from_furthest", offset_from_furtherest_);
  node->get_parameter(plugin_name_ + ".lookahead_dist", lookahead_dist_);
  node->get_parameter(plugin_name_ + ".max_angular_vel", max_angular_vel_);
  node->get_parameter(plugin_name_ + ".transform_tolerance", transform_tolerance);
  transform_tolerance_ = rclcpp::Duration::from_seconds(transform_tolerance);
  global_pub_ = node->create_publisher<nav_msgs::msg::Path>("received_global_plan", 1);
  action_sub_ = node->create_subscription<geometry_msgs::msg::Twist>(
  "/action", rclcpp::QoS(10),
  std::bind(&SACController::actionCallback, this, std::placeholders::_1));
}

void SACController::actionCallback(const geometry_msgs::msg::Twist::SharedPtr msg)
{
  latest_action_ = *msg;
}

void SACController::cleanup() 
{
  RCLCPP_INFO(
    logger_,
    "Cleaning up controller: %s of type SAC_controller::SACController",
    plugin_name_.c_str());
  global_pub_.reset();
}

void SACController::activate()
{
  RCLCPP_INFO(
    logger_,
    "Activating controller: %s of type SAC_controller::SACController",
    plugin_name_.c_str());
  global_pub_->on_activate();
  RCLCPP_INFO(
    logger_,
    "global publisher is activated");
}

void SACController::deactivate()
{
  RCLCPP_INFO(
    logger_,
    "Deactivating controller: %s of type SAC_controller::SACController",
    plugin_name_.c_str());
  global_pub_->on_deactivate();
}

void SACController::setSpeedLimit(const double& speed_limit, const bool& percentage)
{
  (void) speed_limit;
  (void) percentage;
}

void SACController::computeVelocityCommands(
  const geometry_msgs::msg::PoseStamped & pose,
  const geometry_msgs::msg::Twist & velocity,
  nav2_core::GoalChecker * goal_checker)
{
  (void)goal_checker;
  (void)velocity;
  geometry_msgs::msg::PoseStamped robot_pose;
  auto transformed_plan = transformGlobalPlan(pose, robot_pose);
  double linear_vel = latest_action_.linear.x;
  double angular_vel = latest_action_.angular.z;
  // Create and publish a TwistStamped message with the desired velocity
  geometry_msgs::msg::TwistStamped cmd_vel;
  cmd_vel.header.frame_id = pose.header.frame_id;
  cmd_vel.header.stamp = clock_->now();
  cmd_vel.twist.linear.x = linear_vel;
  cmd_vel.twist.angular.z = angular_vel;
  // return cmd_vel;
}

void SACController::setPlan(const nav_msgs::msg::Path & path)
{
  global_pub_->publish(path);
  global_plan_ = path;
}

nav_msgs::msg::Path
SACController::transformGlobalPlan(
  const geometry_msgs::msg::PoseStamped & pose,
  geometry_msgs::msg::PoseStamped & robot_pose)
{
  // Original implementation taken from nav2_dwb_controller

  if (global_plan_.poses.empty()) {
    throw nav2_core::PlannerException("Received plan with zero length");
  }

  // Let's get the pose of the robot in the frame of the plan
  if (!transformPose(
      tf_, global_plan_.header.frame_id, pose,
      robot_pose, transform_tolerance_))
  {
    throw nav2_core::PlannerException("Unable to transform robot pose into global plan's frame");
  }
  // We'll discard points on the plan that are outside the local costmap
  nav2_costmap_2d::Costmap2D * costmap = costmap_ros_->getCostmap();
  double dist_threshold = std::max(costmap->getSizeInCellsX(), costmap->getSizeInCellsY()) *
    costmap->getResolution() / 2.0;

  // First find the closest pose on the path to the robot
  auto transformation_begin =
    min_by(
    global_plan_.poses.begin(), global_plan_.poses.end(),
    [&robot_pose](const geometry_msgs::msg::PoseStamped & ps) {
      return euclidean_distance(robot_pose, ps);
    });

  // From the closest point, look for the first point that's further than dist_threshold from the
  // robot. These points are definitely outside of the costmap, so we won't transform them.
  auto transformation_end = std::find_if(
    transformation_begin, end(global_plan_.poses),
    [&](const auto & global_plan_pose) {
      return euclidean_distance(robot_pose, global_plan_pose) > dist_threshold;
    });

  // Helper function for the transform below. Transforms a PoseStamped from global frame to local
  auto transformGlobalPoseToLocal = [&](const auto & global_plan_pose) {
      // We took a copy of the pose, let's lookup the transform at the current time
      geometry_msgs::msg::PoseStamped stamped_pose, transformed_pose;
      stamped_pose.header.frame_id = global_plan_.header.frame_id;
      stamped_pose.header.stamp = pose.header.stamp;
      stamped_pose.pose = global_plan_pose.pose;
      transformPose(
        tf_, costmap_ros_->getBaseFrameID(),
        stamped_pose, transformed_pose, transform_tolerance_);
      return transformed_pose;
    };

  // Transform the near part of the global plan into the robot's frame of reference.
  nav_msgs::msg::Path transformed_plan;
  std::transform(
    transformation_begin, transformation_end,
    std::back_inserter(transformed_plan.poses),
    transformGlobalPoseToLocal);
  transformed_plan.header.frame_id = costmap_ros_->getBaseFrameID();
  transformed_plan.header.stamp = pose.header.stamp;

  // Remove the portion of the global plan that we've already passed so we don't
  // process it on the next iteration (this is called path pruning)
  global_plan_.poses.erase(begin(global_plan_.poses), transformation_begin);
  global_pub_->publish(transformed_plan);

  if (transformed_plan.poses.empty()) {
    throw nav2_core::PlannerException("Resulting plan has 0 poses in it.");
  }

  return transformed_plan;
}

bool SACController::transformPose(
  const std::shared_ptr<tf2_ros::Buffer> tf,
  const std::string frame,
  const geometry_msgs::msg::PoseStamped & in_pose,
  geometry_msgs::msg::PoseStamped & out_pose,
  const rclcpp::Duration & transform_tolerance
) const
{

  if (in_pose.header.frame_id == frame) {
    out_pose = in_pose;
    return true;
  }

  try {
    tf->transform(in_pose, out_pose, frame);
    return true;
  } catch (tf2::ExtrapolationException & ex) {
    auto transform = tf->lookupTransform(
      frame,
      in_pose.header.frame_id,
      tf2::TimePointZero
    );
    if (
      (rclcpp::Time(in_pose.header.stamp) - rclcpp::Time(transform.header.stamp)) >
      transform_tolerance)
    {
      RCLCPP_ERROR(
        rclcpp::get_logger("tf_help"),
        "Transform data too old when converting from %s to %s",
        in_pose.header.frame_id.c_str(),
        frame.c_str()
      );
      RCLCPP_ERROR(
        rclcpp::get_logger("tf_help"),
        "Data time: %ds %uns, Transform time: %ds %uns",
        in_pose.header.stamp.sec,
        in_pose.header.stamp.nanosec,
        transform.header.stamp.sec,
        transform.header.stamp.nanosec
      );
      return false;
    } else {
      tf2::doTransform(in_pose, out_pose, transform);
      return true;
    }
  } catch (tf2::TransformException & ex) {
    RCLCPP_ERROR(
      rclcpp::get_logger("tf_help"),
      "Exception in transformPose: %s",
      ex.what()
    );
    return false;
  }
  return false;
}

}  // namespace nav2_sac_controller

// Register this controller as a nav2_core plugin
#include "pluginlib/class_list_macros.hpp"
PLUGINLIB_EXPORT_CLASS(nav2_sac_controller::SACController, nav2_core::Controller)