// Copyright (c) 2022 Samsung Research America, @artofnothingness Alexey Budyakov
// Copyright (c) 2023 Open Navigation LLC
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.


#include <algorithm>
#include <chrono>
#include <string>
#include <limits>
#include <memory>
#include <vector>



#include "angles/angles.h"

#include "tf2/utils.h"
#include "tf2_geometry_msgs/tf2_geometry_msgs.hpp"

#include "geometry_msgs/msg/twist_stamped.hpp"
#include "nav_msgs/msg/path.hpp"
#include "visualization_msgs/msg/marker_array.hpp"

#include "rclcpp/rclcpp.hpp"
#include "rclcpp_lifecycle/lifecycle_node.hpp"

#include "nav2_util/node_utils.hpp"
#include "nav2_core/goal_checker.hpp"
#include "geometry_msgs/msg/twist_stamped.hpp"
#include "geometry_msgs/msg/pose_stamped.hpp"
#include "std_msgs/msg/float64.hpp"
#include "nav_msgs/msg/path.hpp"
#include "nav2_costmap_2d/costmap_2d_ros.hpp"



#define M_PIF 3.141592653589793238462643383279502884e+00F
#define M_PIF_2 1.5707963267948966e+00F

namespace utils
{
 
    /**
     * @brief evaluate angle from pose (have angle) to point (no angle)
     * @param pose pose
     * @param point_x Point to find angle relative to X axis
     * @param point_y Point to find angle relative to Y axis
     * @param forward_preference If reversing direction is valid
     * @return Angle between two points
     */
    inline float posePointAngle(
    const geometry_msgs::msg::Pose & pose, const geometry_msgs::msg::Pose & refPose, bool forward_preference )
    {
        float pose_x = pose.position.x;
        float pose_y = pose.position.y;
        float pose_yaw = tf2::getYaw(pose.orientation);

        float point_y = refPose.position.y;
        float point_x = refPose.position.x;

        float yaw = atan2f(point_y - pose_y, point_x - pose_x);

        // If no preference for forward, return smallest angle either in heading or 180 of heading
        if (!forward_preference) {
            return std::min(
            fabs(angles::shortest_angular_distance(yaw, pose_yaw)),
            fabs(angles::shortest_angular_distance(yaw, angles::normalize_angle(pose_yaw + M_PIF))));
        }

        return fabs(angles::shortest_angular_distance(yaw, pose_yaw));
    }
}

