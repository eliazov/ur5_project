#include <rclcpp/rclcpp.hpp>
#include <control_msgs/action/follow_joint_trajectory.hpp>
#include <control_msgs/msg/joint_tolerance.hpp>
#include <trajectory_msgs/msg/joint_trajectory.hpp>
#include <rclcpp_action/rclcpp_action.hpp>
#include <sensor_msgs/msg/joint_state.hpp>
#include <std_srvs/srv/trigger.hpp>
#include <geometry_msgs/msg/pose.hpp>
#include <std_msgs/msg/string.hpp>
#include <Eigen/Dense>
#include <chrono>
#include <vector>
#include <cmath>

using namespace std::chrono_literals;

// Custom service messages (you'll need to create these in your package)
// For now, I'll use basic geometry_msgs
struct PickPlaceCommand
{
    geometry_msgs::msg::Pose pick_pose;
    geometry_msgs::msg::Pose place_pose;
    std::string object_id;
};

class MotionPlannerNode : public rclcpp::Node
{
public:
    using FollowJointTrajectory = control_msgs::action::FollowJointTrajectory;
    using GoalHandleFollowJointTrajectory = rclcpp_action::ClientGoalHandle<FollowJointTrajectory>;

    MotionPlannerNode() : Node("motion_planner_node")
    {
        node_initialized_ = false;
        is_busy_ = false;

        // Define joint names
        joint_names_ = {"shoulder_pan_joint", "shoulder_lift_joint", "elbow_joint",
                        "wrist_1_joint", "wrist_2_joint", "wrist_3_joint"};

        // Create action client
        action_client_ = rclcpp_action::create_client<FollowJointTrajectory>(
            this, "/scaled_joint_trajectory_controller/follow_joint_trajectory");

        // Subscribe to joint states
        subscription_ = this->create_subscription<sensor_msgs::msg::JointState>(
            "joint_states", 10, std::bind(&MotionPlannerNode::joint_state_callback, this, std::placeholders::_1));

        // Subscribe to pick-place commands from high-level planner
        command_subscription_ = this->create_subscription<geometry_msgs::msg::Pose>(
            "/pick_place_command", 10, std::bind(&MotionPlannerNode::pick_place_command_callback, this, std::placeholders::_1));

        // Publisher for status updates
        status_publisher_ = this->create_publisher<std_msgs::msg::String>("/motion_status", 10);

        // Create gripper service clients
        open_gripper_client_ = this->create_client<std_srvs::srv::Trigger>("open_gripper");
        close_gripper_client_ = this->create_client<std_srvs::srv::Trigger>("close_gripper");

        // Wait for action server
        if (!action_client_->wait_for_action_server(10s))
        {
            RCLCPP_ERROR(this->get_logger(), "Action server not available after waiting");
            rclcpp::shutdown();
            return;
        }

        // Wait for gripper services (optional)
        if (!open_gripper_client_->wait_for_service(5s))
        {
            RCLCPP_WARN(this->get_logger(), "Gripper service not available, continuing without gripper control");
        }

        time_between_points_ = 0.3;
        RCLCPP_INFO(this->get_logger(), "OK");
        RCLCPP_INFO(this->get_logger(), "Motion Planner Node started. Waiting for joint states and commands...");
    }

private:
    rclcpp_action::Client<FollowJointTrajectory>::SharedPtr action_client_;
    rclcpp::Subscription<sensor_msgs::msg::JointState>::SharedPtr subscription_;
    rclcpp::Subscription<geometry_msgs::msg::Pose>::SharedPtr command_subscription_;
    rclcpp::Publisher<std_msgs::msg::String>::SharedPtr status_publisher_;
    rclcpp::Client<std_srvs::srv::Trigger>::SharedPtr open_gripper_client_;
    rclcpp::Client<std_srvs::srv::Trigger>::SharedPtr close_gripper_client_;

    double time_between_points_;
    std::vector<trajectory_msgs::msg::JointTrajectory> trajectories_;
    size_t current_trajectory_index_;
    std::vector<std::string> joint_names_;
    std::vector<double> current_joint_positions_;
    std::vector<double> home_position_;
    bool node_initialized_;
    bool is_busy_;

    // Current pick-place command
    PickPlaceCommand current_command_;
    enum class MotionState
    {
        IDLE,
        MOVING_TO_PICK,
        PICKING,
        MOVING_TO_PLACE,
        PLACING,
        RETURNING_HOME
    };
    MotionState current_state_ = MotionState::IDLE;

    void joint_state_callback(const sensor_msgs::msg::JointState::SharedPtr msg)
    {
        // Check for NaN values
        for (size_t i = 0; i < msg->position.size(); i++)
        {
            if (std::isnan(msg->position[i]))
            {
                RCLCPP_WARN(this->get_logger(), "Joint state message contains NaN values");
                return;
            }
        }

        if (msg->position.size() >= 6 && !node_initialized_)
        {
            RCLCPP_INFO(this->get_logger(), "Initializing motion planner node");
            node_initialized_ = true;

            // Extract current joint positions
            current_joint_positions_.clear();
            for (const auto &joint_name : joint_names_)
            {
                for (size_t j = 0; j < msg->name.size(); j++)
                {
                    if (msg->name[j] == joint_name)
                    {
                        current_joint_positions_.push_back(msg->position[j]);
                        break;
                    }
                }
            }

            // Set home position (current position at startup)
            home_position_ = current_joint_positions_;

            RCLCPP_INFO(this->get_logger(), "Node initialized at position: %.3f %.3f %.3f %.3f %.3f %.3f",
                        current_joint_positions_[0], current_joint_positions_[1], current_joint_positions_[2],
                        current_joint_positions_[3], current_joint_positions_[4], current_joint_positions_[5]);

            RCLCPP_INFO(this->get_logger(), "OKX2");
            publish_status("READY");
        }
        else if (node_initialized_)
        {
            // Update current position
            for (const auto &joint_name : joint_names_)
            {
                for (size_t j = 0; j < msg->name.size(); j++)
                {
                    if (msg->name[j] == joint_name)
                    {
                        current_joint_positions_[j] = msg->position[j];
                        break;
                    }
                }
            }
        }
    }

    void pick_place_command_callback(const geometry_msgs::msg::Pose::SharedPtr msg)
    {
        if (!node_initialized_)
        {
            RCLCPP_WARN(this->get_logger(), "Node not initialized yet, ignoring command");
            return;
        }

        if (is_busy_)
        {
            RCLCPP_WARN(this->get_logger(), "Motion planner is busy, ignoring new command");
            return;
        }

        RCLCPP_INFO(this->get_logger(), "Received pick-place command");

        // For this example, we'll use the pose as pick position and derive place position
        // In a real implementation, you'd receive both pick and place poses
        current_command_.pick_pose = *msg;
        current_command_.place_pose = *msg;
        current_command_.place_pose.position.x += 0.2; // Offset for place position
        current_command_.object_id = "object_" + std::to_string(msg->position.x);

        execute_pick_place_sequence();
    }

    void execute_pick_place_sequence()
    {
        is_busy_ = true;
        current_state_ = MotionState::MOVING_TO_PICK;

        trajectories_.clear();

        // Generate complete pick-place sequence
        generate_pick_place_trajectories();

        // Start execution
        current_trajectory_index_ = 0;
        send_next_trajectory();
    }

    void generate_pick_place_trajectories()
    {
        RCLCPP_INFO(this->get_logger(), "Generating pick-place trajectories");

        // Convert poses to joint configurations (inverse kinematics)
        // For this example, I'll use simple pre-defined positions
        // In a real implementation, you'd use proper inverse kinematics

        std::vector<double> pick_approach_config = pose_to_joint_config(current_command_.pick_pose, true);
        std::vector<double> pick_config = pose_to_joint_config(current_command_.pick_pose, false);
        std::vector<double> place_approach_config = pose_to_joint_config(current_command_.place_pose, true);
        std::vector<double> place_config = pose_to_joint_config(current_command_.place_pose, false);

        // 1. Move to pick approach position
        trajectory_msgs::msg::JointTrajectory traj1 = generate_trajectory_segment(
            current_joint_positions_, pick_approach_config, 15, 3.0);
        trajectories_.push_back(traj1);

        // 2. Move down to pick position
        trajectory_msgs::msg::JointTrajectory traj2 = generate_trajectory_segment(
            pick_approach_config, pick_config, 10, 2.0);
        trajectories_.push_back(traj2);

        // 3. Move up from pick position
        trajectory_msgs::msg::JointTrajectory traj3 = generate_trajectory_segment(
            pick_config, pick_approach_config, 10, 2.0);
        trajectories_.push_back(traj3);

        // 4. Move to place approach position
        trajectory_msgs::msg::JointTrajectory traj4 = generate_trajectory_segment(
            pick_approach_config, place_approach_config, 15, 3.0);
        trajectories_.push_back(traj4);

        // 5. Move down to place position
        trajectory_msgs::msg::JointTrajectory traj5 = generate_trajectory_segment(
            place_approach_config, place_config, 10, 2.0);
        trajectories_.push_back(traj5);

        // 6. Move up from place position
        trajectory_msgs::msg::JointTrajectory traj6 = generate_trajectory_segment(
            place_config, place_approach_config, 10, 2.0);
        trajectories_.push_back(traj6);

        // 7. Return to home position
        trajectory_msgs::msg::JointTrajectory traj7 = generate_trajectory_segment(
            place_approach_config, home_position_, 15, 3.0);
        trajectories_.push_back(traj7);

        RCLCPP_INFO(this->get_logger(), "Generated %zu trajectories for pick-place sequence", trajectories_.size());
    }

    std::vector<double> pose_to_joint_config(const geometry_msgs::msg::Pose &pose, bool is_approach)
    {
        // Simplified inverse kinematics - replace with proper IK solver
        // This is just an example implementation
        std::vector<double> config = home_position_;

        // Simple mapping based on pose position (this is NOT proper inverse kinematics!)
        config[0] = atan2(pose.position.y, pose.position.x); // Base rotation
        config[1] = -M_PI_2 + (pose.position.z - 0.1) * 0.5; // Shoulder lift based on height
        config[2] = -M_PI_4;                                 // Elbow
        config[3] = -M_PI_2;                                 // Wrist 1
        config[4] = 0.0;                                     // Wrist 2
        config[5] = 0.0;                                     // Wrist 3

        if (is_approach)
        {
            // Approach position is higher
            config[1] += 0.2;
        }

        return config;
    }

    trajectory_msgs::msg::JointTrajectory generate_trajectory_segment(
        const std::vector<double> &start_config,
        const std::vector<double> &end_config,
        int num_points,
        double duration_seconds)
    {
        trajectory_msgs::msg::JointTrajectory traj_msg;
        traj_msg.joint_names = joint_names_;

        for (int i = 0; i <= num_points; i++)
        {
            trajectory_msgs::msg::JointTrajectoryPoint point;
            double t = (i * duration_seconds) / num_points;

            for (size_t j = 0; j < start_config.size(); j++)
            {
                double interpolation_factor = (num_points == 0) ? 1.0 : static_cast<double>(i) / num_points;
                double interpolated_position = start_config[j] + interpolation_factor * (end_config[j] - start_config[j]);
                point.positions.push_back(interpolated_position);
            }

            point.velocities.resize(joint_names_.size(), 0.0);
            point.accelerations.resize(joint_names_.size(), 0.0);
            point.time_from_start = rclcpp::Duration::from_seconds(t);
            traj_msg.points.push_back(point);
        }

        return traj_msg;
    }

    void send_next_trajectory()
    {
        if (current_trajectory_index_ >= trajectories_.size())
        {
            RCLCPP_INFO(this->get_logger(), "Pick-place sequence completed successfully");
            is_busy_ = false;
            current_state_ = MotionState::IDLE;
            publish_status("READY");
            return;
        }

        // Update state based on trajectory index
        update_motion_state();

        auto goal_msg = FollowJointTrajectory::Goal();
        goal_msg.trajectory = trajectories_[current_trajectory_index_];
        goal_msg.goal_time_tolerance.nanosec = 500000000;

        RCLCPP_INFO(this->get_logger(), "Executing trajectory %zu/%zu - State: %s",
                    current_trajectory_index_ + 1, trajectories_.size(),
                    motion_state_to_string(current_state_).c_str());

        auto send_goal_options = rclcpp_action::Client<FollowJointTrajectory>::SendGoalOptions();

        send_goal_options.goal_response_callback =
            [this](const GoalHandleFollowJointTrajectory::SharedPtr &goal_handle)
        {
            if (!goal_handle)
            {
                RCLCPP_ERROR(this->get_logger(), "Goal was rejected by the server");
                publish_status("ERROR");
            }
        };

        send_goal_options.result_callback =
            [this](const GoalHandleFollowJointTrajectory::WrappedResult &result)
        {
            switch (result.code)
            {
            case rclcpp_action::ResultCode::SUCCEEDED:
                handle_trajectory_success();
                break;
            case rclcpp_action::ResultCode::ABORTED:
                RCLCPP_ERROR(this->get_logger(), "Goal was aborted");
                publish_status("ERROR");
                is_busy_ = false;
                break;
            case rclcpp_action::ResultCode::CANCELED:
                RCLCPP_WARN(this->get_logger(), "Goal was canceled");
                publish_status("CANCELED");
                is_busy_ = false;
                break;
            default:
                RCLCPP_ERROR(this->get_logger(), "Unknown result code");
                publish_status("ERROR");
                is_busy_ = false;
                break;
            }
        };

        action_client_->async_send_goal(goal_msg, send_goal_options);
    }

    void update_motion_state()
    {
        switch (current_trajectory_index_)
        {
        case 0:
            current_state_ = MotionState::MOVING_TO_PICK;
            break;
        case 1:
            current_state_ = MotionState::PICKING;
            break;
        case 2:
            current_state_ = MotionState::PICKING;
            break;
        case 3:
            current_state_ = MotionState::MOVING_TO_PLACE;
            break;
        case 4:
            current_state_ = MotionState::PLACING;
            break;
        case 5:
            current_state_ = MotionState::PLACING;
            break;
        case 6:
            current_state_ = MotionState::RETURNING_HOME;
            break;
        default:
            current_state_ = MotionState::IDLE;
            break;
        }
    }

    void handle_trajectory_success()
    {
        // Handle gripper operations based on current state
        handle_gripper_operations();

        // Move to next trajectory
        current_trajectory_index_++;

        // Small delay between trajectories
        std::this_thread::sleep_for(300ms);

        // Send next trajectory
        send_next_trajectory();
    }

    void handle_gripper_operations()
    {
        switch (current_state_)
        {
        case MotionState::MOVING_TO_PICK:
            call_gripper_service(open_gripper_client_, "Opening gripper for pick");
            break;
        case MotionState::PICKING:
            if (current_trajectory_index_ == 1) // After moving down to pick
            {
                call_gripper_service(close_gripper_client_, "Closing gripper to grasp object");
            }
            break;
        case MotionState::PLACING:
            if (current_trajectory_index_ == 4) // After moving down to place
            {
                call_gripper_service(open_gripper_client_, "Opening gripper to release object");
            }
            break;
        default:
            break;
        }
    }

    void call_gripper_service(rclcpp::Client<std_srvs::srv::Trigger>::SharedPtr client, const std::string &action_name)
    {
        if (!client || !client->service_is_ready())
        {
            RCLCPP_WARN(this->get_logger(), "Gripper service not available, skipping %s", action_name.c_str());
            return;
        }

        RCLCPP_INFO(this->get_logger(), "%s", action_name.c_str());
        auto request = std::make_shared<std_srvs::srv::Trigger::Request>();
        client->async_send_request(request);
        std::this_thread::sleep_for(1s);
    }

    void publish_status(const std::string &status)
    {
        auto msg = std_msgs::msg::String();
        msg.data = status;
        status_publisher_->publish(msg);
        RCLCPP_INFO(this->get_logger(), "Published motion status: %s", status.c_str()); // <--- AGGIUNGI QUESTO LOG
    }

    std::string motion_state_to_string(MotionState state)
    {
        switch (state)
        {
        case MotionState::IDLE:
            return "IDLE";
        case MotionState::MOVING_TO_PICK:
            return "MOVING_TO_PICK";
        case MotionState::PICKING:
            return "PICKING";
        case MotionState::MOVING_TO_PLACE:
            return "MOVING_TO_PLACE";
        case MotionState::PLACING:
            return "PLACING";
        case MotionState::RETURNING_HOME:
            return "RETURNING_HOME";
        default:
            return "UNKNOWN";
        }
    }
};

int main(int argc, char **argv)
{
    rclcpp::init(argc, argv);
    rclcpp::spin(std::make_shared<MotionPlannerNode>());
    rclcpp::shutdown();
    return 0;
}