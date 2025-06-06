#include <rclcpp/rclcpp.hpp>
#include <geometry_msgs/msg/pose.hpp>
#include <geometry_msgs/msg/pose_array.hpp>
#include <std_msgs/msg/string.hpp>
#include <std_srvs/srv/trigger.hpp>
#include <vector>
#include <map>
#include <string>
#include <queue>
#include <chrono>

using namespace std::chrono_literals;

struct DetectedObject
{
    std::string class_name;
    geometry_msgs::msg::Pose pose;
    std::string object_id;
    bool processed = false;
};

struct TargetLocation
{
    std::string class_name;
    geometry_msgs::msg::Pose pose;
    bool occupied = false;
};

class HighLevelPlannerNode : public rclcpp::Node
{
public:
    HighLevelPlannerNode() : Node("high_level_planner_node")
    {
        planning_active_ = false;
        motion_ready_ = true;
        processing_object_ = false;

        // Subscribe to vision system for detected objects
        objects_subscription_ = this->create_subscription<geometry_msgs::msg::PoseArray>(
            "/detected_objects", 10,
            std::bind(&HighLevelPlannerNode::objects_callback, this, std::placeholders::_1));

        // Subscribe to object classification (if separate from poses)
        classification_subscription_ = this->create_subscription<std_msgs::msg::String>(
            "/object_classification", 10,
            std::bind(&HighLevelPlannerNode::classification_callback, this, std::placeholders::_1));

        // Subscribe to motion planner status
        motion_status_subscription_ = this->create_subscription<std_msgs::msg::String>(
            "/motion_status", 10,
            std::bind(&HighLevelPlannerNode::motion_status_callback, this, std::placeholders::_1));

        // Publisher for sending commands to motion planner
        motion_command_publisher_ = this->create_publisher<geometry_msgs::msg::Pose>(
            "/pick_place_command", 10);

        // Publisher for planning status
        status_publisher_ = this->create_publisher<std_msgs::msg::String>(
            "/planning_status", 10);

        // Service to start planning process
        start_planning_service_ = this->create_service<std_srvs::srv::Trigger>(
            "/start_planning",
            std::bind(&HighLevelPlannerNode::start_planning_callback, this,
                      std::placeholders::_1, std::placeholders::_2));

        // Service to stop planning process
        stop_planning_service_ = this->create_service<std_srvs::srv::Trigger>(
            "/stop_planning",
            std::bind(&HighLevelPlannerNode::stop_planning_callback, this,
                      std::placeholders::_1, std::placeholders::_2));

        // Timer for periodic planning updates
        planning_timer_ = this->create_wall_timer(
            1s, std::bind(&HighLevelPlannerNode::planning_timer_callback, this));

        // Initialize target locations (predefined positions for each object class)
        initialize_target_locations();

        RCLCPP_INFO(this->get_logger(), "High Level Planner Node started");
        RCLCPP_INFO(this->get_logger(), "Waiting for vision system and motion planner to be ready...");

        publish_status("INITIALIZED");
    }

private:
    // Subscriptions
    rclcpp::Subscription<geometry_msgs::msg::PoseArray>::SharedPtr objects_subscription_;
    rclcpp::Subscription<std_msgs::msg::String>::SharedPtr classification_subscription_;
    rclcpp::Subscription<std_msgs::msg::String>::SharedPtr motion_status_subscription_;

    // Publishers
    rclcpp::Publisher<geometry_msgs::msg::Pose>::SharedPtr motion_command_publisher_;
    rclcpp::Publisher<std_msgs::msg::String>::SharedPtr status_publisher_;

    // Services
    rclcpp::Service<std_srvs::srv::Trigger>::SharedPtr start_planning_service_;
    rclcpp::Service<std_srvs::srv::Trigger>::SharedPtr stop_planning_service_;

    // Timer
    rclcpp::TimerBase::SharedPtr planning_timer_;

    // Data structures
    std::vector<DetectedObject> detected_objects_;
    std::map<std::string, TargetLocation> target_locations_;
    std::queue<DetectedObject> planning_queue_;

    // State variables
    bool planning_active_;
    bool motion_ready_;
    std::string current_motion_status_;
    DetectedObject current_object_;
    bool processing_object_;

    void objects_callback(const geometry_msgs::msg::PoseArray::SharedPtr msg)
    {
        RCLCPP_INFO(this->get_logger(), "Received %zu detected objects", msg->poses.size());

        // Clear previous detections
        detected_objects_.clear();

        // Process each detected pose
        for (size_t i = 0; i < msg->poses.size(); i++)
        {
            DetectedObject obj;
            obj.pose = msg->poses[i];
            obj.object_id = "object_" + std::to_string(i);
            obj.class_name = "unknown"; // Will be updated by classification
            obj.processed = false;

            detected_objects_.push_back(obj);
        }

        // Update planning queue if planning is active
        if (planning_active_)
        {
            update_planning_queue();
        }
    }

    void classification_callback(const std_msgs::msg::String::SharedPtr msg)
    {
        // Parse classification message (format: "object_id:class_name")
        std::string data = msg->data;
        size_t delimiter_pos = data.find(':');

        if (delimiter_pos != std::string::npos)
        {
            std::string object_id = data.substr(0, delimiter_pos);
            std::string class_name = data.substr(delimiter_pos + 1);

            // Update corresponding object
            for (auto &obj : detected_objects_)
            {
                if (obj.object_id == object_id)
                {
                    obj.class_name = class_name;
                    RCLCPP_INFO(this->get_logger(), "Updated object %s classification: %s",
                                object_id.c_str(), class_name.c_str());
                    break;
                }
            }
        }
    }

    void motion_status_callback(const std_msgs::msg::String::SharedPtr msg)
    {
        current_motion_status_ = msg->data;

        if (msg->data == "READY")
        {
            motion_ready_ = true;
            processing_object_ = false;
            RCLCPP_INFO(this->get_logger(), "Motion planner is ready");
        }
        else if (msg->data == "ERROR" || msg->data == "CANCELED")
        {
            motion_ready_ = true;
            processing_object_ = false;
            RCLCPP_WARN(this->get_logger(), "Motion planner reported: %s", msg->data.c_str());
        }
        else
        {
            motion_ready_ = false;
        }
    }

    void start_planning_callback(const std::shared_ptr<std_srvs::srv::Trigger::Request> request,
                                 std::shared_ptr<std_srvs::srv::Trigger::Response> response)
    {
        (void)request; // Suppress unused parameter warning

        if (planning_active_)
        {
            response->success = false;
            response->message = "Planning is already active";
            return;
        }

        planning_active_ = true;
        processing_object_ = false;

        // Clear and update planning queue
        while (!planning_queue_.empty())
            planning_queue_.pop();
        update_planning_queue();

        response->success = true;
        response->message = "Planning started successfully";

        RCLCPP_INFO(this->get_logger(), "Planning started - %zu objects in queue", planning_queue_.size());
        publish_status("PLANNING_ACTIVE");
    }

    void stop_planning_callback(const std::shared_ptr<std_srvs::srv::Trigger::Request> request,
                                std::shared_ptr<std_srvs::srv::Trigger::Response> response)
    {
        (void)request; // Suppress unused parameter warning

        planning_active_ = false;
        processing_object_ = false;

        // Clear planning queue
        while (!planning_queue_.empty())
            planning_queue_.pop();

        response->success = true;
        response->message = "Planning stopped successfully";

        RCLCPP_INFO(this->get_logger(), "Planning stopped");
        publish_status("PLANNING_STOPPED");
    }

    void planning_timer_callback()
    {
        if (!planning_active_)
        {
            return;
        }

        // Check if we can process the next object
        if (motion_ready_ && !processing_object_ && !planning_queue_.empty())
        {
            process_next_object();
        }

        // Check if planning is complete
        if (planning_queue_.empty() && !processing_object_)
        {
            if (all_objects_processed())
            {
                RCLCPP_INFO(this->get_logger(), "All objects have been processed successfully!");
                planning_active_ = false;
                publish_status("PLANNING_COMPLETE");
            }
        }
    }

    void initialize_target_locations()
    {
        // Define target locations for different object classes
        // These should match the silhouettes mentioned in the project description

        TargetLocation loc1;
        loc1.class_name = "cube";
        loc1.pose.position.x = 0.5;
        loc1.pose.position.y = 0.3;
        loc1.pose.position.z = 0.05;
        loc1.pose.orientation.w = 1.0;
        target_locations_["cube"] = loc1;

        TargetLocation loc2;
        loc2.class_name = "cylinder";
        loc2.pose.position.x = 0.5;
        loc2.pose.position.y = 0.0;
        loc2.pose.position.z = 0.05;
        loc2.pose.orientation.w = 1.0;
        target_locations_["cylinder"] = loc2;

        TargetLocation loc3;
        loc3.class_name = "sphere";
        loc3.pose.position.x = 0.5;
        loc3.pose.position.y = -0.3;
        loc3.pose.position.z = 0.05;
        loc3.pose.orientation.w = 1.0;
        target_locations_["sphere"] = loc3;

        TargetLocation loc4;
        loc4.class_name = "prism";
        loc4.pose.position.x = 0.3;
        loc4.pose.position.y = 0.3;
        loc4.pose.position.z = 0.05;
        loc4.pose.orientation.w = 1.0;
        target_locations_["prism"] = loc4;

        RCLCPP_INFO(this->get_logger(), "Initialized %zu target locations", target_locations_.size());
    }

    void update_planning_queue()
    {
        // Clear current queue
        while (!planning_queue_.empty())
            planning_queue_.pop();

        // Add unprocessed objects with known classifications to queue
        for (const auto &obj : detected_objects_)
        {
            if (!obj.processed && obj.class_name != "unknown" &&
                target_locations_.find(obj.class_name) != target_locations_.end())
            {
                planning_queue_.push(obj);
            }
        }

        RCLCPP_INFO(this->get_logger(), "Updated planning queue with %zu objects", planning_queue_.size());
    }

    void process_next_object()
    {
        if (planning_queue_.empty())
        {
            return;
        }

        current_object_ = planning_queue_.front();
        planning_queue_.pop();
        processing_object_ = true;

        RCLCPP_INFO(this->get_logger(), "Processing object %s of type %s",
                    current_object_.object_id.c_str(), current_object_.class_name.c_str());

        // Find target location for this object class
        auto target_it = target_locations_.find(current_object_.class_name);
        if (target_it == target_locations_.end())
        {
            RCLCPP_ERROR(this->get_logger(), "No target location defined for class: %s",
                         current_object_.class_name.c_str());
            processing_object_ = false;
            return;
        }

        // Check if target location is available
        if (target_it->second.occupied)
        {
            RCLCPP_WARN(this->get_logger(), "Target location for %s is occupied, skipping",
                        current_object_.class_name.c_str());
            processing_object_ = false;
            return;
        }

        // Send pick-place command to motion planner
        send_pick_place_command(current_object_.pose, target_it->second.pose);

        // Mark target as occupied
        target_it->second.occupied = true;

        // Mark object as processed
        for (auto &obj : detected_objects_)
        {
            if (obj.object_id == current_object_.object_id)
            {
                obj.processed = true;
                break;
            }
        }

        publish_status("PROCESSING_OBJECT");
    }

    void send_pick_place_command(const geometry_msgs::msg::Pose &pick_pose,
                                 const geometry_msgs::msg::Pose &place_pose)
    {
        // For this implementation, we'll send the pick pose
        // The motion planner will handle the pick-place sequence
        // In a more sophisticated system, you might send both poses
        geometry_msgs::msg::Pose command_pose = pick_pose;

        RCLCPP_INFO(this->get_logger(), "Sending pick command for object at [%.3f, %.3f, %.3f] to [%.3f, %.3f, %.3f]",
                    pick_pose.position.x, pick_pose.position.y, pick_pose.position.z,
                    place_pose.position.x, place_pose.position.y, place_pose.position.z);

        motion_command_publisher_->publish(command_pose);
    }

    bool all_objects_processed()
    {
        for (const auto &obj : detected_objects_)
        {
            if (!obj.processed && obj.class_name != "unknown" &&
                target_locations_.find(obj.class_name) != target_locations_.end())
            {
                return false;
            }
        }
        return true;
    }

    void publish_status(const std::string &status)
    {
        auto msg = std_msgs::msg::String();
        msg.data = status;
        status_publisher_->publish(msg);
        RCLCPP_INFO(this->get_logger(), "Status: %s", status.c_str());
    }
};

int main(int argc, char **argv)
{
    rclcpp::init(argc, argv);
    rclcpp::spin(std::make_shared<HighLevelPlannerNode>());
    rclcpp::shutdown();
    return 0;
}