#include <ros/ros.h>
#include <std_msgs/String.h>
#include <chrono>
#include <cstdlib>
#include <string>
#include <vector>
#include <iostream>

#include <nlohmann/json.hpp>
using json = nlohmann::json;

struct Detection
{
    int class_id;
    std::string class_name;
    double confidence;
    int x1, y1, x2, y2;
};

class YoloJsonListener
{
public:
    YoloJsonListener(ros::NodeHandle& nh)
        : last_msg_time_(std::chrono::steady_clock::now())
        , empty_times_(0)
    {
        std::string detections_topic;
        nh.param<std::string>("detections_topic", detections_topic, std::string("/yolo/detections"));

        sub_ = nh.subscribe(detections_topic, 10, &YoloJsonListener::jsonCallback, this);
        timeout_timer_ = nh.createTimer(ros::Duration(1.0), &YoloJsonListener::timeoutCallback, this);

        ROS_INFO("YoloJsonListener started. Subscribing to: %s", detections_topic.c_str());
    }

private:
    ros::Subscriber sub_;
    std::chrono::steady_clock::time_point last_msg_time_;
    int empty_times_;
    ros::Timer timeout_timer_;

    void timeoutCallback(const ros::TimerEvent&)
    {
        const auto now = std::chrono::steady_clock::now();
        const auto idle_seconds =
            std::chrono::duration_cast<std::chrono::seconds>(now - last_msg_time_).count();

        if (idle_seconds >= 20)
        {
            ros::param::set("/stop_flag", 0);
            ROS_WARN("YOLO JSON listener timed out after %ld seconds without new detection messages.", idle_seconds);
            ros::shutdown();
        }
    }

    void jsonCallback(const std_msgs::String::ConstPtr& msg)
    {
        last_msg_time_ = std::chrono::steady_clock::now();

        const std::string& json_str = msg->data;

        try
        {
            json j = json::parse(json_str);

            std::string frame_id = "";
            double stamp_sec = 0.0;

            if (j.contains("header") && j["header"].is_object())
            {
                auto j_header = j["header"];
                if (j_header.contains("frame_id"))
                    frame_id = j_header["frame_id"].get<std::string>();
                if (j_header.contains("stamp"))
                    stamp_sec = j_header["stamp"].get<double>();
            }

            std::vector<Detection> detections;

            if (j.contains("detections") && j["detections"].is_array())
            {
                for (const auto& j_det : j["detections"])
                {
                    Detection det;

                    det.class_id  = j_det.value("class_id", -1);
                    det.class_name = j_det.value("class_name", "");
                    det.confidence = j_det.value("confidence", 0.0);

                    if (j_det.contains("bbox"))
                    {
                        const auto& j_bbox = j_det["bbox"];

                        if (j_bbox.is_array() && j_bbox.size() == 4)
                        {
                            det.x1 = j_bbox.at(0).get<int>();
                            det.y1 = j_bbox.at(1).get<int>();
                            det.x2 = j_bbox.at(2).get<int>();
                            det.y2 = j_bbox.at(3).get<int>();
                        }
                        else if (j_bbox.is_object())
                        {
                            // 兼容旧格式
                            det.x1 = j_bbox.value("x1", 0);
                            det.y1 = j_bbox.value("y1", 0);
                            det.x2 = j_bbox.value("x2", 0);
                            det.y2 = j_bbox.value("y2", 0);
                        }
                        else
                        {
                            det.x1 = det.y1 = det.x2 = det.y2 = 0;
                        }
                    }
                    else
                    {
                        det.x1 = det.y1 = det.x2 = det.y2 = 0;
                    }

                    detections.push_back(det);
                }
            }

            ROS_INFO("---- YOLO JSON received ----");
            ROS_INFO("Frame: %s, stamp: %.6f, num detections: %zu",
                     frame_id.c_str(), stamp_sec, detections.size());
//////////////////////////////////////////////////////////////////////////////////////////////
            int best_id = 0;
            if(detections.empty())
            {
                ++empty_times_;
                return;
            }
            if(empty_times_ >= 40)
            {
                ros::param::set("/stop_flag", 0);
                ROS_WARN("YOLO JSON listener received 40 consecutive empty detection messages.");
                ros::shutdown();
                return;
            }

            empty_times_ = 0;

            for (size_t i = 0; i < detections.size(); ++i)
            {
                const auto& d = detections[i];
                ROS_INFO("  #%zu: class_id=%d, class_name=%s, conf=%.3f, bbox=(%d,%d,%d,%d)",
                         i,
                         d.class_id,
                         d.class_name.c_str(),
                         d.confidence,
                         d.x1, d.y1, d.x2, d.y2);
                if(d.confidence > detections[best_id].confidence)
                {
                    best_id = i;
                }
            }
            if(detections[best_id].class_id)
            {
                ros::param::set("/stop_flag", 1);
            }
            else
            {
                ros::param::set("/stop_flag", 0);
            }
            
//////////////////////////////////////////////////////////////////////////////////////////////////
        }
        catch (const std::exception& e)
        {
            ROS_WARN("Failed to parse JSON message: %s", e.what());
        }
    }
};

int main(int argc, char** argv)
{
    ros::init(argc, argv, "yolo_json_listener");
    ros::NodeHandle nh("~");

    YoloJsonListener listener(nh);

    ros::spin();
    return 0;
}

