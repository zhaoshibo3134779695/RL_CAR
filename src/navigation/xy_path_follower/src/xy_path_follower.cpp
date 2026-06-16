#include <ros/ros.h>
#include <geometry_msgs/Twist.h>
#include <nav_msgs/Path.h>
#include <tf2_ros/transform_listener.h>
#include <tf2_geometry_msgs/tf2_geometry_msgs.h>
#include <tf2/utils.h>
#include <fstream>
#include <sstream>
#include <vector>
#include <cmath>
#include <algorithm>
#include <ros/package.h>
#include <limits>
#include <stdexcept>
#include <std_msgs/String.h>

// 单个路径点：只用 x、y，yaw 目前不参与控制
struct Waypoint {
  double x;
  double y;
  double yaw;  // 可为 NaN
};

// 简单限幅
static inline double clamp(double v, double lo, double hi) {
  return std::max(lo, std::min(v, hi));
}

// 路径跟随节点（纯平移控制版本）
class PurePursuit {
public:
  PurePursuit(ros::NodeHandle& nh, ros::NodeHandle& pnh)
    : tf_buffer_(ros::Duration(10.0)), tf_listener_(tf_buffer_)
  {
    // ========== 模块 0：参数读取 ==========
    pnh.param<std::string>("frame_map",  frame_map_,  "map");
    pnh.param<std::string>("frame_base", frame_base_, "base_link");
    pnh.param<double>("rate",           rate_hz_,     20.0);

    pnh.param<double>("lookahead",      lookahead_,      0.35);
    pnh.param<double>("goal_xy_tol",    goal_xy_tol_,    0.05);
    pnh.param<double>("goal_yaw_tol",   goal_yaw_tol_,   0.10); // 保留字段
    pnh.param<double>("slow_down_dist", slow_down_dist_, 0.8);
    pnh.param<bool>  ("reverse_allowed", reverse_allowed_, false);
    pnh.param<double>("reach_stop_time", reach_stop_time_, 1.0);

    pnh.param<double>("v_max",      v_max_,      0.35);  // 保留字段
    pnh.param<double>("w_max",      w_max_,      1.2);
    pnh.param<double>("acc_lim_v",  acc_lim_v_,  0.5);   // 保留字段
    pnh.param<double>("acc_lim_w",  acc_lim_w_,  2.0);
    pnh.param<double>("vxy_max",    vxy_max_,    0.35);
    pnh.param<double>("vx_max",     vx_max_,     vxy_max_); // 不再单轴限幅，但保留参数
    pnh.param<double>("vy_max",     vy_max_,     vxy_max_);
    pnh.param<double>("acc_lim_x",  acc_lim_x_,  0.4);
    pnh.param<double>("acc_lim_y",  acc_lim_y_,  0.4);

    pnh.param<double>("k_xy",   k_xy_,   0.7);
    pnh.param<double>("k_yaw",  k_yaw_,  0.5);  // 保留字段

    pnh.param<double>("yaw_stop_thresh",   yaw_stop_thresh_,   20.0 * M_PI / 180.0);
    pnh.param<double>("yaw_slow_thresh",   yaw_slow_thresh_,   10.0 * M_PI / 180.0);
    pnh.param<double>("min_lin_when_turn", min_lin_when_turn_, 0.0);

    std::string default_csv = ros::package::getPath("xy_path_follower") + "/path/ros_path.csv";
    pnh.param<std::string>("csv_path", csv_path_, default_csv);

    // 2========== 模块 1：加载路径 ==========
    if (!loadCSV(csv_path_)) {
      ROS_FATAL("Failed to load CSV waypoints from %s", csv_path_.c_str());
      throw std::runtime_error("no csv");
    }

    // ========== 模块 2：ROS 通信 ==========
    pub_cmd_    = nh.advertise<geometry_msgs::Twist>("/cmd_vel", 10);
    pub_finish_ = nh.advertise<std_msgs::String>("/move_finish_follow_trace1", 1, true);

    ROS_INFO("Loaded %zu waypoints from %s", path_.size(), csv_path_.c_str());
  }

  // 主循环
  void spin() {
    ros::Rate r(rate_hz_);
    ros::Time last = ros::Time::now();
    while (ros::ok()) {
      ros::spinOnce();
      ros::Time now = ros::Time::now();
      double dt = (now - last).toSec();
      if (dt <= 1e-3) dt = 1.0 / rate_hz_;
      step(dt);
      last = now;
      r.sleep();
    }
  }

private:
  // ========== 模块 1：CSV 路径加载 ==========
  bool loadCSV(const std::string& file) {
    std::ifstream fin(file.c_str());
    if (!fin.is_open()) return false;

    std::string line;
    while (std::getline(fin, line)) {
      if (line.empty())   continue;
      if (line[0] == '#') continue;

      std::stringstream ss(line);
      std::string tok;
      std::vector<std::string> toks;
      while (std::getline(ss, tok, ',')) {
        tok.erase(0, tok.find_first_not_of(" \t\r\n"));
        tok.erase(tok.find_last_not_of(" \t\r\n") + 1);
        toks.push_back(tok);
      }
      if (toks.size() < 2) continue;

      Waypoint wp;
      wp.x = std::stod(toks[0]);
      wp.y = std::stod(toks[1]);
      if (toks.size() >= 3 && !toks[2].empty())
        wp.yaw = std::stod(toks[2]);
      else
        wp.yaw = std::numeric_limits<double>::quiet_NaN();
      path_.push_back(wp);
    }
    return !path_.empty();
  }

  // ========== 模块 2：机器人位姿获取 ==========
  bool getRobotPose(double& x, double& y, double& yaw) {
    try {
      geometry_msgs::TransformStamped tf =
        tf_buffer_.lookupTransform(frame_map_, frame_base_, ros::Time(0), ros::Duration(0.2));
      x   = tf.transform.translation.x;
      y   = tf.transform.translation.y;
      yaw = tf2::getYaw(tf.transform.rotation);
      return true;
    }
    catch (const std::exception& e) {
      ROS_WARN_THROTTLE(1.0, "TF lookup failed: %s", e.what());
      return false;
    }
  }

  // ========== 模块 3：路径索引与前瞻点选择 ==========
  size_t nearestIndex(double x, double y, size_t start) {
    size_t best    = start;
    double best_d2 = std::numeric_limits<double>::infinity();
    int    search_span = 50;

    size_t i0 = (start > 5) ? (start - 5) : 0;
    size_t i1 = std::min(path_.size() - 1, start + static_cast<size_t>(search_span));

    for (size_t i = i0; i <= i1; ++i) {
      double dx = path_[i].x - x;
      double dy = path_[i].y - y;
      double d2 = dx * dx + dy * dy;
      if (d2 < best_d2) {
        best_d2 = d2;
        best    = i;
      }
    }
    return best;
  }

  std::pair<std::pair<double,double>, size_t>
  findLookahead(double x, double y, size_t start, double Ld) {
    if (path_.empty()) return {{x, y}, start};

    size_t i = std::max<size_t>(0, start);
    double px = path_[i].x;
    double py = path_[i].y;
    double acc = std::hypot(px - x, py - y);

    while (i + 1 < path_.size() && acc < Ld) {
      double nx = path_[i + 1].x;
      double ny = path_[i + 1].y;
      acc += std::hypot(nx - px, ny - py);
      px = nx;
      py = ny;
      ++i;
    }
    return {{px, py}, i};
  }

  // ========== 模块 4：速度指令发布 ==========
  void publishCmd(double vx, double vy, double w) {
    geometry_msgs::Twist tw;
    tw.linear.x  = vx;
    tw.linear.y  = vy;
    tw.angular.z = w;
    pub_cmd_.publish(tw);
  }

  void stopCmd() {
    vx_cmd_ = 0.0;
    vy_cmd_ = 0.0;
    w_cmd_  = 0.0;
    publishCmd(vx_cmd_, vy_cmd_, w_cmd_);
  }

  void finishCmd() {
    std_msgs::String fin;
    fin.data = "fin";
    // 如需通知上层，可打开下面这一行
    // pub_finish_.publish(fin);
  }

  // ========== 模块 5：控制主逻辑（纯平移跟随） ==========
  void step(double dt) {
    if (path_.empty()) {
      stopCmd();
      return;
    }

    double rx, ry, ryaw;
    if (!getRobotPose(rx, ry, ryaw)) {
      stopCmd();
      return;
    }

    const Waypoint& goal = path_.back();
    double dist_goal = std::hypot(goal.x - rx, goal.y - ry);

    // ---- 终点判断：只看 XY，不管 yaw ----
    bool pos_ok = (dist_goal < goal_xy_tol_);
    if (pos_ok) {
      if (!reached_) {
        reached_     = true;
        reach_stamp_ = ros::Time::now();
        ROS_INFO("Goal reached (XY only). Hold stop for %.2fs", reach_stop_time_);
        finishCmd();
      }

      if ((ros::Time::now() - reach_stamp_).toSec() < reach_stop_time_) {
        stopCmd();
        return;
      } else {
        stopCmd();

        ros::Time t0 = ros::Time::now();
        while (pub_finish_.getNumSubscribers() == 0 &&
               (ros::Time::now() - t0).toSec() < 2.0 && ros::ok()) {
          ros::Duration(0.05).sleep();
          ros::spinOnce();
        }
        ros::Duration(0.2).sleep();

        ROS_INFO("SHUTDOWN Follow_Trace1.");
        ros::shutdown();
        ROS_INFO("Follow_Trace1 Node shutdown complete.");
        std::exit(0);
        return;
      }
    }
    reached_ = false;

    // ---- 最近点 + 前瞻点 ----
    idx_ = nearestIndex(rx, ry, idx_);
    double Ld = lookahead_;
    auto   lk = findLookahead(rx, ry, idx_, Ld);
    double lx = lk.first.first;
    double ly = lk.first.second;
    size_t iL = lk.second;

    // ---- 世界坐标 → 机体系 ----
    double dx = lx - rx;
    double dy = ly - ry;
    double cos_yaw = std::cos(-ryaw);
    double sin_yaw = std::sin(-ryaw);
    double local_x =  cos_yaw * dx - sin_yaw * dy;
    double local_y =  sin_yaw * dx + cos_yaw * dy;

    // ---- 终点附近减速（只缩放模长） ----
    double v_scale = 1.0;
    if (dist_goal < slow_down_dist_) {
      v_scale = clamp(dist_goal / slow_down_dist_, 0.1, 1.0);
    }

    // ---- 位置误差 → 期望机体系平移速度 ----
    double vx_des = k_xy_ * local_x * v_scale;
    double vy_des = k_xy_ * local_y * v_scale;
    double w_des  = 0.0;  // 不再控制朝向

    // ---- 等比例限幅：按合速度缩放 XY ----
    double vnorm = std::hypot(vx_des, vy_des);
    if (vnorm > vxy_max_) {
      double s = vxy_max_ / std::max(vnorm, 1e-6);
      vx_des *= s;
      vy_des *= s;
    }

    // ---- 加速度限制（逐轴，但最终再等比例保护） ----
    double dvx_max = acc_lim_x_ * dt;
    double dvy_max = acc_lim_y_ * dt;
    double dw_max  = acc_lim_w_ * dt;

    vx_cmd_ = clamp(vx_des, vx_cmd_ - dvx_max, vx_cmd_ + dvx_max);
    vy_cmd_ = clamp(vy_des, vy_cmd_ - dvy_max, vy_cmd_ + dvy_max);
    w_cmd_  = clamp(w_des,  w_cmd_  - dw_max,  w_cmd_  + dw_max);

    // 再做一次等比例限幅，保证最终速度向量不超 vxy_max_
    double vcmd_norm = std::hypot(vx_cmd_, vy_cmd_);
    if (vcmd_norm > vxy_max_) {
      double s = vxy_max_ / std::max(vcmd_norm, 1e-6);
      vx_cmd_ *= s;
      vy_cmd_ *= s;
    }
    w_cmd_ = clamp(w_cmd_, -w_max_, w_max_);

    // ---- 发布指令 ----
    publishCmd(vx_cmd_, vy_cmd_, w_cmd_);

    // 防止索引回退
    idx_ = std::max(idx_, iL);
  }

private:
  // 参数
  std::string frame_map_;
  std::string frame_base_;
  std::string csv_path_;

  double rate_hz_;
  double lookahead_;
  double v_max_;
  double w_max_;
  double acc_lim_v_;
  double acc_lim_w_;
  double goal_xy_tol_;
  double goal_yaw_tol_;
  double slow_down_dist_;
  double reach_stop_time_;
  bool   reverse_allowed_;

  double vxy_max_;
  double vx_max_;
  double vy_max_;
  double acc_lim_x_;
  double acc_lim_y_;
  double k_xy_;
  double k_yaw_;

  double yaw_stop_thresh_;
  double yaw_slow_thresh_;
  double min_lin_when_turn_;

  // 状态
  std::vector<Waypoint> path_;
  size_t idx_ = 0;
  double vx_cmd_ = 0.0;
  double vy_cmd_ = 0.0;
  double w_cmd_  = 0.0;
  bool   reached_ = false;
  ros::Time reach_stamp_;

  // ROS 通信
  ros::Publisher pub_cmd_;
  ros::Publisher pub_finish_;
  tf2_ros::Buffer tf_buffer_;
  tf2_ros::TransformListener tf_listener_;
};

// ========== 模块 6：节点入口 ==========
int main(int argc, char** argv) {
  ros::init(argc, argv, "xy_path_follower");
  ros::NodeHandle nh;
  ros::NodeHandle pnh("~");
  try {
    ROS_INFO("Pure-pursuit (translation-only) node started.");
    PurePursuit node(nh, pnh);
    node.spin();
  }
  catch (const std::exception& e) {
    ROS_ERROR("Fatal: %s", e.what());
    return 1;
  }
  return 0;
}
