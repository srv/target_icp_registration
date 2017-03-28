#include <ros/ros.h>
#include <tf/transform_datatypes.h>
#include <tf/transform_broadcaster.h>
#include <tf_conversions/tf_eigen.h>
#include <sensor_msgs/PointCloud2.h>
#include <std_srvs/Empty.h>

#include <pcl_ros/point_cloud.h>
#include <pcl_ros/transforms.h>
#include <pcl_conversions/pcl_conversions.h>

#include <pcl/point_types.h>
#include <pcl/common/common.h>
#include <pcl/registration/icp.h>
#include <pcl/filters/passthrough.h>
#include <pcl/filters/approximate_voxel_grid.h>
#include <pcl/filters/radius_outlier_removal.h>

typedef pcl::PointXYZ Point;
typedef pcl::PointXYZRGB PointRGB;
typedef pcl::PointCloud<Point> PointCloud;
typedef pcl::PointCloud<PointRGB> PointCloudRGB;
typedef pcl::IterativeClosestPoint<Point, Point> IterativeClosestPoint;

class IcpRegistration {
  ros::NodeHandle nh_;
  ros::NodeHandle nh_private_;

  ros::Subscriber point_cloud_sub_;
  ros::Publisher dbg_reg_cloud_pub_;
  ros::ServiceServer enable_srv_;
  ros::ServiceServer disable_srv_;
  tf::TransformBroadcaster tf_broadcaster_;

  // Params
  double min_range_;
  double max_range_;
  double voxel_size_;
  std::string target_file_;
  std::string target_frame_id_;

  // Operational variables
  PointCloud::Ptr original_target_;
  bool target_readed_;
  bool enable_;

 public:
  IcpRegistration() : nh_private_("~"), original_target_(new PointCloud),
                      target_readed_(false), enable_(false) {
    // Read params
    nh_private_.param("min_range",        min_range_,       0.5);
    nh_private_.param("max_range",        max_range_,       4.5);
    nh_private_.param("voxel_size",       voxel_size_,      0.015);
    nh_private_.param("target",           target_file_,     std::string(""));
    nh_private_.param("target_frame_id",  target_frame_id_, std::string("target"));

    // Subscribers and publishers
    point_cloud_sub_ = nh_.subscribe(
      "input_cloud", 1, &IcpRegistration::pointCloudCb, this);
    dbg_reg_cloud_pub_ = nh_private_.advertise<sensor_msgs::PointCloud2>(
      "dbg_reg_cloud", 1);

    // Services
    enable_srv_ = nh_private_.advertiseService("enable",
      &IcpRegistration::enable, this);
    disable_srv_ = nh_private_.advertiseService("disable",
      &IcpRegistration::disable, this);
  }

  void pointCloudCb(const sensor_msgs::PointCloud2::ConstPtr& in_cloud) {
    if (!enable_) {
      ROS_INFO_THROTTLE(15, "[IcpRegistration]: Not enabled.");
      return;
    }

    // Copy
    PointCloud::Ptr cloud(new PointCloud);
    fromROSMsg(*in_cloud, *cloud);

    if (cloud->points.size() < 100) {
      ROS_WARN("[IcpRegistration]: Input cloud has less than 100 points.");
      return;
    }

    // Filter input cloud
    filter(cloud);

    if (!target_readed_) {
      ROS_INFO_STREAM("[IcpRegistration]: Loading target for the first time...");

      // Opening target
      if (pcl::io::loadPCDFile<Point>(target_file_, *original_target_) == -1) {
        ROS_ERROR_STREAM("[IcpRegistration]: Couldn't read file " <<
          target_file_);
        return;
      }

      // Filter target
      filter(original_target_, false);
      target_readed_ = true;
    }

    PointCloud::Ptr target(new PointCloud);
    pcl::copyPointCloud(*original_target_, *target);

    // Move to center
    Eigen::Vector4f centroid;
    pcl::compute3DCentroid(*cloud, centroid);
    tf::Transform tf_01;
    tf_01.setIdentity();
    tf_01.setOrigin(tf::Vector3(centroid[0], centroid[1], centroid[2]));
    move(target, tf_01);

    // Registration
    bool converged;
    double score;
    tf::Transform target_pose;
    pairAlign(target, cloud, target_pose, converged, score);
    double dist = eucl(tf_01, target_pose);
    if (converged && dist < 2.0 && score < 0.0001) {
      ROS_INFO_STREAM("[IcpRegistration]: Target found with score of " <<
        score << ", and a distance to pointcloud center of " <<
        dist << "m.");

      // Publish tf
      tf::Transform cam_to_target = target_pose * tf_01;
      tf_broadcaster_.sendTransform(
        tf::StampedTransform(cam_to_target,
                             in_cloud->header.stamp,
                             in_cloud->header.frame_id,
                             target_frame_id_));

      // Debug cloud
      if (dbg_reg_cloud_pub_.getNumSubscribers() > 0) {
        PointCloudRGB::Ptr dbg_cloud(new PointCloudRGB);
        pcl::copyPointCloud(*cloud, *dbg_cloud);

        // Add target to debug cloud
        move(target, target_pose);

        PointCloudRGB::Ptr target_color(new PointCloudRGB);
        for (uint i=0; i < target->size(); i++) {
          Point p = target->points[i];
          PointRGB prgb(0, 255, 0);
          prgb.x = p.x; prgb.y = p.y; prgb.z = p.z;
          target_color->push_back(prgb);
        }
        *dbg_cloud += *target_color;

        // Publish
        sensor_msgs::PointCloud2 dbg_cloud_ros;
        toROSMsg(*dbg_cloud, dbg_cloud_ros);
        dbg_cloud_ros.header = in_cloud->header;
        dbg_reg_cloud_pub_.publish(dbg_cloud_ros);
      }
    } else {
      ROS_WARN_STREAM("[IcpRegistration]: Target not found in the input " <<
        "pointcloud. Trying again...");
    }

    // Publish debug cloud
    if (dbg_reg_cloud_pub_.getNumSubscribers() > 0) {
      PointCloudRGB::Ptr dbg_cloud(new PointCloudRGB);
      pcl::copyPointCloud(*cloud, *dbg_cloud);

      // Publish
      sensor_msgs::PointCloud2 dbg_cloud_ros;
      toROSMsg(*dbg_cloud, dbg_cloud_ros);
      dbg_cloud_ros.header = in_cloud->header;
      dbg_reg_cloud_pub_.publish(dbg_cloud_ros);
    }

  }

  void pairAlign(PointCloud::Ptr src,
                 PointCloud::Ptr tgt,
                 tf::Transform &output,
                 bool &converged,
                 double &score) {
    ROS_INFO_STREAM("[IcpRegistration]: Target pointcloud " << src->size() <<
      " points. Scene pointcloud " << tgt->size() << " points.");
    // Align
    PointCloud::Ptr aligned(new PointCloud);
    IterativeClosestPoint icp;
    icp.setMaxCorrespondenceDistance(0.07);
    icp.setRANSACOutlierRejectionThreshold(0.001);
    icp.setTransformationEpsilon(0.00001);
    icp.setEuclideanFitnessEpsilon(0.001);
    icp.setMaximumIterations(80);
    icp.setInputSource(src);
    icp.setInputTarget(tgt);
    icp.align(*aligned);

    // The transform
    output = matrix4fToTf(icp.getFinalTransformation());

    // The measures
    converged = icp.hasConverged();
    score = icp.getFitnessScore();
  }

  void filter(PointCloud::Ptr cloud, bool passthrough = false) {
    std::vector<int> indicies;
    pcl::removeNaNFromPointCloud(*cloud, *cloud, indicies);

    if (passthrough) {
      pcl::PassThrough<Point> pass;
      pass.setFilterFieldName("z");
      pass.setFilterLimits(min_range_, max_range_);
      pass.setInputCloud(cloud);
      pass.filter(*cloud);
    }

    pcl::ApproximateVoxelGrid<Point> grid;
    grid.setLeafSize(voxel_size_, voxel_size_, voxel_size_);
    grid.setDownsampleAllData(true);
    grid.setInputCloud(cloud);
    grid.filter(*cloud);
  }

  tf::Transform matrix4fToTf(Eigen::Matrix4f in) {
    tf::Vector3 t_out;
    t_out.setValue(static_cast<double>(in(0,3)),
                   static_cast<double>(in(1,3)),
                   static_cast<double>(in(2,3)));

    tf::Matrix3x3 tf3d;
    tf3d.setValue(static_cast<double>(in(0,0)), static_cast<double>(in(0,1)), static_cast<double>(in(0,2)),
                  static_cast<double>(in(1,0)), static_cast<double>(in(1,1)), static_cast<double>(in(1,2)),
                  static_cast<double>(in(2,0)), static_cast<double>(in(2,1)), static_cast<double>(in(2,2)));

    tf::Quaternion q_out;
    tf3d.getRotation(q_out);
    tf::Transform out(q_out, t_out);
    return out;
  }

  void move(const PointCloud::Ptr& cloud, const tf::Transform trans) {
    Eigen::Affine3d trans_eigen;
    transformTFToEigen(trans, trans_eigen);
    pcl::transformPointCloud(*cloud, *cloud, trans_eigen);
  }

  double eucl(const tf::Transform& a, const tf::Transform& b) {
    return sqrt( (a.getOrigin().x() - b.getOrigin().x())*(a.getOrigin().x() - b.getOrigin().x()) +
                 (a.getOrigin().y() - b.getOrigin().y())*(a.getOrigin().y() - b.getOrigin().y()) +
                 (a.getOrigin().z() - b.getOrigin().z())*(a.getOrigin().z() - b.getOrigin().z()) );
  }

  bool enable(std_srvs::Empty::Request  &req,
              std_srvs::Empty::Response &res) {
    ROS_INFO("[IcpRegistration]: Enabled!");
    enable_ = true;
  }
  bool disable(std_srvs::Empty::Request  &req,
               std_srvs::Empty::Response &res) {
    ROS_INFO("[IcpRegistration]: Disabled!");
    enable_ = false;
  }
};

int main(int argc, char** argv) {
  ros::init(argc, argv, "icp_registration");
  IcpRegistration node;
  ros::spin();
  return 0;
}
