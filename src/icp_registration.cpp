#include <icp_registration.h>


IcpRegistration::IcpRegistration() :
  nh_private_("~"), original_target_(new PointCloud), target_readed_(false),
  enable_(false), in_clouds_num_(0), last_detection_(ros::Time(-100)) {
  // Read params
  nh_private_.param("min_range",        min_range_,       0.5);
  nh_private_.param("max_range",        max_range_,       4.5);
  nh_private_.param("voxel_size",       voxel_size_,      0.015);
  nh_private_.param("target",           target_file_,     std::string(""));
  nh_private_.param("world_frame_id",   world_frame_id_,  std::string("world"));
  nh_private_.param("target_frame_id",  target_frame_id_, std::string("target"));
  nh_private_.param("save_in_clouds",   save_in_clouds_,  false);
  nh_private_.param("save_dir",         save_dir_,        std::string("~/icp_registration"));

  // Subscribers and publishers
  point_cloud_sub_ = nh_.subscribe(
    "input_cloud", 1, &IcpRegistration::pointCloudCb, this);
  dbg_reg_cloud_pub_ = nh_private_.advertise<sensor_msgs::PointCloud2>(
    "dbg_reg_cloud", 1);
  target_pose_pub_ = nh_private_.advertise<geometry_msgs::PoseStamped>(
    "target_pose", 1);

  // Services
  enable_srv_ = nh_private_.advertiseService("enable",
    &IcpRegistration::enable, this);
  disable_srv_ = nh_private_.advertiseService("disable",
    &IcpRegistration::disable, this);
}

void IcpRegistration::pointCloudCb(const sensor_msgs::PointCloud2::ConstPtr& in_cloud) {
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

  if (save_in_clouds_) {
    std::string filename = save_dir_ + "/cloud_" +
      boost::lexical_cast<std::string>(in_clouds_num_) + ".pcd";
    pcl::io::savePCDFileBinary(filename, *cloud);
    in_clouds_num_++;
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

  // Move target
  tf::Transform tmp_pose;
  double elapsed = fabs(last_detection_.toSec() - ros::Time::now().toSec());
  if (elapsed > 2.0) {
    // Move to center
    Eigen::Vector4f centroid;
    pcl::compute3DCentroid(*cloud, centroid);
    tf::Transform tf_01;
    tf_01.setIdentity();
    tf_01.setOrigin(tf::Vector3(centroid[0], centroid[1], centroid[2]));
    move(target, tf_01);
    last_pose_ = tf_01;
  } else {
    // Move to last detected position
    move(target, last_pose_);
  }

  // Registration
  bool converged;
  double score;
  tf::Transform target_pose;
  pairAlign(target, cloud, target_pose, converged, score);
  double dist = eucl(last_pose_, target_pose);
  if (converged && dist < 2.0 && score < 0.0001) {
    ROS_INFO_STREAM("[IcpRegistration]: Target found with score of " <<
      score);

    last_pose_ = target_pose * last_pose_;
    last_detection_ = ros::Time::now();

    // Publish tf and message
    publish(last_pose_, in_cloud->header);

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

void IcpRegistration::pairAlign(PointCloud::Ptr src,
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

void IcpRegistration::filter(PointCloud::Ptr cloud, const bool& passthrough) {
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

void IcpRegistration::publish(const tf::Transform& cam_to_target,
             const std_msgs::Header& header) {
  // Publish tf
  tf_broadcaster_.sendTransform(
    tf::StampedTransform(cam_to_target,
                         header.stamp,
                         header.frame_id,
                         target_frame_id_));

  // Publish geometry message from world frame id
  if (target_pose_pub_.getNumSubscribers() > 0) {
    try {
      ros::Time now = ros::Time::now();
      tf::StampedTransform world2camera;
      tf_listener_.waitForTransform(world_frame_id_,
                                    header.frame_id,
                                    now, ros::Duration(1.0));
      tf_listener_.lookupTransform(world_frame_id_,
          header.frame_id, now, world2camera);

      // Compose the message
      geometry_msgs::PoseStamped pose_stamped;
      tf::Transform world2target = world2camera * cam_to_target;
      pose_stamped.pose.position.x = world2target.getOrigin().x();
      pose_stamped.pose.position.y = world2target.getOrigin().y();
      pose_stamped.pose.position.z = world2target.getOrigin().z();
      pose_stamped.header.stamp = header.stamp;
      pose_stamped.header.frame_id = world_frame_id_;
      target_pose_pub_.publish(pose_stamped);

    } catch (tf::TransformException ex) {
      ROS_WARN_STREAM("[IcpRegistration]: Cannot find the tf between " <<
        "world frame id and camera. " << ex.what());
    }
  }
}

tf::Transform IcpRegistration::matrix4fToTf(const Eigen::Matrix4f& in) {
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

void IcpRegistration::move(const PointCloud::Ptr& cloud, const tf::Transform& trans) {
  Eigen::Affine3d trans_eigen;
  transformTFToEigen(trans, trans_eigen);
  pcl::transformPointCloud(*cloud, *cloud, trans_eigen);
}

double IcpRegistration::eucl(const tf::Transform& a, const tf::Transform& b) {
  return sqrt( (a.getOrigin().x() - b.getOrigin().x())*(a.getOrigin().x() - b.getOrigin().x()) +
               (a.getOrigin().y() - b.getOrigin().y())*(a.getOrigin().y() - b.getOrigin().y()) +
               (a.getOrigin().z() - b.getOrigin().z())*(a.getOrigin().z() - b.getOrigin().z()) );
}

bool IcpRegistration::enable(std_srvs::Empty::Request& req,
                             std_srvs::Empty::Response& res) {
  ROS_INFO("[IcpRegistration]: Enabled!");
  enable_ = true;
}
bool IcpRegistration::disable(std_srvs::Empty::Request& req,
                              std_srvs::Empty::Response& res) {
  ROS_INFO("[IcpRegistration]: Disabled!");
  enable_ = false;
}