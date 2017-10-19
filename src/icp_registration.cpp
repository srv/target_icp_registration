#include <icp_registration.h>


IcpRegistration::IcpRegistration() :
  nh_private_("~"), original_target_(new PointCloud), target_readed_(false),
  enable_(false), in_clouds_num_(0), last_detection_(ros::Time(-100)),
  robot2camera_init_(false) {
  // Read params
  nh_private_.param("min_range",        min_range_,       1.0);
  nh_private_.param("max_range",        max_range_,       2.0);
  nh_private_.param("voxel_size",       voxel_size_,      0.02);
  nh_private_.param("target",           target_file_,     std::string("target.pcd"));
  nh_private_.param("robot_frame_id",   robot_frame_id_,  std::string("robot"));
  nh_private_.param("world_frame_id",   world_frame_id_,  std::string("world"));
  nh_private_.param("target_frame_id",  target_frame_id_, std::string("target"));
  nh_private_.param("remove_ground",    remove_ground_,   true);
  nh_private_.param("ground_height",    ground_height_,   0.09);
  nh_private_.param("max_icp_dist",     max_icp_dist_,    2.0);
  nh_private_.param("max_icp_score",    max_icp_score_,   0.0001);

  // Subscribers and publishers
  point_cloud_sub_ = nh_.subscribe(
    "input_cloud", 1, &IcpRegistration::pointCloudCb, this);
  dbg_reg_cloud_pub_ = nh_private_.advertise<sensor_msgs::PointCloud2>(
    "dbg_reg_cloud", 1);
  dbg_obj_cloud_pub_ = nh_private_.advertise<sensor_msgs::PointCloud2>(
    "dbg_obj_cloud", 1);
  target_pose_pub_ = nh_private_.advertise<geometry_msgs::PoseStamped>(
    "target_pose", 1);

  // Services
  enable_srv_ = nh_private_.advertiseService("enable",
    &IcpRegistration::enable, this);
  disable_srv_ = nh_private_.advertiseService("disable",
    &IcpRegistration::disable, this);
}

void IcpRegistration::pointCloudCb(const sensor_msgs::PointCloud2::ConstPtr& in_cloud) {
  if (!target_readed_) {
    ROS_INFO_STREAM("[IcpRegistration]: Loading target for the first time...");

    // Opening target
    if (pcl::io::loadPCDFile<Point>(target_file_, *original_target_) == -1) {
      ROS_ERROR_STREAM("[IcpRegistration]: Couldn't read file " <<
        target_file_);
      return;
    }

    // Filter target
    filter(original_target_);
    target_readed_ = true;
  }

  if (!enable_) {
    ROS_INFO_THROTTLE(15, "[IcpRegistration]: Not enabled.");
    return;
  }

  // Copy
  PointCloud::Ptr original(new PointCloud);
  PointCloud::Ptr cloud(new PointCloud);
  fromROSMsg(*in_cloud, *cloud);
  fromROSMsg(*in_cloud, *original);

  if (cloud->points.size() < 100) {
    ROS_WARN("[IcpRegistration]: Input cloud has less than 100 points.");
    return;
  }

  // Translate the cloud to the robot frame id to remove the camera orientation effect
  if (!robot2camera_init_) {
    bool ok = getRobot2Camera(in_cloud->header.frame_id);
    if (!ok) return;
  }
  move(cloud, robot2camera_);

  // Filter input cloud
  filter(cloud, true, true);

  // Remove ground
  if (remove_ground_)
    removeGround(cloud, in_cloud->header.stamp);

  if (cloud->points.size() < 100) {
    ROS_WARN("[IcpRegistration]: Input cloud has not enough points after filtering.");
    return;
  }

  // Move target
  PointCloud::Ptr target(new PointCloud);
  pcl::copyPointCloud(*original_target_, *target);
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
  if (converged) {
    ROS_INFO_STREAM("[IcpRegistration]: Icp converged. Score: " <<
      score << ". Dist: " << dist);
  }
  if (converged && dist < max_icp_dist_ && score < max_icp_score_) {
    ROS_INFO_STREAM("[IcpRegistration]: Target found with score of " <<
      score);

    last_pose_ = target_pose * last_pose_;
    last_detection_ = ros::Time::now();

    // Publish tf and message
    publish(last_pose_, in_cloud->header.stamp);

    // Debug cloud
    if (dbg_reg_cloud_pub_.getNumSubscribers() > 0) {
      move(original, robot2camera_);
      PointCloud::Ptr dbg_cloud(new PointCloud);
      pcl::copyPointCloud(*original, *dbg_cloud);

      // Add target to debug cloud
      move(target, target_pose);

      PointCloud::Ptr target_color(new PointCloud);
      for (uint i=0; i < target->size(); i++) {
        Point prgb(0, 255, 0);
        prgb.x = target->points[i].x;
        prgb.y = target->points[i].y;
        prgb.z = target->points[i].z;
        target_color->push_back(prgb);
      }
      *dbg_cloud += *target_color;

      // Publish
      sensor_msgs::PointCloud2 dbg_cloud_ros;
      toROSMsg(*dbg_cloud, dbg_cloud_ros);
      dbg_cloud_ros.header.stamp = in_cloud->header.stamp;
      dbg_cloud_ros.header.frame_id = robot_frame_id_;
      dbg_reg_cloud_pub_.publish(dbg_cloud_ros);
    }
  } else {
    ROS_WARN_STREAM("[IcpRegistration]: Target not found in the input " <<
      "pointcloud. Trying again...");
  }

  // Publish debug cloud
  if (dbg_reg_cloud_pub_.getNumSubscribers() > 0) {
    move(original, robot2camera_);
    PointCloud::Ptr dbg_cloud(new PointCloud);
    pcl::copyPointCloud(*original, *dbg_cloud);

    // Publish
    sensor_msgs::PointCloud2 dbg_cloud_ros;
    toROSMsg(*dbg_cloud, dbg_cloud_ros);
    dbg_cloud_ros.header.stamp = in_cloud->header.stamp;
    dbg_cloud_ros.header.frame_id = robot_frame_id_;
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
  icp.setMaximumIterations(100);
  icp.setInputSource(src);
  icp.setInputTarget(tgt);
  icp.align(*aligned);

  // The transform
  output = matrix4fToTf(icp.getFinalTransformation());

  // The measures
  converged = icp.hasConverged();
  score = icp.getFitnessScore();
}

void IcpRegistration::filter(PointCloud::Ptr cloud,
                             const bool& passthrough,
                             const bool& statistical) {
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

  if (statistical) {
    pcl::RadiusOutlierRemoval<Point> outrem;
    outrem.setInputCloud(cloud);
    outrem.setRadiusSearch(0.2);
    outrem.setMinNeighborsInRadius(100);
    outrem.filter(*cloud);
  }
}

void IcpRegistration::removeGround(PointCloud::Ptr cloud, const ros::Time& stamp) {
  pcl::ModelCoefficients::Ptr coefficients (new pcl::ModelCoefficients);
  pcl::PointIndices::Ptr inliers (new pcl::PointIndices);
  pcl::SACSegmentation<Point> seg;
  seg.setModelType(pcl::SACMODEL_PLANE);
  seg.setMethodType(pcl::SAC_RANSAC);
  seg.setDistanceThreshold(ground_height_);
  seg.setInputCloud(cloud);
  seg.segment(*inliers, *coefficients);

  double mean_z;
  PointCloud::Ptr ground(new PointCloud);
  PointCloud::Ptr objects(new PointCloud);
  for (int i=0; i < (int)cloud->size(); i++) {
    if (std::find(inliers->indices.begin(), inliers->indices.end(), i) == inliers->indices.end()) {
      objects->push_back(cloud->points[i]);
    } else {
      ground->push_back(cloud->points[i]);
      mean_z += cloud->points[i].z;
    }
  }
  mean_z = mean_z / ground->size();

  // Filter outliers in the objects
  PointCloud::Ptr object_inliers(new PointCloud);
  for (uint i=0; i < objects->size(); i++) {
    if ( fabs(objects->points[i].z - mean_z) < 0.35)
      object_inliers->push_back(objects->points[i]);
  }

  pcl::copyPointCloud(*object_inliers, *cloud);

  if (dbg_obj_cloud_pub_.getNumSubscribers() > 0) {
    for (uint i=0; i < object_inliers->size(); i++) {
      object_inliers->points[i].r = 255;
      object_inliers->points[i].g = 0;
      object_inliers->points[i].b = 0;
    }
    *ground += *object_inliers;
    sensor_msgs::PointCloud2 dbg_cloud_ros;
    toROSMsg(*ground, dbg_cloud_ros);
    dbg_cloud_ros.header.stamp = stamp;
    dbg_cloud_ros.header.frame_id = robot_frame_id_;
    dbg_obj_cloud_pub_.publish(dbg_cloud_ros);
  }
}

void IcpRegistration::publish(const tf::Transform& cam_to_target,
                              const ros::Time& stamp) {
  // Publish tf
  tf_broadcaster_.sendTransform(
    tf::StampedTransform(cam_to_target,
                         stamp,
                         robot_frame_id_,
                         target_frame_id_));

  // Publish geometry message from world frame id
  if (target_pose_pub_.getNumSubscribers() > 0) {
    try {
      ros::Time now = ros::Time::now();
      tf::StampedTransform world2robot;
      tf_listener_.waitForTransform(world_frame_id_,
                                    robot_frame_id_,
                                    now, ros::Duration(1.0));
      tf_listener_.lookupTransform(world_frame_id_,
          robot_frame_id_, now, world2robot);

      // Compose the message
      geometry_msgs::PoseStamped pose_stamped;
      tf::Transform world2target = world2robot * cam_to_target;
      pose_stamped.pose.position.x = world2target.getOrigin().x();
      pose_stamped.pose.position.y = world2target.getOrigin().y();
      pose_stamped.pose.position.z = world2target.getOrigin().z();
      pose_stamped.header.stamp = stamp;
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

void IcpRegistration::move(const PointCloud::Ptr& cloud,
                           const tf::Transform& trans) {
  Eigen::Affine3d trans_eigen;
  transformTFToEigen(trans, trans_eigen);
  pcl::transformPointCloud(*cloud, *cloud, trans_eigen);
}

double IcpRegistration::eucl(const tf::Transform& a, const tf::Transform& b) {
  return sqrt( (a.getOrigin().x() - b.getOrigin().x())*(a.getOrigin().x() - b.getOrigin().x()) +
               (a.getOrigin().y() - b.getOrigin().y())*(a.getOrigin().y() - b.getOrigin().y()) +
               (a.getOrigin().z() - b.getOrigin().z())*(a.getOrigin().z() - b.getOrigin().z()) );
}

bool IcpRegistration::getRobot2Camera(const std::string& camera_frame_id) {
  try {
    ros::Time now = ros::Time::now();
    tf_listener_.waitForTransform(robot_frame_id_,
                                  camera_frame_id,
                                  now, ros::Duration(1.0));
    tf_listener_.lookupTransform(robot_frame_id_,
        camera_frame_id, now, robot2camera_);
    robot2camera_init_ = true;
    return true;
  } catch (tf::TransformException ex) {
    ROS_WARN_STREAM("[IcpRegistration]: Cannot find the tf between " <<
      "robot frame id and camera. " << ex.what());
    return false;
  }
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
