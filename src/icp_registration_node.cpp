#include "icp_registration.h"

int main(int argc, char** argv) {
  ros::init(argc, argv, "icp_registration");
  IcpRegistration node;
  ros::spin();
  return 0;
}
