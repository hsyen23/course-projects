from utils import *

imu_loader = load_KITTI_txt('../KITTI/oxts/data', [8,9,10,17,18,19])
gps_loader = load_KITTI_txt('../KITTI/oxts/data', [0,1])
number_of_data = get_KITTI_size('../KITTI/oxts/data')

save_KITTI_to_np(imu_loader, number_of_data, '../Odometry_data/imu')
save_KITTI_to_np(gps_loader, number_of_data, '../Odometry_data/gps')