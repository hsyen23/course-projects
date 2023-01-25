import glob
import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import expm
from utils import *

imu_loader = load_KITTI_txt('../KITTI/oxts/data', [8,9,10,17,18,19])
gps_loader = load_KITTI_txt('../KITTI/oxts/data', [0,1])
number_of_data = get_KITTI_size('../KITTI/oxts/data')

IMU_hz = 10
pose_init = np.array([[1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,1]])
pos = np.zeros([number_of_data, 2])
# plot deadrecon by IMU
for i in range(number_of_data):
    pos[i,:] = np.array([pose_init[0,3], pose_init[1,3]])
    # update SE3
    IMU = next(imu_loader)
    pose_init = pose_init @ expm(twistHat(IMU)/IMU_hz)

# plot deadrecon by Visual odometry
VO = np.load('../Odometry_data/vo.npy')
VO_hz = 10
pose_init = np.array([[1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,1]])
pos_vo = np.zeros([number_of_data, 2])
for i in range(VO.shape[0]):
    pos_vo[i,:] = np.array([pose_init[0,3], pose_init[1,3]])
    # update SE3
    pose_init = pose_init @ expm(twistHat(VO[i])/VO_hz)

# plot GPS
r = 6.3781 * (10**6)
gt_pose = np.zeros([number_of_data, 2])
pre_GPS = next(gps_loader)
for i in range(1,number_of_data):
    cur_GPS = next(gps_loader)
    if i != 0:
        dx_th = (cur_GPS[1] - pre_GPS[1]) * np.pi/180
        dy_th = (cur_GPS[0] - pre_GPS[0]) * np.pi/180
        gt_pose[i] = gt_pose[i-1] + np.array([dx_th, dy_th]) * r
    pre_GPS = cur_GPS


# plot dead of IMU
plt.figure()
plt.scatter(pos[:,0], pos[:,1], s = 1)
plt.title('IMU trajectory')
# plot dead of VO
plt.figure()
plt.scatter(pos_vo[:,0], pos_vo[:,1], s = 1)
plt.title('VIO trajectory')
# plot GPS
plt.figure()
plt.scatter(gt_pose[:,0], gt_pose[:,1], s = 1)
plt.title('GPS trajectory')
plt.axis('equal')
plt.show()
