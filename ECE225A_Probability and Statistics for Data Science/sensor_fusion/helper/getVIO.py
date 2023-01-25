from stereo_visual_odometry import VisualOdometry
import glob
import numpy as np
import matplotlib.pyplot as plt

data_dir = '../KITTI'
vo = VisualOdometry(data_dir)

# find number of images
img_dir = '../KITTI/image_00/data'
img_names = glob.glob(img_dir + '/*.png')
n = len(img_names)

# time interval between two images
hz = 10

def decompose_homogeneous_translation(T):
    d = T[0:3, 3]
    R = T[0:3, 0:3]
    roll = np.arctan2(R[2,1],R[2,2])
    pitch = np.arctan2(-R[2,0],np.sqrt(R[2,1]**2 + R[2,2]**2))
    yaw = np.arctan2(R[1,0],R[0,0])
    return np.array([d[0], d[1], d[2], roll, pitch, yaw])

c_R_r = np.array([[0,0,1,0],[-1,0,0,0],[0,-1,0,0],[0,0,0,1]])

initial_pose = np.eye(4)
pose = np.zeros([n,2])
pre_state = initial_pose
VO_data = np.zeros([n, 6])
for i in range(1, n):
    transf = vo.get_pose(i)
    initial_pose =  initial_pose @ transf
    wolrd_frame_pose = c_R_r @ initial_pose
    pose[i,0] = wolrd_frame_pose[0,3]
    pose[i,1] = wolrd_frame_pose[1,3]

    camera_VO_data = decompose_homogeneous_translation(transf)
    correct_VO_data = np.array([camera_VO_data[2], -camera_VO_data[0], -camera_VO_data[1], camera_VO_data[5], -camera_VO_data[3], -camera_VO_data[4]])
    VO_data[i-1] = correct_VO_data

    pre_state = wolrd_frame_pose

VO_data[-1] = VO_data[-2]
np.save('../Odometry_data/vo', VO_data*hz)

# plot the trajectory based on VO
# plt.scatter(pose[:,0], pose[:,1])
# plt.show()