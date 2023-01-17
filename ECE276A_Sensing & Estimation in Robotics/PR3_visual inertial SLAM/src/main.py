import numpy as np
from pr3_utils import *
from setup import *
from scipy.linalg import expm
import matplotlib.pyplot as plt
from numpy.linalg import inv

if __name__ == '__main__':

    # Load the measurements
    filename = "./data/03.npz"
    t,features,linear_velocity,angular_velocity,K,b,imu_T_cam = load_data(filename)
    
    ###### hyperparameters
    pct_landmarks = 1
    initial_pose = np.array([[1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,1]])
    # get random landmark indices and reshape our features data
    if pct_landmarks == 1:
        landmark_indices = range(features.shape[1])
    else:
        landmark_indices = get_random_landmark(features.shape[1], pct_landmarks)
    #landmark_indices = np.load('indices.npy')
    features = features[:,landmark_indices,:]
    number_of_landmark = len(landmark_indices)
    
    landmarks = landmark(number_of_landmark)
    
    R_flip_1 = np.array([[-1,0,0,0],[0,1,0,0],[0,0,-1,0],[0,0,0,1]])
    R_flip_2 = np.array([[1,0,0,0],[0,-1,0,0],[0,0,-1,0],[0,0,0,1]])
    R_flip = np.matmul(R_flip_1, R_flip_2)
    imu_T_cam = np.matmul(imu_T_cam,R_flip)
    
    '''
    landmark_mean = np.empty([3*number_of_landmark])
    landmark_mean[:] = np.nan
    landmark_covariance = np.empty([3*number_of_landmark, 3*number_of_landmark])
    landmark_covariance[:,:] = np.nan
    observed_landmark = set()
    '''
    
    se3 = np.concatenate((linear_velocity, angular_velocity), axis=0)
    twist_matrix = hatmap(se3)
    skinny_matrix = skinny_hat(se3)
    
    # creat robot pose array
    # Here, pose means the mean of probability that robot at this timestamp
    pose = np.empty([4,4,t.shape[1]+1])
    pose[:,:,0] = initial_pose
    pose_covariance = np.zeros([6,6, t.shape[1]+1])
    pose_covariance[:,:,:] = 0
    position_noise = 0.3 #0.3
    orientation_noise = 0.05 #0.05
    noise_W = np.block([
        [position_noise*np.eye(3), np.zeros([3,3])],
        [np.zeros([3,3]), orientation_noise*np.eye(3)]
        ]
        )
    # loop through timestamp
    for i in range(t.shape[1]):
        # this part we simply apply motion model to get T_t+1.
        # By doing this, we can roughly know how robot moves, and examinate our mapping method for landmarks
        
        dt = t[0,i] - t[0,i-1] if i > 0 else t[0,1] - t[0,0]

        pose[:,:,i+1] = np.matmul(pose[:,:,i] , expm(dt * twist_matrix[:,:,i]))
        # update pose_covariance
     #   pose_covariance[:,:, i+1] = ( np.matmul(np.matmul( expm(-dt * skinny_matrix[:,:,i]), pose_covariance[:,:, i] ), np.transpose(expm(-dt * skinny_matrix[:,:,i]))) ) + noise_W
        # here, we know where is our robot, so we update the landmark prior
        
        #landmarks.kalman_filter_update(features[:,:,i], pose[:,:,i], imu_T_cam, K, b)
        
        landmarks.update_landmark_prior(features[:,:,i], pose[:,:,i], inv(imu_T_cam), K, b)
        if not i == (t.shape[1] - 1):
            landmarks.kalman_filter_update(features[:,:,i+1], pose[:,:,i+1], inv(imu_T_cam), K, b)
            # EKF update step for robot state
     #       pose[:,:,i+1], pose_covariance[:,:,i+1] = landmarks.kalman_filter_robot_update(features[:,:,i+1], pose[:,:,i+1], inv(imu_T_cam), K, b, pose_covariance[:,:,i+1])
        
        
    fig, ax = visualize_trajectory_2d_with_landmarks(pose, landmarks)
    #fig, ax = visualize_trajectory_2d(pose)
    # (a) IMU Localization via EKF Prediction

    # (b) Landmark Mapping via EKF Update

    # (c) Visual-Inertial SLAM

    # You can use the function below to visualize the robot pose over time
    # visualize_trajectory_2d(world_T_imu, show_ori = True)


