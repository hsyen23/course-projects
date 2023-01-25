import numpy as np
import matplotlib.pyplot as plt

class EKF:
    def __init__(self):
        self.state = np.zeros([18])
        self.covariance = np.eye(18) * 1e-9
        self.state_noise = self.get_prediction_noise_covariance()

    def prediction_step(self,delta_t):
        # get jacobian
        F = self.get_prediction_jacobian(self.state, delta_t)
        # omi-direction kinematics
        R = self.get_rotation_matrix()
        self.state[0:3] += (R @ self.state[6:9]) * delta_t + (R @ self.state[12:15]) * 0.5 * delta_t**2
        self.state[3:6] += self.state[9:12] * delta_t + self.state[15:] * 0.5 * delta_t**2
        self.state[6:9] += self.state[12:15] * delta_t
        self.state[9:12] += self.state[15:] * delta_t

        # update covariance
        self.covariance = F @ self.covariance @ F.T + self.state_noise

    def update_step(self, observation, bool_list, noise_covariance = None):
        size = sum(bool_list)
        bool_array = np.array(bool_list)
        mask = np.outer(bool_array, bool_array)
        sub_covariance = self.covariance[mask].reshape([size, size])
        sub_mean = self.state[bool_array]
        if noise_covariance is None:
            noise_covariance = self.state_noise[mask].reshape([size,size])
        # kalman gain
        K = sub_covariance @ np.linalg.inv(sub_covariance + noise_covariance)
        # update mean
        updated_mean = sub_mean + K @ (observation - sub_mean)
        # update covariance
        sub_covariance = (np.eye(size) - K)@sub_covariance
        # put back to normal state
        counter = 0
        for i in range(18):
            if bool_array[i]:
                self.state[i] = updated_mean[counter]
                counter += 1
        # put back to normal covariance
        counter = 0
        for i in range(18):
            if bool_array[i]:
                inner_counter = 0
                for j in range(18):
                    if bool_array[j]:
                        self.covariance[i,j] = sub_covariance[counter, inner_counter]
                        inner_counter += 1
                counter += 1
        

    def get_rotation_matrix(self):
        roll = self.state[3]
        pitch = self.state[4]
        yaw = self.state[5]
        R = np.array([
        [np.cos(pitch) * np.cos(yaw), (np.sin(roll) * np.sin(pitch) * np.cos(yaw) - np.cos(roll * np.sin(yaw))), (np.cos(roll)*np.sin(pitch)*np.cos(yaw) + np.sin(roll)*np.sin(yaw))],
        [np.cos(pitch) * np.sin(yaw), (np.sin(roll) * np.sin(pitch) * np.sin(yaw) + np.cos(roll * np.cos(yaw))), (np.cos(roll)*np.sin(pitch)*np.sin(yaw) - np.sin(roll)*np.cos(yaw))],
        [-np.sin(roll), np.sin(roll) * np.cos(pitch), np.cos(roll) * np.cos(pitch)]
        ])
        return R
    
    def get_prediction_noise_covariance(self):
        return np.array([
            [0.05, 0,    0,    0,    0,    0,    0,     0,     0,    0,    0,    0,    0,    0,    0,    0,    0,    0],
            [0,    0.05, 0,    0,    0,    0,    0,     0,     0,    0,    0,    0,    0,    0,    0,    0,    0,    0],
            [0,    0,    0.06, 0,    0,    0,    0,     0,     0,    0,    0,    0,    0,    0,    0,    0,    0,    0],
            [0,    0,    0,    0.03, 0,    0,    0,     0,     0,    0,    0,    0,    0,    0,    0,    0,    0,    0],
            [0,    0,    0,    0,    0.03, 0,    0,     0,     0,    0,    0,    0,    0,    0,    0,    0,    0,    0],
            [0,    0,    0,    0,    0,    0.06, 0,     0,     0,    0,    0,    0,    0,    0,    0,    0,    0,    0],
            [0,    0,    0,    0,    0,    0,    0.025, 0,     0,    0,    0,    0,    0,    0,    0,    0,    0,    0],
            [0,    0,    0,    0,    0,    0,    0,     0.025, 0,    0,    0,    0,    0,    0,    0,    0,    0,    0],
            [0,    0,    0,    0,    0,    0,    0,     0,     0.04, 0,    0,    0,    0,    0,    0,    0,    0,    0],
            [0,    0,    0,    0,    0,    0,    0,     0,     0,    0.01, 0,    0,    0,    0,    0,    0,    0,    0],
            [0,    0,    0,    0,    0,    0,    0,     0,     0,    0,    0.01, 0,    0,    0,    0,    0,    0,    0],
            [0,    0,    0,    0,    0,    0,    0,     0,     0,    0,    0,    0.02, 0,    0,    0,    0,    0,    0],
            [0,    0,    0,    0,    0,    0,    0,     0,     0,    0,    0,    0,    0.01, 0,    0,    0,    0,    0],
            [0,    0,    0,    0,    0,    0,    0,     0,     0,    0,    0,    0,    0,    0.01, 0,    0,    0,    0],
            [0,    0,    0,    0,    0,    0,    0,     0,     0,    0,    0,    0,    0,    0,    0.015,0,    0,    0],
            [0,    0,    0,    0,    0,    0,    0,     0,     0,    0,    0,    0,    0,    0,    0,    0.001,0,    0],
            [0,    0,    0,    0,    0,    0,    0,     0,     0,    0,    0,    0,    0,    0,    0,    0,    0.001,0],
            [0,    0,    0,    0,    0,    0,    0,     0,     0,    0,    0,    0,    0,    0,    0,    0,    0,    0.001]
        ])
    def get_prediction_jacobian(self, state, delta_t):
        dxdroll = delta_t * ((np.cos(state[3])*np.sin(state[4])*np.cos(state[5]) + np.sin(state[3])*np.sin(state[5]))*state[7] + (-np.sin(state[3])*np.sin(state[4])*np.cos(state[5])+np.cos(state[3])*np.sin(state[5]))*state[8]) + \
             0.5*delta_t**2*((np.cos(state[3])*np.sin(state[4])*np.cos(state[5]) + np.sin(state[3])*np.sin(state[5]))*state[13] + (-np.sin(state[3])*np.sin(state[4])*np.cos(state[5])+np.cos(state[3])*np.sin(state[5]))*state[14])
        dxdpitch = delta_t * ((-np.sin(state[4])*np.cos(state[5]))*state[6] + (np.sin(state[3])*np.cos(state[4])*np.cos(state[5]))*state[7] + (np.cos(state[3])*np.cos(state[4])*np.cos(state[5]))*state[8]) +\
             0.5*delta_t**2* ((-np.sin(state[4])*np.cos(state[5]))*state[12] + (np.sin(state[3])*np.cos(state[4])*np.cos(state[5]))*state[13] + (np.cos(state[3])*np.cos(state[4])*np.cos(state[5]))*state[14])
        dxdyaw = delta_t * ((-np.cos(state[4])*np.sin(state[5]))*state[6] + (-np.sin(state[3])*np.sin(state[4])*np.sin(state[5])-np.cos(state[3])*np.cos(state[5]))*state[7] + (-np.cos(state[3])*np.sin(state[4])*np.sin(state[5])+np.sin(state[3])*np.cos(state[5]))*state[8]) +\
            0.5*delta_t**2*((-np.cos(state[4])*np.sin(state[5]))*state[12] + (-np.sin(state[3])*np.sin(state[4])*np.sin(state[5])-np.cos(state[3])*np.cos(state[5]))*state[13] + (-np.cos(state[3])*np.sin(state[4])*np.sin(state[5])+np.sin(state[3])*np.cos(state[5]))*state[14])
        dxdvx = delta_t * np.cos(state[4])*np.cos(state[5])
        dxdvy = delta_t * (np.sin(state[3])*np.sin(state[4])*np.cos(state[5]) - np.cos(state[3])*np.sin(state[5]))
        dxdvz = delta_t * (np.cos(state[3])*np.sin(state[4])*np.cos(state[5]) + np.sin(state[3])*np.sin(state[5]))
        dxdax = dxdvx*0.5*delta_t
        dxday = dxdvy*0.5*delta_t
        dxdaz = dxdvz*0.5*delta_t

        dydroll = delta_t * ((np.cos(state[3])*np.sin(state[4])*np.sin(state[5])-np.sin(state[3])*np.cos(state[5]))*state[7] + (-np.sin(state[3])*np.sin(state[4])*np.sin(state[5])-np.cos(state[3])*np.cos(state[5]))*state[8]) +\
            0.5*delta_t**2* ((np.cos(state[3])*np.sin(state[4])*np.sin(state[5])-np.sin(state[3])*np.cos(state[5]))*state[13] + (-np.sin(state[3])*np.sin(state[4])*np.sin(state[5])-np.cos(state[3])*np.cos(state[5]))*state[14])
        dydpitch = delta_t *((-np.sin(state[4])*np.sin(state[5]))*state[6] + (np.sin(state[3])*np.cos(state[4])*np.sin(state[5]))*state[7] + (np.cos(state[3])*np.cos(state[4])*np.sin(state[5]))*state[8]) +\
            0.5*delta_t**2* ((-np.sin(state[4])*np.sin(state[5]))*state[12] + (np.sin(state[3])*np.cos(state[4])*np.sin(state[5]))*state[13] + (np.cos(state[3])*np.cos(state[4])*np.sin(state[5]))*state[14])
        dydyaw = delta_t * ((np.cos(state[4])*np.cos(state[5]))*state[6] + (np.sin(state[3])*np.sin(state[4])*np.cos(state[5])-np.cos(state[3])*np.sin(state[5]))*state[7] + (np.cos(state[3])*np.sin(state[4])*np.cos(state[5])+np.sin(state[3])*np.sin(state[5]))*state[8]) +\
            0.5*delta_t**2* ((np.cos(state[4])*np.cos(state[5]))*state[12] + (np.sin(state[3])*np.sin(state[4])*np.cos(state[5])-np.cos(state[3])*np.sin(state[5]))*state[13] + (np.cos(state[3])*np.sin(state[4])*np.cos(state[5])+np.sin(state[3])*np.sin(state[5]))*state[14])
        dydvx = delta_t * np.cos(state[4])*np.sin(state[5])
        dydvy = delta_t * (np.sin(state[3])*np.sin(state[4])*np.sin(state[5]) + np.cos(state[3])*np.cos(state[5]))
        dydvz = delta_t * (np.cos(state[3])*np.sin(state[4])*np.sin(state[5]) - np.sin(state[3])*np.cos(state[5]))
        dydax = dydvx * 0.5 * delta_t
        dyday = dydvy * 0.5 * delta_t
        dydaz = dydvz * 0.5 * delta_t

        dzdroll = delta_t * ((np.cos(state[3])*np.cos(state[4]))*state[7] + (-np.sin(state[3])*np.cos(state[4]))*state[8]) +\
            0.5*delta_t**2* ((np.cos(state[3])*np.cos(state[4]))*state[13] + (-np.sin(state[3])*np.cos(state[4]))*state[14])
        dzdpitch = delta_t * ((-np.cos(state[4]))*state[6] + (-np.sin(state[3])*np.sin(state[4]))*state[7] + (-np.cos(state[3])*np.sin(state[4]))*state[8]) +\
            0.5*delta_t**2* ((-np.cos(state[4]))*state[12] + (-np.sin(state[3])*np.sin(state[4]))*state[13] + (-np.cos(state[3])*np.sin(state[4]))*state[14])
        dzdyaw = 0
        dzdvx = delta_t * -np.sin(state[4])
        dzdvy = delta_t * np.sin(state[3]) * np.cos(state[4])
        dzdvz = delta_t * np.cos(state[3]) * np.cos(state[4])
        dzdax = dzdvx * 0.5 * delta_t
        dzday = dzdvy * 0.5 * delta_t
        dzdaz = dzdvz * 0.5 * delta_t
        F = np.array([
            [1, 0, 0, dxdroll, dxdpitch, dxdyaw, dxdvx, dxdvy, dxdvz, 0, 0, 0, dxdax, dxday, dxdaz, 0, 0, 0],
            [0, 1, 0, dydroll, dydpitch, dydyaw, dydvx, dydvy, dydvz, 0, 0, 0, dydax, dyday, dydaz, 0, 0, 0],
            [0, 0, 1, dzdroll, dzdpitch, dzdyaw, dzdvx, dzdvy, dzdvz, 0, 0, 0, dzdax, dzday, dzdaz, 0, 0, 0],
            [0, 0, 0, 1, 0, 0, 0, 0, 0, delta_t, 0, 0, 0, 0, 0, 0.5*delta_t**2, 0, 0],
            [0, 0, 0, 0, 1, 0, 0, 0, 0, 0, delta_t, 0, 0, 0, 0, 0, 0.5*delta_t**2, 0],
            [0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, delta_t, 0, 0, 0, 0, 0, 0.5*delta_t**2],
            [0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, delta_t, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, delta_t, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, delta_t, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, delta_t, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, delta_t, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0 ,0, 1, 0, 0, 0, 0, 0, delta_t],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1]
        ])
        return F
if __name__ == "__main__":
    ekf = EKF()
    VO = np.load('./Odometry_data/vo.npy')
    IMU = np.load('./Odometry_data/imu.npy')

    bool_list = [False, False, False, False, False, False, True, True, True, True, True, True, False, False, False, False, False, False]

    time_size = VO.shape[0]

    pose = np.zeros([time_size, 2])

    VO_covariance = np.array([
        [0.2,  0,    0,    0,    0,    0],
        [   0,  0.3, 0,    0,    0,    0],
        [   0,  0,    0.2, 0,    0,    0],
        [   0,  0,    0,    0.01, 0,    0],
        [   0,  0,    0,    0,    0.7, 0],
        [   0,  0,    0,    0,    0,    0.012]
    ])
    IMU_covariance = np.array([
        [0.05,  0,    0,    0,    0,    0],
        [   0,  0.01, 0,    0,    0,    0],
        [   0,  0,    0.06, 0,    0,    0],
        [   0,  0,    0,    0.04, 0,    0],
        [   0,  0,    0,    0,    0.05, 0],
        [   0,  0,    0,    0,    0,    0.025]
    ])

    for i in range(time_size):
        pose[i] = ekf.state[0:2]
        # prediction step
        ekf.prediction_step(0.1)
        # update step
        ekf.update_step(VO[i], bool_list, noise_covariance = VO_covariance)
        ekf.update_step(IMU[i], bool_list, noise_covariance = IMU_covariance)

    plt.scatter(pose[:,0], pose[:,1], s = 1)
    #plt.axis('equal')
    plt.show()
