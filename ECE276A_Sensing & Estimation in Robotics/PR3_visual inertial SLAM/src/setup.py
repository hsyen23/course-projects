# -*- coding: utf-8 -*-
"""
Created on Sat Mar  5 20:47:56 2022

@author: Yen
"""
import numpy as np
from numpy.linalg import inv
from scipy.linalg import expm

def get_random_landmark(n, val_pct):
    n_val = int(n * val_pct)
    indices = np.random.permutation(n)
    return indices[:n_val]

def hatmap(x_hat):
    x_T = np.zeros([4,4,x_hat.shape[1]])
    for i in range(x_hat.shape[1]):
        theta_map = np.array([[0, -x_hat[5,i], x_hat[4,i]],[x_hat[5,i],0,-x_hat[3,i]],[-x_hat[4,i],x_hat[3,i],0]])
        p = np.array([[x_hat[0,i]],[x_hat[1,i]],[x_hat[2,i]]])
        x_T[:,:,i] = np.block([
            [theta_map, p],
            [np.zeros((1, 3)),0]
            ])
    return x_T

def skinny_hat(se3):
    output = np.zeros([6,6,se3.shape[1]])
    for i in range(se3.shape[1]):
        theta_map = np.array([[0, -se3[5,i], se3[4,i]],[se3[5,i],0,-se3[3,i]],[-se3[4,i],se3[3,i],0]])
        v_map = np.array([[0, -se3[2,i], se3[1,i]],[se3[2,i],0,-se3[0,i]],[-se3[1,i],se3[0,i],0]])
        output[:,:,i] = np.block([
            [theta_map, v_map],
            [np.zeros((3, 3)),theta_map]
            ])
    return output

def skinny_hat(se3):
    output = np.zeros([6,6,se3.shape[1]])
    for i in range(se3.shape[1]):
        theta_map = np.array([[0, -se3[5,i], se3[4,i]],[se3[5,i],0,-se3[3,i]],[-se3[4,i],se3[3,i],0]])
        v_map = np.array([[0, -se3[2,i], se3[1,i]],[se3[2,i],0,-se3[0,i]],[-se3[1,i],se3[0,i],0]])
        output[:,:,i] = np.block([
            [theta_map, v_map],
            [np.zeros((3, 3)),theta_map]
            ])
    return output

def sigle_twist_map(x_hat):
    x_T = np.zeros([4,4])
    theta_map = np.array([[0, -x_hat[5], x_hat[4]],[x_hat[5],0,-x_hat[3]],[-x_hat[4],x_hat[3],0]])
    p = np.array([[x_hat[0]],[x_hat[1]],[x_hat[2]]])
    x_T[:,:] = np.block([
        [theta_map, p],
        [np.zeros((1, 3)),0]
        ])
    return x_T

class landmark:
    def __init__(self, number_of_landmark):
        self.landmark_mean = np.empty([3,number_of_landmark])
        self.landmark_mean[:,:] = np.nan
        self.landmark_covariance = np.empty([3, 3, number_of_landmark])
        self.landmark_covariance[:,:,:] = np.nan
        self.observed_landmark = set()
        self.number_of_landmark = number_of_landmark
        # up to 3 (when it comes 4 it will create singular)
        self.observation_noise = 4*np.eye(4)
        self.observation_noise_factor = 1
    def update_landmark_prior(self, current_feature, current_robot_pose, c_T_imu, K, b):
        if len(self.observed_landmark) == self.number_of_landmark:
            return
        fsu = K[0,0]
        cu = K[0,2]
        fsv = K[1,1]
        cv = K[1,2]
        fsub = b * fsu
        
        imu_T_w = inv(current_robot_pose)
        #o_T_r = np.array([[0,-1,0,0],[0,0,-1,0],[1,0,0,0],[0,0,0,1]])
        #comined = np.matmul(np.matmul(o_T_r, c_T_imu) , imu_T_w)
        comined = np.matmul(c_T_imu, imu_T_w)
        
        for i in range(current_feature.shape[1]):
            if current_feature[0, i] == -1 or (i in self.observed_landmark):
                continue
            
            ul = current_feature[0,i]
            vl = current_feature[1,i]
            ur = current_feature[2,i]
            left_side_eq = np.array([[fsu,0,cu-ul],[0,fsv,cv-vl],[fsu,0,cu-ur]])
            right_side_eq = np.array([[0],[0],[fsub]])
            optical_frame = np.matmul(inv(left_side_eq) , right_side_eq)
            optical_frame = np.concatenate((optical_frame, np.array([[1]])), axis=0)
    
            world_frame = np.matmul(inv(comined) , optical_frame)
            # add world frame to prior
            self.landmark_mean[:, i] = world_frame[0:3].reshape((3))
            # change covariance from nan to zeros
            #print('Hello, I am dealing with {}th landmark'.format(i))
            #print('Before:')
            #print(self.landmark_covariance[3*i : 3*(i+1), 3*i : 3*(i+1)])
            #print('After :')
            
            # up to 3 (when it comes 4 it will create singular)
            self.landmark_covariance[:, :, i] = 1*np.eye(3)
            #print(self.landmark_covariance[3*i : 3*(i+1), 3*i : 3*(i+1)])
            # add to observed set
            self.observed_landmark.add(i)
            '''
            # testify whether matrix mul correct
            Q = np.matmul(comined, world_frame)
            q3 = Q[2]
            K_s = np.array([[fsu,0,cu,0],[0,fsv,cv,0],[fsu,0,cu,-fsub],[0,fsv,cv,0]])
            pixel = np.matmul(K_s,(Q/q3))
            print('This is {}th pixel, original coordinate:'.format(i))
            print('ul: {}, vl: {}, ur: {}, vr: {}'.format(current_feature[0,i],current_feature[1,i],current_feature[2,i],current_feature[3,i]))
            print('Calculated pixel coordinat:')
            print('ul: {}, vl: {}, ur: {}, vr: {}'.format(pixel[0],pixel[1],pixel[2],pixel[3]))
            '''

    def kalman_filter_update(self,current_feature, current_robot_pose, c_T_imu, K, b):
        # because we proceed prior update before, we can assure that each landmark that camera hit has probability already.
        
        imu_T_w = inv(current_robot_pose)
        #o_T_r = np.array([[0,-1,0,0],[0,0,-1,0],[1,0,0,0],[0,0,0,1]])
        #combined = np.matmul(np.matmul(o_T_r, c_T_imu) , imu_T_w)
        combined = np.matmul(c_T_imu, imu_T_w)
        fsu = K[0,0]
        cu = K[0,2]
        fsv = K[1,1]
        cv = K[1,2]
        fsub = b * fsu
        K_s = np.array([[fsu,0,cu,0],[0,fsv,cv,0],[fsu,0,cu,-fsub],[0,fsv,cv,0]])
        P_transpose = np.array([[1,0,0],[0,1,0],[0,0,1],[0,0,0]])
        homogeneous_mean = np.concatenate((self.landmark_mean, np.ones([1,self.number_of_landmark])),axis=0 )
        
        for i in range(current_feature.shape[1]):
            if current_feature[0, i] == -1 or np.isnan(self.landmark_mean[0,i]):
                continue
            # first comput H
            # current mean in optical frame = q
            current_homo_mean = homogeneous_mean[:,i]
            
            q = np.matmul(combined, current_homo_mean)
            
            dpi_dq = (1/q[2]) * np.array([[1,0,-q[0]/q[2],0],[0,1,-q[1]/q[2],0],[0,0,0,0],[0,0,-q[3]/q[2],1]])
            H = np.matmul(K_s, np.matmul(dpi_dq, np.matmul(combined, P_transpose)))
            # now comput its kalman gain
            pre_cal = np.matmul(self.landmark_covariance[:,:,i], np.transpose(H))
            
            matrix2inv = np.matmul(H, pre_cal) + self.observation_noise
            while not (np.linalg.matrix_rank(matrix2inv) == 4):
                print('singular happen')
                matrix2inv = matrix2inv + 0.01 * np.eye(4)
                
            inv_matrix = inv(matrix2inv)
            kalman_gain = np.matmul(pre_cal, inv_matrix)
            q_after_pi = q / q[2]
            predicted_z = np.matmul(K_s, q_after_pi)
            new_mean = self.landmark_mean[:,i] + np.matmul(kalman_gain, current_feature[:,i]-predicted_z)
            new_covariance = np.matmul((np.eye(3) - np.matmul(kalman_gain, H)) , self.landmark_covariance[:,:,i])
            # reassign value
            self.landmark_mean[:,i] = new_mean
            self.landmark_covariance[:,:,i] = new_covariance
            
    def kalman_filter_robot_update(self,current_feature, current_robot_pose, c_T_imu, K, b, robot_covariance):
        # because we proceed prior update before, we can assure that each landmark that camera hit has probability already.
        
        imu_T_w = inv(current_robot_pose)
        #o_T_r = np.array([[0,-1,0,0],[0,0,-1,0],[1,0,0,0],[0,0,0,1]])
        #combined = np.matmul(np.matmul(o_T_r, c_T_imu) , imu_T_w)
        combined = np.matmul(c_T_imu, imu_T_w)
        fsu = K[0,0]
        cu = K[0,2]
        fsv = K[1,1]
        cv = K[1,2]
        fsub = b * fsu
        K_s = np.array([[fsu,0,cu,0],[0,fsv,cv,0],[fsu,0,cu,-fsub],[0,fsv,cv,0]])
        
        #P_transpose = np.array([[1,0,0],[0,1,0],[0,0,1],[0,0,0]])
        homogeneous_mean = np.concatenate((self.landmark_mean, np.ones([1,self.number_of_landmark])),axis=0 )
        
        #Overall_H = np.array([]) # this maybe not be used in here
        landmark_record = []
        stacked_truth_z = np.array([])
        stacked_predicted_z = np.array([])
        detached_H_for_state = np.array([])

        def add_to_big_H(small_H, Big_H):
            if len(Big_H) == 0:
                Big_H = small_H
            else:
                small_width = small_H.shape[1]
                big_height = Big_H.shape[0]
                small_height = small_H.shape[0]
                big_width = Big_H.shape[1]
                
                Big_H = np.block([
                    [Big_H, np.zeros([big_height, small_width])],
                    [np.zeros([small_height, big_width]), small_H]
                    ])
            return Big_H
        
        def add_stacked_z(small_z, big_z):
            if len(big_z) == 0:
                big_z = small_z
            else:
                big_z = np.concatenate((big_z, small_z),axis=0 )
            return big_z
        
        def circle_hat(x):
            s_hat = np.array([[0,-x[2],x[1]],[x[2],0,-x[0]],[-x[1],x[0],0]])
            output = np.block([
                [np.eye(3),-s_hat],
                [np.zeros([1,3]),np.zeros([1,3])]
                ])
            return output
        
        # first deal with observation value part, then deal with robot state part
        for i in range(current_feature.shape[1]):
            if current_feature[0, i] == -1 or np.isnan(self.landmark_mean[0,i]):
                continue
            # first filter out not observered point and point that didn't have prior now.
            
            # first comput H
            # current mean in optical frame = q
            current_homo_mean = homogeneous_mean[:,i]
            
            q = np.matmul(combined, current_homo_mean)
            
            dpi_dq = (1/q[2]) * np.array([[1,0,-q[0]/q[2],0],[0,1,-q[1]/q[2],0],[0,0,0,0],[0,0,-q[3]/q[2],1]])
            repeated_part = np.matmul(K_s, dpi_dq)
            # I don't need this H here
            #H = np.matmul(repeated_part, np.matmul(combined, P_transpose))
            # now we have H for this observed point, add it to big H
            
            #Overall_H = add_to_big_H(H, Overall_H)

            # add to record which landmark point is placed
            landmark_record.append(i)
            
            
            # stack ground truth z and predicted z
            stacked_truth_z = add_stacked_z(current_feature[:, i], stacked_truth_z)
            q_after_pi = q / q[2]
            predicted_z = np.matmul(K_s, q_after_pi)
            stacked_predicted_z = add_stacked_z(predicted_z, stacked_predicted_z)
            
            # here compute H for robot state part
            H_for_state =  np.matmul(-repeated_part, np.matmul(c_T_imu, circle_hat(np.matmul(imu_T_w, current_homo_mean))))
            detached_H_for_state = add_stacked_z(H_for_state, detached_H_for_state)
            '''
            # now comput its kalman gain
            pre_cal = np.matmul(self.landmark_covariance[:,:,i], np.transpose(H))
            
            matrix2inv = np.matmul(H, pre_cal) + self.observation_noise
            while not (np.linalg.matrix_rank(matrix2inv) == 4):
                print('singular happen')
                matrix2inv = matrix2inv + 0.01 * np.eye(4)
                
            inv_matrix = inv(matrix2inv)
            kalman_gain = np.matmul(pre_cal, inv_matrix)
            q_after_pi = q / q[2]
            predicted_z = np.matmul(K_s, q_after_pi)
            new_mean = self.landmark_mean[:,i] + np.matmul(kalman_gain, current_feature[:,i]-predicted_z)
            new_covariance = np.matmul((np.eye(3) - np.matmul(kalman_gain, H)) , self.landmark_covariance[:,:,i])
            # reassign value
            self.landmark_mean[:,i] = new_mean
            self.landmark_covariance[:,:,i] = new_covariance
            '''
            
        # debug???
        if len(landmark_record) == 0:
            return current_robot_pose, robot_covariance
            
        # now for last part of H, the robot state part
        # last part still will be compute in landmark loop, here we just concatenate it to big H
        Overall_H = detached_H_for_state
        
        # decompose the whole big covariance matrix into only observed one with state covariance
        # we use record list to extract fraction covariance here
        
        number_of_observed = len(landmark_record)
        '''
        fraction_covariance = np.zeros([3*number_of_observed + 6, 3*number_of_observed + 6])
        for obs_index in range(number_of_observed):
            cor_index = landmark_record[obs_index]
            fraction_covariance[3*obs_index : 3*(obs_index+1), 3*obs_index : 3*(obs_index+1)] = self.landmark_covariance[3*cor_index : 3*(cor_index+1), 3*cor_index : 3*(cor_index+1)]
            # also need cor with state part(the part at bottom and right)
            fraction_covariance[3*obs_index : 3*(obs_index+1), -6:] = self.landmark_covariance[3*cor_index : 3*(cor_index+1), -6:]
            fraction_covariance[-6:, 3*obs_index : 3*(obs_index+1)] = self.landmark_covariance[-6:, 3*cor_index : 3*(cor_index+1)]
        # add state covariance
        fraction_covariance[-6: , -6:] = self.landmark_covariance[-6: , -6:]
        '''
        fraction_covariance = robot_covariance
        
        # finally, compute kalman gain and update mean and covariance
        ##################debug#################
        #print(fraction_covariance)
        #print('#####')
        #print(np.transpose(Overall_H))
        ####################debug###############
        kal_repeated_part = np.matmul(fraction_covariance, np.transpose(Overall_H))
        # define obs noise
        observation_noise = self.observation_noise_factor*np.eye(4*number_of_observed)
        kalman_gain = np.matmul(kal_repeated_part, inv(  np.matmul(Overall_H, kal_repeated_part)   + observation_noise  ))
        # update the covariance and then assign back to class_covariance instance
        fraction_covariance = np.matmul((np.eye(6) - np.matmul(kalman_gain, Overall_H)) , fraction_covariance)
        '''
        for obs_index in range(number_of_observed):
            cor_index = landmark_record[obs_index]
            self.landmark_covariance[3*cor_index : 3*(cor_index+1), 3*cor_index : 3*(cor_index+1)] = fraction_covariance[3*obs_index : 3*(obs_index+1), 3*obs_index : 3*(obs_index+1)]
            # also need cor with state part(the part at bottom and right)
            self.landmark_covariance[3*cor_index : 3*(cor_index+1), -6:] = fraction_covariance[3*obs_index : 3*(obs_index+1), -6:]
            self.landmark_covariance[-6:, 3*cor_index : 3*(cor_index+1)] = fraction_covariance[-6:, 3*obs_index : 3*(obs_index+1)]
        # add state covariance
        self.landmark_covariance[-6: , -6:] = fraction_covariance[-6: , -6:]
        '''
        # now update the mean, and assign back
        innovation = np.matmul(kalman_gain, stacked_truth_z - stacked_predicted_z)
        # based on innovation update the mean
        '''
        for obs_index in range(number_of_observed):
            cor_index = landmark_record[obs_index]
            self.landmark_mean[:, cor_index] = self.landmark_mean[:, cor_index] + innovation[3*obs_index : 3*(obs_index+1)]
        '''
        # ok, final part, update the mean for state
        return np.matmul(current_robot_pose ,expm(sigle_twist_map(innovation))), fraction_covariance
            
        