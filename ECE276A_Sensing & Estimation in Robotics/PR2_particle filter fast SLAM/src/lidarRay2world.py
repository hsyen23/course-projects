# -*- coding: utf-8 -*-
"""
Created on Sat Feb 19 19:40:18 2022

@author: Yen
"""
import numpy as np
def lidarRay2world(lidar_data, robot_pose):
    theta = robot_pose[2]
    wRb = np.array([[np.cos(theta), -np.sin(theta)],[np.sin(theta), np.cos(theta)]])
    angles = np.linspace(-5, 185, 286) / 180 * np.pi
    
    ranges = lidar_data
    indValid = np.logical_and((ranges < 80),(ranges> 0.1))
    ranges = ranges[indValid]
    angles = angles[indValid]
    # xy position in the sensor frame
    sensor_xs0 = ranges*np.cos(angles)
    sensor_ys0 = ranges*np.sin(angles)
    # change xy position from sensor frame to robot frame
    # I made a assumption lidar center is car center
    xs0 = np.cos(-np.pi/2) * sensor_xs0 - np.sin(-np.pi/2) * sensor_ys0
    ys0 = np.sin(-np.pi/2) * sensor_xs0 + np.cos(-np.pi/2) * sensor_ys0
    # 0.599 is lidar offset to robot
    p_world = []
    # convert to world frame
    for i in range(len(xs0)):
        p_world.append(np.matmul(wRb, np.array([xs0[i], ys0[i]]))+np.array([robot_pose[0],robot_pose[1]]))
    
    list_x = []
    list_y = []
    for each in p_world:
        list_x.append(each[0])
        list_y.append(each[1])
    list_x = np.array(list_x)
    list_y = np.array(list_y)
    return np.stack((list_x, list_y))