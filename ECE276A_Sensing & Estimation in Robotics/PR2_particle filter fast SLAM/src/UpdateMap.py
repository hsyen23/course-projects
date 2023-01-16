# -*- coding: utf-8 -*-
"""
Created on Fri Feb 18 21:01:28 2022

@author: Yen
"""
import numpy as np
import pr2_utils

def update_map(MAP,lidar_data, robot_pose, inverse_odd_ratio):
    '''
    

    '''
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
    
    ##### comment is default, I try to use provided pose
    
    xs0 = np.cos(-np.pi/2) * sensor_xs0 - np.sin(-np.pi/2) * sensor_ys0
    ys0 = np.sin(-np.pi/2) * sensor_xs0 + np.cos(-np.pi/2) * sensor_ys0
    
    p_world = []
    # convert to world frame
    for i in range(len(xs0)):
        p_world.append(np.matmul(wRb, np.array([xs0[i], ys0[i]]))+np.array([robot_pose[0],robot_pose[1]]))
    
    # get cell coordinates in grid
    B_cell = MAP.meter2cell(robot_pose[0], robot_pose[1])
    p_cell = []
    for each in p_world:
        py, px = MAP.meter2cell(each[0], each[1])
        p_cell.append([py, px])
    
    # so far so good
    
    # using bresenham2D and update the path
    for each in p_cell:
        sx = B_cell[1]
        sy = B_cell[0]
        ex = each[1]
        ey = each[0]
        path = pr2_utils.bresenham2D(sx, sy, ex, ey)
        # before end point we increase Log O
        for i in range(len(path[0])-1):
            idx = int(path[0,i])
            idy = int(path[1,i])
            if 0 <= idx < MAP.sizex and 0 <= idy < MAP.sizey:
                MAP.updateLogO(idx,idy,-inverse_odd_ratio)
        # update for last point which is blocked
        idx = int(path[0,-1])
        idy = int(path[1,-1])
        if 0 <= idx < MAP.sizex and 0 <= idy < MAP.sizey:
            MAP.updateLogO(idx,idy,inverse_odd_ratio)