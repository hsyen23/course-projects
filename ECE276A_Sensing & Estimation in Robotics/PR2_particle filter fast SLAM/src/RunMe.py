# -*- coding: utf-8 -*-
"""
Created on Sat Feb 19 23:06:34 2022

@author: Yen
"""

import setup
import UpdateMap
import lidarRay2world
import pr2_utils
import numpy as np
import math
import matplotlib.pyplot as plt;


##### hyperparameters ###########################
# init robot pose
P = [0,0] # unit: meters
Angle = 0 # unit: rad
lidar_sample_interval = 50
number_of_particle = 1
# iteration up to 6400
iteration = 2400
mapcorrelation_offset_x = np.arange(-2,2,1)
mapcorrelation_offset_y = np.arange(2,-2,-1)
v_noise = 0.1
w_noise = 0.03
#mapcorrelation_offset_x = np.arange(1)
#mapcorrelation_offset_y = np.arange(1)
LogO = 4
##### end of hyperparameters ####################



robot_post = [P[0], P[1], Angle]
lidar_data = pr2_utils.read_data_from_csv('data/sensor_data/lidar.csv')
encoder_data = pr2_utils.read_data_from_csv('data/sensor_data/encoder.csv')
fog_data = pr2_utils.read_data_from_csv('data/sensor_data/fog.csv')

M = setup.MAP()
Particles = setup.Particle(number_of_particle, robot_post, v_noise, w_noise)
Inputs = setup.Input(lidar_sample_interval)
next_input = Inputs.next_input()

# using first observation data to create map
UpdateMap.update_map(M, next(next_input)[1], robot_post, LogO)

#
particle_hist = []
#

for i in range(iteration):
    u_t, u_z, u_v, u_o = next(next_input)
    # first move particles
    # prediction step
    Particles.moveParticles(u_t, u_v, u_o)
    
    # update step
    
    # each point's obervation to world frame

    alpah_normalization = 0
    
    for j in range(number_of_particle):
        particle_pose = np.copy(Particles.list[j][0])
        lidar_data_in_this_particle = lidarRay2world.lidarRay2world(u_z, particle_pose)
        C = M.mapCorrelation(lidar_data_in_this_particle,mapcorrelation_offset_x,mapcorrelation_offset_y)
        # loop C to find largest value with x and y coordinate
        tem_buffer = -1
        largest_xid = -1
        largest_yid = -1
        for tem_j in range(len(C)):
            for tem_i in range(len(C[0])):
                value_here = C[tem_j][tem_i]
                if value_here > tem_buffer:
                    tem_buffer = value_here
                    largest_xid = tem_i
                    largest_yid = tem_j
                    
        # update position by this offset
        ##### if I common this line, we only take mapcor value, but not update its position
        particle_pose = particle_pose + np.array([mapcorrelation_offset_x[largest_xid], mapcorrelation_offset_y[largest_yid], 0])
        #####
        alpah = tem_buffer * Particles.list[j][1]
        
        Particles.list[j] = [particle_pose, alpah]
        alpah_normalization += alpah
    
    # apply normalize alpha
        
    for j in range(number_of_particle):
        Particles.list[j][1] = Particles.list[j][1] / alpah_normalization
    #print('print alpah_normalization to check problem: {}'.format(alpah_normalization) )
    # pick highest particle to update map
    
    robot_post, isprobability = Particles.get_mostlikely_pose()
    particle_hist.append([robot_post[0], robot_post[1]])
    UpdateMap.update_map(M, u_z, robot_post, LogO)
    
    # resample here
    #Particles.special_resample(0.001)
    #Particles.reset_probability()
    # done!
#M.plot_conceret_map()
M.plot_map()
print('The end point is: ')
print(Particles.get_mostlikely_pose())
'''
# plot robot's trajectory
for each in particle_hist:
    plt.scatter(each[0],each[1])
'''