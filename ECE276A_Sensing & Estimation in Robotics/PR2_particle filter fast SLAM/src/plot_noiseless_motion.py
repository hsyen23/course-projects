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
# init robot pose
P = [0,0] # unit: meters
Angle = 0 # unit: rad
lidar_sample_interval = 50
number_of_particle = 1
iteration = 2317

robot_post = [P[0], P[1], Angle]
lidar_data = pr2_utils.read_data_from_csv('data/sensor_data/lidar.csv')
encoder_data = pr2_utils.read_data_from_csv('data/sensor_data/encoder.csv')
fog_data = pr2_utils.read_data_from_csv('data/sensor_data/fog.csv')

M = setup.MAP()
Particles = setup.Particle(number_of_particle, robot_post, 0.0, 0.00)
Inputs = setup.Input(lidar_sample_interval)
next_input = Inputs.next_input()

# using first observation data to create map
UpdateMap.update_map(M, next(next_input)[1], robot_post, 4)


#Y = lidarRay2world.lidarRay2world(lidar_data, [0,0,np.pi/2])
#x_range = np.arange(-0.4,0.5,0.1)
#y_range = np.arange(0.4,-0.5,-0.1)
#M.plot_map()
#M.plot_conceret_map()
#print(M.mapCorrelation(Y, x_range, y_range))

#

#
for i in range(iteration):
    u_t, u_z, u_v, u_o = next(next_input)
    # first move particles
    Particles.moveParticles(u_t, u_v, u_o)
    # plot trajectory for one point
    x = Particles.list[0][0][0]
    y = Particles.list[0][0][1]
    plt.scatter(x,y)
    