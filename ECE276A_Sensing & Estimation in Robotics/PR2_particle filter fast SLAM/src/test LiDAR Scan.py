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

class MAP:
    def __init__(self):
        self.res = 0.1 #meters
        self.xmin = -50 #meters
        self.ymin = -50
        self.xmax = 50
        self.ymax = 50
        self.sizex = int(np.ceil((self.xmax - self.xmin) / self.res + 1)) #cells
        self.sizey =int(np.ceil((self.ymax - self.ymin) / self.res + 1))
        self.map = np.zeros((self.sizex,self.sizey),dtype=np.int8) #DATA TYPE: char or int8
        self.colormap = np.zeros((self.sizex,self.sizey, 3),dtype=np.int8)
        # try comment this time
        self.logOmin = -200
        self.logOmax = 200
    def meter2cell(self, x_meters, y_meters):
        # return indices of cell
        return [np.ceil((y_meters - self.ymin) / self.res ).astype(np.int16)-1, np.ceil((x_meters - self.xmin) / self.res ).astype(np.int16)-1]
    def updateLogO(self, idx, idy, value):
        self.map[idy, idx] += value
        if self.map[idy, idx] < self.logOmin:
           self.map[idy, idx] = self.logOmin
        if self.map[idy, idx] > self.logOmax:
            self.map[idy, idx] = self.logOmax
    def plot_map(self):
        # plot the map in correct coordinate as usual
        # first convert log O back to probability
        probability = np.copy(self.map)
        probability = probability.astype(float)
        for j in range(len(probability)):
            for i in range(len(probability[0])):
                probability[j][i] = math.exp(probability[j][i]) / (1+math.exp(probability[j][i]))
        plt.figure()
        plt.imshow(probability, origin='lower',extent=[self.xmin,self.xmax,self.ymin,self.ymax])
    def concrete_map(self):
        # return concrete_map with 0 or 1
        # 1 for occupied
        return (self.map > 0).astype(int)
    def plot_concrete_map(self):
        plt.figure()
        plt.imshow(self.concrete_map(), origin='lower',extent=[self.xmin,self.xmax,self.ymin,self.ymax], cmap='gray')
    
    def color_map(self, cell_index_y, cell_index_x, color):
        # color all index
        # color is RGB order
        self.colormap[cell_index_y][cell_index_x][0] = color[0]
        self.colormap[cell_index_y][cell_index_x][1] = color[1]
        self.colormap[cell_index_y][cell_index_x][2] = color[2]
    
    def plot_colormap(self):
        plt.figure()
        plt.imshow(self.colormap, origin='lower',extent=[self.xmin,self.xmax,self.ymin,self.ymax])
        
    def plot_colormap_on_border(self):
        border_map = np.copy(self.colormap)
        for j in range(self.sizey):
            for i in range(self.sizex):
                if self.map[j][i] <= 0:
                    border_map[j][i] = np.array([0,0,0])
        
        plt.figure()
        plt.imshow(border_map, origin='lower',extent=[self.xmin,self.xmax,self.ymin,self.ymax])

    def mapCorrelation(self, Y, offx_range, offy_range):
        # Y is list of ray point in real world frame(unit: meters)
        
        # important offset range for y is large to small, for x is small to large
        counter = 0
        c_map = self.concrete_map()
        
        l_x = len(offx_range)
        l_y = len(offy_range)
        
        C = np.zeros([l_y, l_x])
        
        
        for ioy in range(l_y):
            oy = offy_range[ioy]
            for iox in range(l_x):
                ox = offx_range[iox]
                offseted_Y = np.array([Y[0] + ox,Y[1] + oy])
                counter = 0
                for i in range(len(offseted_Y[0])):
                    dy, dx = self.meter2cell(offseted_Y[0][i], offseted_Y[1][i])
                    if 0 <= dy < self.sizey and 0 <= dx < self.sizex:
                        if c_map[dy][dx] == 1:
                            counter += 1
                C[ioy, iox] = counter
        return C
    
    def replace_map(self, builded_map):
        self.map = builded_map
    
# init robot pose
P = [0,0] # unit: meters
Angle = 0 # unit: rad
lidar_sample_interval = 20
number_of_particle = 1
LogO = 4

robot_post = [P[0], P[1], Angle]
lidar_data = pr2_utils.read_data_from_csv('data/sensor_data/lidar.csv')
encoder_data = pr2_utils.read_data_from_csv('data/sensor_data/encoder.csv')
fog_data = pr2_utils.read_data_from_csv('data/sensor_data/fog.csv')

M = MAP()

check_map = np.copy(M.map)

Particles = setup.Particle(number_of_particle, robot_post, 0.00, 0.000)
Inputs = setup.Input(lidar_sample_interval)
next_input = Inputs.next_input()

# using first observation data to create map
UpdateMap.update_map(M, next(next_input)[1], robot_post, LogO)
M.plot_map()

aa, bb = pr2_utils.compute_stereo(1544582648735466220, [0,0,0],M)
for j in range(560):
    for i in range(1280):
        if np.isnan(aa[j][i][0]):
            continue
        else:
            j_index = int(aa[j][i][0])
            i_index = int(aa[j][i][1])
            # extract RGB value
            Red = bb[j][i][2]
            Green = bb[j][i][1]
            Blue = bb[j][i][0]
            M.color_map(j_index,i_index,[Red, Green, Blue])

M.plot_colormap()
M.plot_colormap_on_border()