# -*- coding: utf-8 -*-
"""
Created on Fri Feb 18 19:34:37 2022

@author: Yen
"""

import pr2_utils
import numpy as np
import math
import matplotlib.pyplot as plt
import pr2_utils
# first create grip map, using log odd ratio to present element in the matrix.
# init MAP
class MAP:
    def __init__(self):
        self.res = 2 #meters
        self.xmin = -1500 #meters
        self.ymin = -1500
        self.xmax = 1500
        self.ymax = 1500
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
        
class Particle:
    def __init__(self, n, robot_pose, vnoise, wnoise):
        self.list = []
        self.number = n
        self.vnoise = vnoise
        self.wnoise = wnoise
        for i in range(n):
            self.list.append([np.array([robot_pose[0],robot_pose[1],robot_pose[2]]), 1 / n])
            
    def moveParticles(self, t, v, w):
        for i in range (len(self.list)):
            theta = self.list[i][0][2]
            v_noise = np.random.normal(0, self.vnoise)
            w_noise = np.random.normal(0, self.wnoise)
            noised_v = v + v_noise
            noised_w = w + w_noise
            self.list[i][0] = self.list[i][0] + t * np.array([noised_v*np.cos(theta), noised_v*np.sin(theta), noised_w])
                 
    
    def get_mostlikely_pose(self):
        pro = 0
        most_likelyOne = -1
        for i in range (len(self.list)):
            if self.list[i][1] > pro:
                pro = self.list[i][1]
                most_likelyOne = i
        return self.list[most_likelyOne][0], self.list[most_likelyOne][1]
    
    def reset_probability(self):
        for i in range (len(self.list)):
            self.list[i][1] = 1 / self.number
            
    def special_resample(self, thresthold):
        the_most_likelyPose, itpro = self.get_mostlikely_pose()
        
        # u can print info
        #acc = 0
        
        for i in range (len(self.list)):
            if self.list[i][1] <= thresthold:
                self.list[i][0] = np.copy(the_most_likelyPose)
                #acc += 1
        #print("This time we resample : {} points".format(acc))
    
class Input:
    def __init__(self, lidar_sample_interval):
        self.lidar_data = pr2_utils.read_data_from_csv('data/sensor_data/lidar.csv')
        self.encoder_data = pr2_utils.read_data_from_csv('data/sensor_data/encoder.csv')
        self.fog_data = pr2_utils.read_data_from_csv('data/sensor_data/fog.csv')
        self.lidar_index = 0
        self.encoder_index = 0
        self.fog_data_index = 0
        self.lidar_sample_interval = lidar_sample_interval
    def next_input(self):
        # first get newest z (lidar)
        # (index here means not read yet!, we will update index in the end)
        
        # first time, only yiled z
        time_interval = None
        lidar_output = self.lidar_data[1][self.lidar_index]
        velocity_output = None
        omega_output = None
        pre_time = self.lidar_data[0][self.lidar_index]
        currentTimestamp = pre_time
        yield time_interval, lidar_output, velocity_output, omega_output, currentTimestamp
        self.lidar_index += self.lidar_sample_interval
        
        while True:
            # check whether need to stop
            #if self.lidar_index >= len(self.lidar_data[0]) or self.encoder_index >= len(self.encoder_data[0]) or self.fog_index >= len(self.fog_data[0]):
            #    break
            
            # lidar part
            time = self.lidar_data[0][self.lidar_index]
            currentTimestamp = time
            time_interval = (time - pre_time)/ 1000000000
            pre_time = self.lidar_data[0][self.lidar_index]
            
            lidar_output = self.lidar_data[1][self.lidar_index]
            self.lidar_index += self.lidar_sample_interval
            
            # velocity part
            v_start_time = self.encoder_data[0][self.encoder_index]
            encoder_start_left = self.encoder_data[1][self.encoder_index][0]
            encoder_start_right = self.encoder_data[1][self.encoder_index][1]
            while (self.encoder_data[0][self.encoder_index+1] < time):
                self.encoder_index += 1
            v_end_time = self.encoder_data[0][self.encoder_index]
            encoder_end_left = self.encoder_data[1][self.encoder_index][0]
            encoder_end_right = self.encoder_data[1][self.encoder_index][1]
            
            dt = (v_end_time - v_start_time) / 1000000000
            v_left = np.pi * 0.623479 * (encoder_end_left - encoder_start_left) / (4096 * dt)
            v_right = np.pi * 0.622806 * (encoder_end_right - encoder_start_right) / (4096 * dt)
            
            velocity_output = (v_left + v_right)/2
            self.encoder_index += 1
            # omega part
            o_start_time = self.fog_data[0][self.fog_data_index]
            d_yaw = 0
            while (self.fog_data[0][self.fog_data_index] < time):
                d_yaw += self.fog_data[1][self.fog_data_index][2]
                self.fog_data_index += 1
            o_end_time = self.fog_data[0][self.fog_data_index-1]
            dt = (o_end_time - o_start_time) / 1000000000
            omega_output = d_yaw / dt
            
            yield time_interval, lidar_output, velocity_output, omega_output, currentTimestamp
                
            
        