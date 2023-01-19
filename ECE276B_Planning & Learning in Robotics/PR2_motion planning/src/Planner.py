# -*- coding: utf-8 -*-
"""
Created on Mon May  9 09:09:15 2022

@author: Yen
"""

import numpy as np
import math
import time

class node:
    def __init__(self, x, y, state, targetpos):
        self.x = x
        self.y = y
        self.state = state
        self.g = np.inf
        self.h = np.sqrt((targetpos[0] - self.x)**2 + (targetpos[1] - self.y)**2)
        self.children = []
        self.cost2children = []
        
        self.flag = False
        self.parent = None
    def __lt__(self, other):
        return (self.g + self.h) < (other.g + other.h)
    
    def __repr__(self):
        return 'p_{}_{}'.format(self.x, self.y)
    
    def connect(self, listOfNodes):
        if self.state == 1:
            return
        for each in listOfNodes:
            if each.state == 1:
                continue
            self.children.append(each)
            dis = (self.x - each.x)**2 + (self.y - each.y)**2
            if dis > 1:
                self.cost2children.append(np.sqrt(2))
            else:
                self.cost2children.append(1.0)
                
    def update_h(self, targetpos):
        new_dist = np.sqrt((targetpos[0] - self.x)**2 + (targetpos[1] - self.y)**2)
        self.h = max(self.h, new_dist)
        if self.flag == False:
            self.h = new_dist
class Planner:
    def __init__(self, _map, _robotpos, _targetpos):
        self.map = []
        self.robotpos = _robotpos
        self.targetpos = _targetpos
        self.initiate_map(_map)
        
    def initiate_map(self, _map):
        # create node map
        rows = _map.shape[0]
        cols = _map.shape[1]
        for r in range(rows):
            r_list = []
            for c in range(cols):
                r_list.append(node(r,c,_map[r,c],self.targetpos))
            self.map.append(r_list)
        print('map created!')
        # connect node map
        for r in range(rows):
            for c in range(cols):
                current_node = self.map[r][c]
                listOfNodes = []
                # loop surrounding
                for x in range(-1,2,1):
                    for y in range(-1,2,1):
                        x_pos = r + x
                        y_pos = c + y
                        if x_pos < 0 or x_pos >= rows or y_pos < 0 or y_pos >= cols or (x == 0 and y == 0):
                            continue
                        listOfNodes.append(self.map[x_pos][y_pos])
                current_node.connect(listOfNodes)
        print('all node connected!')
        
    def robotplanner(self, targetpos, n):
        open_list = []
        close_list = []
        
        start_node = self.map[self.robotpos[0]][self.robotpos[1]]
        start_node.g = 0
        
        target_node = self.map[targetpos[0]][targetpos[1]]
        target_node.h = 0
        
        open_list.append(start_node)
        
        # A*
        while(len(open_list) != 0):
            open_list.sort()
            current_node = open_list[0]
            open_list.remove(current_node)
            close_list.append(current_node)
            
            for i in range(len(current_node.children)):
                nb_node = current_node.children[i]
                if nb_node in close_list :
                    continue
                if current_node.g + current_node.cost2children[i] <= nb_node.g:
                    if current_node.g + current_node.cost2children[i] <= target_node.g:
                        nb_node.g = current_node.g + current_node.cost2children[i]
                        nb_node.parent = current_node
                        if nb_node not in open_list:
                            open_list.append(nb_node)
                        
            if target_node in close_list or len(close_list)>= n:
                break
        
        # A* finished
        #print('Expand {} nodes, {} in OPEN'.format(len(close_list), len(open_list)))
        
        # update h in open list first
        for each in open_list:
            each.update_h(targetpos)
        open_list.sort()
        
        #print('The min f is {}'.format(open_list[0]))
        #print('g: {}, h:{}'.format(open_list[0].g , open_list[0].h))
        
        min_f = open_list[0].g + open_list[0].h
        min_node = open_list[0]
        
        # below is using real-time adaptive A*
        # update expanded nodes
        for each in close_list:
            #each.h = min_f - each.g
            #each.h = max(each.h, min_f - each.g)
            each.h = min_f - each.g
            each.flag = True
            # reset g process
            each.g = np.inf
        
        # try to update by LRTA*
        
        
        # rest g process
        for each in open_list:
            each.g = np.inf
        '''
        # choose movement
        tem_list = start_node.children
        move_node = tem_list[0]

        for each in tem_list:
            if each.h < move_node.h and each in close_list:
                move_node = each
        '''
        # choose movement v.2
        #print('start Node = {}'.format(start_node))
        
        while min_node.parent != start_node:
            #print("{} -> ".format(min_node), end = ' ')
            min_node = min_node.parent
        #print("{} -> ".format(min_node))
        
        #print('Move to {}'.format(min_node))
        self.robotpos = np.array([min_node.x, min_node.y])
        return np.array([min_node.x, min_node.y])
        
    '''
        print('Move to {}'.format(move_node))
        self.robotpos = np.array([move_node.x, move_node.y])
        return np.array([move_node.x, move_node.y])
    '''