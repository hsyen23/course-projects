import numpy as np
import gym
from utils import *
from example import example_use_of_gym_env

MF = 0 # Move Forward
TL = 1 # Turn Left
TR = 2 # Turn Right
PK = 3 # Pickup Key
UD = 4 # Unlock Door

def doorkey_problem(env, info):
    '''
    You are required to find the optimal path in
        doorkey-5x5-normal.env
        doorkey-6x6-normal.env
        doorkey-8x8-normal.env
        
        doorkey-6x6-direct.env
        doorkey-8x8-direct.env
        
        doorkey-6x6-shortcut.env
        doorkey-8x8-shortcut.env
        
    Feel Free to modify this fuction
    '''
    # initial setting
    initial_pos = env.agent_pos
    initial_dir = env.dir_vec
    door = env.grid.get(info['door_pos'][0], info['door_pos'][1])
    door_pos = door.init_pos
    is_open = door.is_open
    is_locked = door.is_locked
    is_carrying = env.carrying is not None
    grid_width = info['width']
    grid_height = info['height']
    key_pos = info['key_pos']
    goal_pose = info['goal_pos']
    
    
    # facing direction list
    # 0 up
    # 1 rihgt
    # 2 down
    # 3 left
    dir_list = [np.array([0,-1]), np.array([1,0]), np.array([0,1]), np.array([-1, 0])]
    
    # start info
    start_having_key = 1 if is_carrying else 0
    start_open_door = 1 if is_open else 0
    start_y_cor = initial_pos[1]
    start_x_cor = initial_pos[0]
    start_facing_dir = -1
    for i in range(len(dir_list)):
        if np.array_equal(dir_list[i] , initial_dir):
            start_facing_dir = i
    
    value_list = []
    motion_list = []
    while(True):
        # build value matrix for all states
        # having key or not, door open or not, y coordinate, x coordinate, facing direction
        v = np.zeros([2,2,grid_height, grid_width, 4])
        # build motion matrix for all states
        m = np.ones([2, 2,grid_height, grid_width, 4]) * -1
        # check whether it's terminal
        if len(value_list) == 0:
            # terminal
            v[:] = np.inf
            v[:,:, goal_pose[1], goal_pose[0], :] = 0
        else:
            # stage
            # using loop to get through all stage
            for having_key in range(2):
                # 0 = don't have key, 1 = have key
                for opening_door in range(2):
                    # 0 = not open, 1 = open
                    for y_cor in range(grid_height):
                        for x_cor in range(grid_width):
                            for facing_dir in range(4):
                                # get front_cell pos
                                front_pose = np.array([x_cor, y_cor]) + dir_list[facing_dir]
                                # find minimum motion
                                #--------------------
                                # 0. move forward
                                
                                # new version
                                if front_pose[1] < 0 or front_pose[1] >= grid_height or front_pose[0] < 0 or front_pose[0] >= grid_width:
                                    cost1 = np.inf
                                elif env.grid.get(front_pose[0], front_pose[1]) == None:
                                    cost1 = 1 + value_list[-1][having_key,opening_door, front_pose[1], front_pose[0], facing_dir]
                                elif env.grid.get(front_pose[0], front_pose[1]).type == 'key' and having_key == 1:
                                    cost1 = 1 + value_list[-1][having_key,opening_door, front_pose[1], front_pose[0], facing_dir]
                                elif env.grid.get(front_pose[0], front_pose[1]).type == 'goal':
                                    cost1 = 1 + value_list[-1][having_key,opening_door, front_pose[1], front_pose[0], facing_dir]
                                elif env.grid.get(front_pose[0], front_pose[1]).type == 'door' and opening_door == 1:
                                    cost1 = 1 + value_list[-1][having_key,opening_door, front_pose[1], front_pose[0], facing_dir]
                                else:
                                    cost1 = np.inf
                                
                                # 1. turn left
                                cost2 = 1
                                new_facing_dir = facing_dir - 1
                                if new_facing_dir < 0:
                                    new_facing_dir = new_facing_dir + 4
                                cost2 = cost2 + value_list[-1][having_key,opening_door, y_cor, x_cor, new_facing_dir]
                                
                                # 2. turn right
                                cost3 = 1
                                new_facing_dir = facing_dir + 1
                                if new_facing_dir > 3:
                                    new_facing_dir = new_facing_dir - 4
                                cost3 = cost3 + value_list[-1][having_key,opening_door, y_cor, x_cor, new_facing_dir]
                                
                                # 3. pick key up
                                cost4 = np.inf
                                # out of bound
                                if front_pose[1] < 0 or front_pose[1] >= grid_height or front_pose[0] < 0 or front_pose[0] >= grid_width:
                                    cost4 = np.inf
                                elif env.grid.get(front_pose[0], front_pose[1]) == None:
                                    cost4 = np.inf
                                elif having_key == 0 and env.grid.get(front_pose[0], front_pose[1]).type == 'key':
                                    cost4 = 1
                                cost4 = cost4 + value_list[-1][1,opening_door, y_cor, x_cor, facing_dir]
                                
                                # 4. unlock door
                                cost5 = np.inf
                                if front_pose[1] < 0 or front_pose[1] >= grid_height or front_pose[0] < 0 or front_pose[0] >= grid_width:
                                    cost5 = np.inf
                                elif env.grid.get(front_pose[0], front_pose[1]) == None:
                                    cost5 = np.inf
                                elif having_key == 1 and env.grid.get(front_pose[0], front_pose[1]).type == 'door' and opening_door == 0:
                                    cost5 = 1
                                cost5 = cost5 + value_list[-1][1,1, y_cor, x_cor, facing_dir]
                                #--------------------
                                # find the minimum value function
                                cost_list = np.array([cost1, cost2, cost3, cost4, cost5])
                                v[having_key, opening_door, y_cor, x_cor, facing_dir] = np.min(cost_list)
                                if np.isinf(np.min(cost_list)):
                                    m[having_key, opening_door, y_cor, x_cor, facing_dir] = -1
                                else:
                                    m[having_key, opening_door, y_cor, x_cor, facing_dir] = np.argmin(cost_list)
                                
                                
        value_list.append(v)
        motion_list.append(m)
        print('Time:', len(value_list))
        # check exit condition
        if np.isinf(v[start_having_key, start_open_door, start_y_cor, start_x_cor, start_facing_dir]) == False:
            print('value at start: ', v[start_having_key, start_open_door, start_y_cor, start_x_cor, start_facing_dir])
            break
    # debug
    #print(len(value_list))
    
    
    
    # now using table to get optimal control
    optim_act_seq = []
    current_having_key = start_having_key
    current_open_door = start_open_door
    current_y_cor = start_y_cor
    current_x_cor = start_x_cor
    current_facing_dir = start_facing_dir
    #debug

    for i in range(len(motion_list)-1, 0, -1):
        m = motion_list[i]
        act = m[current_having_key, current_open_door, current_y_cor, current_x_cor, current_facing_dir]
        optim_act_seq.append(act)
        # get next state
        if act == 0:
            current_having_key = current_having_key
            current_open_door = current_open_door
            current_y_cor = current_y_cor + dir_list[current_facing_dir][1]
            current_x_cor = current_x_cor + dir_list[current_facing_dir][0]
            current_facing_dir = current_facing_dir
        elif act == 1:
            current_having_key = current_having_key
            current_open_door = current_open_door
            current_y_cor = current_y_cor
            current_x_cor = current_x_cor
            current_facing_dir = current_facing_dir - 1
            if current_facing_dir < 0:
                current_facing_dir = current_facing_dir + 4
        elif act == 2:
            current_having_key = current_having_key
            current_open_door = current_open_door
            current_y_cor = current_y_cor
            current_x_cor = current_x_cor
            current_facing_dir = current_facing_dir + 1
            if current_facing_dir > 3:
                current_facing_dir = current_facing_dir - 4
        elif act == 3:
            current_having_key = 1
            current_open_door = current_open_door
            current_y_cor = current_y_cor
            current_x_cor = current_x_cor
            current_facing_dir = current_facing_dir
        elif act == 4:
            current_having_key = current_having_key
            current_open_door = 1
            current_y_cor = current_y_cor
            current_x_cor = current_x_cor
            current_facing_dir = current_facing_dir
        else:
            print('error: not exsiting action!')
        
    
    return optim_act_seq

def doorkey_problem_rnd(env, info):
    '''
    You are required to find the optimal path in
        doorkey-5x5-normal.env
        doorkey-6x6-normal.env
        doorkey-8x8-normal.env
        
        doorkey-6x6-direct.env
        doorkey-8x8-direct.env
        
        doorkey-6x6-shortcut.env
        doorkey-8x8-shortcut.env
        
    Feel Free to modify this fuction
    '''
    initial_pos = env.agent_pos
    initial_dir = env.dir_vec
    #door = env.grid.get(info['door_pos'][0], info['door_pos'][1])
    #door_pos = door.init_pos
    #is_open = door.is_open
    #is_locked = door.is_locked
    door1_is_open = info['door_open'][0]
    door2_is_open = info['door_open'][1]
    is_carrying = env.carrying is not None
    grid_width = info['width']
    grid_height = info['height']
    key_pos = info['key_pos']
    goal_pose = info['goal_pos']
    
    
    # facing direction list
    # 0 up
    # 1 rihgt
    # 2 down
    # 3 left
    dir_list = [np.array([0,-1]), np.array([1,0]), np.array([0,1]), np.array([-1, 0])]
    
    # start info
    start_having_key = 1 if is_carrying else 0
    start_open_door_1 = 1 if door1_is_open else 0
    start_open_door_2 = 1 if door2_is_open else 0
    start_y_cor = initial_pos[1]
    start_x_cor = initial_pos[0]
    start_facing_dir = -1
    for i in range(len(dir_list)):
        if np.array_equal(dir_list[i] , initial_dir):
            start_facing_dir = i
    
    value_list = []
    motion_list = []
    while(True):
        # build value matrix for all states
        # having key or not, door_1 open or not, door_2 open or not, y coordinate, x coordinate, facing direction
        v = np.zeros([2,2,2,grid_height, grid_width, 4])
        # build motion matrix for all states
        m = np.ones([2, 2,2,grid_height, grid_width, 4]) * -1
        # check whether it's terminal
        if len(value_list) == 0:
            # terminal
            v[:] = np.inf
            v[:,:,:, goal_pose[1], goal_pose[0], :] = 0
        else:
            # stage
            # using loop to get through all stage
            for having_key in range(2):
                # 0 = don't have key, 1 = have key
                for opening_door_1 in range(2):
                    # 0 = not open, 1 = open
                    for opening_door_2 in range(2):
                        for y_cor in range(grid_height):
                            for x_cor in range(grid_width):
                                for facing_dir in range(4):
                                    # get front_cell pos
                                    front_pose = np.array([x_cor, y_cor]) + dir_list[facing_dir]
                                    # find minimum motion
                                    #--------------------
                                    # 0. move forward
                                    
                                    # new version
                                    if front_pose[1] < 0 or front_pose[1] >= grid_height or front_pose[0] < 0 or front_pose[0] >= grid_width:
                                        cost1 = np.inf
                                    elif env.grid.get(front_pose[0], front_pose[1]) == None:
                                        cost1 = 1 + value_list[-1][having_key,opening_door_1, opening_door_2, front_pose[1], front_pose[0], facing_dir]
                                    elif env.grid.get(front_pose[0], front_pose[1]).type == 'key' and having_key == 1:
                                        cost1 = 1 + value_list[-1][having_key,opening_door_1, opening_door_2, front_pose[1], front_pose[0], facing_dir]
                                    elif env.grid.get(front_pose[0], front_pose[1]).type == 'goal':
                                        cost1 = 1 + value_list[-1][having_key,opening_door_1, opening_door_2, front_pose[1], front_pose[0], facing_dir]
                                    elif env.grid.get(front_pose[0], front_pose[1]).type == 'door':
                                        if front_pose[1] == 2 and opening_door_1 == 1:
                                            # door 1 case
                                            cost1 = 1 + value_list[-1][having_key,opening_door_1, opening_door_2, front_pose[1], front_pose[0], facing_dir]
                                        elif front_pose[1] == 5 and opening_door_2 == 1:
                                            # door 2 case
                                            cost1 = 1 + value_list[-1][having_key,opening_door_1, opening_door_2, front_pose[1], front_pose[0], facing_dir]
                                        else:
                                            cost1 = np.inf
                                    else:
                                        cost1 = np.inf
                                    
                                    
                                    '''
                                    cost1 = 1
                                    # out of bound case
                                    if front_pose[1] < 0 or front_pose[1] >= grid_height or front_pose[0] < 0 or front_pose[0] >= grid_width:
                                        cost1 = np.inf
                                    elif env.grid.get(front_pose[0], front_pose[1]) == None:
                                        cost1 = cost1 + value_list[-1][having_key,opening_door, front_pose[1], front_pose[0], facing_dir]
                                    elif env.grid.get(front_pose[0], front_pose[1]).type == 'goal':
                                        cost1 = cost1 + value_list[-1][having_key,opening_door, front_pose[1], front_pose[0], facing_dir]
                                    elif env.grid.get(front_pose[0], front_pose[1]).type == 'wall':
                                        cost1 = np.inf
                                    elif env.grid.get(front_pose[0], front_pose[1]).type == 'door':
                                        # no door open
                                        if opening_door == 0:
                                            cost1 = np.inf
                                        else:
                                            cost1 = cost1 + value_list[-1][having_key,opening_door, front_pose[1], front_pose[0], facing_dir]
                                    '''
                                    # 1. turn left
                                    cost2 = 1
                                    new_facing_dir = facing_dir - 1
                                    if new_facing_dir < 0:
                                        new_facing_dir = new_facing_dir + 4
                                    cost2 = cost2 + value_list[-1][having_key,opening_door_1, opening_door_2, y_cor, x_cor, new_facing_dir]
                                    # 2. turn right
                                    cost3 = 1
                                    new_facing_dir = facing_dir + 1
                                    if new_facing_dir > 3:
                                        new_facing_dir = new_facing_dir - 4
                                    cost3 = cost3 + value_list[-1][having_key,opening_door_1, opening_door_2, y_cor, x_cor, new_facing_dir]
                                    # 3. pick key up
                                    cost4 = np.inf
                                    # out of bound
                                    if front_pose[1] < 0 or front_pose[1] >= grid_height or front_pose[0] < 0 or front_pose[0] >= grid_width:
                                        cost4 = np.inf
                                    elif env.grid.get(front_pose[0], front_pose[1]) == None:
                                        cost4 = np.inf
                                    elif having_key == 0 and env.grid.get(front_pose[0], front_pose[1]).type == 'key':
                                        cost4 = 1
                                    cost4 = cost4 + value_list[-1][1,opening_door_1, opening_door_2, y_cor, x_cor, facing_dir]
                                    # 4. unlock door
                                    cost5 = np.inf
                                    if front_pose[1] < 0 or front_pose[1] >= grid_height or front_pose[0] < 0 or front_pose[0] >= grid_width:
                                        cost5 = np.inf
                                    elif env.grid.get(front_pose[0], front_pose[1]) == None:
                                        cost5 = np.inf
                                    elif having_key == 1 and env.grid.get(front_pose[0], front_pose[1]).type == 'door':
                                        if front_pose[1] == 2:
                                            # door 1 case
                                            if opening_door_1 == 0:
                                                cost5 = 1 + value_list[-1][1,1,opening_door_2 ,y_cor, x_cor, facing_dir]
                                            else:
                                                cost5 = np.inf
                                        else:
                                            # door 2 case
                                            if opening_door_2 == 0:
                                                cost5 = 1 + value_list[-1][1,opening_door_1,1,y_cor, x_cor, facing_dir]
                                            else:
                                                cost5 = np.inf
                                    #cost5 = cost5 + value_list[-1][1,1, y_cor, x_cor, facing_dir]
                                    #--------------------
                                    cost_list = np.array([cost1, cost2, cost3, cost4, cost5])
                                    v[having_key, opening_door_1,opening_door_2, y_cor, x_cor, facing_dir] = np.min(cost_list)
                                    if np.isinf(np.min(cost_list)):
                                        m[having_key, opening_door_1, opening_door_2, y_cor, x_cor, facing_dir] = -1
                                    else:
                                        m[having_key, opening_door_1, opening_door_2, y_cor, x_cor, facing_dir] = np.argmin(cost_list)
                                
                                
        value_list.append(v)
        motion_list.append(m)
        print('Time:', len(value_list))
        # check exit condition
        if np.isinf(v[start_having_key, start_open_door_1,start_open_door_2, start_y_cor, start_x_cor, start_facing_dir]) == False:
            print('value at start: ', v[start_having_key, start_open_door_1,start_open_door_2, start_y_cor, start_x_cor, start_facing_dir])
            break
    # debug
    #print(len(value_list))
    
    
    
    # now using table to get optimal control
    optim_act_seq = []
    current_having_key = start_having_key
    current_open_door_1 = start_open_door_1
    current_open_door_2 = start_open_door_2
    current_y_cor = start_y_cor
    current_x_cor = start_x_cor
    current_facing_dir = start_facing_dir
    
    for i in range(len(motion_list)-1, 0, -1):
        m = motion_list[i]
        act = m[current_having_key, current_open_door_1, current_open_door_2, current_y_cor, current_x_cor, current_facing_dir]
        optim_act_seq.append(act)
        # get next state
        if act == 0:
            current_having_key = current_having_key
            current_open_door_1 = current_open_door_1
            current_open_door_2 = current_open_door_2
            current_y_cor = current_y_cor + dir_list[current_facing_dir][1]
            current_x_cor = current_x_cor + dir_list[current_facing_dir][0]
            current_facing_dir = current_facing_dir
        elif act == 1:
            current_having_key = current_having_key
            current_open_door_1 = current_open_door_1
            current_open_door_2 = current_open_door_2
            current_y_cor = current_y_cor
            current_x_cor = current_x_cor
            current_facing_dir = current_facing_dir - 1
            if current_facing_dir < 0:
                current_facing_dir = current_facing_dir + 4
        elif act == 2:
            current_having_key = current_having_key
            current_open_door_1 = current_open_door_1
            current_open_door_2 = current_open_door_2
            current_y_cor = current_y_cor
            current_x_cor = current_x_cor
            current_facing_dir = current_facing_dir + 1
            if current_facing_dir > 3:
                current_facing_dir = current_facing_dir - 4
        elif act == 3:
            current_having_key = 1
            current_open_door_1 = current_open_door_1
            current_open_door_2 = current_open_door_2
            current_y_cor = current_y_cor
            current_x_cor = current_x_cor
            current_facing_dir = current_facing_dir
        elif act == 4:
            current_having_key = current_having_key
            current_y_cor = current_y_cor
            current_x_cor = current_x_cor
            current_facing_dir = current_facing_dir
            # find which door is unlocked then assign 1 to it
            # get front cell
            front_cell_y_cor = current_y_cor + dir_list[current_facing_dir][1]
            if front_cell_y_cor == 2:
                current_open_door_1 = 1
                current_open_door_2 = current_open_door_2
            else:
                current_open_door_1 = current_open_door_1
                current_open_door_2 = 1
        else:
            print('error: not exsiting action!')
        
    
    return optim_act_seq

def partA():
    env_path = './envs/doorkey-8x8-normal.env'
    env, info = load_env(env_path) # load an environment
    plot_env(env)
    seq = doorkey_problem(env, info) # find the optimal action sequence
    print('action: ',seq)
    draw_gif_from_seq(seq, load_env(env_path)[0]) # draw a GIF & save
    # plot state at time t under optimal control policy
    t_interval = 2
    current_t = 0
    while(True):
        draw_state_at_time_t(seq, load_env(env_path)[0], current_t)
        current_t = current_t + t_interval
        if current_t > len(seq):
            break
    
def partB():
    env_folder = './envs/random_envs'
    env, info, env_path = load_random_env(env_folder)
    plot_env(env)
    seq = doorkey_problem_rnd(env, info)
    print('action: ',seq)
    draw_gif_from_seq(seq, load_env(env_path)[0])
    # plot state at time t under optimal control policy
    t_interval = 2
    current_t = 0
    while(True):
        draw_state_at_time_t(seq, load_env(env_path)[0], current_t)
        current_t = current_t + t_interval
        if current_t > len(seq):
            break
if __name__ == '__main__':
    #example_use_of_gym_env()
    #partA()
    partB()

        
        
    
