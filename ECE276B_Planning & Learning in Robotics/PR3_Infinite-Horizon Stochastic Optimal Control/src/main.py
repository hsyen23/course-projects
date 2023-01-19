from time import time
import numpy as np
from utils import visualize
from casadi import *

# Simulation params
np.random.seed(10)
time_step = 0.5 # time between steps in seconds
sim_time = 120    # simulation time

# Car params
x_init = 1.5
y_init = 0.0
theta_init = np.pi/2
v_max = 1
v_min = 0
w_max = 1
w_min = -1

# hyperparameters
gamma = 0.85
processing_time = 20
R = 1
Q = 1
q = 1
collision = True
# This function returns the reference point at time step k
def lissajous(k):
    xref_start = 0
    yref_start = 0
    A = 2
    B = 2
    a = 2*np.pi/50
    b = 3*a
    T = np.round(2*np.pi/(a*time_step))
    k = k % T
    delta = np.pi/2
    xref = xref_start + A*np.sin(a*k*time_step + delta)
    yref = yref_start + B*np.sin(b*k*time_step)
    v = [A*a*np.cos(a*k*time_step + delta), B*b*np.cos(b*k*time_step)]
    thetaref = np.arctan2(v[1], v[0])
    return [xref, yref, thetaref]

# This function implements a simple P controller
def simple_controller(cur_state, ref_state):
    k_v = 0.55
    k_w = 1.0
    v = k_v*np.sqrt((cur_state[0] - ref_state[0])**2 + (cur_state[1] - ref_state[1])**2)
    v = np.clip(v, v_min, v_max)
    angle_diff = ref_state[2] - cur_state[2]
    angle_diff = (angle_diff + np.pi) % (2 * np.pi ) - np.pi
    w = k_w*angle_diff
    w = np.clip(w, w_min, w_max)
    return [v,w]

def next_error(current_state, control, t):
    p_x = current_state[0]
    p_y = current_state[1]
    p_theta = current_state[2]
    v = control[0]
    w = control[1]
    r_x, r_y, r_theta = lissajous(t)
    r_x_next, r_y_next, r_theta_next = lissajous(t+1)
    next_state = []
    next_state.append(p_x + time_step * cos(p_theta + r_theta) * v + r_x - r_x_next)
    next_state.append(p_y + time_step * sin(p_theta + r_theta) * v + r_y - r_y_next)
    next_state.append(p_theta + time_step * w + r_theta - r_theta_next)
    return next_state[0], next_state[1], next_state[2]

def CEC_controller(t, cur_state, ref_state):
    opti = Opti()
    T = sim_time / time_step
    n_t = int(T - t)
    if n_t > processing_time:
        n_t = processing_time
    u = opti.variable(2, n_t)
    L = 0
    p_x = cur_state[0] - ref_state[0]
    p_y = cur_state[1] - ref_state[1]
    p_theta = cur_state[2] - ref_state[2]
    for i in range(n_t):
        # get error state at t
        f = gamma ** i * (Q*p_x * p_x + Q*p_y * p_y + q*(1 - 2*cos(p_theta) + cos(p_theta) * cos(p_theta)) + R*u[0, i] * u[0, i] + R*u[1, i] * u[1, i])
        L = L + f
        p_x, p_y, p_theta = next_error([p_x, p_y, p_theta], [u[0, i], u[1, i]], i + t)
        
        r_x, r_y, r_theta = lissajous(i+t+1)
        robot_x = p_x + r_x
        robot_y = p_y + r_y
        condition_1 = (robot_x + 2) * (robot_x + 2) + (robot_y + 2) * (robot_y + 2)
        condition_2 = (robot_x - 1) * (robot_x - 1) + (robot_y - 2) * (robot_y - 2)
        opti.subject_to(robot_x > -3)
        opti.subject_to(robot_x < 3)
        opti.subject_to(robot_y > -3)
        opti.subject_to(robot_y < 3)
        if collision:
            opti.subject_to(condition_1 >= 0.27)
            opti.subject_to(condition_2 >= 0.27)
        
    #terminal_cost = p_x * p_x + p_y * p_y
    L = L
    '''
    nlp = {}                 # NLP declaration
    nlp['x']= u # decision vars
    nlp['f'] = L            # objective
    F = nlpsol('F','ipopt',nlp)
    tem = [0] * n_t
    ans = F(x0=tem,ubg=1,lbg=-1)
    print(ans['x'][:,0])
    '''
    opti.minimize(L)
    for i in range(n_t):
        opti.subject_to(-1 <= u[1,i])
        opti.subject_to(1 >= u[1,i])
        opti.subject_to(0 <= u[0,i])
        opti.subject_to(1 >= u[0,i])

    opti.solver('ipopt')
    #sol = opti.solve()
    try:
        sol = opti.solve()
    except:
        print(opti.debug.show_infeasibilities)
        print('iteration:{}'.format(t))
    if n_t == 1:
       return [sol.value(u)[0], sol.value(u)[1]]
    return [sol.value(u)[0,0], sol.value(u)[1,0]]
# This function implement the car dynamics
def car_next_state(time_step, cur_state, control, noise = True):
    theta = cur_state[2]
    rot_3d_z = np.array([[np.cos(theta), 0], [np.sin(theta), 0], [0, 1]])
    f = rot_3d_z @ control
    mu, sigma = 0, 0.04 # mean and standard deviation for (x,y)
    w_xy = np.random.normal(mu, sigma, 2)
    mu, sigma = 0, 0.004  # mean and standard deviation for theta
    w_theta = np.random.normal(mu, sigma, 1)
    w = np.concatenate((w_xy, w_theta))
    if noise:
        return cur_state + time_step*f.flatten() + w
    else:
        return cur_state + time_step*f.flatten()

if __name__ == '__main__':
    # %matplotlib qt
    # Obstacles in the environment
    obstacles = np.array([[-2,-2,0.5], [1,2,0.5]])
    # Params
    traj = lissajous
    ref_traj = []
    error = 0.0
    car_states = []
    times = []
    # Start main loop
    main_loop = time()  # return time in sec
    # Initialize state
    cur_state = np.array([x_init, y_init, theta_init])
    cur_iter = 0
    # Main loop
    while (cur_iter * time_step < sim_time):
        t1 = time()
        # Get reference state
        cur_time = cur_iter*time_step
        cur_ref = traj(cur_iter)
        # Save current state and reference state for visualization
        ref_traj.append(cur_ref)
        car_states.append(cur_state)

        ################################################################
        # Generate control input
        # TODO: Replace this simple controller with your own controller
        #control = simple_controller(cur_state, cur_ref)
        #print("[v,w]", control)
        control = CEC_controller(cur_iter,cur_state, cur_ref)
        # return control need to be numpy
        
        ################################################################

        # Apply control input
        next_state = car_next_state(time_step, cur_state, control, noise=True)
        # Update current state
        cur_state = next_state
        # Loop time
        t2 = time()
        print(cur_iter)
        print(t2-t1)
        times.append(t2-t1)
        error = error + np.linalg.norm(cur_state - cur_ref)
        cur_iter = cur_iter + 1
    
    main_loop_time = time()
    print('\n\n')
    print('Total time: ', main_loop_time - main_loop)
    print('Average iteration time: ', np.array(times).mean() * 1000, 'ms')
    print('Final error: ', error)

    # Visualization
    ref_traj = np.array(ref_traj)
    car_states = np.array(car_states)
    times = np.array(times)
    visualize(car_states, ref_traj, obstacles, times, time_step, save=True)
    
