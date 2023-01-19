Run "main.py" to print the result and save gif animation.

There are several hyperparameters can be tuned in "main.py" under the annotation of #hyperparameters.
hyperparameters:
'gamma' is discount factor between 0 and 1.
'processing_time' is time interval in receding-horizon CEC control (variable k in my report).
'R' is a scalar factor for stage cost in using excessive control effort.
'Q' is a scalar factor for stage cost in deviating from the reference position trajectory.
'q' is a scalar factor for stage cost in deviating from the reference orientation trajectory.
'collision' is a bool to control whether enable collision checking in optimal policy.

The path should be:
--Code
-main.py
-utils.py
---fig


#### functions in 'main.py' ####
def lissajous(k): 
input: 
    timestamp: {k}
output: 
    desired reference position and orientation: [xref, yref, thetaref]

def next_error(current_state, control, t):
input: 
    list of current robot position and orientation: {current_state}
    list of control signal: {control}
    timestamp: {t}
output:
    list of error state at timestamp t+1: [e_x, e_y, e_theta]

def CEC_controller(t, cur_state, ref_state):
input:
    timestamp: {t}
    list of current robot position and orientation: {cur_state}
    list of reference state from 'lissajous': {ref_state}
output:
    a list of optimal control policy at timestame t [v,w]

def car_next_state(time_step, cur_state, control, noise):
input:
    time interval between two reference sample: {time_step}
    a list of current robot position and orientation: {cur_state}
    a list of control policy for robot: {control}
    a bool to control whether introduce noise to the motion model: {noise}
output:
    a list of robot position and orientation at next timestamp [x,y,theta]
