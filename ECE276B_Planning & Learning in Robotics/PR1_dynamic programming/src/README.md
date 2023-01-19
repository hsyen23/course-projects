There are two main functions in "doorkey.py", one is "doorkey_problem(env, info)", another is "doorkey_problem_rnd(env, info)".
Both functions will return sequence of optimal action. We can use "partA()" or "partB()" to write the .gif file and plot the optimal path on console.
The sequence will follow the dictionary: {0 : MF, 1 : TL, 2 : TR, 3 : PK, 4 : UD}.

To perform Part A or Part B, directly execute "doorkey.py" (you can comment "partB()" to perform partA() and otherwise).

### parameter ###
"t_interval" is a hyperparameter in "partA()" and "partB()", it makes plot with a period of "t_interval" steps.
Changing "env_path" in "partA()" can change the environment.

### hardcode ###
I hardcode cost function in "doorkey.py" and "doorkey_problem(env, info)". 
As represented in report, all resonable action will make the cost:1, and not reasonable action will make the cost:infinity.

### additional function in "utils.py" ###
Add "draw_state_at_time_t(seq, env, t)" in "utils.py" to help plot the state under the optimal policy at time t (with value function displayed on top-left).