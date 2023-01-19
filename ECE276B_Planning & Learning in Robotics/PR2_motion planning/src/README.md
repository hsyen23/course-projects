I use class "node" and "Planner" to store information about environment which is written in "Planner.py".

Run "main.py" to test the algorithm on different MAP.
It will generate a series of motion plot in "gif" folder, we can run "save_gif_test.py" to create gif animation with name "motion.gif".
For better visualization on how robot move, please refer to gif animation in "gif animation" folder.

There is one hyperparameter can be tuned in "main.py", in function "runtest" we can change the value n to expand n cells during each iteration.

Procedure: change "test_map0()" in "main.py" -> run "main.py" -> run "save_gif_test.py" to create gif animation.


The path should be:
--Code
-main.py
-Planner.py
-targetplanner.py
-save_gif_test.py
---maps
---gif
---gif animation

#### What node class can do #### ("Planner.py")
node(x, y, state, targetpos): generate a node with following properties:
self.x = x -> x position for the node in 2D grid map.
self.y = y -> y position for the node in 2D grid map.
self.state = state -> 0 as free, 1 as obstacle.
self.g = inifinity.
self.h = euclidean distance between the node and target. 
self.children = [] -> it will be connected later
self.cost2children = [] -> it will be calculated once a child added.

connect(listOfNodes): append all nodes in listOfNodes in self.children, and assign corresponding cost in self.cost2children.
update_h(targetpos): based on the new input targetpos to update h to deal with moving target case.

#### What Planner class can do #### ("Planner.py")
Planner(_map, _robotpos, _targetpos): it will call initiate_map(_map) to create the environment, and store current robot and target position based on _robotpos and _targetpos.
initiate_map(_map): a function help to create environment based on _map information (it will create nodes and connects them automatically).
robotplanner(targetpos, n): it will generate next position the robot should go based on the real-time adaptive A* presented in report. (it will automatically reset g for each nodes to prepare the environment for next call)