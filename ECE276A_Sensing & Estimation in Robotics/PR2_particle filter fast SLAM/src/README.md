Three major components MAP, Particle, and Input handeling are written as class in "setup.py"

I use class MAP to store the grid map. Only map's hyperparamters need to be changed in __int__ of 'class MAP'.

Other hyperparamters like robot's initial position or value of noise can be tune in "RunMe.py"

(1) "test LiDAR Scan.py" will use smaller grid map to test first Lidar data and visualize.

(2) "plot_noiseless_motion.py" will plot robot's trajectory without noise (Dead-reckoning).

(3) Perform "RunMe.py", it will start particle filter SLAM based on provided data.

(4) "RunMeWithTexture.py" basically same as "RumMe.py" but with texture mapping loop.

Depends on "lidar_sample_interval" and "iteration" these two variable will decide when to stop computation.
If the number of data isn't sufficient to compute, it will use index error to stop.
But, we still can use M.plot_map() and M.plot_concrete_map() to plot log-odd map or binary map.

The path should be:
--Code
-setup.py
-RunMe.py
-UpdateMap.py
-pr2_utils.py
-lidarRay2world.py
-test LiDAR Scan.py
-plot_noiseless_motion.py
-RunMeWithTexture.py
 --data
   --sensor_data
     -encoder.csv
     -fog.csv
     -lidar.csv

##### What MAP class can do #####
MAP(): build a map.
MAP.meter2cell(x_meters, y_meters): turn physical position (unit: meters) into cell position in the map.
MAP.updateLogO(idx, idy, value): update log-odd value of cell at (idx, idy) with value.
MAP.plot_map(): plot log-odd map.
MAP.concrete_map(): return binary map as 2D array.
MAP.plot_concrete_map(): plot binary map.
MAP.mapCorrelation(Y, offx_range, offy_range): use a list of ray point in real world frame(unit: meters) to calculate map-correlation value.
MAP.replace_map(builded_map): load builded_map as current map.
MAP.plot_colormap(): plot texture mapping.
MAP.plot_colormap_on_border(): plot texture mapping but only color the occupied cells.

##### What Particle class can do #####
Particle(n, robot_pose, vnoise, wnoise): build n particles, and all particles initially have same robot_pose.
Particle .moveParticles(t,v,w): move all particles based on t: time period, v: velocity, w: angular velocity
Particle.get_mostlikely_pose(): return the particle pose with highest probability.
Particle.reset_probability(): reset all particles' probability back to 1/(n).

##### What Input class can do #####
Input(lidar_sample_interval): build input handeling, and lidar_sample_interval defines interval between lidar data.
Input.next_input(): it's a generator. call next(Input.next_input()) will return [t,z,v,w]. t: time period, z: lidar observation data, v: velocity, w: angular velocity.