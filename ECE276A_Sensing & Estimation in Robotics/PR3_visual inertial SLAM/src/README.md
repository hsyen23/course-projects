I use class to store information about landmarks which is written in "setup.py".

Observation noise and pripor covariance of landmarks can be tunned in 'landmark class' inside "setup.py".

Initial pose, motion noise for robot and percentage of landmarks can be tunned in "main.py".

(1),(2) execute "main.py" will predict robot trajectory by input only, then using the trajectory estimates position of landmarks.

(3) "part_c.py" is similar to "main.py"; however, this time update step work simultaneously on robot pose and landmark position.

Therefore, I slightly change "setup.py" into "setupforpartc.py" which records combined covariance(robot pose and landmark position).

The path should be:
--Code
-setup.py
-main.py
-pr3_utils.py
-part_c.py
-setupforpartc.py

#### What landmark class can do #### (setup.py)
landmark(number_of_landmark): based on number_of_landmark to create array to store mean and covariance.
update_landmark_prior(current_feature, current_robot_pose, c_T_imu, K, b): based on current_feature(feature data from stereo) create pripor distribution for landmarks.
kalman_filter_update(current_feature, current_robot_pose, c_T_imu, K, b): perform EKF update by current_feature for landmark only.

#### What landmark class can do #### ("setupforpartc.py" )
landmark(number_of_landmark): same as above, but covariance is mixed with pose elements.
update_landmark_prior(current_feature, current_robot_pose, c_T_imu, K, b): same as above.
kalman_filter_update(current_feature, current_robot_pose, c_T_imu, K, b): perform EKF update by current_feature robot pose and landmark position simultaneously.