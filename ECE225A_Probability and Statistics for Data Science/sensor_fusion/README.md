Python version of robot_localization pkg in ROS.

`ekf.py` has class `ekf` which implements sensor fusion in robot_localization.

`helper/getVIO.py` and `helper/getOdometry.py` generate `imu.npy` and `vo.npy` for running an example in `ekf`.

Explanation of algorithm is written in report pdf.


# Example

## IMU trajectory
![Alt text](./pic/imu_trajectory.png "imu_trajectory")

## Visual Odometry trajectory
![Alt text](./pic/vio_trajectory.png "vio_trajectory")

## Sensor fusion (ekf)
![Alt text](./pic/VIO+IMU.png "VIO+IMU")
