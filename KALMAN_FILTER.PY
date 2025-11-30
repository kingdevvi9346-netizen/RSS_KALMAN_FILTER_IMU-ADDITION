#!/usr/bin/env python3
import rospy
import numpy as np
from nav_msgs.msg import Odometry
from geometry_msgs.msg import Twist
from sensor_msgs.msg import Imu
from tf.transformations import quaternion_from_euler, euler_from_quaternion
from std_msgs.msg import Float64MultiArray
import math

class SimpleKalmanFilterNode:
    def __init__(self):
        rospy.init_node('kalman_filter_simple', anonymous=True)
        self.user_name = rospy.get_param('~user_name', 'default_name')

        # --- EKF Tuning Parameters ---
        self.dt = rospy.get_param('~dt', 0.1)
        self.Q = np.array([
            [rospy.get_param('~q_x', 0.01), 0, 0],
            [0, rospy.get_param('~q_y', 0.01), 0],
            [0, 0, rospy.get_param('~q_yaw', 0.01)]
        ])
        self.R_gps = np.array([
            [rospy.get_param('~r_gps_x', 0.25), 0],
            [0, rospy.get_param('~r_gps_y', 0.25)]
        ])
        self.R_odom1 = np.array([
            [rospy.get_param('~r_odom_x', 1.0), 0, 0],
            [0, rospy.get_param('~r_odom_y', 1.0), 0],
            [0, 0, rospy.get_param('~r_odom_yaw', 0.1)]
        ])
        self.R_imu = rospy.get_param('~r_imu_yaw', 0.01)

        # --- Sensor Mode ---
        # 0: odometry only, 1: odometry+GPS, 2: odometry+GPS+IMU
        self.sensor_mode = rospy.get_param('~sensor_mode', 2)

        # Subscribers
        rospy.Subscriber('/fake_gps', Odometry, self.gps_callback)
        rospy.Subscriber('/odom1', Odometry, self.odom1_callback)
        rospy.Subscriber('/cmd_vel', Twist, self.cmd_vel_callback)
        rospy.Subscriber('/imu', Imu, self.imu_callback)

        # Publishers
        self.pub = rospy.Publisher('/kalman_estimate', Odometry, queue_size=10)
        self.pub_gaussian = rospy.Publisher('/gaussian_info', Float64MultiArray, queue_size=10)
        self.pub_error = rospy.Publisher('/localization_error', Float64MultiArray, queue_size=10)
        # State: [x, y, yaw]
        self.x = np.zeros((3,1))
        self.P = np.eye(3) * 0.1
        # Latest command velocities
        self.vx = 0.0
        self.vy = 0.0
        self.yaw_rate = 0.0

        # Latest sensor measurements
        self.gps = None
        self.odom1 = None
        self.imu_yaw_rate = None
        # For error analysis
        self.true_pose = None

        rospy.loginfo("Waiting for simulation clock to start...")
        while rospy.Time.now() == rospy.Time(0) and not rospy.is_shutdown():
            rospy.sleep(0.1)
        rospy.loginfo("Simulation clock active!")
        rospy.Timer(rospy.Duration(self.dt), self.update_kalman)
        rospy.loginfo("Kalman Filter with Gaussian representation started - Vivek Devarapalli")

    def cmd_vel_callback(self, msg):
        self.vx = msg.linear.x
        self.vy = msg.linear.y
        self.yaw_rate = msg.angular.z

    def gps_callback(self, msg):
        self.gps = np.array([
            [msg.pose.pose.position.x],
            [msg.pose.pose.position.y]
        ])
        # For error analysis, assume /fake_gps is ground truth
        self.true_pose = np.array([
            [msg.pose.pose.position.x],
            [msg.pose.pose.position.y],
            [self.x[2,0]]
        ])

    def odom1_callback(self, msg):
        orientation_q = msg.pose.pose.orientation
        orientation_list = [orientation_q.x, orientation_q.y, orientation_q.z, orientation_q.w]
        (_, _, yaw) = euler_from_quaternion(orientation_list)
        self.odom1 = np.array([
            [msg.pose.pose.position.x],
            [msg.pose.pose.position.y],
            [yaw]
        ])

    def imu_callback(self, msg):
        # Use yaw rate from IMU (z axis angular velocity)
        self.imu_yaw_rate = msg.angular_velocity.z

    def update_kalman(self, event):
        # --- Prediction Step ---
        yaw = float(self.x[2])
        vx_world = self.vx * np.cos(yaw) - self.vy * np.sin(yaw)
        vy_world = self.vx * np.sin(yaw) + self.vy * np.cos(yaw)
        # Use IMU yaw rate if available and mode==2
        if self.sensor_mode == 2 and self.imu_yaw_rate is not None:
            yaw_rate = self.imu_yaw_rate
        else:
            yaw_rate = self.yaw_rate
        x_pred = np.array([
            [self.x[0] + vx_world * self.dt],
            [self.x[1] + vy_world * self.dt],
            [self.x[2] + yaw_rate * self.dt]
        ])
        x_pred[2] = np.arctan2(np.sin(x_pred[2]), np.cos(x_pred[2]))
        F = np.array([
            [1.0, 0.0, -(self.vx * np.sin(yaw) + self.vy * np.cos(yaw)) * self.dt],
            [0.0, 1.0,  (self.vx * np.cos(yaw) - self.vy * np.sin(yaw)) * self.dt],
            [0.0, 0.0,  1.0]
        ])
        P_pred = np.dot(np.dot(F, self.P), F.T) + self.Q
        self.x = x_pred
        self.P = P_pred

        # --- Correction Step ---
        # Odometry only
        if self.sensor_mode == 0 and self.odom1 is not None:
            self.kalman_update(self.odom1, np.eye(3), self.R_odom1)
            self.odom1 = None
        # Odometry + GPS
        elif self.sensor_mode == 1:
            if self.odom1 is not None:
                self.kalman_update(self.odom1, np.eye(3), self.R_odom1)
                self.odom1 = None
            if self.gps is not None:
                H_gps = np.array([[1,0,0],[0,1,0]])
                self.kalman_update(self.gps, H_gps, self.R_gps)
            self.gps = None
        # Odometry + GPS + IMU
        elif self.sensor_mode == 2:
            if self.odom1 is not None:
                self.kalman_update(self.odom1, np.eye(3), self.R_odom1)
                self.odom1 = None
            if self.gps is not None:
                H_gps = np.array([[1,0,0],[0,1,0]])
                self.kalman_update(self.gps, H_gps, self.R_gps)
                self.gps = None
            # IMU yaw rate update (1D)
            if self.imu_yaw_rate is not None:
                H_imu = np.array([[0,0,1]])
                z_imu = np.array([[self.x[2,0] + self.imu_yaw_rate * self.dt]])
                self.kalman_update(z_imu, H_imu, np.array([[self.R_imu]]))

        self.P = (self.P + self.P.T) / 2.0
        self.publish_estimate()
        self.publish_gaussian_info()
        self.publish_error()

    def kalman_update(self, z, H, R):
        y = z - np.dot(H, self.x)
        if y.shape[0] == 1:
            y[0] = np.arctan2(np.sin(y[0]), np.cos(y[0]))
        elif y.shape[0] == 3:
            y[2] = np.arctan2(np.sin(y[2]), np.cos(y[2]))
        S = np.dot(np.dot(H, self.P), H.T) + R
        K = np.dot(np.dot(self.P, H.T), np.linalg.inv(S))
        self.x = self.x + np.dot(K, y)
        self.x[2] = np.arctan2(np.sin(self.x[2]), np.cos(self.x[2]))
        I = np.eye(3)
        self.P = np.dot((I - np.dot(K, H)), self.P)

    def publish_estimate(self):
        msg = Odometry()
        msg.header.stamp = rospy.Time.now()
        msg.header.frame_id = "odom"
        msg.pose.pose.position.x = float(self.x[0])
        msg.pose.pose.position.y = float(self.x[1])
        msg.pose.pose.position.z = 0.0
        q = quaternion_from_euler(0, 0, float(self.x[2]))
        msg.pose.pose.orientation.x = q[0]
        msg.pose.pose.orientation.y = q[1]
        msg.pose.pose.orientation.z = q[2]
        msg.pose.pose.orientation.w = q[3]
        msg.pose.covariance[0] = float(self.P[0,0])
        msg.pose.covariance[1] = float(self.P[0,1])
        msg.pose.covariance[6] = float(self.P[1,0])
        msg.pose.covariance[7] = float(self.P[1,1])
        msg.pose.covariance[35] = float(self.P[2,2])
        msg.twist.twist.linear.x = float(self.vx)
        msg.twist.twist.linear.y = float(self.vy)
        msg.twist.twist.angular.z = float(self.yaw_rate)
        self.pub.publish(msg)

    def publish_gaussian_info(self):
        msg = Float64MultiArray()
        std_devs = np.sqrt(np.diag(self.P))
        msg.data = [
            float(self.x[0]), float(self.x[1]), float(self.x[2]),
            float(self.P[0,0]), float(self.P[1,1]), float(self.P[2,2]),
            float(self.P[0,1]),
            float(std_devs[0]), float(std_devs[1]), float(std_devs[2]),
            float(np.linalg.det(self.P)), float(np.trace(self.P)),
            0.0 # Reserved for author
        ]
        rospy.loginfo(f"Gaussian info published by: {self.user_name}")
        self.pub_gaussian.publish(msg)

    def publish_error(self):
        if self.true_pose is not None:
            error = np.linalg.norm(self.x[:2] - self.true_pose[:2])
            msg = Float64MultiArray()
            msg.data = [float(error)]
            self.pub_error.publish(msg)

if __name__ == '__main__':
    try:
        node = SimpleKalmanFilterNode()
        rospy.spin()
    except rospy.ROSInterruptException:
        pass

