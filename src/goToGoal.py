#!/usr/bin/env python

import rospy
import rospkg
import time
import numpy as np
from tf.transformations import euler_from_quaternion

from nav_msgs.msg import Odometry
from sensor_msgs.msg import LaserScan
from geometry_msgs.msg import Point
from geometry_msgs.msg import Twist
from std_msgs.msg import Int8


class Tracker:
    def __init__(self):
        rospy.init_node("wp_tracker", anonymous=True)
        rospack = rospkg.RosPack()
        self.kt = [rospy.get_param("~k_pt"), rospy.get_param(
            "~k_it"), rospy.get_param("~k_dt")]
        self.previous_error = 0
        self.previous_stamp = rospy.Time.now()
        self.e_int = 0
        self.ImageClass_current = -1
        self.ImageClass_mode = -1
        self.odom = None
        self.dist_to_virtual_goal = 0.075                     # to be tuned
        self.dist_image = 0.325
        self.dist_wall_min = 0.15
        self.angle_error_tolerance = 0.03
        self.maneuver_completed = 0
        self.lidar_front_dist = 10
        self.prev_turn = 0
        # FSM
        self.state = 1
        self.mode_image_class = np.array([0, 0, 0, 0, 0, 0])

        # create ROS tools
        self.goal_pub = rospy.Publisher("/goal", Point, queue_size=1)
        self.cmd_pub = rospy.Publisher("/cmd_vel", Twist, queue_size=1)

        self.image_sub = rospy.Subscriber(
            "/ImageClass", Int8, self.ImageClassReceived, queue_size=1)
        rospy.wait_for_message("/ImageClass", Int8)
        self.odom_sub = rospy.Subscriber(
            "/odom", Odometry, self.odomCallback, queue_size=1)
        self.scan_sub = rospy.Subscriber(
            "/scan", LaserScan, self.scanCallback, queue_size=1)

    def odomCallback(self, msg):
        self.odom = msg

    def ImageClassReceived(self, msg):
        self.ImageClass_current = msg.data
        # print("ImageClass = ", self.ImageClass)
        if 0.35 < self.lidar_front_dist < 0.55 and (self.state == 1 or self.state == 5):
            self.mode_image_class[self.ImageClass_current] += 1
            self.mode_image_class[0] = 1

    def scanCallback(self, scan):
        if self.odom is None:
            return
        print("state = ", self.state)

        d = np.array(scan.ranges)
        d[d <= 0.05] = 1000
        self.lidar_min_dist = np.min(d)
        # print("Lidar min dist = ", self.lidar_min_dist)
        angle_to_consider_front = 15
        d_front = np.concatenate(
            (d[:angle_to_consider_front], d[-angle_to_consider_front:]))
        self.lidar_front_dist = np.min(d_front)
        self.ImgAngle = np.argmin(d_front).astype(float)
        self.ObstacleAngle = np.argmin(d).astype(float)
        print("Lidar front dist = ", self.lidar_front_dist)
        # print("Before conversion ", self.ImgAngle)
        # Convert image position and closest obstacle position to radians
        if self.ImgAngle < angle_to_consider_front:
            self.ImgAngle = self.ImgAngle/180*np.pi
        else:
            self.ImgAngle = (-2*angle_to_consider_front +
                             self.ImgAngle)/180*np.pi
        # print("Image Angle after conversion = ", self.ImgAngle*180/np.pi)

        # if self.ObstacleAngle > 180:
        #     self.ObstacleAngle = self.ObstacleAngle - 360
        # print("Obstacle Angle = ", self.ObstacleAngle)
        self.ObstacleAngle = self.ObstacleAngle/180*np.pi

        # FSM
        # State = 1 -> go straight
        # State = 2 -> read sign
        # State = 3 -> execute sign instructions
        # State = 4 -> no img found at wall, so rotate in place
        # State = 5 -> avoid wall

        # safe, go straight till new img is encountered
        if self.state == 1:
            # print("Enter state 1 code")
            # check transitions
            if self.lidar_front_dist < self.dist_image:
                print("Convert to state 2")
                self.state = 2
                self.readSign()
            if self.lidar_min_dist < self.dist_wall_min:
                print("Convert to state 5")
                self.state = 5
                self.avoidObstacle()
            # print("Sending goal for state 1")
            goal_msg = Point()
            goal_msg.x = self.dist_to_virtual_goal
            goal_msg.y = 0
            self.goal_pub.publish(goal_msg)
        if self.state == 2:
            if self.maneuver_completed == 1:
                self.state = 1
            else:
                self.readSign()

        if self.state == 5:
            self.avoidObstacle()

    def getOdometry(self):
        quat = (self.odom.pose.pose.orientation.x,
                self.odom.pose.pose.orientation.y,
                self.odom.pose.pose.orientation.z,
                self.odom.pose.pose.orientation.w)
        euler = euler_from_quaternion(quat)
        self.theta = euler[2]
        # print("self theta = ", self.theta)

    def avoidObstacle(self):
        print("enter avoid obstacle")
        print("Obstacle angle ", self.ObstacleAngle)
        if 0.1 < self.lidar_min_dist < 0.2:
            alpha = 0.95
        else:
            alpha = 1 - np.exp(-9*self.lidar_min_dist)
        evasion_vector = np.array([-np.cos(self.ObstacleAngle),
                                   -np.sin(self.ObstacleAngle)])
        print("Evasion Vector", evasion_vector, "alpha ", alpha)
        straight_vector = np.array([1, 0])
        avoidance_vector = alpha*straight_vector + (1-alpha)*evasion_vector
        goal_msg = Point()
        goal_msg.x = avoidance_vector[0]*self.dist_to_virtual_goal
        goal_msg.y = avoidance_vector[1]*self.dist_to_virtual_goal
        self.goal_pub.publish(goal_msg)
        if self.lidar_min_dist > self.dist_wall_min:
            self.state = 1

    def readSign(self):
        goal_msg = Point()
        self.goal_pub.publish(goal_msg)
        self.ImageClass = np.argmax(self.mode_image_class)
        # instant 2nd turn after 1st, too close to populate histogram
        if self.ImageClass == 0:
            self.ImageClass = self.ImageClass_current
        print("Histogram -> ", self.mode_image_class)
        self.mode_image_class = self.mode_image_class*0
        print("read sign", self.ImageClass)
        self.getOdometry()
        self.start_angle = self.theta + self.ImgAngle
        if self.ImageClass == 1:
            print("Turning Left")
            self.turnLeft()
        elif self.ImageClass == 2:
            print("Turning Right")
            self.turnRight()
        elif self.ImageClass == 3 or self.ImageClass == 4:
            print("Turning around")
            self.turnAround()
        elif self.ImageClass == 5:
            print("Reached Goal")
            self.goToGoal()
        elif self.ImageClass == 0:
            print("Look for signs")
            self.rotateOnPoint()

    def turnLeft(self):
        self.theta_desired = self.start_angle + np.pi/2
        self.prev_turn = 1
        self.runPID()

    def turnRight(self):
        self.theta_desired = self.start_angle - np.pi/2
        self.prev_turn = 2
        self.runPID()

    def turnAround(self):
        self.theta_desired = self.start_angle + np.pi
        print("theta_desired for turn around", self.theta_desired)
        self.runPID()

    def goToGoal(self):
        msg = Twist()
        self.cmd_pub.publish(msg)
        goal_msg = Point()
        self.goal_pub.publish(goal_msg)
        while (1):
            print("!!!!REACHED GOAL!!!!")

    def rotateOnPoint(self):
        if self.prev_turn == 1:
            self.theta_desired = self.start_angle - np.pi/2
        elif self.prev_turn == 2:
            self.theta_desired = self.start_angle + np.pi/2
        self.theta_desired = self.start_angle + np.pi/2
        self.runPID()

    def runPID(self):
        print("IN PID")
        self.previous_stamp = rospy.Time.now()
        self.e_theta = self.theta_desired - self.theta
        # print("Theta desired ", self.theta_desired)
        # print("Initial Angular error ", self.e_theta)
        # self.e_theta = np.remainder(self.e_theta, (2*np.pi))
        if self.e_theta > 2*np.pi:
            self.e_theta -= 2*np.pi
        if self.e_theta < -2*np.pi:
            self.e_theta += 2*np.pi

        if self.e_theta > np.pi:
            self.e_theta -= 2*np.pi
        if self.e_theta <= -np.pi:
            self.e_theta += 2*np.pi

        print("Final Angular error ", self.e_theta)
        # print("State before loop = ", self.state)
        self.e_int = 0
        while abs(self.e_theta) > self.angle_error_tolerance:
            if self.state == 2 and self.ImageClass == 0:
                if self.ImageClass_current != 0:
                    return
            # print("In while")
            msg = Twist()
            Ts = (rospy.Time.now()-self.previous_stamp).to_sec()
            self.e_int += self.e_theta*Ts
            if abs(self.e_int) > 0.2:
                self.e_int = self.e_int/abs(self.e_int)*0.2
            msg.angular.z = self.kt[0]*self.e_theta + self.kt[1]*self.e_int
            if abs(msg.angular.z) > 0.25:
                msg.angular.z = msg.angular.z/abs(msg.angular.z)*0.25
            # print("angular speed ", msg.angular.z)
            self.previous_stamp = rospy.Time.now()
            self.cmd_pub.publish(msg)
            self.getOdometry()
            self.e_theta = self.theta_desired - self.theta
            if self.e_theta > 2*np.pi:
                self.e_theta -= 2*np.pi
            if self.e_theta < -2*np.pi:
                self.e_theta += 2*np.pi

            if self.e_theta > np.pi:
                self.e_theta -= 2*np.pi
            if self.e_theta < -np.pi:
                self.e_theta += 2*np.pi
            # print("Angular error = ", self.e_theta)

        print("Angular error = ", self.e_theta)
        self.e_theta = 0
        self.maneuver_completed = 1
        self.state = 1


tr = Tracker()
rospy.spin()
