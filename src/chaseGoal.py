#!/usr/bin/env python
import rospy
import numpy as np
import cv2 as cv

from geometry_msgs.msg import Point
from geometry_msgs.msg import Twist


class chaseGoal:
    def __init__(self):
        rospy.init_node("chase_goal", anonymous=True)
        # most recent command
        self.goal = None
        self.cmd_pub = rospy.Publisher("/cmd_vel", Twist, queue_size=1)
        self.timer = rospy.Timer(rospy.Duration(2), self.timerCallback)

        # PID related
        self.kd = [rospy.get_param("~k_pd"), rospy.get_param(
            "~k_id"), rospy.get_param("~k_dd")]
        self.kt = [rospy.get_param("~k_pt"), rospy.get_param(
            "~k_it"), rospy.get_param("~k_dt")]

        self.previous_error = [0, 0]
        self.previous_stamp = rospy.Time.now()
        self.e_int = np.array([0, 0])

        self.goal_sub = rospy.Subscriber(
            "/goal", Point, self.goalCallback, queue_size=1)

    def goalCallback(self, goal):
        self.goal = goal

    def timerCallback(self, event):
        # print("Entered timercallback before check")
        if self.goal is None:
            return

        # print("Entered timercallback")

        # compute errors
        r_goal = np.array([[self.goal.x], [self.goal.y]])
        e_distance = np.linalg.norm(r_goal)
        e_theta = np.arctan2(r_goal[1], r_goal[0])

        # compute PID
        Ts = (rospy.Time.now()-self.previous_stamp).to_sec()
        self.e_int[0] += e_distance*Ts
        self.e_int[1] += e_theta*Ts

        ed_distance = (Ts > 0)*(e_distance - self.previous_error[0])/Ts
        ed_theta = (Ts > 0)*(e_theta - self.previous_error[1])/Ts

        # update previous iteration
        self.previous_stamp = rospy.Time.now()
        self.previous_error[0] = e_distance
        self.previous_error[1] = e_theta

        # clip integral term
        self.e_int = self.e_int.clip(-0.2, 0.2)
        # publish out
        msg = Twist()
        # print("e_distance = ", e_distance)
        # stop if close enough
        if e_distance > 0.05:
            # print("Far from goal")
            msg.linear.x = self.kd[0]*e_distance + \
                self.kd[1]*self.e_int[0] + self.kd[2]*ed_distance
            msg.angular.z = self.kt[0]*e_theta + \
                self.kt[1]*self.e_int[1] + self.kt[2]*ed_theta

            # apply forward acceleration only if well-aligned
            msg.linear.x = msg.linear.x*np.exp(-1.5*abs(e_theta))

            # clip accelerations
            msg.linear.x = min(msg.linear.x, 0.1)
            # print("cmd vel ", msg.linear.x, " ", msg.angular.z)
            #msg.angular.z = min(abs(msg.angular.z), 0.2)*msg.angular.z/abs(msg.angular.z)
        self.cmd_pub.publish(msg)


cG = chaseGoal()
rospy.spin()
