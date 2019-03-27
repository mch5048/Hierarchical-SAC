#!/usr/bin/env python
# ROS Imports
import rospy
from geometry_msgs.msg import Pose


class GoalPublisher(object):
    def __init__(self):
        self.goal_pub = rospy.Publisher('/teacher/ik_vel/', Pose, queue_size=3)
        self.ee_pose_sub = rospy.Subscriber('/robot/limb/right/end_effector_pose', Pose, self.ee_pose_cb)

        self.is_set_target = False

        self.mode = input("Select Mode (0: safe init, 1 : safe home, 2 : init position, 3: home position 4: user define)")


        rate = rospy.Rate(100)
        while not rospy.is_shutdown():
            self.goal_publish()
            rate.sleep()

    def goal_publish(self):

        if self.mode == 0 :
            if self.is_set_target == False:
                self.control_start_time = rospy.get_rostime().secs + rospy.get_rostime().nsecs*10**-9
                self.target_x = 0.290
                self.target_y = 0.0
                self.target_z = 0.203
                self.is_set_target = True
            self.now = rospy.get_rostime().secs + rospy.get_rostime().nsecs*10**-9
            self.goal_x = self.cubic(self.now, self.control_start_time, self.control_start_time+2.0, self.init_x, self.target_x, 0.0, 0.0)
            self.goal_y = self.cubic(self.now, self.control_start_time, self.control_start_time+2.0, self.init_y, self.target_y, 0.0, 0.0)
            self.goal_z = self.cubic(self.now, self.control_start_time, self.control_start_time+2.0, self.init_z, self.target_z, 0.0, 0.0)

        elif self.mode == 1 :
            if self.is_set_target == False:
                self.control_start_time = rospy.get_rostime().secs + rospy.get_rostime().nsecs*10**-9
                self.target_x = 0.138
                self.target_y = 0.0
                self.target_z = 0.239
                self.is_set_target = True
            self.now = rospy.get_rostime().secs + rospy.get_rostime().nsecs*10**-9
            self.goal_x = self.cubic(self.now, self.control_start_time, self.control_start_time+2.0, self.init_x, self.target_x, 0.0, 0.0)
            self.goal_y = self.cubic(self.now, self.control_start_time, self.control_start_time+2.0, self.init_y, self.target_y, 0.0, 0.0)
            self.goal_z = self.cubic(self.now, self.control_start_time, self.control_start_time+2.0, self.init_z, self.target_z, 0.0, 0.0)

        elif self.mode == 2:
            self.goal_x = 0.138
            self.goal_y = 0.0
            self.goal_z = 0.239

        elif self.mode == 3:
            self.goal_x = 0.290
            self.goal_y = 0.0
            self.goal_z = 0.203
            self.control_start_time = rospy.get_rostime().secs + rospy.get_rostime().nsecs*10**-9

        elif self.mode == 4:
            self.goal_x, self.goal_y, self.goal_z = [float(goal) for goal in raw_input("Enter goal position x, y, z: ").split()]

        
        goal_pose = Pose()
        goal_pose.position.x = self.goal_x
        goal_pose.position.y = self.goal_y
        goal_pose.position.z = self.goal_z    
        self.goal_pub.publish(goal_pose)

    def ee_pose_cb(self, ee_pose):
        self.init_x = ee_pose.position.x
        self.init_y = ee_pose.position.y
        self.init_z = ee_pose.position.z
        self.init_z = ee_pose.position.z
        self.init_z = ee_pose.position.z
        self.init_z = ee_pose.position.z
        self.init_z = ee_pose.position.z


    def cubic(self, time,time_0, time_f, x_0, x_f, x_dot_0, x_dot_f):
        x_t = x_0

        if (time < time_0):
            x_t = x_0

        elif (time > time_f):
            x_t = x_f
        else :
            elapsed_time = time - time_0
            total_time = time_f - time_0
            total_time2 = total_time * total_time
            total_time3 = total_time2 * total_time
            total_x    = x_f - x_0

            x_t = x_0 + x_dot_0 * elapsed_time \
                + (3 * total_x / total_time2 \
                - 2 * x_dot_0 / total_time \
                - x_dot_f / total_time) \
                * elapsed_time * elapsed_time \
                + (-2 * total_x / total_time3 + \
                (x_dot_0 + x_dot_f) / total_time2) \
                * elapsed_time * elapsed_time * elapsed_time

        return x_t


def main():
    rospy.init_node('goal_publisher')

    try:
        gp = GoalPublisher()
    except rospy.ROSInterruptException:
        pass

    rospy.spin()

if __name__ == '__main__':
    main()
