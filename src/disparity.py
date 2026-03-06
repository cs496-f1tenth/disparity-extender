#!/usr/bin/env python3
import numpy as np
import math
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import LaserScan
from ackermann_msgs.msg import AckermannDriveStamped
from nav_msgs.msg import Odometry
from rtec_f1tenth_comm.msg import Waypoint, Trajectory

class DisparityExtender(Node):

    CAR_WIDTH = 0.45
    DIFFERENCE_THRESHOLD = 2.
    STRAIGHTS_SPEED = 4.0
    CORNERS_SPEED = 3.0
    DRAG_SPEED = 4.0
    SAFETY_PERCENTAGE = 900.

    def __init__(self):
        super().__init__('disparity_extender_node')

        self.STEERING_SENSITIVITY = 3.
        self.COEFFICIENT = 2.5
        self.EXP_COEFFICIENT = 0.02
        self.X_POWER = 1.8
        self.QUADRANT_FACTOR = 3.5

        self.speed = 4.0  # Initial speed
        self.radians_per_point = 0.0

        lidarscan_topic = '/scan'
        odom_topic = '/vesc/odom'
        drive_topic = '/vesc/low_level/ackermann_cmd_mux/input/teleop'

        self.odom_sub = self.create_subscription(
            Odometry, odom_topic, self.odom_cb, 1)
        self.lidar_sub = self.create_subscription(
            LaserScan, lidarscan_topic, self.process_lidar, 1)
        self.drive_pub = self.create_publisher(
            AckermannDriveStamped, drive_topic, 1)
        self.traj_pub = self.create_publisher(Trajectory, '/trajectory', 1)

    def odom_cb(self, data):
        self.speed = data.twist.twist.linear.x

    def preprocess_lidar(self, ranges):
        ranges = np.clip(ranges, 0, 16)
        eighth = int(len(ranges) / self.QUADRANT_FACTOR)
        return np.array(ranges[eighth:-eighth])

    def get_differences(self, ranges):
        differences = [0.]
        for i in range(1, len(ranges)):
            differences.append(abs(ranges[i] - ranges[i - 1]))
        return differences

    def get_disparities(self, differences, threshold):
        disparities = []
        for index, difference in enumerate(differences):
            if difference > threshold:
                disparities.append(index)
        return disparities

    def get_num_points_to_cover(self, dist, width):
        angle = 1.5 * np.arctan(width / (2 * dist))
        num_points = int(np.ceil(angle / self.radians_per_point))
        return num_points

    def cover_points(self, num_points, start_idx, cover_right, ranges):
        new_dist = ranges[start_idx]
        if cover_right:
            for i in range(num_points):
                next_idx = start_idx + 1 + i
                if next_idx >= len(ranges):
                    break
                if ranges[next_idx] > new_dist:
                    ranges[next_idx] = new_dist
        else:
            for i in range(num_points):
                next_idx = start_idx - 1 - i
                if next_idx < 0:
                    break
                if ranges[next_idx] > new_dist:
                    ranges[next_idx] = new_dist
        return ranges

    def extend_disparities(self, disparities, ranges, car_width, extra_pct):
        width_to_cover = 0.155 * (1 + extra_pct / 100)
        for index in disparities:
            first_idx = index - 1
            points = ranges[first_idx:first_idx + 2]
            close_idx = first_idx + np.argmin(points)
            far_idx = first_idx + np.argmax(points)
            close_dist = ranges[close_idx]
            num_points_to_cover = self.get_num_points_to_cover(
                close_dist, width_to_cover)
            cover_right = close_idx < far_idx
            ranges = self.cover_points(
                num_points_to_cover, close_idx, cover_right, ranges)
        return ranges

    def get_steering_angle(self, range_index, range_len):
        lidar_angle = (range_index - (range_len / 2)) * self.radians_per_point
        steering_angle = np.clip(
            lidar_angle, np.radians(-90), np.radians(90)
        ) / self.STEERING_SENSITIVITY
        return steering_angle
    
    def publish_trajectory(self, steering_angle, speed):
        waypoint = Waypoint()
        waypoint.s = 0.0
        waypoint.x = 0.0
        waypoint.y = 0.0
        waypoint.psi = steering_angle
        waypoint.kappa = 0.0
        waypoint.v = speed
        waypoint.a = 0.0

        traj_msg = Trajectory()
        traj_msg.waypoints = [waypoint]
        self.traj_pub.publish(traj_msg)

    def process_lidar(self, data):
        ranges = data.ranges
        self.radians_per_point = (2 * np.pi) / len(ranges)

        proc_ranges = self.preprocess_lidar(ranges)
        differences = self.get_differences(proc_ranges)
        disparities = self.get_disparities(differences, self.DIFFERENCE_THRESHOLD)
        proc_ranges = self.extend_disparities(
            disparities, proc_ranges, self.CAR_WIDTH, self.SAFETY_PERCENTAGE)

        steering_angle = self.get_steering_angle(
            proc_ranges.argmax(), len(proc_ranges))

        x = max(proc_ranges[227:237])
        speed = self.COEFFICIENT * math.exp(self.EXP_COEFFICIENT * (x ** self.X_POWER))
        self.get_logger().info(f'x: {x}, speed: {speed}')

        drive_msg = AckermannDriveStamped()
        drive_msg.header.stamp = self.get_clock().now().to_msg()
        drive_msg.drive.steering_angle = steering_angle
        drive_msg.drive.speed = speed
        self.drive_pub.publish(drive_msg)

        self.drive_pub.publish(drive_msg)
        self.publish_trajectory(steering_angle, speed)  # <-- add this


def main(args=None):
    rclpy.init(args=args)
    node = DisparityExtender()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()