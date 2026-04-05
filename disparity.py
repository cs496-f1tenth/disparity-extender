#!/usr/bin/env python3
import numpy as np
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import LaserScan
from ackermann_msgs.msg import AckermannDriveStamped
from nav_msgs.msg import Odometry


class DisparityExtender(Node):

    CAR_WIDTH = 0.27
    DIFFERENCE_THRESHOLD = 2.
    STRAIGHTS_SPEED = 1.0
    CORNERS_SPEED = 1.0
    DRAG_SPEED = 1.0
    SAFETY_PERCENTAGE = 3.00

    def __init__(self):
        super().__init__('disparity_extender_node')

        self.STEERING_SENSITIVITY = 5.0
        self.COEFFICIENT = 1.0
        self.EXP_COEFFICIENT = 0.02
        self.X_POWER = 1.8
        self.QUADRANT_FACTOR = 5.0
        self.GAUSSIAN_SIG_PCT = 0.6

        self.speed = 1.0  # Initial speed
        self.radians_per_point = 0.0

        # namespace
        ns = self.get_namespace()
        if ns == '/':
            self.get_logger().warn(
                "Node running in global namespace. "
                "For multi-vehicle, run with __ns:=/vehicleX"
            )

        self.declare_parameter('odom_topic', 'odom')
        self.declare_parameter('scan_topic', 'scan')
        self.declare_parameter('drive_topic', 'drive')

        odom_topic = self.get_parameter('odom_topic').value
        scan_topic = self.get_parameter('scan_topic').value
        drive_topic = self.get_parameter('drive_topic').value

        self.odom_sub = self.create_subscription(
            Odometry, odom_topic, self.odom_cb, 2)
        self.lidar_sub = self.create_subscription(
            LaserScan, scan_topic, self.process_lidar, 1)
        self.drive_pub = self.create_publisher(
            AckermannDriveStamped, drive_topic, 1)

#        lidarscan_topic = '/scan'
#        odom_topic = '/vesc/odom'
#        drive_topic = '/vesc/low_level/ackermann_cmd_mux/input/teleop'
#
#        self.odom_sub = self.create_subscription(
#            Odometry, odom_topic, self.odom_cb, 2)
#        self.lidar_sub = self.create_subscription(
#            LaserScan, lidarscan_topic, self.process_lidar, 1)
#        self.drive_pub = self.create_publisher(
#            AckermannDriveStamped, drive_topic, 1)

    def odom_cb(self, data):
        self.speed = data.twist.twist.linear.x

    def preprocess_lidar(self, ranges):
        ranges = np.clip(ranges, 0, 8)
        eighth = int(len(ranges) / self.QUADRANT_FACTOR)
        return np.array(ranges[eighth:-eighth])

    def get_differences(self, ranges):
        return np.concatenate(([0.], np.abs(np.diff(ranges))))

    def get_disparities(self, differences, threshold):
        return np.where(differences > threshold)[0]

    def get_num_points_to_cover(self, dist, width):
        angle = 1.5 * np.arctan(width / (2 * dist))
        num_points = int(np.ceil(angle / self.radians_per_point))
        return num_points

    def cover_points(self, num_points, start_idx, cover_right, ranges):
        new_dist = ranges[start_idx]
        if cover_right:
            end = min(start_idx + 1 + num_points, len(ranges))
            ranges[start_idx + 1:end] = np.minimum(ranges[start_idx + 1:end], new_dist)
        else:
            start = max(0, start_idx - num_points)
            ranges[start:start_idx] = np.minimum(ranges[start:start_idx], new_dist)
        return ranges

    def extend_disparities(self, disparities, ranges, car_width, extra_pct):
        width_to_cover = car_width * extra_pct
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
            lidar_angle, np.radians(-90), np.radians(90)) / self.STEERING_SENSITIVITY
        return steering_angle

    def process_lidar(self, data):
        ranges = data.ranges
        self.radians_per_point = data.angle_increment

        proc_ranges = self.preprocess_lidar(ranges)
        differences = self.get_differences(proc_ranges)
        disparities = self.get_disparities(differences, self.DIFFERENCE_THRESHOLD)
        proc_ranges = self.extend_disparities(
            disparities, proc_ranges, self.CAR_WIDTH, self.SAFETY_PERCENTAGE)
        

        #ADDED THINGS TO STEERING ANGLE CALCULATIOn
        center = len(proc_ranges) // 2
        weights = np.exp(-0.5 * ((np.arange(len(proc_ranges)) - center) / (len(proc_ranges) * self.GAUSSIAN_SIG_PCT)) ** 2)
        weighted_ranges = proc_ranges * weights
        steering_angle = self.get_steering_angle(weighted_ranges.argmax(), len(proc_ranges))
        #---------------------------

        center = len(proc_ranges) // 2
        window = 5; #width to read around center
        x = np.max(proc_ranges[center - window : center + window])
        speed = self.COEFFICIENT * np.exp(self.EXP_COEFFICIENT * (x ** self.X_POWER))
        self.get_logger().info(f'x: {x}, speed: {speed}')

        #Makes the car backup and turn towards the goal point if there are no good paths.
#        if(x <= 2.0):
#            speed *= -1
#            steering_angle *= -1
        
        drive_msg = AckermannDriveStamped()
        drive_msg.header.stamp = self.get_clock().now().to_msg()
        drive_msg.drive.steering_angle = steering_angle
        drive_msg.drive.speed = speed
        self.drive_pub.publish(drive_msg)


def main(args=None):
    rclpy.init(args=args)
    node = DisparityExtender()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
