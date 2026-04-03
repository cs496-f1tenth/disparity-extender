#!/usr/bin/env python3
import numpy as np
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import LaserScan
from ackermann_msgs.msg import AckermannDriveStamped
from nav_msgs.msg import Odometry
from datetime import datetime


class DisparityExtender(Node):

    CAR_WIDTH = 0.27
    DIFFERENCE_THRESHOLD = 2.
    SAFETY_MULT = 3.00
    VIEW_RANGE = 8

    #PD controller variables
    MAX_SPEED = 6.0
    KP = 1.8
    KD = 1.8
    PD_MAX_OUTPUT = 20.0
    MAX_DERIVATIVE = 5.0

    #Used for derivative low pass filtering, should add up to 1.
    # Greater old = more smoothing
    # Greater new = more responsive but less noise suppression
    PD_FILTER_OLD = 0.8
    PD_FILTER_NEW = 0.2
    SPEED_FILTER_OLD = 0.75
    SPEED_FILTER_NEW = 0.25

    # Percent of beams used for gaussian spread,
    # Higher = less preference for center (wider curve)
    # Lower = more preference for center (steeper curve)
    GAUSSIAN_SIG_PCT = 0.6 #NOTE: 0.6 was what worked before adding PD speed, I believe this is too high in practice when combined

    def __init__(self):
        super().__init__('disparity_extender_node')

        self.STEERING_SENSITIVITY = 3.0
        self.QUADRANT_FACTOR = 3.5

        #PD controller variables
        self.pd_output = 0.0
        self.prev_error = 0.0
        self.prev_time = 0.0
        self.is_first_run = True
        self.filtered_derivative = 0.0
        self.filtered_speed = 0.0

        self.speed = 2.0  # Initial speed
        self.radians_per_point = 0.0

        lidarscan_topic = '/scan'
        odom_topic = '/vesc/odom'
        drive_topic = '/vesc/low_level/ackermann_cmd_mux/input/teleop'

        self.odom_sub = self.create_subscription(
            Odometry, odom_topic, self.odom_cb, 2)
        self.lidar_sub = self.create_subscription(
            LaserScan, lidarscan_topic, self.process_lidar, 1)
        self.drive_pub = self.create_publisher(
            AckermannDriveStamped, drive_topic, 1)

    #PD controller returns a "danger value"
    def pd_controller_update(self, forward_clearance):
        current_time = self.get_clock().now().nanoseconds * 1e-9
        error = self.VIEW_RANGE - forward_clearance
        derivative = 0.0

        if(self.is_first_run):
            self.is_first_run = False
        else:
            dt = current_time - self.prev_time
            if(dt > 1e-6):
                derivative = (error - self.prev_error)/dt

        #filter derivative in order to prevent noise from throwing off the controller
        #we also clamp the derivative value to ensure reasonable adjustments
        self.filtered_derivative = np.clip((self.PD_FILTER_OLD * self.filtered_derivative + self.PD_FILTER_NEW * derivative), -self.MAX_DERIVATIVE, self.MAX_DERIVATIVE);
        raw_pd = (self.KP * error) + (self.KD * self.filtered_derivative)
        
        self.prev_error = error
        self.prev_time = current_time

        self.get_logger().info(f'raw_pd: {raw_pd:.3f}, error: {error:.3f}, derivative: {self.filtered_derivative:.3f}')
        return np.clip(raw_pd, 0.0, self.PD_MAX_OUTPUT)
            
    #uses the danger value returned by the PD controller to give a speed value
    def danger_to_speed(self, danger):
        #consider adding a speed floor so that the car doesn't stop completely
        #normalize the danger value 0..1 and get the inverse to scale speed
        new_speed = self.MAX_SPEED * (1 - (danger/self.PD_MAX_OUTPUT))
        #prevent jerking speed by apply a low pass filter to the speed change
        self.filtered_speed = (self.filtered_speed * self.SPEED_FILTER_OLD) + (new_speed * self.SPEED_FILTER_NEW)
        return self.filtered_speed

    def odom_cb(self, data):
        self.speed = data.twist.twist.linear.x

    def preprocess_lidar(self, ranges):
        ranges = np.clip(ranges, 0, self.VIEW_RANGE)
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
            disparities, proc_ranges, self.CAR_WIDTH, self.SAFETY_MULT)
       

        center = len(proc_ranges) // 2

        # GAUSSIAN CURVE FOR STEERING ANGLE CALCULATION
        weights = np.exp(-0.5 * ((np.arange(len(proc_ranges)) - center) / (len(proc_ranges) * self.GAUSSIAN_SIG_PCT)) ** 2)
        weighted_ranges = proc_ranges * weights
        steering_angle = self.get_steering_angle(weighted_ranges.argmax(), len(proc_ranges))
        #--------------------

        window = 6 #width to read around center
        x = np.mean(proc_ranges[center-window : center+window+1]) #forward clearance around center

        danger = self.pd_controller_update(x)
        speed = self.danger_to_speed(danger)
        
        self.get_logger().info(f'x: {x}, speed: {speed}')
        
        #Makes the car backup and turn towards the goal point if there are no good paths.
        #if(x <= 0.35):
        #    speed *= -1
        #    steering_angle *= -1
        
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
