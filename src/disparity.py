#!/usr/bin/env python3
import math
import numpy as np

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import LaserScan
from nav_msgs.msg import Odometry
from rtec_f1tenth_comm.msg import Waypoint, Trajectory


class DisparityExtender(Node):
    # =========================
    # CONFIG YOU CAN TUNE
    # =========================

    # --- Physical constants ---
    CAR_WIDTH       = 0.45
    WHEELBASE       = 0.30
    SAFETY_MARGIN   = 0.05

    # --- Disparity detection ---
    DISPARITY_THRESHOLD = 0.20

    # --- Speed control ---
    MIN_SPEED           = 0.5
    MAX_SPEED           = 4.0
    STOP_DISTANCE       = 0.5
    FULL_SPEED_DISTANCE = 3.0

    # --- Side obstacle override (optional, keep conservative) ---
    SIDE_OBSTACLE_DISTANCE     = 0.8
    SIDE_SECTOR_HALF_ANGLE_DEG = 20.0

    # --- Forward view limits (IMPORTANT on tracks) ---
    FORWARD_ARC_DEG = 80.0   # was +-90; reduce to ignore side openings on track (try 60–80)

    # --- Target selection (THIS FIXES “GOES IN CIRCLES”) ---
    USE_GAP_CENTERING = True     # True: pick center of a good open region; False: score beams
    GAP_CLEARANCE_M   = 1.2      # beams >= this are considered “open” for gap finding (try 0.8–1.6)
    GAP_MIN_WIDTH_DEG = 8.0      # ignore tiny gaps (try 6–15)
    ANGLE_PENALTY     = 2.0      # meters per rad: higher = prefer straight (try 0.8–2.0)
    RANGE_POWER       = 1.0      # >1 biases toward far points (try 1.0–2.0)

    # --- Best point stability ---
    BEST_POINT_WINDOW_BEAMS = 6

    # --- Steering feel ---
    STEER_DEADBAND_DEG = 2.0
    STEER_GAIN         = 0.60
    MAX_STEER_DEG      = 1.0
    STEER_SMOOTH_ALPHA = 0.20

    # --- Speed reduction in turns ---
    CURVATURE_SPEED_REDUCE = 0.6

    # --- Sign convention ---
    INVERT_STEERING_SIGN = False

    # --- Trajectory rollout ---
    TRAJ_HORIZON = 3.0
    TRAJ_DT      = 0.1

    # =========================
    # END CONFIG
    # =========================

    def __init__(self):
        super().__init__('disparity_extender_node')

        self.HALF_CAR_WIDTH = self.CAR_WIDTH / 2.0

        self.radians_per_point = None
        self.center_idx = None

        self.x = 0.0
        self.y = 0.0
        self.yaw = 0.0
        self.vel = 0.0
        self.pose_ready = False

        self._steer_filt = 0.0

        self.odom_sub = self.create_subscription(Odometry, '/odom', self.handle_odometry, 10)
        self.lidar_sub = self.create_subscription(LaserScan, '/scan', self.process_lidar, 10)
        self.traj_pub = self.create_publisher(Trajectory, '/trajectory', 10)

        self.get_logger().info('Disparity Extender node started.')

    def handle_odometry(self, msg: Odometry) -> None:
        self.x = msg.pose.pose.position.x
        self.y = msg.pose.pose.position.y
        self.yaw = self.quaternion_to_yaw(
            msg.pose.pose.orientation.w,
            msg.pose.pose.orientation.x,
            msg.pose.pose.orientation.y,
            msg.pose.pose.orientation.z,
        )
        self.vel = math.sqrt(msg.twist.twist.linear.x ** 2 + msg.twist.twist.linear.y ** 2)
        self.pose_ready = True

    def preprocess_lidar(self, ranges: np.ndarray) -> np.ndarray:
        MAX_RANGE = 10.0
        ranges = np.where(np.isfinite(ranges), ranges, MAX_RANGE)
        ranges = np.clip(ranges, 0.0, MAX_RANGE)
        return ranges

    def get_disparities(self, ranges: np.ndarray) -> list:
        diffs = np.abs(np.diff(ranges))
        indices = np.where(diffs > self.DISPARITY_THRESHOLD)[0] + 1
        return indices.tolist()

    def num_points_to_cover(self, distance: float) -> int:
        width = self.HALF_CAR_WIDTH + self.SAFETY_MARGIN
        if distance <= 1e-6:
            return 0
        arg = np.clip(width / distance, -1.0, 1.0)
        angle = 2.0 * np.arcsin(arg)
        return int(np.ceil(angle / self.radians_per_point))

    def extend_disparity(self, ranges: np.ndarray, idx: int) -> np.ndarray:
        left_dist = ranges[idx - 1]
        right_dist = ranges[idx]

        if left_dist <= right_dist:
            close_dist = left_dist
            start, step = idx, 1
        else:
            close_dist = right_dist
            start, step = idx - 1, -1

        if close_dist < 0.01:
            return ranges

        n = self.num_points_to_cover(close_dist)
        for i in range(n):
            j = start + step * i
            if j < 0 or j >= len(ranges):
                break
            if ranges[j] > close_dist:
                ranges[j] = close_dist
        return ranges

    def extend_all_disparities(self, ranges: np.ndarray) -> np.ndarray:
        out = ranges.copy()
        for idx in self.get_disparities(ranges):  # compute from original scan
            out = self.extend_disparity(out, idx)
        return out

    def _forward_window(self, n: int):
        half_forward = int(np.floor(math.radians(self.FORWARD_ARC_DEG) / self.radians_per_point))
        lo = max(0, self.center_idx - half_forward)
        hi = min(n - 1, self.center_idx + half_forward)
        return lo, hi

    def _pick_target_gap_center(self, ranges: np.ndarray) -> int:
        """
        Find widest gap of beams >= GAP_CLEARANCE_M in forward arc.
        Return its center index. If none found, fall back to score-based.
        """
        n = len(ranges)
        lo, hi = self._forward_window(n)
        seg = ranges[lo:hi + 1]

        open_mask = seg >= float(self.GAP_CLEARANCE_M)
        if not np.any(open_mask):
            return self._pick_target_scored(ranges)

        # Find contiguous True segments
        idxs = np.where(open_mask)[0]
        # Split into contiguous runs
        runs = np.split(idxs, np.where(np.diff(idxs) != 1)[0] + 1)

        min_width_beams = int(round(math.radians(self.GAP_MIN_WIDTH_DEG) / self.radians_per_point))
        best_run = None
        best_run_score = -1e9

        for run in runs:
            if len(run) < max(1, min_width_beams):
                continue

            a = int(run[0])
            b = int(run[-1])

            # Center of run, but also evaluate with angle penalty so we prefer straighter runs.
            center_local = int(round((a + b) / 2))
            center_idx = lo + center_local
            ang = (center_idx - self.center_idx) * self.radians_per_point

            # Prefer wide gaps and near-straight
            run_width = (b - a + 1) * self.radians_per_point
            score = (run_width * 1.0) - float(self.ANGLE_PENALTY) * abs(ang)

            if score > best_run_score:
                best_run_score = score
                best_run = (a, b)

        if best_run is None:
            return self._pick_target_scored(ranges)

        a, b = best_run
        center_local = int(round((a + b) / 2))
        return lo + center_local

    def _pick_target_scored(self, ranges: np.ndarray) -> int:
        """
        Score beams by: score = range^p - ANGLE_PENALTY*|angle|
        This prevents “track circling” by preferring forward-ish points.
        """
        n = len(ranges)
        lo, hi = self._forward_window(n)
        idxs = np.arange(lo, hi + 1)

        angles = (idxs - self.center_idx) * self.radians_per_point
        r = ranges[lo:hi + 1]

        r_term = np.power(r, float(self.RANGE_POWER))
        score = r_term - float(self.ANGLE_PENALTY) * np.abs(angles)

        i0 = lo + int(np.argmax(score))

        # Weighted centroid around i0 for stability
        w_beams = int(self.BEST_POINT_WINDOW_BEAMS)
        a = max(lo, i0 - w_beams)
        b = min(hi, i0 + w_beams)
        idxs2 = np.arange(a, b + 1, dtype=np.float64)
        w = ranges[a:b + 1].astype(np.float64)
        s = float(np.sum(w))
        if s <= 1e-9:
            return i0
        return int(round(float(np.sum(idxs2 * w) / s)))

    def get_best_point_index(self, ranges: np.ndarray) -> int:
        if self.USE_GAP_CENTERING:
            return self._pick_target_gap_center(ranges)
        return self._pick_target_scored(ranges)

    def index_to_steering_angle(self, idx: int) -> float:
        lidar_angle = (idx - self.center_idx) * self.radians_per_point
        lidar_angle = float(np.clip(lidar_angle, -math.pi / 2.0, math.pi / 2.0))
        if self.INVERT_STEERING_SIGN:
            lidar_angle *= -1.0
        return lidar_angle

    def side_obstacle_override(self, raw_ranges: np.ndarray, steering_angle: float) -> float:
        if abs(steering_angle) < 1e-9:
            return steering_angle

        n = len(raw_ranges)
        ninety = int(round((math.pi / 2.0) / self.radians_per_point))
        half_sector = int(round(math.radians(self.SIDE_SECTOR_HALF_ANGLE_DEG) / self.radians_per_point))

        center = self.center_idx + ninety if steering_angle > 0.0 else self.center_idx - ninety
        a = max(0, center - half_sector)
        b = min(n - 1, center + half_sector)
        if a >= b:
            return steering_angle

        side_ranges = raw_ranges[a:b + 1]
        if np.any(side_ranges < self.SIDE_OBSTACLE_DISTANCE):
            return 0.0
        return steering_angle

    def compute_speed(self, filtered_ranges: np.ndarray) -> float:
        window = max(1, int(np.floor(np.radians(5.0) / self.radians_per_point)))
        lo = max(0, self.center_idx - window)
        hi = min(len(filtered_ranges) - 1, self.center_idx + window)
        forward_dist = float(np.mean(filtered_ranges[lo:hi + 1]))

        if forward_dist <= self.STOP_DISTANCE:
            return self.MIN_SPEED
        if forward_dist >= self.FULL_SPEED_DISTANCE:
            return self.MAX_SPEED

        t = (forward_dist - self.STOP_DISTANCE) / (self.FULL_SPEED_DISTANCE - self.STOP_DISTANCE)
        return self.MIN_SPEED + t * (self.MAX_SPEED - self.MIN_SPEED)

    def tame_steering(self, steering_angle: float) -> float:
        steering_angle *= float(self.STEER_GAIN)

        max_steer = math.radians(self.MAX_STEER_DEG)
        steering_angle = float(np.clip(steering_angle, -max_steer, max_steer))

        a = float(self.STEER_SMOOTH_ALPHA)
        self._steer_filt = (1.0 - a) * self._steer_filt + a * steering_angle
        return self._steer_filt

    def build_trajectory(self, x0: float, y0: float, yaw0: float,
                         steering_angle: float, speed: float) -> Trajectory:

        kappa = math.tan(steering_angle) / self.WHEELBASE
        dt = self.TRAJ_DT
        steps = int(self.TRAJ_HORIZON / dt)
        ds = speed * dt

        waypoints = []
        x, y, psi = x0, y0, yaw0
        s = 0.0

        for _ in range(steps):
            wp = Waypoint()
            wp.s = s
            wp.x = x
            wp.y = y
            wp.psi = psi
            wp.kappa = kappa
            wp.v = speed
            wp.a = 0.0
            waypoints.append(wp)

            dpsi = kappa * ds
            psi_mid = psi + dpsi / 2.0
            x += ds * math.cos(psi_mid)
            y += ds * math.sin(psi_mid)
            psi += dpsi
            s += ds

        traj = Trajectory()
        traj.waypoints = waypoints
        return traj

    def process_lidar(self, msg: LaserScan) -> None:
        self.radians_per_point = float(msg.angle_increment)
        if abs(self.radians_per_point) < 1e-9:
            return

        raw_ranges = np.array(msg.ranges, dtype=np.float64)

        # True forward index with rounding (reduces bias)
        self.center_idx = int(round((-float(msg.angle_min)) / self.radians_per_point))
        self.center_idx = int(np.clip(self.center_idx, 0, len(raw_ranges) - 1))

        proc_ranges = self.preprocess_lidar(raw_ranges)
        filtered_ranges = self.extend_all_disparities(proc_ranges)

        best_idx = self.get_best_point_index(filtered_ranges)
        steering_angle = self.index_to_steering_angle(best_idx)

        steering_angle = self.side_obstacle_override(proc_ranges, steering_angle)

        deadband = math.radians(self.STEER_DEADBAND_DEG)
        if abs(steering_angle) < deadband:
            steering_angle = 0.0

        steering_angle = self.tame_steering(steering_angle)

        speed = self.compute_speed(filtered_ranges)

        # Reduce speed when turning
        max_steer = math.radians(self.MAX_STEER_DEG)
        turn_frac = min(1.0, abs(steering_angle) / max_steer) if max_steer > 1e-9 else 0.0
        speed *= (1.0 - float(self.CURVATURE_SPEED_REDUCE) * turn_frac)
        speed = float(np.clip(speed, self.MIN_SPEED, self.MAX_SPEED))

        if not self.pose_ready:
            return

        traj = self.build_trajectory(self.x, self.y, self.yaw, steering_angle, speed)
        self.traj_pub.publish(traj)

    @staticmethod
    def quaternion_to_yaw(w: float, x: float, y: float, z: float) -> float:
        t3 = 2.0 * (w * z + x * y)
        t4 = 1.0 - 2.0 * (y * y + z * z)
        return math.atan2(t3, t4)


def main(args=None):
    rclpy.init(args=args)
    node = DisparityExtender()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
