import cv2
import numpy as np
from src.detectors.base import LaneDetector
import time
from src.utils import get_perspective_transform, warp_image, get_x_from_fit
from src.detectors.viz_utils import (
    draw_text_outlined,
    get_confidence_color,
    get_accuracy_color,
    calculate_accuracy_exponential,
    get_latency_color,
    combine_grid,
)


class SlidingWindowDetector(LaneDetector):
    """Lane detection using Bird's Eye View transformation and sliding window search."""

    def __init__(self, debug: bool = False):
        super().__init__(debug)
        self.left_fit_average = None
        self.right_fit_average = None
        self.smooth_factor = 0.1
        self.avg_steering_offset = 0.0

        self.M = None
        self.Minv = None
        self.initialized = False

        self.nwindows = 9
        self.margin = 100
        self.minpix = 50
        self.white_mask_limit = 200
        self.yellow_mask_lower = np.array([15, 30, 100], dtype=np.uint8)
        self.yellow_mask_upper = np.array([35, 204, 255], dtype=np.uint8)

    def detect(self, image: np.ndarray, steering: float = 0.0) -> dict:
        t0 = time.time()
        height, width = image.shape[:2]

        if not self.initialized:
            self.M, self.Minv = get_perspective_transform(width, height)
            self.initialized = True

        alpha = 0.1
        self.avg_steering_offset = (alpha * steering) + (
            (1 - alpha) * self.avg_steering_offset
        )

        # Color thresholding in HLS
        hls = cv2.cvtColor(image, cv2.COLOR_BGR2HLS)

        l_channel = hls[:, :, 1]
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        hls[:, :, 1] = clahe.apply(l_channel)

        lower_white = np.array([0, self.white_mask_limit, 0], dtype=np.uint8)
        upper_white = np.array([180, 255, 255], dtype=np.uint8)
        white_mask = cv2.inRange(hls, lower_white, upper_white)

        yellow_mask = cv2.inRange(hls, self.yellow_mask_lower, self.yellow_mask_upper)

        combined_binary = cv2.bitwise_or(white_mask, yellow_mask)

        binary_warped = warp_image(combined_binary, self.M)

        # Histogram-based lane starting point detection
        histogram = np.sum(binary_warped[binary_warped.shape[0] // 2 :, :], axis=0)

        midpoint = int(histogram.shape[0] // 2)
        leftx_base = np.argmax(histogram[:midpoint])
        rightx_base = np.argmax(histogram[midpoint:]) + midpoint

        window_height = int(height // self.nwindows)

        nonzero = binary_warped.nonzero()
        nonzeroy = np.array(nonzero[0])
        nonzerox = np.array(nonzero[1])

        leftx_current = leftx_base
        rightx_current = rightx_base

        left_lane_inds = []
        right_lane_inds = []

        out_img = np.dstack((binary_warped, binary_warped, binary_warped)) * 255

        for window in range(self.nwindows):
            win_y_low = height - (window + 1) * window_height
            win_y_high = height - window * window_height
            win_xleft_low = leftx_current - self.margin
            win_xleft_high = leftx_current + self.margin
            win_xright_low = rightx_current - self.margin
            win_xright_high = rightx_current + self.margin

            cv2.rectangle(
                out_img,
                (win_xleft_low, win_y_low),
                (win_xleft_high, win_y_high),
                (0, 255, 0),
                2,
            )
            cv2.rectangle(
                out_img,
                (win_xright_low, win_y_low),
                (win_xright_high, win_y_high),
                (0, 255, 0),
                2,
            )

            good_left_inds = (
                (nonzeroy >= win_y_low)
                & (nonzeroy < win_y_high)
                & (nonzerox >= win_xleft_low)
                & (nonzerox < win_xleft_high)
            ).nonzero()[0]
            good_right_inds = (
                (nonzeroy >= win_y_low)
                & (nonzeroy < win_y_high)
                & (nonzerox >= win_xright_low)
                & (nonzerox < win_xright_high)
            ).nonzero()[0]

            left_lane_inds.append(good_left_inds)
            right_lane_inds.append(good_right_inds)

            if len(good_left_inds) > self.minpix:
                leftx_current = int(np.mean(nonzerox[good_left_inds]))
            if len(good_right_inds) > self.minpix:
                rightx_current = int(np.mean(nonzerox[good_right_inds]))

        left_lane_inds = np.concatenate(left_lane_inds)
        right_lane_inds = np.concatenate(right_lane_inds)

        leftx = nonzerox[left_lane_inds]
        lefty = nonzeroy[left_lane_inds]
        rightx = nonzerox[right_lane_inds]
        righty = nonzeroy[right_lane_inds]

        # Fit linear polynomial: x = m*y + b
        left_fit = None
        right_fit = None

        if len(leftx) > 0:
            left_fit = np.polyfit(lefty, leftx, 1)
        if len(rightx) > 0:
            right_fit = np.polyfit(righty, rightx, 1)

        def smooth_poly(curr, avg):
            if curr is None:
                return avg
            if avg is None:
                return curr
            return (self.smooth_factor * curr) + ((1 - self.smooth_factor) * avg)

        self.left_fit_average = smooth_poly(left_fit, self.left_fit_average)
        self.right_fit_average = smooth_poly(right_fit, self.right_fit_average)

        lane_polygon_points = []
        confidence = 0.0

        max_pixels = height * 10
        conf_l = min(len(leftx), max_pixels) / max_pixels
        conf_r = min(len(rightx), max_pixels) / max_pixels
        confidence = (conf_l + conf_r) / 2.0

        y_vals = np.linspace(0, height - 1, height)

        if self.left_fit_average is not None and self.right_fit_average is not None:
            left_fitx = self.left_fit_average[0] * y_vals + self.left_fit_average[1]
            right_fitx = self.right_fit_average[0] * y_vals + self.right_fit_average[1]

            pts_left = np.array([np.transpose(np.vstack([left_fitx, y_vals]))])
            pts_right = np.array(
                [np.flipud(np.transpose(np.vstack([right_fitx, y_vals])))]
            )
            pts = np.hstack((pts_left, pts_right))

            pts_flat = pts.reshape(-1, 1, 2).astype(np.float32)
            pts_unwarped = cv2.perspectiveTransform(pts_flat, self.Minv).astype(
                np.int32
            )

            lane_polygon_points = [pts_unwarped]

        trajectory_points = []
        steering_trajectory_points = []

        if self.left_fit_average is not None and self.right_fit_average is not None:
            lx_top = self.left_fit_average[1]
            rx_top = self.right_fit_average[1]
            cx_bev_top = (lx_top + rx_top) / 2

            target_pt_bev = np.array([[[cx_bev_top, 0]]], dtype=np.float32)
            target_pt_persp = cv2.perspectiveTransform(target_pt_bev, self.Minv)[0][0]

            end_x = int(target_pt_persp[0])

            roi_bottom_y = height
            roi_top_y = int(height * 0.45)
            start_x = width // 2

            t_pts = []
            n_pts = 20
            for i in range(n_pts + 1):
                t = i / n_pts
                curr_y = int(roi_bottom_y - (roi_bottom_y - roi_top_y) * t)
                curr_x = int((1 - t**2) * start_x + (t**2) * end_x)
                t_pts.append([curr_x, curr_y])
            trajectory_points = [np.array(t_pts, np.int32)]

            steer_shift = int(self.avg_steering_offset * (width * 0.75))
            steer_end_x = (width // 2) + steer_shift

            lane_center_x = cx_bev_top

            diff = abs(lane_center_x - steer_end_x)
            decay_constant = width * 0.1
            accuracy = calculate_accuracy_exponential(diff, decay_constant)

            steer_traj = []
            for i in range(n_pts + 1):
                t = i / n_pts
                curr_y = int(roi_bottom_y - (roi_bottom_y - roi_top_y) * t)
                curr_x = int((1 - t**2) * start_x + (t**2) * steer_end_x)
                steer_traj.append([curr_x, curr_y])
            steering_trajectory_points = [np.array(steer_traj, np.int32)]

        else:
            accuracy = 0.0

        t1 = time.time()
        latency_ms = (t1 - t0) * 1000

        out_img[lefty, leftx] = [0, 0, 255]
        out_img[righty, rightx] = [255, 0, 0]

        if self.left_fit_average is not None:
            left_fitx = self.left_fit_average[0] * y_vals + self.left_fit_average[1]
            pts_l = np.array([np.transpose(np.vstack([left_fitx, y_vals]))]).astype(
                np.int32
            )
            cv2.polylines(out_img, pts_l, False, (255, 255, 0), 2)

        if self.right_fit_average is not None:
            right_fitx = self.right_fit_average[0] * y_vals + self.right_fit_average[1]
            pts_r = np.array([np.transpose(np.vstack([right_fitx, y_vals]))]).astype(
                np.int32
            )
            cv2.polylines(out_img, pts_r, False, (255, 255, 0), 2)

        return {
            "lane_polygon_points": lane_polygon_points,
            "trajectory_points": trajectory_points,
            "steering_trajectory_points": steering_trajectory_points,
            "latency": latency_ms,
            "confidence": confidence,
            "accuracy": accuracy,
            "bev_viz": out_img.astype(np.uint8),
            "binary_mask": combined_binary,
        }

    def visualize(self, image: np.ndarray, prediction: dict) -> np.ndarray:
        lane_polygon_points = prediction.get("lane_polygon_points", [])
        trajectory_points = prediction.get("trajectory_points", [])
        steering_trajectory_points = prediction.get("steering_trajectory_points", [])
        latency = prediction.get("latency", 0.0)
        confidence = prediction.get("confidence", 0.0)
        accuracy = prediction.get("accuracy", 0.0)
        bev_viz = prediction.get("bev_viz")

        vis_img = image.copy()
        height, width = image.shape[:2]

        conf_color = get_confidence_color(confidence)

        if lane_polygon_points and confidence > 0.1:
            overlay = np.zeros_like(vis_img)
            cv2.fillPoly(overlay, lane_polygon_points, conf_color)
            vis_img = cv2.addWeighted(vis_img, 1, overlay, 0.4, 0)
            cv2.polylines(vis_img, lane_polygon_points, True, conf_color, 2)

        if trajectory_points:
            cv2.polylines(vis_img, trajectory_points, False, conf_color, 4)
        if steering_trajectory_points:
            cv2.polylines(vis_img, steering_trajectory_points, False, (255, 0, 0), 3)

        if not self.debug:
            return vis_img

        panel1 = image.copy()
        draw_text_outlined(panel1, "1. Raw Image", (10, 30))
        c_lat = get_latency_color(latency)
        draw_text_outlined(panel1, f"Lat: {latency:.1f}ms", (10, 60), 0.6, c_lat)

        bin_mask = prediction.get(
            "binary_mask", np.zeros((height, width), dtype=np.uint8)
        )
        panel2 = cv2.cvtColor(bin_mask, cv2.COLOR_GRAY2BGR)

        pts = np.array(
            [
                [int(width * 0.45), int(height * 0.55)],
                [int(width * 0.55), int(height * 0.55)],
                [int(width), int(height)],
                [int(0), int(height)],
            ],
            np.int32,
        )
        cv2.polylines(panel2, [pts], True, (0, 255, 255), 2)

        draw_text_outlined(panel2, "2. Color Threshold + ROI", (10, 30))

        if bev_viz is None:
            bev_viz = np.zeros_like(image)
        panel3 = cv2.resize(bev_viz, (width, height))
        draw_text_outlined(panel3, "3. Sliding Window (BEV)", (10, 30))

        if self.left_fit_average is not None:
            x_bot = get_x_from_fit(height, self.left_fit_average)
            x_top = get_x_from_fit(0, self.left_fit_average)
            if x_bot is not None and x_top is not None:
                cv2.line(panel3, (x_bot, height), (x_top, 0), (0, 0, 255), 3)

        if self.right_fit_average is not None:
            x_bot = get_x_from_fit(height, self.right_fit_average)
            x_top = get_x_from_fit(0, self.right_fit_average)
            if x_bot is not None and x_top is not None:
                cv2.line(panel3, (x_bot, height), (x_top, 0), (0, 0, 255), 3)

        panel4 = vis_img.copy()
        draw_text_outlined(panel4, "4. Final Overlay", (10, 30))
        draw_text_outlined(
            panel4, f"Conf: {int(confidence * 100)}%", (10, 60), 0.6, (0, 255, 0)
        )
        acc_c = get_accuracy_color(accuracy)
        draw_text_outlined(panel4, f"Acc: {accuracy:.1f}%", (10, 90), 0.6, acc_c)

        panels = [panel1, panel2, panel3, panel4]
        return combine_grid(panels, (2, 2), scale=0.5)
