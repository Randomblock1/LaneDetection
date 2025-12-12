from .base import LaneDetector
import numpy as np
import cv2
import time
from src.detectors.viz_utils import (
    draw_text_outlined,
    get_confidence_color,
    get_accuracy_color,
    calculate_accuracy_exponential,
    get_latency_color,
    combine_grid,
    get_steering_x_at_y,
)
from src.utils import get_x_from_fit


class YoloLaneDetector(LaneDetector):
    """Lane detection using YOLOv8 segmentation model."""

    def __init__(self, model_path: str, debug: bool = False):
        super().__init__(debug)
        self.model_path = model_path
        self.model = None

        self.left_fit_average = None
        self.right_fit_average = None
        self.smooth_factor = 0.1
        self.avg_steering_offset = 0.0

    def load_model(self):
        try:
            from ultralytics import YOLO

            self.model = YOLO(self.model_path)
            print(f"Loaded YOLO model from {self.model_path}")
        except Exception as e:
            print(f"Failed to load YOLO model: {e}")
            self.model = None

    def detect(self, image: np.ndarray, steering: float = 0.0) -> dict:
        t0 = time.time()

        if self.model is None:
            return {"raw_results": None, "latency": 0}

        height, width = image.shape[:2]

        alpha = 0.1
        self.avg_steering_offset = (alpha * steering) + (
            (1 - alpha) * self.avg_steering_offset
        )

        results = self.model(image, verbose=False, stream=False)
        result = results[0]

        left_points_list = []
        right_points_list = []
        confidences = []

        mask_viz = np.zeros_like(image)
        binary_mask = np.zeros((height, width), dtype=np.uint8)

        if result.masks is not None:
            for i, segment in enumerate(result.masks.xy):
                if len(segment) < 2:
                    continue

                cls_id = int(result.boxes.cls[i])
                conf = float(result.boxes.conf[i])

                if cls_id not in [1, 2]:
                    continue

                pts = segment.astype(np.int32)
                centroid_x = np.mean(segment[:, 0])

                color = (0, 0, 0)
                if centroid_x < width / 2:
                    left_points_list.extend(segment)
                    confidences.append(conf)
                    color = (255, 255, 0)
                else:
                    right_points_list.extend(segment)
                    confidences.append(conf)
                    color = (255, 255, 255)

                cv2.fillPoly(mask_viz, [pts], color)
                cv2.fillPoly(binary_mask, [pts], 255)

        left_points = np.array(left_points_list) if len(left_points_list) > 0 else None
        right_points = (
            np.array(right_points_list) if len(right_points_list) > 0 else None
        )

        # Fit linear: x = m*y + b
        current_left_fit = None
        current_right_fit = None

        if left_points is not None and len(left_points) > 10:
            try:
                current_left_fit = np.polyfit(left_points[:, 1], left_points[:, 0], 1)
            except np.linalg.LinAlgError:
                pass

        if right_points is not None and len(right_points) > 10:
            try:
                current_right_fit = np.polyfit(
                    right_points[:, 1], right_points[:, 0], 1
                )
            except np.linalg.LinAlgError:
                pass

        def smooth(curr, avg):
            if curr is None:
                return avg
            if avg is None:
                return curr
            return (self.smooth_factor * curr) + ((1 - self.smooth_factor) * avg)

        self.left_fit_average = smooth(current_left_fit, self.left_fit_average)
        self.right_fit_average = smooth(current_right_fit, self.right_fit_average)

        lane_polygon_points = []
        trajectory_points = []
        steering_trajectory_points = []

        def get_x(y, fit):
            if fit is None:
                return None
            return int(fit[0] * y + fit[1])

        y_bottom = height
        y_top = int(height * 0.45)

        valid_lanes = (
            self.left_fit_average is not None and self.right_fit_average is not None
        )

        points_viz = np.zeros_like(image)
        fit_viz = np.zeros_like(image)

        if left_points is not None:
            for p in left_points:
                cv2.circle(points_viz, (int(p[0]), int(p[1])), 1, (0, 255, 255), -1)
        if right_points is not None:
            for p in right_points:
                cv2.circle(points_viz, (int(p[0]), int(p[1])), 1, (255, 255, 255), -1)

        if self.left_fit_average is not None:
            x_bot = get_x(y_bottom, self.left_fit_average)
            x_top = get_x(y_top, self.left_fit_average)
            cv2.line(fit_viz, (x_bot, y_bottom), (x_top, y_top), (0, 0, 255), 3)

        if self.right_fit_average is not None:
            x_bot = get_x(y_bottom, self.right_fit_average)
            x_top = get_x(y_top, self.right_fit_average)
            cv2.line(fit_viz, (x_bot, y_bottom), (x_top, y_top), (0, 0, 255), 3)

        if valid_lanes:
            x_l_bot = get_x(y_bottom, self.left_fit_average)
            x_l_top = get_x(y_top, self.left_fit_average)
            x_r_bot = get_x(y_bottom, self.right_fit_average)
            x_r_top = get_x(y_top, self.right_fit_average)

            pts = np.array(
                [
                    [x_l_bot, y_bottom],
                    [x_l_top, y_top],
                    [x_r_top, y_top],
                    [x_r_bot, y_bottom],
                ],
                np.int32,
            )
            lane_polygon_points = [pts]

            start_x = width // 2
            end_x = (x_l_top + x_r_top) // 2

            traj_pts = []
            for i in range(21):
                t = i / 20.0
                curr_y = int(y_bottom - (y_bottom - y_top) * t)
                curr_x = int((1 - t**2) * start_x + (t**2) * end_x)
                traj_pts.append([curr_x, curr_y])
            trajectory_points = [np.array(traj_pts, np.int32)]

        steer_start_x = width // 2
        steer_shift = int(self.avg_steering_offset * (width * 0.75))
        steer_end_x = steer_start_x + steer_shift

        steer_traj = []
        for i in range(21):
            t = i / 20.0
            curr_y = int(y_bottom - (y_bottom - y_top) * t)
            curr_x = int((1 - t**2) * steer_start_x + (t**2) * steer_end_x)
            steer_traj.append([curr_x, curr_y])

        steering_trajectory_points = [np.array(steer_traj, np.int32)]

        accuracy = 0.0
        if valid_lanes:
            if self.left_fit_average is not None:
                lx_top = get_x_from_fit(y_top, self.left_fit_average)
            else:
                lx_top = 0

            if self.right_fit_average is not None:
                rx_top = get_x_from_fit(y_top, self.right_fit_average)
            else:
                rx_top = width

            if lx_top is not None and rx_top is not None:
                lane_center_top = (lx_top + rx_top) / 2

                y_target = int(height * 0.45)
                target_steer_x = get_steering_x_at_y(
                    y_target, width, height, self.avg_steering_offset
                )

                diff = abs(lane_center_top - target_steer_x)
                decay = width * 0.1
                accuracy = calculate_accuracy_exponential(diff, decay)

        final_conf = float(np.mean(confidences)) if confidences else 0.0

        t1 = time.time()
        latency_ms = (t1 - t0) * 1000

        return {
            "lane_polygon_points": lane_polygon_points,
            "trajectory_points": trajectory_points,
            "steering_trajectory_points": steering_trajectory_points,
            "accuracy": accuracy,
            "confidence": final_conf,
            "latency": latency_ms,
            "mask_viz": mask_viz,
            "binary_mask": binary_mask,
            "points_viz": points_viz,
            "fit_viz": fit_viz,
        }

    def visualize(self, image: np.ndarray, prediction: dict) -> np.ndarray:
        lane_polygon_points = prediction.get("lane_polygon_points", [])
        trajectory_points = prediction.get("trajectory_points", [])
        steering_trajectory_points = prediction.get("steering_trajectory_points", [])
        accuracy = prediction.get("accuracy", 0.0)
        confidence = prediction.get("confidence", 0.0)
        latency = prediction.get("latency", 0.0)

        vis_img = image.copy()
        height, width = image.shape[:2]

        conf_color = get_confidence_color(confidence)

        if lane_polygon_points and confidence > 0.2:
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

        panel2 = prediction.get("mask_viz", np.zeros_like(image))
        draw_text_outlined(panel2, "2. YOLO Masks", (10, 30))

        panel3 = prediction.get("points_viz", np.zeros_like(image))

        if self.left_fit_average is not None:
            x_bot = get_x_from_fit(height, self.left_fit_average)
            x_top = get_x_from_fit(int(height * 0.45), self.left_fit_average)

            if x_bot is not None and x_top is not None:
                cv2.line(
                    panel3, (x_bot, height), (x_top, int(height * 0.45)), (0, 0, 255), 3
                )

        if self.right_fit_average is not None:
            x_bot = get_x_from_fit(height, self.right_fit_average)
            x_top = get_x_from_fit(int(height * 0.45), self.right_fit_average)

            if x_bot is not None and x_top is not None:
                cv2.line(
                    panel3, (x_bot, height), (x_top, int(height * 0.45)), (0, 0, 255), 3
                )

        if trajectory_points:
            cv2.polylines(panel3, trajectory_points, False, (255, 255, 255), 2)
        if steering_trajectory_points:
            cv2.polylines(panel3, steering_trajectory_points, False, (255, 0, 0), 2)

        draw_text_outlined(panel3, "3. Points & Linear Fit", (10, 30))
        steering_text = f"Steering: {self.avg_steering_offset:.2f}"
        draw_text_outlined(panel3, steering_text, (10, 70), 0.6, (0, 255, 255))

        panel4 = vis_img.copy()
        draw_text_outlined(panel4, "4. Final Overlay", (10, 30))

        conf_text = f"Conf: {int(confidence * 100)}%"
        draw_text_outlined(panel4, conf_text, (10, 60), 0.6, (0, 255, 0))

        acc_color = get_accuracy_color(accuracy)
        acc_text = f"Steering Accuracy: {accuracy:.1f}%"
        draw_text_outlined(panel4, acc_text, (10, 90), 1.0, acc_color, 2)

        panels = [panel1, panel2, panel3, panel4]
        return combine_grid(panels, (2, 2), scale=0.5)
