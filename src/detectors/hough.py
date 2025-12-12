import cv2
import numpy as np
from src.detectors.base import LaneDetector
from src.detectors.viz_utils import (
    draw_text_outlined,
    generate_parabola,
    get_confidence_color,
    get_accuracy_color,
    calculate_accuracy_exponential,
    get_latency_color,
    combine_grid,
    get_steering_x_at_y,
)
from src.utils import get_x_from_fit


class HoughLaneDetector(LaneDetector):
    """Lane detection using Hough Transform with color masking and dynamic ROI."""

    def __init__(self, debug: bool = False):
        super().__init__(debug)
        self.avg_steering_offset = 0.0

        self.left_fit_average = None
        self.right_fit_average = None
        self.smooth_factor = 0.1

        # Tunable parameters (adjustable via keyboard in debug mode)
        self.white_mask_limit = 200
        self.canny_low = 0
        self.canny_high = 255
        self.hough_threshold = 65
        self.min_line_length = 30
        self.slope_threshold = 0.2

    def detect(self, image: np.ndarray, steering: float = 0.0) -> dict:
        import time

        t0 = time.time()

        # Color masking in HLS space
        hls = cv2.cvtColor(image, cv2.COLOR_BGR2HLS)

        l_channel = hls[:, :, 1]
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        cl1 = clahe.apply(l_channel)
        hls[:, :, 1] = cl1

        lower_white = np.array([0, self.white_mask_limit, 0], dtype=np.uint8)
        upper_white = np.array([180, 255, 255], dtype=np.uint8)
        white_mask = cv2.inRange(hls, lower_white, upper_white)

        lower_yellow = np.array([15, 30, 100], dtype=np.uint8)
        upper_yellow = np.array([35, 204, 255], dtype=np.uint8)
        yellow_mask = cv2.inRange(hls, lower_yellow, upper_yellow)

        color_mask = cv2.bitwise_or(white_mask, yellow_mask)

        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        gray_masked = cv2.bitwise_and(gray, gray, mask=color_mask)

        blur = cv2.GaussianBlur(gray_masked, (9, 9), 0)
        edges = cv2.Canny(blur, self.canny_low, self.canny_high)

        height, width = image.shape[:2]

        # Dynamic ROI based on steering
        alpha = 0.1
        self.avg_steering_offset = (alpha * steering) + (
            (1 - alpha) * self.avg_steering_offset
        )

        max_shift = width * 0.75
        pixel_shift = int(self.avg_steering_offset * max_shift)

        roi_top_center_x = (width / 2) + pixel_shift
        roi_top_width = width * 0.2
        roi_top_left_x = int(roi_top_center_x - (roi_top_width / 2))
        roi_top_right_x = int(roi_top_center_x + (roi_top_width / 2))

        roi_top_left_x = np.clip(roi_top_left_x, 0, width)
        roi_top_right_x = np.clip(roi_top_right_x, 0, width)

        mask = np.zeros_like(edges)
        roi_bottom_y = height
        roi_narrowing_curr = int(height * 0.70)
        roi_top_y = int(height * 0.45)

        polygon = np.array(
            [
                [
                    (0, roi_bottom_y),
                    (width, roi_bottom_y),
                    (width, roi_narrowing_curr),
                    (roi_top_right_x, roi_top_y),
                    (roi_top_left_x, roi_top_y),
                    (0, roi_narrowing_curr),
                ]
            ],
            np.int32,
        )
        cv2.fillPoly(mask, polygon, 255)
        masked_edges = cv2.bitwise_and(edges, mask)

        lines = cv2.HoughLinesP(
            masked_edges,
            1,
            np.pi / 180,
            self.hough_threshold,
            minLineLength=self.min_line_length,
            maxLineGap=100,
        )

        left_points = []
        right_points = []
        n_left_lines = 0
        n_right_lines = 0

        rejected_lines_slope = []
        rejected_lines_length = []

        if lines is not None:
            for line in lines:
                x1, y1, x2, y2 = line[0]
                if x2 == x1:
                    continue

                line_len = np.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)
                avg_y = (y1 + y2) / 2
                total_roi_height = roi_bottom_y - roi_top_y

                # Minimum length increases with distance from horizon
                if total_roi_height > 0:
                    ratio = (avg_y - roi_top_y) / total_roi_height
                    ratio = np.clip(ratio, 0, 1)
                    required_len = 25 + (75 * ratio)
                else:
                    required_len = 25

                if line_len < required_len:
                    rejected_lines_length.append(line)
                    continue

                slope = (y2 - y1) / (x2 - x1)

                if abs(slope) > self.slope_threshold:
                    if slope < 0:
                        left_points.append((x1, y1))
                        left_points.append((x2, y2))
                        n_left_lines += 1
                    else:
                        right_points.append((x1, y1))
                        right_points.append((x2, y2))
                        n_right_lines += 1
                else:
                    rejected_lines_slope.append(line)

        # Fit lines: polyfit(y, x, 1) gives x = k*y + c
        current_left_fit = None
        current_right_fit = None

        if len(left_points) > 2:
            pts = np.array(left_points)
            try:
                current_left_fit = np.polyfit(pts[:, 1], pts[:, 0], 1)
            except np.linalg.LinAlgError:
                pass

        if len(right_points) > 2:
            pts = np.array(right_points)
            try:
                current_right_fit = np.polyfit(pts[:, 1], pts[:, 0], 1)
            except np.linalg.LinAlgError:
                pass

        # Exponential moving average smoothing
        if current_left_fit is not None:
            if self.left_fit_average is None:
                self.left_fit_average = current_left_fit
            else:
                self.left_fit_average = (self.smooth_factor * current_left_fit) + (
                    (1 - self.smooth_factor) * self.left_fit_average
                )

        if current_right_fit is not None:
            if self.right_fit_average is None:
                self.right_fit_average = current_right_fit
            else:
                self.right_fit_average = (self.smooth_factor * current_right_fit) + (
                    (1 - self.smooth_factor) * self.right_fit_average
                )

        lane_polygon_points = []
        clustered_lines_viz = []

        max_lines = 10.0

        def get_exp_conf(n):
            return min(n, max_lines) / max_lines

        conf_left = get_exp_conf(n_left_lines)
        conf_right = get_exp_conf(n_right_lines)
        confidence = (conf_left + conf_right) / 2.0

        if (
            self.left_fit_average is not None
            and self.right_fit_average is not None
            and n_left_lines > 0
            and n_right_lines > 0
        ):
            y_bottom = roi_bottom_y
            x_left_bottom = get_x_from_fit(y_bottom, self.left_fit_average)
            x_right_bottom = get_x_from_fit(y_bottom, self.right_fit_average)

            # Find lane intersection point
            k_l, c_l = self.left_fit_average
            k_r, c_r = self.right_fit_average

            denom = k_l - k_r
            if abs(denom) < 1e-4:
                intersection_y = roi_top_y
            else:
                intersection_y = (c_r - c_l) / denom

            y_top = max(int(intersection_y), roi_top_y)

            if y_top > roi_bottom_y:
                y_top = roi_top_y

            x_left_top = get_x_from_fit(y_top, self.left_fit_average)
            x_right_top = get_x_from_fit(y_top, self.right_fit_average)

            # Validation
            if None not in [x_left_bottom, x_right_bottom, x_left_top, x_right_top]:
                pts = np.array(
                    [
                        [x_left_bottom, y_bottom],
                        [x_left_top, y_top],
                        [x_right_top, y_top],
                        [x_right_bottom, y_bottom],
                    ],
                    np.int32,
                )
                lane_polygon_points = [pts]

            clustered_lines_viz.append([[x_left_bottom, y_bottom, x_left_top, y_top]])
            clustered_lines_viz.append([[x_right_bottom, y_bottom, x_right_top, y_top]])

        trajectory_points = []
        steering_trajectory_points = []

        if lane_polygon_points and confidence > 0.2:
            start_x = width // 2
            end_x = (x_left_top + x_right_top) // 2
            trajectory_points = generate_parabola(start_x, end_x, roi_bottom_y, y_top)

        steering_shift = int(self.avg_steering_offset * (width * 0.75))
        steer_start_x = width // 2
        steer_end_x = (width // 2) + steering_shift
        steering_trajectory_points = generate_parabola(
            steer_start_x, steer_end_x, roi_bottom_y, roi_top_y
        )

        accuracy = 0.0
        if lane_polygon_points:
            lane_center_x = (x_left_top + x_right_top) // 2
            target_steer_x = get_steering_x_at_y(
                y_top, width, height, self.avg_steering_offset
            )
            diff = abs(lane_center_x - target_steer_x)
            decay_constant = width * 0.1
            accuracy = calculate_accuracy_exponential(diff, decay_constant)

        # ROI heatmap visualization
        filter_viz = np.zeros((height, width, 3), dtype=np.uint8)
        y_indices = np.arange(height).reshape(-1, 1)
        gradient_col = (y_indices - roi_top_y) / (roi_bottom_y - roi_top_y)
        gradient_col = np.clip(gradient_col, 0, 1)
        heatmap_gray_col = (gradient_col * 255).astype(np.uint8)
        heatmap_col = cv2.applyColorMap(heatmap_gray_col, cv2.COLORMAP_JET)
        heatmap = cv2.resize(
            heatmap_col, (width, height), interpolation=cv2.INTER_NEAREST
        )
        mask_poly = np.zeros((height, width), dtype=np.uint8)
        cv2.fillPoly(mask_poly, polygon, 255)
        filter_viz = cv2.bitwise_and(heatmap, heatmap, mask=mask_poly)

        t1 = time.time()
        latency_ms = (t1 - t0) * 1000

        return {
            "lines": lines,
            "lane_polygon_points": lane_polygon_points,
            "clustered_lines": clustered_lines_viz,
            "trajectory_points": trajectory_points,
            "steering_trajectory_points": steering_trajectory_points,
            "rejected_slope": rejected_lines_slope,
            "rejected_length": rejected_lines_length,
            "edges": edges,
            "masked_edges": masked_edges,
            "color_mask": color_mask,
            "roi_polygon": polygon,
            "filter_viz": filter_viz,
            "latency": latency_ms,
            "confidence": confidence,
            "accuracy": accuracy,
            "num_left": n_left_lines,
            "num_right": n_right_lines,
        }

    def visualize(self, image: np.ndarray, prediction: dict) -> np.ndarray:
        lines = prediction.get("lines")
        roi_polygon = prediction.get("roi_polygon")
        filter_viz = prediction.get("filter_viz")
        lane_polygon_points = prediction.get("lane_polygon_points", [])
        clustered_lines = prediction.get("clustered_lines", [])
        trajectory_points = prediction.get("trajectory_points", [])
        steering_trajectory_points = prediction.get("steering_trajectory_points", [])
        latency = prediction.get("latency", 0.0)
        confidence = prediction.get("confidence", 0.0)
        num_left = prediction.get("num_left", 0)
        num_right = prediction.get("num_right", 0)
        accuracy = prediction.get("accuracy", 0.0)

        overlay_img = image.copy()
        conf_color = get_confidence_color(confidence)

        if lane_polygon_points and confidence > 0.2:
            lane_overlay = np.zeros_like(overlay_img)
            cv2.fillPoly(lane_overlay, lane_polygon_points, conf_color)
            overlay_img = cv2.addWeighted(overlay_img, 1, lane_overlay, 0.3, 0)
            cv2.polylines(
                overlay_img,
                lane_polygon_points,
                isClosed=True,
                color=conf_color,
                thickness=2,
            )

        if steering_trajectory_points:
            cv2.polylines(
                overlay_img, steering_trajectory_points, False, (255, 0, 0), 3
            )

        if trajectory_points:
            cv2.polylines(overlay_img, trajectory_points, False, conf_color, 4)

        if not self.debug:
            return overlay_img

        height, width = image.shape[:2]

        raw = image.copy()
        draw_text_outlined(raw, "1. Raw Image", (10, 30))
        c_lat = get_latency_color(latency)
        draw_text_outlined(raw, f"Lat: {latency:.1f}ms", (10, 60), 0.6, c_lat)

        color_mask = prediction.get(
            "color_mask", np.zeros((height, width), dtype=np.uint8)
        )
        color_mask_bgr = cv2.cvtColor(color_mask, cv2.COLOR_GRAY2BGR)
        draw_text_outlined(color_mask_bgr, "2. Color Mask", (10, 30))
        draw_text_outlined(
            color_mask_bgr,
            f"White L: {self.white_mask_limit} (n-/m+)",
            (10, height - 20),
            0.5,
            (0, 255, 255),
        )

        edges = prediction.get("edges", np.zeros((height, width), dtype=np.uint8))
        edges_bgr = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)
        draw_text_outlined(edges_bgr, "3. Canny Edges", (10, 30))
        draw_text_outlined(
            edges_bgr,
            f"Low: {self.canny_low} (u-/i+)",
            (10, height - 40),
            0.5,
            (0, 255, 255),
        )
        draw_text_outlined(
            edges_bgr,
            f"High: {self.canny_high} (o-/p+)",
            (10, height - 20),
            0.5,
            (0, 255, 255),
        )

        masked_edges = prediction.get(
            "masked_edges", np.zeros((height, width), dtype=np.uint8)
        )
        masked_edges_bgr = cv2.cvtColor(masked_edges, cv2.COLOR_GRAY2BGR)
        draw_text_outlined(masked_edges_bgr, "4. ROI Masked Edges", (10, 30))

        filter_frame = (
            filter_viz.copy()
            if filter_viz is not None
            else np.zeros((height, width, 3), dtype=np.uint8)
        )
        filter_frame = (filter_frame * 0.5).astype(np.uint8)

        if roi_polygon is not None:
            cv2.polylines(filter_frame, [roi_polygon], True, (0, 255, 255), 2)

        if lines is not None:
            for line in lines:
                x1, y1, x2, y2 = line[0]
                cv2.line(filter_frame, (x1, y1), (x2, y2), (0, 255, 0), 1)

        rejected_length = prediction.get("rejected_length", [])
        for line in rejected_length:
            x1, y1, x2, y2 = line[0]
            cv2.line(filter_frame, (x1, y1), (x2, y2), (255, 0, 0), 1)

        rejected_slope = prediction.get("rejected_slope", [])
        for line in rejected_slope:
            x1, y1, x2, y2 = line[0]
            cv2.line(filter_frame, (x1, y1), (x2, y2), (0, 0, 255), 1)

        if clustered_lines:
            for line in clustered_lines:
                x1, y1, x2, y2 = line[0]
                cv2.line(filter_frame, (x1, y1), (x2, y2), (255, 0, 255), 3)

        if trajectory_points:
            cv2.polylines(filter_frame, trajectory_points, False, (255, 255, 255), 2)
        if steering_trajectory_points:
            cv2.polylines(
                filter_frame, steering_trajectory_points, False, (255, 0, 0), 2
            )

        steering_text = f"Steering: {self.avg_steering_offset:.2f}"
        draw_text_outlined(filter_frame, steering_text, (10, 70), 0.6, (0, 255, 255))

        draw_text_outlined(
            filter_frame,
            f"Hough Thresh: {self.hough_threshold} ([-/]+)",
            (10, 100),
            0.5,
            (0, 255, 255),
        )
        draw_text_outlined(
            filter_frame,
            f"Min Line Len: {self.min_line_length} (k-/l+)",
            (10, 120),
            0.5,
            (0, 255, 255),
        )
        draw_text_outlined(
            filter_frame,
            f"slope Thresh: {self.slope_threshold:.2f} (v-/b+)",
            (10, 140),
            0.5,
            (0, 255, 255),
        )
        draw_text_outlined(
            filter_frame, "5. Lane(Mag), Raw(Grn), Rej(Red/Blu)", (10, 30)
        )

        draw_text_outlined(overlay_img, "6. Final Overlay", (10, 30))
        conf_text = f"Conf: {int(confidence * 100)}% (L={num_left} R={num_right})"
        draw_text_outlined(overlay_img, conf_text, (10, 60), 0.6, (0, 255, 0))

        acc_color = get_accuracy_color(accuracy)
        acc_text = f"Steering Accuracy (exponential): {accuracy:.1f}%"
        draw_text_outlined(overlay_img, acc_text, (10, 90), 1.0, acc_color, 2)

        panels = [
            raw,
            color_mask_bgr,
            edges_bgr,
            masked_edges_bgr,
            filter_frame,
            overlay_img,
        ]
        return combine_grid(panels, (2, 3), scale=0.5)
