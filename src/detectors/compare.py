from .base import LaneDetector
import numpy as np
import cv2
from .hough import HoughLaneDetector
from .sliding_window import SlidingWindowDetector
from .yolo import YoloLaneDetector
from .viz_utils import (
    combine_grid,
    draw_text_outlined,
    get_latency_color,
    get_accuracy_color,
)


class ComparisonDetector(LaneDetector):
    """Runs all detection methods and displays them in a 2x2 comparison grid."""

    def __init__(self, yolo_model_path: str, debug: bool = False):
        super().__init__(debug)
        self.detectors = {
            "hough": HoughLaneDetector(debug=False),
            "sliding": SlidingWindowDetector(debug=False),
            "yolo": YoloLaneDetector(yolo_model_path, debug=False),
        }
        self.detectors["yolo"].load_model()

    def detect(self, image: np.ndarray, steering: float = 0.0) -> dict:
        results = {}
        for name, detector in self.detectors.items():
            results[name] = detector.detect(image, steering)
        return results

    def visualize(self, image: np.ndarray, prediction: dict) -> np.ndarray:
        h_res = prediction["hough"]
        s_res = prediction["sliding"]
        y_res = prediction["yolo"]

        raw_panel = image.copy()
        draw_text_outlined(raw_panel, "1. Input / Latency", (10, 30))

        y_offset = 60
        total_lat = 0
        for name, res in [("Hough", h_res), ("Slid", s_res), ("YOLO", y_res)]:
            lat = res.get("latency", 0.0)
            total_lat += lat
            c = get_latency_color(lat)
            draw_text_outlined(
                raw_panel, f"{name}: {lat:.1f}ms", (10, y_offset), 0.6, c
            )
            y_offset += 30

        draw_text_outlined(
            raw_panel,
            f"Total: {total_lat:.1f}ms",
            (10, y_offset + 10),
            0.7,
            (0, 0, 255),
        )

        h_viz = self.detectors["hough"].visualize(image, h_res)
        draw_text_outlined(h_viz, "2. Hough Transform", (10, 30), 1.0, (0, 255, 255))
        h_acc = h_res.get("accuracy", 0.0)
        h_c = get_accuracy_color(h_acc)
        draw_text_outlined(h_viz, f"Acc: {h_acc:.1f}%", (10, 60), 0.8, h_c, 2)

        s_viz = self.detectors["sliding"].visualize(image, s_res)
        draw_text_outlined(s_viz, "3. Sliding Window", (10, 30), 1.0, (0, 255, 255))
        s_acc = s_res.get("accuracy", 0.0)
        s_c = get_accuracy_color(s_acc)
        draw_text_outlined(s_viz, f"Acc: {s_acc:.1f}%", (10, 60), 0.8, s_c, 2)

        y_viz = self.detectors["yolo"].visualize(image, y_res)
        draw_text_outlined(y_viz, "4. YOLOv8", (10, 30), 1.0, (0, 255, 255))
        y_acc = y_res.get("accuracy", 0.0)
        y_c = get_accuracy_color(y_acc)
        draw_text_outlined(y_viz, f"Acc: {y_acc:.1f}%", (10, 60), 0.8, y_c, 2)

        panels = [raw_panel, h_viz, s_viz, y_viz]
        return combine_grid(panels, (2, 2), scale=0.5)
