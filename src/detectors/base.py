from abc import ABC, abstractmethod
import cv2
import numpy as np


class LaneDetector(ABC):
    """Abstract base class for lane detection algorithms."""

    def __init__(self, debug: bool = False):
        self.debug = debug
        self.last_result = {}

    @abstractmethod
    def detect(self, image: np.ndarray, steering: float = 0.0) -> dict:
        """Process image and return lane information dict with 'left_lane' and 'right_lane'."""
        pass

    def visualize(self, image: np.ndarray, prediction: dict) -> np.ndarray:
        """Draw detected lanes on image. Override in subclasses."""
        vis_img = image.copy()
        cv2.putText(
            vis_img,
            "Base LaneDetector",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 0, 255),
            2,
        )
        return vis_img
