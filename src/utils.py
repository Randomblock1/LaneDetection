import numpy as np
import cv2


def compute_error(predicted_x, gt_x):
    """L1 distance between predicted and ground truth x coordinates."""
    if predicted_x is None or gt_x is None:
        return float("inf")
    return abs(predicted_x - gt_x)


def get_perspective_transform(width, height):
    """Returns M and Minv matrices for Bird's Eye View transformation."""
    src = np.float32(
        [
            (width * 0.45, height * 0.55),
            (width * 0.55, height * 0.55),
            (0, height),
            (width, height),
        ]
    )

    offset = width * 0.25
    dst = np.float32(
        [(offset, 0), (width - offset, 0), (offset, height), (width - offset, height)]
    )

    M = cv2.getPerspectiveTransform(src, dst)
    Minv = cv2.getPerspectiveTransform(dst, src)
    return M, Minv


def warp_image(image, M):
    """Applies perspective transform to image."""
    h, w = image.shape[:2]
    return cv2.warpPerspective(image, M, (w, h), flags=cv2.INTER_LINEAR)


def get_x_from_fit(y: int, fit: np.ndarray) -> int:
    """Returns x coordinate for given y using linear fit (m, b) where x = m*y + b."""
    if fit is None or len(fit) != 2:
        return None
    m, b = fit
    return int(m * y + b)
