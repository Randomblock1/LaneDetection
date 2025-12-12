import cv2
import numpy as np


def draw_text_outlined(
    image: np.ndarray,
    text: str,
    pos: tuple,
    scale: float = 0.6,
    color: tuple = (255, 255, 255),
    thickness: int = 1,
):
    """Draw text with black outline for visibility."""
    cv2.putText(
        image, text, pos, cv2.FONT_HERSHEY_SIMPLEX, scale, (0, 0, 0), thickness + 2
    )
    cv2.putText(image, text, pos, cv2.FONT_HERSHEY_SIMPLEX, scale, color, thickness)


def generate_parabola(start_x, end_x, y_bot, y_top, n_pts=20):
    """Generate parabolic trajectory points for cv2.polylines."""
    t_pts = []
    for i in range(n_pts + 1):
        t = i / n_pts
        curr_y = int(y_bot - (y_bot - y_top) * t)
        curr_x = int((1 - t**2) * start_x + (t**2) * end_x)
        t_pts.append([curr_x, curr_y])
    return [np.array(t_pts, np.int32)]


def get_steering_x_at_y(y, width, height, steering_offset):
    """Calculate expected steering X coordinate at a specific Y level using parabolic model."""
    roi_bottom_y = height
    roi_top_y = int(height * 0.45)

    total_h = roi_bottom_y - roi_top_y
    if total_h == 0:
        return width // 2

    t = (roi_bottom_y - y) / total_h
    t = np.clip(t, 0, 1)

    start_x = width // 2
    max_shift = width * 0.75
    pixel_shift = int(steering_offset * max_shift)
    end_x = start_x + pixel_shift

    curr_x = int((1 - t**2) * start_x + (t**2) * end_x)
    return curr_x


def get_confidence_color(confidence: float) -> tuple:
    """Return BGR color transitioning from red (low) to green (high)."""
    r = int(255 * (1 - confidence))
    g = int(255 * confidence)
    return (0, g, r)


def calculate_accuracy_exponential(diff: float, decay_constant: float) -> float:
    """Calculate accuracy using exponential decay."""
    return 100.0 * np.exp(-diff / decay_constant)


def get_accuracy_color(accuracy: float) -> tuple:
    """Return color based on accuracy: green >80%, yellow >50%, red otherwise."""
    if accuracy > 80:
        return (0, 255, 0)
    elif accuracy > 50:
        return (0, 255, 255)
    else:
        return (0, 0, 255)


def get_latency_color(latency_ms: float) -> tuple:
    """Green if <30ms, orange otherwise."""
    return (0, 255, 0) if latency_ms < 30 else (0, 165, 255)


def combine_grid(
    panels: list,
    shape: tuple,
    base_height: int = None,
    base_width: int = None,
    scale: float = 0.5,
) -> np.ndarray:
    """Combine list of images into a grid with given (rows, cols) shape."""
    if not panels:
        return None

    rows, cols = shape

    if base_height is None or base_width is None:
        base_height, base_width = panels[0].shape[:2]

    resized_panels = []
    for p in panels:
        if len(p.shape) == 2:
            p = cv2.cvtColor(p, cv2.COLOR_GRAY2BGR)
        if p.shape[:2] != (base_height, base_width):
            p = cv2.resize(p, (base_width, base_height))
        resized_panels.append(p)

    total_slots = rows * cols
    while len(resized_panels) < total_slots:
        resized_panels.append(np.zeros((base_height, base_width, 3), dtype=np.uint8))

    grid_rows = []
    for r in range(rows):
        start_idx = r * cols
        end_idx = start_idx + cols
        row_panels = resized_panels[start_idx:end_idx]
        grid_rows.append(np.hstack(row_panels))

    combined = np.vstack(grid_rows)

    if scale != 1.0:
        new_w = int(combined.shape[1] * scale)
        new_h = int(combined.shape[0] * scale)
        combined = cv2.resize(combined, (new_w, new_h))

    return combined
