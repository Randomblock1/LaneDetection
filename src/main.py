import cv2
import argparse
from src.simulation import SimulationManager
from src.detectors.hough import HoughLaneDetector
from src.detectors.sliding_window import SlidingWindowDetector
from src.detectors.yolo import YoloLaneDetector
from src.detectors.compare import ComparisonDetector


def main():
    parser = argparse.ArgumentParser(description="BeamNG Lane Detection")
    parser.add_argument(
        "--method",
        choices=["hough", "sliding", "yolo", "compare"],
        default="hough",
        help="Detection method: 'hough', 'sliding', 'yolo', or 'compare'",
    )
    parser.add_argument(
        "--yolo_path",
        default="yolov8-trained.pt",
        help="Path to YOLO model",
    )
    args = parser.parse_args()

    print("Initializing Simulation...")
    sim = SimulationManager()

    try:
        sim.setup_scenario()

        if args.method == "hough":
            detector = HoughLaneDetector(debug=True)
        elif args.method == "sliding":
            detector = SlidingWindowDetector(debug=True)
        elif args.method == "yolo":
            detector = YoloLaneDetector(args.yolo_path, debug=True)
            detector.load_model()
        elif args.method == "compare":
            print("Initializing Comparison Mode (Loading all models)...")
            detector = ComparisonDetector(args.yolo_path, debug=True)

        print(
            f"Running {args.method} detection. Press 'q' to quit, 'j' to pause/resume."
        )

        if args.method == "hough":
            print(
                "Controls: u/i (Canny Low), o/p (Canny High), [/] (Hough Thresh), k/l (MinLen), n/m (White Mask), v/b (Slope)"
            )

        paused = False
        last_frame = None
        last_steering = 0.0
        vis_frame = None
        params_changed = False

        while True:
            if not paused:
                frame, steering = sim.get_frame()
                if frame is not None:
                    last_frame = frame
                    last_steering = steering

            # Reprocess when live, params changed, or first frame while paused
            should_process = (
                (not paused)
                or params_changed
                or (vis_frame is None and last_frame is not None)
            )

            if last_frame is not None and should_process:
                results = detector.detect(last_frame, steering=last_steering)
                vis_frame = detector.visualize(last_frame, results)
                params_changed = False

            if vis_frame is not None:
                cv2.imshow("Lane Detection", vis_frame)

            key = cv2.waitKey(1) & 0xFF

            if key == ord("q"):
                break
            elif key == ord("j"):
                paused = not paused
                if paused:
                    sim.pause()
                else:
                    sim.resume()

            # Hough parameter tuning via keyboard
            if args.method == "hough" and hasattr(detector, "canny_low"):
                if key == ord("u"):
                    detector.canny_low = max(0, detector.canny_low - 5)
                    params_changed = True
                elif key == ord("i"):
                    detector.canny_low = min(255, detector.canny_low + 5)
                    params_changed = True
                elif key == ord("o"):
                    detector.canny_high = max(0, detector.canny_high - 5)
                    params_changed = True
                elif key == ord("p"):
                    detector.canny_high = min(255, detector.canny_high + 5)
                    params_changed = True
                elif key == ord("["):
                    detector.hough_threshold = max(1, detector.hough_threshold - 5)
                    params_changed = True
                elif key == ord("]"):
                    detector.hough_threshold = min(200, detector.hough_threshold + 5)
                    params_changed = True
                elif key == ord("k"):
                    detector.min_line_length = max(1, detector.min_line_length - 5)
                    params_changed = True
                elif key == ord("l"):
                    detector.min_line_length = min(200, detector.min_line_length + 5)
                    params_changed = True
                elif key == ord("n"):
                    detector.white_mask_limit = max(0, detector.white_mask_limit - 5)
                    params_changed = True
                elif key == ord("m"):
                    detector.white_mask_limit = min(255, detector.white_mask_limit + 5)
                    params_changed = True
                elif key == ord("v"):
                    detector.slope_threshold = max(0.0, detector.slope_threshold - 0.05)
                    params_changed = True
                elif key == ord("b"):
                    detector.slope_threshold = min(1.0, detector.slope_threshold + 0.05)
                    params_changed = True

    except Exception as e:
        print(f"An error occurred: {e}")
        import traceback

        traceback.print_exc()
    finally:
        print("Closing simulation...")
        sim.close()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
