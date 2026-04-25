from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np


WINDOW_NAME = "Actuator Bend Setup"
TRACK_WINDOW_NAME = "Actuator Bend Tracking"
POINT_COUNT = 3


@dataclass
class FrameMeasurement:
    frame_index: int
    time_seconds: float
    angle_degrees: float
    flexion_degrees: float
    p1_x: float
    p1_y: float
    p2_x: float
    p2_y: float
    p3_x: float
    p3_y: float


def parse_args() -> argparse.Namespace:
    repo_root = Path(__file__).resolve().parents[1]
    default_video = repo_root / "data" / "04-15_Actuator_Test.mp4"
    default_output = repo_root / "analysis" / "actuator_bend"

    parser = argparse.ArgumentParser(
        description=(
            "Baseline actuator-bend tracker using OpenCV optical flow. "
            "Click 3 points along the actuator in base-to-tip order. "
            "The middle point is treated as the bend joint."
        )
    )
    parser.add_argument(
        "--video",
        type=Path,
        default=default_video,
        help="Path to the input video. Defaults to data/04-15_Actuator_Test.mp4.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=default_output,
        help="Directory for CSV data, selected-frame snapshot, and annotated video.",
    )
    parser.add_argument(
        "--show-preview",
        action="store_true",
        help="Show the tracked video while processing. Press ESC to stop early.",
    )
    parser.add_argument(
        "--max-frames",
        type=int,
        default=None,
        help="Optional cap for quick tests.",
    )
    parser.add_argument(
        "--smoothing-window",
        type=int,
        default=7,
        help="Window size for the simple moving-average flexion column in the CSV.",
    )
    return parser.parse_args()


def compute_angle_degrees(a: np.ndarray, b: np.ndarray, c: np.ndarray) -> float:
    ba = a - b
    bc = c - b

    ba_norm = np.linalg.norm(ba)
    bc_norm = np.linalg.norm(bc)
    if ba_norm == 0 or bc_norm == 0:
        return float("nan")

    cosine = float(np.dot(ba, bc) / (ba_norm * bc_norm))
    cosine = float(np.clip(cosine, -1.0, 1.0))
    return float(np.degrees(np.arccos(cosine)))


def compute_flexion_degrees(angle_degrees: float) -> float:
    if np.isnan(angle_degrees):
        return float("nan")
    return max(0.0, 180.0 - angle_degrees)


def moving_average(values: list[float], window: int) -> list[float]:
    if window <= 1:
        return values[:]

    smoothed: list[float] = []
    for index in range(len(values)):
        start = max(0, index - window + 1)
        window_values = [value for value in values[start : index + 1] if not np.isnan(value)]
        if not window_values:
            smoothed.append(float("nan"))
            continue
        smoothed.append(float(sum(window_values) / len(window_values)))
    return smoothed


def draw_points(frame: np.ndarray, points: np.ndarray) -> None:
    integer_points = points.astype(int)
    if len(integer_points) >= 2:
        cv2.polylines(frame, [integer_points], isClosed=False, color=(0, 200, 255), thickness=2)

    for index, point in enumerate(integer_points, start=1):
        x, y = int(point[0]), int(point[1])
        cv2.circle(frame, (x, y), 7, (0, 120, 255), -1)
        cv2.circle(frame, (x, y), 10, (255, 255, 255), 2)
        cv2.putText(
            frame,
            f"P{index}",
            (x + 12, y - 12),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )


def annotate_frame(
    frame: np.ndarray,
    points: np.ndarray,
    frame_index: int,
    angle_degrees: float,
    flexion_degrees: float,
) -> np.ndarray:
    annotated = frame.copy()
    draw_points(annotated, points)

    cv2.putText(
        annotated,
        f"Frame: {frame_index}",
        (20, 35),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )
    cv2.putText(
        annotated,
        f"Joint angle: {angle_degrees:6.2f} deg",
        (20, 70),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )
    cv2.putText(
        annotated,
        f"Bending: {flexion_degrees:6.2f} deg",
        (20, 105),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )
    cv2.putText(
        annotated,
        "Straight actuator is close to 0 deg bend.",
        (20, 140),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )
    return annotated


def select_initial_points(frame: np.ndarray) -> np.ndarray:
    selected_points: list[tuple[int, int]] = []
    display_frame = frame.copy()

    def redraw() -> None:
        nonlocal display_frame
        display_frame = frame.copy()
        instructions = [
            "Click 3 points along the actuator in order:",
            "1) point before the bend",
            "2) bend joint",
            "3) point after the bend",
            "Press R to reset, ENTER or SPACE to confirm.",
        ]
        for index, line in enumerate(instructions):
            cv2.putText(
                display_frame,
                line,
                (20, 35 + 28 * index),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (255, 255, 255),
                2,
                cv2.LINE_AA,
            )

        if selected_points:
            point_array = np.array(selected_points, dtype=np.float32)
            draw_points(display_frame, point_array)

    def on_mouse(event: int, x: int, y: int, flags: int, param: object) -> None:
        del flags, param
        if event == cv2.EVENT_LBUTTONDOWN and len(selected_points) < POINT_COUNT:
            selected_points.append((x, y))
            redraw()

    redraw()
    cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
    cv2.setMouseCallback(WINDOW_NAME, on_mouse)

    while True:
        cv2.imshow(WINDOW_NAME, display_frame)
        key = cv2.waitKey(20) & 0xFF
        if key in (13, 32) and len(selected_points) == POINT_COUNT:
            break
        if key in (ord("r"), ord("R")):
            selected_points.clear()
            redraw()
        if key in (27, ord("q"), ord("Q")):
            cv2.destroyWindow(WINDOW_NAME)
            raise KeyboardInterrupt("Point selection cancelled by user.")

    cv2.destroyWindow(WINDOW_NAME)
    return np.array(selected_points, dtype=np.float32)


def create_video_writer(output_path: Path, fps: float, frame_shape: tuple[int, int, int]) -> cv2.VideoWriter:
    height, width = frame_shape[:2]
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(output_path), fourcc, fps, (width, height))
    if not writer.isOpened():
        raise RuntimeError(f"Unable to create output video: {output_path}")
    return writer


def save_measurements_csv(
    output_path: Path,
    measurements: list[FrameMeasurement],
    smoothing_window: int,
) -> None:
    smoothed_flexion = moving_average(
        [measurement.flexion_degrees for measurement in measurements],
        smoothing_window,
    )

    with output_path.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(
            [
                "frame_index",
                "time_seconds",
                "angle_degrees",
                "flexion_degrees",
                "smoothed_flexion_degrees",
                "p1_x",
                "p1_y",
                "p2_x",
                "p2_y",
                "p3_x",
                "p3_y",
            ]
        )

        for measurement, smoothed_value in zip(measurements, smoothed_flexion, strict=True):
            writer.writerow(
                [
                    measurement.frame_index,
                    f"{measurement.time_seconds:.6f}",
                    f"{measurement.angle_degrees:.6f}",
                    f"{measurement.flexion_degrees:.6f}",
                    f"{smoothed_value:.6f}",
                    f"{measurement.p1_x:.3f}",
                    f"{measurement.p1_y:.3f}",
                    f"{measurement.p2_x:.3f}",
                    f"{measurement.p2_y:.3f}",
                    f"{measurement.p3_x:.3f}",
                    f"{measurement.p3_y:.3f}",
                ]
            )


def measure_frame(points: np.ndarray, frame_index: int, fps: float) -> FrameMeasurement:
    angle_degrees = compute_angle_degrees(points[0], points[1], points[2])
    flexion_degrees = compute_flexion_degrees(angle_degrees)
    return FrameMeasurement(
        frame_index=frame_index,
        time_seconds=frame_index / fps,
        angle_degrees=angle_degrees,
        flexion_degrees=flexion_degrees,
        p1_x=float(points[0][0]),
        p1_y=float(points[0][1]),
        p2_x=float(points[1][0]),
        p2_y=float(points[1][1]),
        p3_x=float(points[2][0]),
        p3_y=float(points[2][1]),
    )


def process_video(
    video_path: Path,
    output_dir: Path,
    show_preview: bool,
    max_frames: int | None,
    smoothing_window: int,
) -> None:
    if not video_path.exists():
        raise FileNotFoundError(f"Video not found: {video_path}")

    output_dir.mkdir(parents=True, exist_ok=True)

    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise RuntimeError(f"Unable to open video: {video_path}")

    success, first_frame = capture.read()
    if not success or first_frame is None:
        capture.release()
        raise RuntimeError(f"Unable to read the first frame from: {video_path}")

    fps = capture.get(cv2.CAP_PROP_FPS)
    if fps <= 0:
        fps = 30.0

    selected_points = select_initial_points(first_frame)
    initial_angle = compute_angle_degrees(selected_points[0], selected_points[1], selected_points[2])
    initial_flexion = compute_flexion_degrees(initial_angle)

    initial_snapshot_path = output_dir / "selected_points.png"
    initial_overlay = annotate_frame(
        frame=first_frame,
        points=selected_points,
        frame_index=0,
        angle_degrees=initial_angle,
        flexion_degrees=initial_flexion,
    )
    cv2.imwrite(str(initial_snapshot_path), initial_overlay)

    annotated_video_path = output_dir / "actuator_bend_annotated.mp4"
    writer = create_video_writer(annotated_video_path, fps, first_frame.shape)

    lk_params = dict(
        winSize=(31, 31),
        maxLevel=3,
        criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01),
    )

    measurements: list[FrameMeasurement] = []
    previous_gray = cv2.cvtColor(first_frame, cv2.COLOR_BGR2GRAY)
    tracked_points = selected_points.reshape(-1, 1, 2)

    first_measurement = measure_frame(selected_points, frame_index=0, fps=fps)
    measurements.append(first_measurement)
    writer.write(
        annotate_frame(
            frame=first_frame,
            points=selected_points,
            frame_index=0,
            angle_degrees=first_measurement.angle_degrees,
            flexion_degrees=first_measurement.flexion_degrees,
        )
    )

    frame_index = 1
    while True:
        if max_frames is not None and frame_index >= max_frames:
            break

        success, frame = capture.read()
        if not success or frame is None:
            break

        current_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        next_points, status, _error = cv2.calcOpticalFlowPyrLK(
            previous_gray,
            current_gray,
            tracked_points,
            None,
            **lk_params,
        )

        if next_points is None or status is None or not bool(status.ravel().all()):
            print(f"Tracking stopped at frame {frame_index} because one or more points were lost.")
            break

        tracked_points = next_points
        flat_points = tracked_points.reshape(-1, 2)
        measurement = measure_frame(flat_points, frame_index=frame_index, fps=fps)
        measurements.append(measurement)

        annotated = annotate_frame(
            frame=frame,
            points=flat_points,
            frame_index=frame_index,
            angle_degrees=measurement.angle_degrees,
            flexion_degrees=measurement.flexion_degrees,
        )
        writer.write(annotated)

        if show_preview:
            cv2.imshow(TRACK_WINDOW_NAME, annotated)
            if cv2.waitKey(1) & 0xFF == 27:
                print("Stopped early at user request.")
                break

        previous_gray = current_gray
        frame_index += 1

    capture.release()
    writer.release()
    cv2.destroyAllWindows()

    csv_path = output_dir / "actuator_bend_measurements.csv"
    save_measurements_csv(csv_path, measurements, smoothing_window=smoothing_window)

    flexion_values = [measurement.flexion_degrees for measurement in measurements if not np.isnan(measurement.flexion_degrees)]
    if flexion_values:
        print(f"Processed {len(measurements)} frames from {video_path.name}.")
        print(f"Minimum flexion: {min(flexion_values):.2f} deg")
        print(f"Maximum flexion: {max(flexion_values):.2f} deg")
        print(f"Range of motion: {max(flexion_values) - min(flexion_values):.2f} deg")
    else:
        print(f"Processed {len(measurements)} frames from {video_path.name}, but no valid angles were produced.")

    print(f"CSV saved to: {csv_path}")
    print(f"Annotated video saved to: {annotated_video_path}")
    print(f"Selected-point snapshot saved to: {initial_snapshot_path}")


def main() -> None:
    args = parse_args()
    process_video(
        video_path=args.video.resolve(),
        output_dir=args.output_dir.resolve(),
        show_preview=args.show_preview,
        max_frames=args.max_frames,
        smoothing_window=args.smoothing_window,
    )


if __name__ == "__main__":
    main()
