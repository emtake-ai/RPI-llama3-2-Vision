import argparse
import json
import time
import urllib.error
import urllib.request
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor

import cv2
import numpy as np
from ultralytics import YOLO

from test import RTSP_URL
from yolo_od import get_detected_classes


WINDOW_NAME = "YOLOv11 + Llama Report"
OLLAMA_URL = "http://127.0.0.1:11434/api/generate"
LOG_FILE = "log.txt"


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run YOLOv11 on RTSP video and report detections with Llama."
    )
    parser.add_argument(
        "--model",
        default="yolo11n.pt",
        help="YOLOv11 model path/name. Default: yolo11n.pt",
    )
    parser.add_argument(
        "--llm-model",
        default="llama3.2:3b",
        help="Ollama model name. Default: llama3.2:3b",
    )
    parser.add_argument(
        "--conf",
        type=float,
        default=0.25,
        help="Detection confidence threshold. Default: 0.25",
    )
    parser.add_argument(
        "--device",
        default="cpu",
        help="YOLO inference device. Default: cpu",
    )
    parser.add_argument(
        "--url",
        default=RTSP_URL,
        help="RTSP URL. Defaults to RTSP_URL from test.py.",
    )
    parser.add_argument(
        "--interval",
        type=float,
        default=10.0,
        help="Seconds between Llama reports. Default: 10",
    )
    parser.add_argument(
        "--window-width",
        type=int,
        default=1280,
        help="Initial display window width. Default: 1280",
    )
    parser.add_argument(
        "--window-height",
        type=int,
        default=720,
        help="Initial display window height. Default: 720",
    )
    parser.add_argument(
        "--report-panel-height",
        type=int,
        default=220,
        help="Deprecated. Use --report-panel-width for the left report panel.",
    )
    parser.add_argument(
        "--report-panel-width",
        type=int,
        default=420,
        help="Width of the left report section. Default: 420",
    )
    return parser.parse_args()


def append_log(timestamp, report, elapsed):
    with open(LOG_FILE, "a", encoding="utf-8") as log_file:
        log_file.write(f"[{timestamp}] 10-second report\n")
        log_file.write(f"Window processed in {elapsed:.1f}s\n")
        log_file.write(f"{report}\n\n")


def build_detection_payload(frame_count, class_occurrences, max_objects_per_frame):
    return {
        "frames_analyzed": frame_count,
        "class_frame_occurrences": dict(class_occurrences),
        "max_objects_seen_in_one_frame": dict(max_objects_per_frame),
    }


def ask_llama(llm_model, payload):
    prompt = (
        "You are monitoring a camera using YOLO object detection. "
        "Report what was detected in the last time window. "
        "Be concise and mention object classes and useful counts. "
        "Only mention classes present in the detection data. "
        "Do not mention absent classes or say that specific classes were not detected. "
        "If nothing was detected, say that no objects were detected.\n\n"
        f"Detection data:\n{json.dumps(payload, indent=2, sort_keys=True)}"
    )
    data = {
        "model": llm_model,
        "prompt": prompt,
        "stream": False,
    }
    request = urllib.request.Request(
        OLLAMA_URL,
        data=json.dumps(data).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )

    try:
        with urllib.request.urlopen(request, timeout=60) as response:
            result = json.loads(response.read().decode("utf-8"))
            return result.get("response", "").strip()
    except urllib.error.URLError as exc:
        return f"Could not contact Ollama at {OLLAMA_URL}: {exc}"
    except json.JSONDecodeError as exc:
        return f"Could not parse Ollama response: {exc}"


def get_llama_report(future, started_at):
    elapsed = time.monotonic() - started_at
    try:
        report = future.result()
    except Exception as exc:
        report = f"Llama report failed: {exc}"

    timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
    report = sanitize_llama_report(report) or "No response from Llama."

    print(f"\n[{timestamp}] 10-second report")
    print(f"Window processed in {elapsed:.1f}s")
    print(report)
    print()
    append_log(timestamp, report, elapsed)
    return timestamp, report


def sanitize_llama_report(report):
    return "\n".join(
        line
        for line in report.splitlines()
        if "no objects were detected for" not in line.lower()
    ).strip()


def build_detected_view(result, annotated_frame):
    height, width = annotated_frame.shape[:2]
    if result.boxes is None or len(result.boxes) == 0:
        empty = np.zeros_like(annotated_frame)
        cv2.putText(
            empty,
            "No objects detected",
            (24, max(48, height // 2)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.9,
            (220, 220, 220),
            2,
            cv2.LINE_AA,
        )
        return empty

    boxes = result.boxes.xyxy.cpu().numpy()
    x1 = max(0, int(np.floor(boxes[:, 0].min())) - 24)
    y1 = max(0, int(np.floor(boxes[:, 1].min())) - 24)
    x2 = min(width, int(np.ceil(boxes[:, 2].max())) + 24)
    y2 = min(height, int(np.ceil(boxes[:, 3].max())) + 24)

    detected_view = annotated_frame[y1:y2, x1:x2]
    if detected_view.size == 0:
        return annotated_frame

    view_height, view_width = detected_view.shape[:2]
    scale = height / view_height
    resized_width = max(1, int(view_width * scale))
    return cv2.resize(detected_view, (resized_width, height))


def wrap_text(text, max_width, font, scale, thickness):
    lines = []
    for paragraph in text.splitlines() or [""]:
        words = paragraph.split()
        if not words:
            lines.append("")
            continue

        line = words[0]
        for word in words[1:]:
            candidate = f"{line} {word}"
            width = cv2.getTextSize(candidate, font, scale, thickness)[0][0]
            if width <= max_width:
                line = candidate
            else:
                lines.append(line)
                line = word
        lines.append(line)
    return lines


def build_display_frame(frame, report_text, report_timestamp, status_text, panel_width):
    height, width = frame.shape[:2]
    panel_width = max(280, panel_width)
    display = np.zeros((height, width + panel_width, 3), dtype=np.uint8)
    display[:, :panel_width] = (24, 24, 24)
    display[:, panel_width:] = frame

    cv2.line(display, (panel_width, 0), (panel_width, height), (80, 80, 80), 1)

    font = cv2.FONT_HERSHEY_SIMPLEX
    title = "10-second Llama report"
    if report_timestamp:
        title = f"{title} | {report_timestamp}"

    cv2.putText(
        display,
        title,
        (16, 32),
        font,
        0.62,
        (245, 245, 245),
        2,
        cv2.LINE_AA,
    )
    cv2.putText(
        display,
        status_text,
        (16, 64),
        font,
        0.55,
        (170, 210, 255),
        1,
        cv2.LINE_AA,
    )

    max_text_width = max(80, panel_width - 32)
    lines = wrap_text(report_text, max_text_width, font, 0.58, 1)
    y = 100
    line_height = 24
    max_y = height - 16
    for line in lines:
        if y > max_y:
            cv2.putText(
                display,
                "...",
                (16, max_y),
                font,
                0.58,
                (230, 230, 230),
                1,
                cv2.LINE_AA,
            )
            break
        cv2.putText(
            display,
            line,
            (16, y),
            font,
            0.58,
            (230, 230, 230),
            1,
            cv2.LINE_AA,
        )
        y += line_height

    return display


def main():
    args = parse_args()
    model = YOLO(args.model)

    cap = cv2.VideoCapture(args.url, cv2.CAP_FFMPEG)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open RTSP stream: {args.url}")

    print("Stream opened. Press 'q' or ESC to quit.")
    print(f"YOLO device: {args.device}")
    print(f"Llama model: {args.llm_model}")

    cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(WINDOW_NAME, args.window_width, args.window_height)

    frame_count = 0
    class_occurrences = Counter()
    max_objects_per_frame = defaultdict(int)
    last_report_at = time.monotonic()
    pending_report = None
    pending_started_at = None
    latest_report = "Waiting for the first 10-second detection report."
    latest_report_timestamp = None
    report_status = "Collecting detections..."

    with ThreadPoolExecutor(max_workers=1) as executor:
        while True:
            ok, frame = cap.read()
            if not ok:
                print("Failed to read frame from stream.")
                break

            results = model.predict(
                frame,
                conf=args.conf,
                device=args.device,
                verbose=False,
            )
            result = results[0]
            detected_classes = get_detected_classes(result)

            frame_count += 1
            frame_counts = Counter(detected_classes)
            class_occurrences.update(frame_counts.keys())
            for class_name, count in frame_counts.items():
                max_objects_per_frame[class_name] = max(
                    max_objects_per_frame[class_name],
                    count,
                )

            now = time.monotonic()
            if pending_report is not None and pending_report.done():
                latest_report_timestamp, latest_report = get_llama_report(
                    pending_report,
                    pending_started_at,
                )
                pending_report = None
                pending_started_at = None
                report_status = "Collecting detections..."

            if now - last_report_at >= args.interval:
                payload = build_detection_payload(
                    frame_count,
                    class_occurrences,
                    max_objects_per_frame,
                )

                if pending_report is None:
                    pending_started_at = time.monotonic()
                    pending_report = executor.submit(
                        ask_llama,
                        args.llm_model,
                        payload,
                    )
                    report_status = "Llama is generating a new report..."
                else:
                    print("Skipping report because the previous Llama request is still running.")
                    report_status = "Previous Llama request is still running."

                frame_count = 0
                class_occurrences.clear()
                max_objects_per_frame.clear()
                last_report_at = now

            annotated_frame = result.plot()
            detected_view = build_detected_view(result, annotated_frame)
            display_frame = build_display_frame(
                detected_view,
                latest_report,
                latest_report_timestamp,
                report_status,
                args.report_panel_width,
            )
            cv2.imshow(WINDOW_NAME, display_frame)

            key = cv2.waitKey(1) & 0xFF
            if key == ord("q") or key == 27:
                break

        if pending_report is not None:
            get_llama_report(pending_report, pending_started_at)

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
