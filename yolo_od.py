import argparse
from collections import Counter

import cv2
from ultralytics import YOLO

from test import RTSP_URL


WINDOW_NAME = "YOLOv11 Object Detection"


def parse_args():
    parser = argparse.ArgumentParser(
        description="Open an RTSP stream and run YOLOv11 object detection."
    )
    parser.add_argument(
        "--model",
        default="yolo11n.pt",
        help="YOLOv11 model path/name. Default: yolo11n.pt",
    )
    parser.add_argument(
        "--conf",
        type=float,
        default=0.25,
        help="Detection confidence threshold. Default: 0.25",
    )
    parser.add_argument(
        "--url",
        default=RTSP_URL,
        help="RTSP URL. Defaults to RTSP_URL from test.py.",
    )
    parser.add_argument(
        "--device",
        default="cpu",
        help="Inference device. Default: cpu",
    )
    return parser.parse_args()


def get_detected_classes(result):
    names = result.names
    classes = result.boxes.cls.tolist() if result.boxes is not None else []
    return [names[int(class_id)] for class_id in classes]


def main():
    args = parse_args()
    model = YOLO(args.model)

    cap = cv2.VideoCapture(args.url, cv2.CAP_FFMPEG)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open RTSP stream: {args.url}")

    print("Stream opened. Press 'q' or ESC to quit.")

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

        if detected_classes:
            counts = Counter(detected_classes)
            class_summary = ", ".join(
                f"{class_name}: {count}" for class_name, count in counts.items()
            )
            print(f"Detected classes: {class_summary}")

        annotated_frame = result.plot()
        cv2.imshow(WINDOW_NAME, annotated_frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord("q") or key == 27:
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
