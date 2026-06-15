import cv2


RTSP_URL = "rtsp://admin:emtake145!@192.168.1.7:554/Streaming/Channels/101"
WINDOW_NAME = "RTSP Stream"


def main():
    cap = cv2.VideoCapture(RTSP_URL, cv2.CAP_FFMPEG)

    if not cap.isOpened():
        raise RuntimeError(f"Could not open RTSP stream: {RTSP_URL}")

    print("Stream opened. Press 'q' or ESC to quit.")

    while True:
        ok, frame = cap.read()
        if not ok:
            print("Failed to read frame from stream.")
            break

        cv2.imshow(WINDOW_NAME, frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord("q") or key == 27:
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
