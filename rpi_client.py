#!/usr/bin/env python3
"""Interactive HTTP client for sllm.py running on a Raspberry Pi."""

import argparse
import json
import subprocess
import sys
from pathlib import Path
import urllib.error
import urllib.parse
import urllib.request


DEFAULT_SERVER = "http://127.0.0.1:18080"


def parse_args():
    parser = argparse.ArgumentParser(
        description="Talk to the Office Monitor SLLM HTTP server."
    )
    parser.add_argument(
        "--server",
        default=DEFAULT_SERVER,
        help=f"sllm.py server URL. Default: {DEFAULT_SERVER}",
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=60.0,
        help="HTTP timeout in seconds. Default: 60",
    )
    parser.add_argument(
        "--once",
        help="Ask one question and exit.",
    )
    parser.add_argument(
        "--status",
        action="store_true",
        help="Print the latest detection status and exit.",
    )
    parser.add_argument(
        "--health",
        action="store_true",
        help="Check whether the SLLM server is reachable and exit.",
    )
    parser.add_argument(
        "--image",
        action="store_true",
        help="Download the latest YOLO annotated image and exit.",
    )
    parser.add_argument(
        "--image-output",
        default="image.png",
        help="Image filename for downloads. Default: image.png",
    )
    parser.add_argument(
        "--no-image-display",
        action="store_true",
        help="Save image downloads without opening a display window.",
    )
    parser.add_argument(
        "--display-image-file",
        help=argparse.SUPPRESS,
    )
    return parser.parse_args()


def normalize_server_url(server):
    server = server.strip()
    if not server:
        raise ValueError("server URL is empty")

    if "://" not in server:
        server = f"http://{server}"

    return server.rstrip("/")


def request_text(url, timeout, method="GET", payload=None):
    data = None
    headers = {}

    if payload is not None:
        data = json.dumps(payload).encode("utf-8")
        headers["Content-Type"] = "application/json"

    request = urllib.request.Request(
        url,
        data=data,
        headers=headers,
        method=method,
    )

    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            return response.read().decode("utf-8", errors="replace")
    except urllib.error.HTTPError as exc:
        body = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"HTTP {exc.code} from {url}\n{body}") from exc
    except urllib.error.URLError as exc:
        raise RuntimeError(f"Could not connect to {url}: {exc.reason}") from exc


def request_binary(url, timeout):
    request = urllib.request.Request(url, method="GET")

    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            return response.read()
    except urllib.error.HTTPError as exc:
        body = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"HTTP {exc.code} from {url}\n{body}") from exc
    except urllib.error.URLError as exc:
        raise RuntimeError(f"Could not connect to {url}: {exc.reason}") from exc


def ask_question(server, question, timeout):
    return request_text(
        f"{server}/ask",
        timeout,
        method="POST",
        payload={"question": question},
    )


def display_image_window(image_path):
    try:
        import tkinter as tk
    except ImportError as exc:
        raise RuntimeError("Could not open image window: tkinter is not installed") from exc

    root = tk.Tk()
    root.title(f"Current YOLO image - {image_path.name}")

    image = tk.PhotoImage(file=str(image_path))
    label = tk.Label(root, image=image)
    label.image = image
    label.pack()

    root.mainloop()


def open_image_window(image_path):
    image_path = Path(image_path).resolve()
    try:
        subprocess.Popen(
            [
                sys.executable,
                str(Path(__file__).resolve()),
                "--display-image-file",
                str(image_path),
            ],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
    except OSError as exc:
        raise RuntimeError(f"Could not open image window for {image_path}: {exc}") from exc


def save_current_image(server, timeout, output_path, display=True):
    image_data = request_binary(f"{server}/image", timeout)
    output_file = Path(output_path)
    try:
        output_file.write_bytes(image_data)
    except OSError as exc:
        raise RuntimeError(f"Could not save image to {output_file}: {exc}") from exc

    if display:
        open_image_window(output_file)
        return f"Saved current YOLO image to {output_file} and opened a display window"

    return f"Saved current YOLO image to {output_file}"


def is_image_request(text):
    text = text.lower().strip()

    if text in {"/image", "/picture", "/photo", "/snapshot", "/download-image"}:
        return True

    image_terms = ("image", "picture", "photo", "snapshot")
    action_terms = (
        "get",
        "download",
        "save",
        "send",
        "transfer",
        "receive",
        "show",
        "display",
        "open",
        "view",
        "see",
    )
    current_terms = ("current", "latest", "now", "last")

    has_image_term = any(image_term in text for image_term in image_terms)
    has_action_term = any(action_term in text for action_term in action_terms)
    has_current_term = any(current_term in text for current_term in current_terms)

    return has_image_term and has_action_term and (
        has_current_term
        or text.startswith(("can i ", "could i ", "please ", "show me ", "send me "))
    )


def get_status(server, timeout):
    status_url = f"{server}/status?format=json"
    text = request_text(status_url, timeout)
    try:
        status = json.loads(text)
    except json.JSONDecodeError:
        return text

    return "\n".join(
        [
            f"Time: {status.get('timestamp', 'unknown')}",
            f"Detected objects: {status.get('detected_objects', 'unknown')}",
            f"Person detected: {status.get('person_detected', 'unknown')}",
            f"Situation: {status.get('situation', 'unknown')}",
            f"Possible action: {status.get('possible_action', 'unknown')}",
            f"Recommended application: {status.get('recommended_application', 'unknown')}",
        ]
    )


def run_interactive(server, timeout, image_output, display_image):
    print(f"Connected target: {server}")
    print("Type a question. Commands: /status, /health, /image, /quit")

    while True:
        try:
            question = input("\nYou: ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            return 0

        if not question:
            continue

        command = question.lower()
        if command in {"/q", "/quit", "/exit", "q", "quit", "exit"}:
            return 0

        try:
            if command == "/status":
                print(get_status(server, timeout))
            elif command == "/health":
                print(request_text(f"{server}/health", timeout).strip())
            elif is_image_request(question):
                print(save_current_image(server, timeout, image_output, display_image))
            else:
                print(ask_question(server, question, timeout).strip())
        except RuntimeError as exc:
            print(exc, file=sys.stderr)


def main():
    args = parse_args()

    if args.display_image_file:
        try:
            display_image_window(Path(args.display_image_file))
            return 0
        except RuntimeError as exc:
            print(exc, file=sys.stderr)
            return 1

    try:
        server = normalize_server_url(args.server)
    except ValueError as exc:
        print(exc, file=sys.stderr)
        return 2

    try:
        if args.health:
            print(request_text(f"{server}/health", args.timeout).strip())
            return 0

        if args.image:
            print(
                save_current_image(
                    server,
                    args.timeout,
                    args.image_output,
                    not args.no_image_display,
                )
            )
            return 0

        if args.status:
            print(get_status(server, args.timeout))
            return 0

        if args.once:
            if is_image_request(args.once):
                print(
                    save_current_image(
                        server,
                        args.timeout,
                        args.image_output,
                        not args.no_image_display,
                    )
                )
                return 0

            print(ask_question(server, args.once, args.timeout).strip())
            return 0

        return run_interactive(
            server,
            args.timeout,
            args.image_output,
            not args.no_image_display,
        )
    except RuntimeError as exc:
        print(exc, file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
