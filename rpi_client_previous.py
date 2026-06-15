#!/usr/bin/env python3
"""Interactive HTTP client for sllm.py running on a Raspberry Pi."""

import argparse
import json
import sys
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


def ask_question(server, question, timeout):
    return request_text(
        f"{server}/ask",
        timeout,
        method="POST",
        payload={"question": question},
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


def run_interactive(server, timeout):
    print(f"Connected target: {server}")
    print("Type a question. Commands: /status, /health, /quit")

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
            else:
                print(ask_question(server, question, timeout).strip())
        except RuntimeError as exc:
            print(exc, file=sys.stderr)


def main():
    args = parse_args()

    try:
        server = normalize_server_url(args.server)
    except ValueError as exc:
        print(exc, file=sys.stderr)
        return 2

    try:
        if args.health:
            print(request_text(f"{server}/health", args.timeout).strip())
            return 0

        if args.status:
            print(get_status(server, args.timeout))
            return 0

        if args.once:
            print(ask_question(server, args.once, args.timeout).strip())
            return 0

        return run_interactive(server, args.timeout)
    except RuntimeError as exc:
        print(exc, file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
