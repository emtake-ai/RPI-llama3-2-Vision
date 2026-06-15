#!/usr/bin/env python3
# sllm.py

import os
os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ["QT_QPA_FONTDIR"] = "/usr/share/fonts/truetype/dejavu"
os.environ["QT_LOGGING_RULES"] = "*.debug=false;qt.qpa.*=false"

import cv2
import time
import json
import platform
import re
import threading
import requests
import numpy as np
from collections import Counter
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse


BASE_DIR = Path(__file__).resolve().parent
YOLO_PT_MODEL_PATH = BASE_DIR / "best.pt"
YOLO_FALLBACK_PT_MODEL_PATH = BASE_DIR / "yolo11n.pt"
YOLO_NCNN_MODEL_PATH = BASE_DIR / "best_ncnn_model"
CAMERA_ID = os.environ.get("SLLM_CAMERA_ID", "/dev/video0")

OLLAMA_BASE_URL = os.environ.get("SLLM_OLLAMA_BASE_URL", "http://localhost:11434").rstrip("/")
OLLAMA_GENERATE_URL = os.environ.get("SLLM_OLLAMA_GENERATE_URL", f"{OLLAMA_BASE_URL}/api/generate")
OLLAMA_EMBED_URL = os.environ.get("SLLM_OLLAMA_EMBED_URL", f"{OLLAMA_BASE_URL}/api/embeddings")

LLM_MODEL = os.environ.get("SLLM_LLM_MODEL", "office-monitor:latest")
EMBED_MODEL = os.environ.get("SLLM_EMBED_MODEL", "nomic-embed-text:latest")

RAG_DB_PATH = os.environ.get("SLLM_RAG_DB_PATH", "/home/soo/Work/Sensor_sLLM/Office/rag_db.json")

HTTP_HOST = os.environ.get("SLLM_HTTP_HOST", "0.0.0.0")
HTTP_PORT = int(os.environ.get("SLLM_HTTP_PORT", "18080"))
DETECTION_INTERVAL_SECONDS = float(os.environ.get("SLLM_DETECTION_INTERVAL_SECONDS", "1.0"))
YOLO_IMGSZ = int(os.environ.get("SLLM_YOLO_IMGSZ", "320"))
IS_ARM = platform.machine().lower() in ["aarch64", "arm64", "armv7l", "armv6l"]
DEFAULT_ENABLE_YOLO = "1" if (not IS_ARM or YOLO_NCNN_MODEL_PATH.exists()) else "0"
ENABLE_YOLO = os.environ.get("SLLM_ENABLE_YOLO", DEFAULT_ENABLE_YOLO).lower() not in ["0", "false", "no"]


latest_detected_text = "none"
latest_image_png = None
latest_image_timestamp = None

latest_lock = threading.Lock()

running = True
ready_event = threading.Event()


def format_detected_objects(objects):
    if not objects:
        return "none"

    counts = Counter(objects)
    return ", ".join(
        name if count == 1 else f"{name} x{count}"
        for name, count in counts.items()
    )


def select_yolo_model_path():
    env_model_path = os.environ.get("SLLM_YOLO_MODEL")
    if env_model_path:
        return Path(env_model_path)

    if IS_ARM and YOLO_NCNN_MODEL_PATH.exists():
        return YOLO_NCNN_MODEL_PATH

    if YOLO_PT_MODEL_PATH.exists():
        return YOLO_PT_MODEL_PATH

    if YOLO_FALLBACK_PT_MODEL_PATH.exists():
        return YOLO_FALLBACK_PT_MODEL_PATH

    return YOLO_PT_MODEL_PATH


def detection_loop():
    global latest_detected_text, latest_image_png, latest_image_timestamp, running

    model_path = select_yolo_model_path()
    use_ncnn = model_path.is_dir() and model_path.name.endswith("_ncnn_model")

    print(f"Loading YOLO model: {model_path}")
    from ultralytics import YOLO

    model = YOLO(str(model_path), task="detect")
    if not use_ncnn:
        model.to("cpu")

    print(f"Opening camera {CAMERA_ID}...")
    cap = cv2.VideoCapture(CAMERA_ID)

    if not cap.isOpened():
        print(f"Error: cannot open {CAMERA_ID}")
        running = False
        ready_event.set()
        return

    print("YOLO and camera are ready.")
    ready_event.set()

    while running:
        loop_started_at = time.monotonic()
        ret, frame = cap.read()

        if not ret:
            time.sleep(0.1)
            continue

        try:
            if use_ncnn:
                results = model(frame, verbose=False, imgsz=YOLO_IMGSZ)
            else:
                results = model(frame, verbose=False, device="cpu", imgsz=YOLO_IMGSZ)

            detected = []

            for result in results:
                if result.boxes is None:
                    continue

                for box in result.boxes:
                    cls_id = int(box.cls[0])
                    cls_name = model.names[cls_id]
                    detected.append(cls_name)

            detected_text = format_detected_objects(detected)
            annotated_frame = results[0].plot() if results else frame
            image_ok, image_buffer = cv2.imencode(".png", annotated_frame)

            with latest_lock:
                latest_detected_text = detected_text
                if image_ok:
                    latest_image_png = image_buffer.tobytes()
                    latest_image_timestamp = current_time_text()

        except Exception as e:
            print(f"YOLO detection error: {e}")
            time.sleep(0.5)
            continue

        elapsed = time.monotonic() - loop_started_at
        time.sleep(max(0.0, DETECTION_INTERVAL_SECONDS - elapsed))

    cap.release()


# =========================
# RAG Functions
# =========================

def embed_text(text):
    response = requests.post(
        OLLAMA_EMBED_URL,
        json={
            "model": EMBED_MODEL,
            "prompt": text,
        },
        timeout=120,
    )
    response.raise_for_status()
    return np.array(response.json()["embedding"], dtype=np.float32)


def cosine_similarity(a, b):
    return float(
        np.dot(a, b) /
        (np.linalg.norm(a) * np.linalg.norm(b) + 1e-8)
    )


def retrieve_rag_context(question, detected_text, top_k=3):
    if not os.path.exists(RAG_DB_PATH):
        return ""

    query = f"""
Detected objects: {detected_text}
Question: {question}
"""

    query_vec = embed_text(query)

    with open(RAG_DB_PATH, "r", encoding="utf-8") as f:
        db = json.load(f)

    scored = []

    for item in db:
        vec = np.array(item["embedding"], dtype=np.float32)
        score = cosine_similarity(query_vec, vec)
        scored.append((score, item["chunk"]))

    scored.sort(reverse=True, key=lambda x: x[0])

    contexts = [chunk for score, chunk in scored[:top_k]]

    return "\n\n".join(contexts)


# =========================
# Local Answer Fallback
# =========================

def local_answer_for_none():
    return """Detected Objects:
none

Situational Context:
No objects are currently detected.

Person Awareness:
No person is currently detected.

Person Possible Action:
No person action can be inferred.

Recommended Application:
Continue monitoring the office camera."""


def local_answer_for_known_objects(detected_text):
    object_list = parse_detected_object_names(detected_text)

    person_detected = "person" in object_list

    if person_detected:
        person_awareness = "A person is currently detected."
    else:
        person_awareness = "No person is currently detected."

    if "boxes" in detected_text:
        context = "This appears to be an office storage or equipment area."
        action = "A person may be organizing or checking stored items."
        app = "Storage monitoring or office inventory assistance."

    elif any(x in detected_text for x in ["desk", "monitor", "laptop", "mic", "speaker", "camera"]):
        context = "This appears to be Kevin's office desk or workstation."
        action = "A person may be working, joining a meeting, or testing devices."
        app = "Office monitoring, meeting assistant, or workspace status report."

    elif any(x in detected_text for x in ["soldering iron", "tools"]):
        context = "This appears to be an electronics development area."
        action = "A person may be soldering, repairing, or debugging hardware."
        app = "Engineering assistant and safety monitoring."

    elif any(x in detected_text for x in ["mom-i", "baby bed", "baby doll"]):
        context = "This appears to be a baby monitoring or device test area."
        action = "A person may be testing Mom-I or monitoring baby care equipment."
        app = "Baby monitoring and device test assistance."

    else:
        context = "This appears to be a general office environment."
        action = "A person may be working or interacting with office equipment."
        app = "General office assistance."

    if not person_detected:
        action = "No direct person action can be confirmed, but this area may be used for " + action[17:].lower()

    return f"""Detected Objects:
{detected_text}

Situational Context:
{context}

Person Awareness:
{person_awareness}

Person Possible Action:
{action}

Recommended Application:
{app}"""


def parse_detected_object_names(detected_text):
    if detected_text == "none":
        return []

    return [
        item.strip().split(" x")[0]
        for item in detected_text.split(",")
        if item.strip()
    ]


def parse_detected_object_counts(detected_text):
    counts = Counter()
    if detected_text == "none":
        return counts

    for item in detected_text.split(","):
        item = item.strip()
        if not item:
            continue

        match = re.fullmatch(r"(.+?)\s+x(\d+)", item)
        if match:
            counts[match.group(1).strip()] += int(match.group(2))
        else:
            counts[item] += 1

    return counts


def build_scene_facts(detected_text):
    objects = parse_detected_object_names(detected_text)

    object_set = set(objects)
    person_detected = "person" in object_set
    has_workstation = bool(object_set & {"desk", "monitor", "laptop", "mic", "speaker", "camera"})
    has_storage = "boxes" in object_set
    has_electronics = bool(object_set & {"soldering iron", "tools"})
    has_baby_area = bool(object_set & {"mom-i", "baby bed", "baby doll"})

    if has_baby_area:
        context = "This appears to be a baby monitoring or device test area."
        action_when_person = "A person may be testing Mom-I or checking baby monitoring equipment."
        app = "Baby monitoring and device test assistance."
    elif has_electronics:
        context = "This appears to be an electronics development area."
        action_when_person = "A person may be soldering, repairing, or debugging hardware."
        app = "Engineering assistant and safety monitoring."
    elif has_workstation and has_storage:
        context = "This appears to be an office workstation with nearby stored items."
        action_when_person = "A person may be working at the workstation or checking stored items."
        app = "Office monitoring and inventory assistance."
    elif has_workstation:
        context = "This appears to be Kevin's office desk or workstation."
        action_when_person = "A person may be working, joining a meeting, or testing devices."
        app = "Office monitoring, meeting assistant, or workspace status report."
    elif has_storage:
        context = "This appears to be an office storage or equipment area."
        action_when_person = "A person may be organizing or checking stored items."
        app = "Storage monitoring or office inventory assistance."
    else:
        context = "This appears to be a general office environment."
        action_when_person = "A person may be working or interacting with office equipment."
        app = "General office assistance."

    if person_detected:
        person_awareness = "A person is currently detected."
        action = action_when_person
    else:
        person_awareness = "No person is currently detected."
        action = "No direct person action can be confirmed from the current detection."

    return {
        "objects": objects,
        "counts": parse_detected_object_counts(detected_text),
        "person_detected": person_detected,
        "context": context,
        "person_awareness": person_awareness,
        "action": action,
        "app": app,
    }


def pluralize_object(name, count):
    if count == 1:
        return name
    if name.endswith("s"):
        return name
    return f"{name}s"


def format_count(name, count):
    return f"{count} {pluralize_object(name, count)}"


def format_object_reference(name):
    if name.endswith("s"):
        return name
    article = "an" if name[:1].lower() in {"a", "e", "i", "o", "u"} else "a"
    return f"{article} {name}"


OBJECT_ALIASES = {
    "box": "boxes",
    "boxes": "boxes",
    "package": "boxes",
    "packages": "boxes",
    "laptop": "laptop",
    "laptops": "laptop",
    "notebook": "laptop",
    "notebooks": "laptop",
    "pc": "laptop",
    "computer": "laptop",
    "computers": "laptop",
    "monitor": "monitor",
    "monitors": "monitor",
    "display": "monitor",
    "screen": "monitor",
    "screens": "monitor",
    "person": "person",
    "people": "person",
    "human": "person",
    "someone": "person",
    "chair": "chair",
    "chairs": "chair",
    "camera": "camera",
    "cameras": "camera",
    "mic": "mic",
    "microphone": "mic",
    "microphones": "mic",
    "speaker": "speaker",
    "speakers": "speaker",
    "tool": "tools",
    "tools": "tools",
    "soldering": "soldering iron",
    "soldering iron": "soldering iron",
    "baby bed": "baby bed",
    "baby doll": "baby doll",
    "mom-i": "mom-i",
}


def object_names_from_question(question):
    question_lower = question.lower()
    names = []

    for phrase, object_name in sorted(OBJECT_ALIASES.items(), key=lambda item: len(item[0]), reverse=True):
        if re.search(rf"\b{re.escape(phrase)}\b", question_lower) and object_name not in names:
            names.append(object_name)

    return names


def answer_count_question(detected_text, question, facts):
    question_lower = question.lower()
    counts = facts["counts"]

    pc_count = counts.get("laptop", 0)
    monitor_count = counts.get("monitor", 0)

    if any(term in question_lower for term in ["pc", "computer", "computers"]):
        if pc_count and monitor_count:
            return f"I can see {format_count('laptop/PC', pc_count)} and {format_count('monitor', monitor_count)}. I count the laptop as the PC; the monitor is a display."
        if pc_count:
            return f"I can see {format_count('laptop/PC', pc_count)}."
        if monitor_count:
            return f"I can see {format_count('monitor', monitor_count)}, but I do not detect a laptop or PC tower."
        return f"I do not detect a PC right now. I detect: {detected_text}."

    for object_name in object_names_from_question(question):
        count = counts.get(object_name, 0)
        return f"I can see {format_count(object_name, count)}."

    total = sum(counts.values())
    return f"I can see {total} detected object{'s' if total != 1 else ''}: {detected_text}."


def answer_presence_question(detected_text, question, facts):
    object_names = object_names_from_question(question)

    if not object_names:
        return ""

    counts = facts["counts"]
    present = [name for name in object_names if counts.get(name, 0) > 0]
    missing = [name for name in object_names if counts.get(name, 0) == 0]

    if present and not missing:
        details = ", ".join(format_count(name, counts[name]) for name in present)
        return f"Yes. I can see {details}."

    if present and missing:
        present_details = ", ".join(format_count(name, counts[name]) for name in present)
        missing_details = ", ".join(format_object_reference(name) for name in missing)
        return f"I can see {present_details}, but I do not detect {missing_details}."

    missing_details = ", ".join(format_object_reference(name) for name in missing)
    return f"No. I do not detect {missing_details} right now. Current detections: {detected_text}."


def answer_status_question(detected_text, facts):
    return (
        f"Current status: I detect {detected_text}. "
        f"{facts['person_awareness']} {facts['context']}"
    )


def answer_context_question(detected_text, facts):
    return (
        f"{facts['context']} "
        f"I base that on the current detections: {detected_text}."
    )


def answer_detection_question(detected_text, facts):
    total = sum(facts["counts"].values())
    if total == 0:
        return "I cannot see any detected objects right now."

    return (
        f"I can see {format_detected_objects_from_counts(facts['counts'])}. "
        f"That is {total} detected object{'s' if total != 1 else ''}."
    )


def format_detected_objects_from_counts(counts):
    if not counts:
        return "no objects"

    return ", ".join(format_count(name, count) for name, count in counts.items())


def answer_help_question(detected_text):
    return (
        "You can ask what I see, whether a person is present, how many objects are visible, "
        f"what action may be happening, or what application fits the scene. Current detections: {detected_text}."
    )


def answer_greeting(detected_text, facts):
    return (
        f"Hello. I am monitoring Kevin's office. Right now I detect {detected_text}. "
        f"{facts['person_awareness']}"
    )


def fallback_question_answer(detected_text, question):
    facts = build_scene_facts(detected_text)
    objects = facts["objects"]
    question_type = classify_question(question)

    if question_type["greeting"]:
        return answer_greeting(detected_text, facts)

    if question_type["help"]:
        return answer_help_question(detected_text)

    if not objects and not (
        question_type["status"]
        or question_type["presence"]
        or question_type["count"]
        or question_type["context"]
        or question_type["detection"]
        or question_type["person"]
    ):
        return "I do not detect any objects right now, and no person is currently detected."

    if question_type["count"]:
        return answer_count_question(detected_text, question, facts)

    if question_type["presence"]:
        presence_answer = answer_presence_question(detected_text, question, facts)
        if presence_answer:
            return presence_answer

    if question_type["action"] and facts["person_detected"]:
        return f"{facts['action']} This is inferred from the detected objects: {detected_text}."

    if question_type["action"]:
        return f"I cannot confirm a person's action because no person is currently detected. I detect: {detected_text}."

    if question_type["person"]:
        if facts["person_detected"]:
            return f"Yes, a person is currently detected. I also detect: {detected_text}."
        return f"No, I do not detect a person right now. I detect: {detected_text}."

    if question_type["status"]:
        return answer_status_question(detected_text, facts)

    if question_type["context"]:
        return answer_context_question(detected_text, facts)

    if question_type["detection"]:
        return answer_detection_question(detected_text, facts)

    if question_type["app"]:
        return f"Recommended application: {facts['app']}"

    return f"I see {detected_text}. {facts['context']} {facts['person_awareness']}"


def classify_question(question):
    question_lower = question.lower()
    question_tokens = set(re.findall(r"\b[\w-]+\b", question_lower))

    greeting_question = bool(question_tokens & {"hi", "hello", "hey"})
    help_question = any(term in question_lower for term in [
        "help", "what can you do", "how can i use", "available questions"
    ])
    status_question = any(term in question_lower for term in [
        "status", "update", "summary", "report", "current situation", "what is happening"
    ])
    context_question = any(term in question_lower for term in [
        "where", "where is here", "what place", "what kind of room", "what area",
        "scene", "environment", "location", "which place"
    ])
    count_question = any(term in question_lower for term in [
        "how many", "number of", "count", "total objects", "object count"
    ])
    detection_question = any(term in question_lower for term in [
        "detection", "detections", "detect", "detected", "what are",
        "what do you see", "what did you see", "what can you see",
        "what is your detection", "which objects", "what objects",
        "which things", "what things", "can you see anything",
        "can you see something", "do you see anything", "do you see something"
    ])
    person_question = any(term in question_lower for term in [
        "someone", "person", "human", "people", "anyone", "there someone", "is there"
    ])
    action_question = (
        bool(question_tokens & {"doing", "action", "activity", "work", "working"})
        or "what does he do" in question_lower
        or "what is he doing" in question_lower
        or "what does she do" in question_lower
        or "what is she doing" in question_lower
        or "what do they do" in question_lower
        or "what are they doing" in question_lower
    )
    app_question = any(term in question_lower for term in [
        "application", "app", "recommend", "use case", "assist"
    ])
    presence_question = (
        bool(object_names_from_question(question))
        and any(term in question_lower for term in [
            "do you see", "can you see", "is there", "are there", "visible", "present", "detect"
        ])
    )

    return {
        "greeting": greeting_question,
        "help": help_question,
        "status": status_question,
        "context": context_question,
        "count": count_question,
        "detection": detection_question,
        "person": person_question,
        "action": action_question,
        "app": app_question,
        "presence": presence_question,
    }


def has_repetitive_output(text):
    words = re.findall(r"\b[\w-]+\b", text.lower())
    if not words:
        return True

    repeated_word_count = 1
    previous_word = None
    for word in words:
        if word == previous_word:
            repeated_word_count += 1
            if repeated_word_count >= 4:
                return True
        else:
            repeated_word_count = 1
            previous_word = word

    for phrase_len in (2, 3):
        phrases = [
            tuple(words[index:index + phrase_len])
            for index in range(len(words) - phrase_len + 1)
        ]
        if any(count >= 5 for count in Counter(phrases).values()):
            return True

    lines = [line.strip().lower() for line in text.splitlines() if line.strip()]
    return any(count >= 3 for count in Counter(lines).values())


def answer_violates_person_facts(text, person_detected):
    text_lower = text.lower()
    if any(label in text_lower for label in [
        "person awareness:",
        "recommended application:",
        "situational context:",
        "person possible action:",
        "detected objects:",
    ]):
        return True

    if person_detected:
        return "no person" in text_lower or "do not detect a person" in text_lower

    unsupported_person_claims = [
        "a person is currently detected",
        "person is currently detected",
        "someone is currently detected",
        "yes, a person",
        "a person may be",
        "person may be",
        "person interaction",
        "he is ",
        "he may ",
        "she is ",
        "she may ",
    ]
    return any(claim in text_lower for claim in unsupported_person_claims)


def build_model_prompt(detected_text, question, facts):
    person_detected_text = "yes" if facts["person_detected"] else "no"
    return f"""You are office-monitor, a camera assistant for Kevin's office.

Answer the user's question directly in one or two short sentences.

Facts from YOLO detection:
- Detected objects: {detected_text}
- Person detected: {person_detected_text}
- Scene context: {facts["context"]}
- Possible person action: {facts["action"]}
- Recommended application: {facts["app"]}

Rules:
- Use only the facts above.
- If person detected is no, do not describe what a person/he/she is doing.
- If person detected is yes, action must be described as possible, not certain.
- Do not repeat words, headings, or sections.
- Do not use labels like Situational Context unless the user asks for a detection report.

User question: {question}
Answer:"""


def generate_question_answer(detected_text, question):
    facts = build_scene_facts(detected_text)
    prompt = build_model_prompt(detected_text, question, facts)

    payload = {
        "model": LLM_MODEL,
        "prompt": prompt,
        "stream": False,
        "options": {
            "temperature": 0.1,
            "top_p": 0.7,
            "repeat_penalty": 1.35,
            "num_predict": 70,
            "stop": ["\nQuestion:", "\nUser question:"],
        },
    }

    response = requests.post(
        OLLAMA_GENERATE_URL,
        json=payload,
        timeout=45,
    )
    response.raise_for_status()
    answer = response.json().get("response", "").strip()
    answer = re.sub(r"^(answer:\s*)+", "", answer, flags=re.IGNORECASE).strip()

    if (
        not answer
        or len(answer) > 500
        or has_repetitive_output(answer)
        or answer_violates_person_facts(answer, facts["person_detected"])
    ):
        return ""

    return answer


# =========================
# Office Monitor Answering
# =========================

def ask_office_monitor(detected_text, question):
    question_type = classify_question(question)
    if (
        question_type["greeting"]
        or question_type["help"]
        or question_type["status"]
        or question_type["context"]
        or question_type["count"]
        or question_type["action"]
        or question_type["detection"]
        or question_type["person"]
        or question_type["presence"]
    ):
        return fallback_question_answer(detected_text, question)

    try:
        answer = generate_question_answer(detected_text, question)
        if answer:
            return answer
    except Exception:
        pass

    return fallback_question_answer(detected_text, question)


def get_latest_detected_text():
    with latest_lock:
        return latest_detected_text


def get_latest_image():
    with latest_lock:
        return latest_image_png, latest_image_timestamp


def save_latest_image(output_path="image.png"):
    image_png, _ = get_latest_image()
    if not image_png:
        return "No image is available yet. Wait for YOLO detection to read a frame."

    output_file = Path(output_path)
    try:
        output_file.write_bytes(image_png)
    except OSError as exc:
        return f"Could not save image to {output_file}: {exc}"

    return f"Saved current YOLO image to {output_file}"


def is_image_request(text):
    text = text.lower().strip()

    if text in {"/image", "/picture", "/photo", "/snapshot", "/download-image"}:
        return True

    image_terms = ("image", "picture", "photo", "snapshot")
    action_terms = ("get", "download", "save", "send", "transfer", "receive")
    current_terms = ("current", "latest", "now", "last")

    return (
        any(image_term in text for image_term in image_terms)
        and any(action_term in text for action_term in action_terms)
        and any(current_term in text for current_term in current_terms)
    )


def current_time_text():
    return time.strftime("%Y-%m-%d %H:%M:%S")


def format_console_reply(question, detected_text, answer):
    return f"""
============================================================
Time: {current_time_text()}
Question: {question}
Detected objects: {detected_text}
============================================================

Reply from office-monitor:
{answer}
""".strip() + "\n"


def format_status_update(detected_text):
    facts = build_scene_facts(detected_text)

    return f"""
============================================================
Office Monitor Update
Time: {current_time_text()}
Detected objects: {detected_text}
============================================================

Finding:
{facts["person_awareness"]}

Situation:
{facts["context"]}

Possible Action:
{facts["action"]}

Recommended Application:
{facts["app"]}
""".strip() + "\n"


def build_status_payload(detected_text):
    facts = build_scene_facts(detected_text)
    timestamp = current_time_text()
    return {
        "timestamp": timestamp,
        "detected_objects": detected_text,
        "person_detected": facts["person_detected"],
        "finding": facts["person_awareness"],
        "situation": facts["context"],
        "possible_action": facts["action"],
        "recommended_application": facts["app"],
        "text": format_status_update(detected_text),
    }


class OfficeMonitorHTTPRequestHandler(BaseHTTPRequestHandler):
    server_version = "OfficeMonitorHTTP/1.0"

    def do_GET(self):
        parsed_url = urlparse(self.path)

        if parsed_url.path in ["/", "/health"]:
            self.send_text(
                "office-monitor is running\n"
                "Use GET /ask?question=your%20question\n"
                "Use GET /status for the latest finding and situation\n"
                "Use GET /image for the latest YOLO annotated PNG image\n"
            )
            return

        if parsed_url.path == "/status":
            detected_text = get_latest_detected_text()
            query = parse_qs(parsed_url.query)
            if query.get("format", [""])[0].lower() == "json":
                self.send_json(build_status_payload(detected_text))
            else:
                self.send_text(format_status_update(detected_text))
            return

        if parsed_url.path == "/ask":
            query = parse_qs(parsed_url.query)
            question = query.get("question", query.get("q", [""]))[0].strip()
            self.handle_ask(question)
            return

        if parsed_url.path == "/image":
            self.handle_image()
            return

        self.send_error(404, "Use /ask?question=..., /status, or /image")

    def do_POST(self):
        parsed_url = urlparse(self.path)

        if parsed_url.path == "/status":
            detected_text = get_latest_detected_text()
            self.send_text(format_status_update(detected_text))
            return

        if parsed_url.path != "/ask":
            self.send_error(404, "Use POST /ask")
            return

        content_length = int(self.headers.get("Content-Length", "0"))
        body = self.rfile.read(content_length).decode("utf-8", errors="replace")
        content_type = self.headers.get("Content-Type", "")

        if "application/json" in content_type:
            try:
                payload = json.loads(body) if body else {}
            except json.JSONDecodeError:
                self.send_error(400, "Invalid JSON body")
                return

            question = str(payload.get("question", payload.get("q", ""))).strip()
        else:
            payload = parse_qs(body)
            question = payload.get("question", payload.get("q", [""]))[0].strip()

        self.handle_ask(question)

    def handle_ask(self, question):
        if not question:
            self.send_error(400, "Missing question. Use /ask?question=...")
            return

        detected_text = get_latest_detected_text()
        answer = ask_office_monitor(detected_text, question)
        body = format_console_reply(question, detected_text, answer)
        self.send_text(body)

    def handle_image(self):
        image_png, image_timestamp = get_latest_image()
        if not image_png:
            self.send_error(503, "No image is available yet. Wait for YOLO detection to read a frame.")
            return

        filename = "image.png"
        self.send_bytes(
            image_png,
            content_type="image/png",
            extra_headers={
                "Content-Disposition": f'attachment; filename="{filename}"',
                "X-Image-Timestamp": image_timestamp or "",
            },
        )

    def send_text(self, body, status=200):
        encoded_body = body.encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "text/plain; charset=utf-8")
        self.send_header("Content-Length", str(len(encoded_body)))
        self.send_header("Access-Control-Allow-Origin", "*")
        self.end_headers()
        self.wfile.write(encoded_body)

    def send_json(self, payload, status=200):
        encoded_body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(encoded_body)))
        self.send_header("Access-Control-Allow-Origin", "*")
        self.end_headers()
        self.wfile.write(encoded_body)

    def send_bytes(self, body, content_type, status=200, extra_headers=None):
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Access-Control-Allow-Origin", "*")
        if extra_headers:
            for name, value in extra_headers.items():
                self.send_header(name, value)
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, format, *args):
        print(f"HTTP {self.client_address[0]} - {format % args}")


def http_server_loop():
    global running

    try:
        server = ThreadingHTTPServer((HTTP_HOST, HTTP_PORT), OfficeMonitorHTTPRequestHandler)
    except OSError as e:
        print(f"HTTP server error: cannot bind {HTTP_HOST}:{HTTP_PORT}: {e}")
        return

    server.timeout = 0.5
    print(f"HTTP server is running on http://{HTTP_HOST}:{HTTP_PORT}")
    print(f"Ask endpoint: http://<this-pc-ip>:{HTTP_PORT}/ask?question=what%20do%20you%20see")
    print(f"Status endpoint: http://<this-pc-ip>:{HTTP_PORT}/status")
    print(f"Image endpoint: http://<this-pc-ip>:{HTTP_PORT}/image")

    while running:
        server.handle_request()

    server.server_close()


def main():
    global running

    print("Office Monitor SLLM starting...")
    print("=" * 60)

    if ENABLE_YOLO:
        det_thread = threading.Thread(target=detection_loop, daemon=True)
        det_thread.start()

        ready_event.wait()

        if not running:
            print("Stopped because camera or YOLO failed.")
            return
    else:
        if IS_ARM and "SLLM_ENABLE_YOLO" not in os.environ:
            print("YOLO detection is disabled automatically on ARM/RPi because best_ncnn_model was not found.")
            print("Copy best_ncnn_model next to sllm.py or set SLLM_YOLO_MODEL=/path/to/best_ncnn_model.")
        else:
            print("YOLO detection is disabled by SLLM_ENABLE_YOLO=0.")
        ready_event.set()

    http_thread = threading.Thread(target=http_server_loop, daemon=True)
    http_thread.start()

    print("Office Monitor SLLM started.")
    if ENABLE_YOLO:
        print("YOLO detection is running headless.")
        print(f"YOLO model path: {select_yolo_model_path()}")
        print(f"YOLO image size: {YOLO_IMGSZ}")
    else:
        print("YOLO detection is disabled; detected objects will stay as none.")
    print(f"HTTP GET access is enabled on port {HTTP_PORT}.")
    print("Stable local responses are enabled.")
    print("Type q, quit, or exit in console to stop.")
    print("=" * 60)

    try:
        while running:
            question = input("\nQuestion: ").strip()

            if question.lower() in ["q", "quit", "exit"]:
                break

            if not question:
                continue

            if is_image_request(question):
                print(save_latest_image("image.png"))
                continue

            with latest_lock:
                detected_text = latest_detected_text

            print("\n" + "=" * 60)
            print(f"Question: {question}")
            print(f"Detected objects: {detected_text}")
            print("=" * 60)

            answer = ask_office_monitor(detected_text, question)

            print("\nReply from office-monitor:")
            print(answer)

    except KeyboardInterrupt:
        pass

    finally:
        running = False
        time.sleep(0.2)
        print("\nStopped.")


if __name__ == "__main__":
    main()
