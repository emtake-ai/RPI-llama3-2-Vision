# RPI-llama3-2-Vision

Raspberry Pi Vision-Language AI system using:

- YOLOv11 (Office Fine-Tuned)
- Llama 3.2 1B
- Raspberry Pi Camera
- Local AI Inference
- Office Environment Understanding

---

## Overview

RPI-llama3-2-Vision is an edge AI vision-language system designed to run completely on Raspberry Pi devices.

The project combines:

- Computer Vision (YOLOv11)
- Small Language Models (Llama 3.2 1B)
- Office Environment Understanding
- Natural Language Question Answering

The system captures images from a Raspberry Pi camera, detects office objects using a fine-tuned YOLOv11 model, and uses Llama 3.2 to answer questions about the environment.

---

## Features

### Vision

- Real-time image capture
- Raspberry Pi Camera support
- YOLOv11 object detection
- Office-specific fine-tuned model

### Language Understanding

- Llama 3.2 1B
- Local inference
- Natural language interaction
- Scene understanding

### Edge AI

- No cloud required
- Raspberry Pi deployment
- Lightweight architecture
- Low power consumption

---

## Model Download

Hugging Face:

https://huggingface.co/emtake-ai/llama3-2

```bash
hf download emtake-ai/llama3-2 vision_llama.tar --local-dir .
```

## Clone Repository

```bash
git clone https://github.com/emtake-ai/RPI-llama3-2-Vision.git
cd RPI-llama3-2-Vision
```

## Installation

```bash
pip install -r requirements.txt
```

## Run

```bash
python3 rpi_client.py --server 192.168.1.3:18080 in here, you should modify the ip address for its ip
```

## License

Apache-2.0 License

## Author

EMTake AI
