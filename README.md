# KonosEye
Enterprise-grade AI NVR pipeline for RTSP cameras

# KonosEye: AI-Powered Edge NVR Pipeline

KonosEye is a highly optimized, completely local AI surveillance pipeline that transforms any standard RTSP IP camera into an enterprise-grade, human-detecting security system. 

Designed to operate flawlessly in challenging real-world environments (low light, heavy occlusion, complex shadows), KonosEye runs directly on your local hardware without relying on expensive cloud subscriptions.

---

## High-Level Overview (For Non-Technical Users)

Traditional security cameras struggle with false alarms. A swaying tree, a stray cat, or a moving shadow can easily trigger a standard motion detector. Furthermore, in dark environments or when a person is partially hidden behind a wall/fence, standard systems fail to detect the threat entirely.

**How KonosEye solves this:**
* **Zero False Alarms:** Uses advanced AI (YOLO) combined with a custom Anti-Ghosting algorithm to ensure alerts are only sent for actual humans, ignoring moving shadows or weather conditions.
* **Sees in the Dark:** Automatically brightens human silhouettes in pitch-black environments before the AI scans them, acting like digital night vision.
* **100% Private & Local:** All video processing happens on your own computer. No video streams are uploaded to third-party servers.
* **Real-Time Telegram Alerts:** The moment a threat is verified, you receive a high-quality, processed image directly to your phone via Telegram.
* **Interactive Setup:** Includes `mask.py`, an easy-to-use visual tool that lets you simply draw your critical security zones directly on the camera feed using your mouse.

---

## Technical Architecture (For Developers)

KonosEye is not just an inference script; it is a full, hardware-accelerated computer vision pipeline built with Python, OpenCV, and ONNXRuntime.

###  Core Technical Features:
1. **Hardware Acceleration:** Native support for AMD GPUs via `DmlExecutionProvider` (DirectML) and NVIDIA via `CUDAExecutionProvider`.
2. **Buffer-Bloat Elimination:** Strictly reads the latest RTSP frame (`CAP_PROP_BUFFERSIZE = 1`) to ensure absolute zero-latency inference.
3. **Dynamic CLAHE:** Converts low-light frames to the LAB color space to apply Contrast Limited Adaptive Histogram Equalization strictly on the luminance channel, preserving RGB integrity while exposing hidden features.
4. **Occlusion-Proof Bounding (Letterboxing):** Uses proper padding algorithms instead of brute-force resizing. This maintains the spatial geometry of human heads/shoulders appearing behind physical obstacles (walls/fences).
5. **Mathematical Anti-Ghosting Filter:** Implements spatial tracking across time `((curr_x - start_x)**2 + (curr_y - start_y)**2)**0.5`. If an AI-detected object fails to traverse >15 pixels within a 3-second window, it is mathematically classified as a static pareidolia (e.g., tree shadow) and discarded.
6. **Asynchronous I/O:** Telegram HTTP POST requests are dispatched in daemonized background threads to prevent main-loop blocking.

---

## 📁 Project Structure

All execution happens within a single directory for maximum portability:

```text
KonosEye/
├── ai_model/                # Directory containing the YOLO ONNX/XML weights
│   └── yolo11n.onnx         # Example: YOLO11 Nano model
├── KonosEye.py              # The main asynchronous AI processing engine
├── mask.py                  # GUI tool to visually draw polygons on the stream
└── settings.json            # Dynamic configuration (auto-updated by mask.py)

(Note: The AI models are not included in this repository due to file size limits. You can drop any trained YOLO ONNX model into the ai_model folder).

-Installation & Quick Start
1. Requirements
Python 3.9+

opencv-python, numpy, requests

onnxruntime-directml (for AMD/Windows) OR onnxruntime-gpu (for NVIDIA)

2. Configuration
Rename settings_example.json to settings.json.

Insert your camera's RTSP_URL and your TELEGRAM_TOKEN.

3. Draw Your Zones (The Easy Way)
Run the GUI setup tool: python mask.py

A live feed of your camera will appear.

Click to draw polygons around the areas you want to monitor.

Press Ε to save. KonosEye will automatically update the coordinates in settings.json.

4. Start the Engine
Run the main pipeline: python KonosEye.py

KonosEye will warm up the GPU, establish the zero-latency RTSP connection, and begin guarding your property.

-Future Scalability
While currently optimized strictly for human detection (class_id = 0), KonosEye's logic is highly modular. By adjusting the target class IDs and sensitivity thresholds in settings.json, it can be instantly adapted for vehicle tracking, perimeter loitering, or package delivery detection.
