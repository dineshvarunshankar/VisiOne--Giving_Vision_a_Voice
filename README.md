# Visione: AI Co-Pilot for the Visually Impaired 👁️🤖

**Visione** is a real-time assistive navigation system designed to act as a "digital guide dog." It combines state-of-the-art Computer Vision, Metric Depth Estimation, and Generative AI Audio to perceive the world in 3D and provide intelligent, natural-language guidance to visually impaired users.

Unlike traditional sensors that simply beep, Visione is **predictive** and **conversational**, allowing users to ask questions about their environment and receive realistic voice feedback.

---

## 🚀 Key Features

* **🗣️ Real-Time Query Grounding (OpenAI Whisper):** The user can speak naturally—*"Where is my latte?"* or *"Find the exit"*. The system uses **Whisper** to transcribe the command and instantly updates its visual search logic to find that specific object.
* **🔊 Natural Voice Feedback (Coqui TTS):** Gone are the robotic voices of the past. Visione uses **Coqui XTTS v2** to generate high-quality, realistic speech, making the "AI Co-Pilot" feel like a human companion.
* **⚠️ Physics-Aware Safety Engine:** Uses **Kalman Filters** and **Ego-Motion Compensation** to distinguish between stationary objects and approaching threats (like cars), filtering out false alarms caused by the user's walking motion.
* **🛑 Invisible Obstacle Detection:** Analyzes raw metric depth maps to detect unclassified hazards (walls, boxes, debris) that standard AI models might miss.

---

## 🧠 AI Tech Stack

Visione integrates four distinct AI models into a single synchronized pipeline:

| Component | Model / Technology | Role |
| :--- | :--- | :--- |
| **Vision** | **YOLO-World (v8)** | **Open-Vocabulary Detection.** Identifies safety hazards (cars, stairs) and user-requested objects (keys, cups) without retraining. |
| **Depth** | **Depth Anything V3** | **Monocular Metric Depth.** Converts standard 2D video into a precise 3D depth map to measure distance in meters. |
| **Listening** | **OpenAI Whisper** | **Automatic Speech Recognition (ASR).** Robustly transcribes voice commands even in noisy street environments. |
| **Speaking** | **Coqui TTS (XTTS)** | **Generative Text-to-Speech.** Synthesizes urgent safety warnings and descriptive answers in a realistic, non-fatiguing voice. |

---

## 🛠️ System Architecture

Visione operates on a highly optimized **Sense-Think-Act** loop running at ~15 FPS:

1.  **Sense:**
    * **Visual:** RGB Camera Input.
    * **Audio:** Microphone Input processed by `QueryListener` (Whisper).
2.  **Perceive:**
    * **Tracking:** **ByteTrack** + **Kalman Filtering** maintains object IDs even through occlusions.
    * **Fusion:** The `FusionEngine` combines 2D boxes with 3D depth patches using **5th Percentile Filtering** to determine the precise "leading edge" distance.
3.  **Act:**
    * **Decision:** The `SafetyEngine` prioritizes threats (Speed > Proximity > Query).
    * **Output:** The `VoiceGuide` (Coqui) synthesizes audio in a background process to prevent frame drops.

---

## 📂 Project Structure

```bash
Visione/
├── models/               # Weights for YOLO, Depth, and TTS
├── modules/
│   ├── detector.py       # OpenVocabDetector (YOLO-World)
│   ├── depth.py          # DepthEstimator (DepthAnything)
│   ├── tracker.py        # VelocityTracker (Kalman + Ego-Motion)
│   ├── fusion.py         # FusionEngine (2D+3D merging)
│   ├── safety_engine.py  # Decision logic & priority handling
│   ├── tts.py            # VoiceGuide (Coqui XTTS / System fallback)
│   └── whisper_listener.py # QueryListener (OpenAI Whisper)
├── utils/
│   ├── logger.py         # System logging
│   └── visualizer.py     # OpenCV display & overlays
├── config.py             # Global thresholds & settings
├── main.py               # Main orchestration loop
└── requirements.txt      # Python dependencies
```

## 🔧 Installation

### Prerequisites
* Python 3.10
* CUDA-capable GPU (Highly Recommended for Coqui/YOLO to achieve real-time FPS)
* Microphone & Speakers

### Setup steps
1.  **Clone the repository:**
    ```bash
    git clone https://github.com/dineshvarunshankar/VisiOne-Giving_Vision_a_Voice.git
    cd visione
    ```

2.  **Install Dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

3.  **Install PyTorch (CUDA):**
    * **Crucial:** You must install the version of PyTorch that matches your GPU drivers.
    * Visit [pytorch.org](https://pytorch.org/) to get the correct install command (e.g., `pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118`).

4.  **Model Setup:**
    * **Whisper & Coqui:** The system will automatically download these models on the first run (approx. 2GB).
    * **YOLO-World:** Download `yolov8m-worldv2.pt` and place it in the `models/` folder.
    * **Depth Anything:** Ensure the model weights are accessible in `models/DA3` folder. 
    * Clone the Depth-Anything-3 repository:
    ```bash
    git clone https://github.com/ByteDance-Seed/Depth-Anything-3.git

    #Install no Dependencies:**
    cd depth-anything-3
    pip install . --no-deps
---

## 🏃 Usage

Run the main application:
```bash
python main.py
```

## 🎮 Controls
* **Voice Commands:**
    1.  Say **"Hello"** to wake the system.
    2.  Speak your command naturally:
        * *"Where is my bottle?"* (System searches specifically for "bottle")
        * *"Find the exit."* (System searches for "exit")
        * *"Do you see a laptop?"* (System searches for "laptop")
* **Keyboard:**
    * Press `q` to safely shut down the system. (This ensures background audio processes are killed correctly).

## 🎓 Coursework Context

This project was developed for **24-678: Computer Vision for Engineers** at Carnegie Mellon University. It demonstrates the engineering application of core course concepts:

* **Convolutional Neural Networks (CNNs):** Integration of YOLO for object detection.
* **Motion Analysis:** Custom implementation of **Kalman Filtering** and **Centroid Tracking** .
* **Image Filtering:** Manual implementation of **ROI Pooling**, **Morphological Masking**, and **Statistical Filtering**.
* **3D Reconstruction:** Principles of monocular metric depth estimation.

## 👬 Authors 


**Archit Joshi, Dinesh Varun Shankar Kandiyappan, Hanming Zhang, Yikai Wang**