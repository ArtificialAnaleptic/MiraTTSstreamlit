# 🎵 MiraTTS Studio

A local, high-fidelity AI Voice Cloning web interface powered by **MiraTTS**.

Based on the work of [Yatharth Sharma](https://github.com/ysharma3501) and [his repo here](https://github.com/ysharma3501/MiraTTS). Please be sure to check it out!

MiraTTS is a fine-tune of Spark-TTS, optimized with Lmdeploy and FlashSR. This application provides a user-friendly Streamlit interface to generate 48kHz audio with context-aware voice cloning.

## ✨ Features of MiraTTS

* **High Fidelity:** Generates crystal clear 48kHz audio.
* **Zero-Shot Cloning:** Upload a 5-10 second reference clip to clone any voice.
* **Smart Batching:** Uses Mira's batch generation to synthesize long text rapidly (up to 100x realtime).
* **Stability Handling:** Automatically chunks text by sentences to prevent model hallucinations on long paragraphs.
* **History System:** Automatically saves and displays the last 5 generations (Audio + Text prompt).
* **GPU Optimized:** Caches the model in VRAM to prevent reloading on every request.

## 🛠️ Prerequisites

* **Hardware:** NVIDIA GPU with at least 6GB VRAM (Required for Lmdeploy/FlashSR).
* **Python:** Version 3.10 or higher (3.12 recommended).
* **Package Manager:** `uv` (Recommended) or `pip`.

---

## 🐧 Installation (Linux)

Tested on Ubuntu 22.04/24.04.

### 1. Install System Dependencies
You need `ffmpeg` for audio processing logic.
```bash
sudo apt update
sudo apt install ffmpeg
```

### 2. Set up the Environment
We recommend using `uv` for fast dependency resolution.

# Clone the repository
```
git clone https://github.com/ArtificialAnaleptic/MiraTTSstreamlit.git
cd MiraTTSstreamlit
mkdir -p static/reference_audio
mkdir -p static/output
```

# Initialize virtual environment
```
uv venv
source .venv/bin/activate
```

# Install Python dependencies
```
uv pip install -r requirements.txt
```

---

## 🪟 Installation (Windows)

### 1. Install System Dependencies
1.  **FFmpeg:** Download FFmpeg from [gyan.dev](https://www.gyan.dev/ffmpeg/builds/). Extract it and add the `bin` folder to your System PATH.
2.  **C++ Build Tools:** You may need "Desktop development with C++" from the Visual Studio Installer if creating the environment fails.

### 2. Set up the Environment (PowerShell)

```
# Clone the repository
git clone https://github.com/ArtificialAnaleptic/MiraTTSstreamlit.git
cd MiraTTSstreamlit
md static\reference_audio
md static\output
```

# Initialize virtual environment
```
uv venv
.venv\Scripts\activate
```

# Install Python dependencies
```
uv pip install -r requirements.txt
```

*Note: If you encounter issues with PyTorch not finding your GPU on Windows, install the specific CUDA version manually:*
```
uv pip install torch torchaudio --index-url [https://download.pytorch.org/whl/cu121](https://download.pytorch.org/whl/cu121)
```

---

## 🚀 Usage

1.  **Activate your environment** (if not already active).
2.  **Run the App:**

```bash
uv run streamlit run app_mira.py
```

3.  Open your browser to the URL shown (usually `http://localhost:8501`).

### First Run Note
On the very first launch, the app will download the MiraTTS model weights (~2-3 GB) from Hugging Face. This process happens automatically and may take a few minutes depending on your internet speed. Watch your terminal for progress bars.

## 📂 Project Structure

* `app_mira.py`: The main application logic.
* `static/reference_audio/`: Stores your uploaded voice clips.
* `static/output/`: Stores the generated `.wav` and `.txt` files.
* `requirements.txt`: Dependency list.
