## 🚀 How to Run the App (One-Click Setup)

This project is fully automated and optimized to build correctly on local development machines, including low-RAM hardware down to 4GB.

### For Windows Judges:
1. Clone this repository and open the project folder.
2. Double-click the `run_win.bat` file. This will automatically create an isolated environment, configure your dependencies, download the 0.5B model, and launch the application interface.

### For Mac & Linux Judges:
1. Clone this repository and open your terminal inside the project folder.
2. Make the script executable and run it:
   ```bash
   chmod +x run_mac_linux.sh
   ./run_mac_linux.sh
   ```

# adtc-laptop-llm
Offline llm challenge
# 🌾 Smart Farm Assistant: Offline Deep-Tech AI

An end-to-end, **100% offline, on-device AI assistant** designed explicitly for resource-constrained environments in Africa. Built for the Africa Deep Tech Challenge (ADTC), this application runs entirely on a laptop CPU without internet dependencies, cloud APIs, or data bundles. 

By combining an ultra-lightweight **0.5B Parameter LLM** with an on-device **Whisper Speech-to-Text engine**, it provides critical agricultural triage, automated financial tracking, and cultural connection features—fully accessible via voice or text in both **English and Hausa**.

---

## 🚀 Key Features

### 🌐 1. System & Language Controls
* **100% Offline Execution**: Runs entirely local with zero cloud dependencies or data consumption.
* **Instant Language Toggle**: Smooth UI dropdown switches the layout instantly between English and Hausa.
* **Auto-Model Provisioning**: Automatically checks local storage on first boot and downloads required model weights from Hugging Face if missing.
* **Zero Storage Bloat**: Built-in runtime cleanup routines purge temp files to maintain a permanent installation footprint under 450 MB.

### 🎙️ 2. Voice & Accessibility Core
* **Voice-First Input**: Allows non-literate farmers to dictate questions or logging updates naturally.
* **Local Speech Transcription**: Uses an optimized Whisper-Tiny engine to transcribe English/Hausa audio on standard laptop CPUs.
* **Data Privacy & Clean Disks**: Instantly deletes temporary recorded audio samples post-transcription to prevent drive bloat.
* **Native Read Aloud (TTS)**: Reads farming answers back to the user utilizing the host machine's native system text-to-speech engine.

### 🤖 3. AI Triage & Advisory Core
* **Crop Symptom Diagnostics**: Leverages a localized 0.5B model to assess crop descriptions and suggest immediate remedies.
* **Hallucination-Free Local RAG**: Grounded directly by native agricultural texts (`maize_guide.txt`, `cassava_guide.txt`) to guarantee 100% accurate recommendations.
* **RAM Guard Rails**: Restricts the maximum model context to 1024 tokens to safely run inside 4GB or 8GB RAM profiles without hanging.

### 📅 4. Farm Tracking & Calendar Management
* **Regional Crop Timelines**: Dedicated scheduling structures specifically tailored to regional Maize and Cassava life cycles.
* **Python-Driven Milestone Calculations**: Accepts a planting date and calculates crucial agronomic milestone dates automatically.
* **Fertilizer Push Notifications**: Generates clear, structured alerts defining exactly when to execute first and second fertilizer stages.
* **Harvest Prediction Windows**: Calculates explicit calendar ranges indicating optimal harvesting periods.

### 💰 5. Farm Economics & Visual Analytics
* **Voice-to-Ledger Accounting**: Automatically extracts transaction categories, items, and currencies from spoken farm logs.
* **Local Profit Margin Calculators**: Maintains an isolated financial spreadsheet calculating Net Revenue vs Expenses locally.
* **Accessible Visual Dashboards**: Renders intuitive offline charts with strong color coding and shapes for easy reading by non-literate users.
* **Local Desktop Exports**: Features single-button exports generating clean text reports of calendars and ledgers straight to the laptop desktop.

### 🌍 6. Cultural Connection Features
* **Traditional Wisdom Engine**: Displays a rotating dashboard feed of authentic farming proverbs (Yoruba, Ashanti, Hausa, Igbo, Swahili).
* **Motivational Crop Poetry**: Shares short, localized poems designed to uplift and encourage through difficult drought or harvest cycles.
* **Cultural AI Wrap-Ups**: Appends a culturally relevant, encouraging remark to the conclusion of all agricultural advisory text.

---

## 📁 Directory Layout

Your project repository should follow this exact structural format:

```text
├── models/
│   └── .gitkeep             # Placeholder to ensure folder tracking on GitHub
├── knowledge/
│   ├── maize_guide.txt      # Local agricultural text for RAG grounding
│   └── cassava_guide.txt    # Local agricultural text for RAG grounding
├── app.py                   # Main python execution application file
├── .gitignore               # System ignore files configuration
├── requirements.txt         # Minimalist Python package dependencies
└── README.md                # Project documentation file
```

---

## 🛠️ Optimization Strategy for 8GB/4GB PCs

To fulfill the strict hackathon constraints of low memory execution and a sub-450MB storage footprint, this application adopts a deep-tech architectural design avoiding heavy ML frameworks (like heavy PyTorch or Transformers):

1. **Model Quantization**: The 0.5B LLM is compressed using **4-bit INT4 quantization (GGUF format)**, lowering runtime RAM to roughly ~350MB.
2. **C++ Inference Bindings**: Utilizing `llama-cpp-python` and native CPU bindings for Whisper (`whisper.cpp`/`faster-whisper`) ensures execution speeds are fast without installing multi-gigabyte library environments.
3. **Streamlined UI Framework**: Employs minimalist UI tooling to minimize the final dependency chain and keep installation steps fast and lightweight.

---

## 📝 License
This project is developed for the Africa Deep Tech Challenge 2026. Distributed under the MIT License.
