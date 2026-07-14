# 🌾 Smart Farm Assistant: Offline Deep-Tech AI
### ADTC 2026 Submission Template Fork (`gatsby100m`)

An end-to-end, **100% offline, on-device AI assistant** designed explicitly for resource-constrained environments in Africa. Built for the Africa Deep Tech Challenge (ADTC) 2026, this application runs entirely on a laptop CPU without internet dependencies, cloud APIs, or data bundles. 

By combining an ultra-lightweight **0.5B Parameter Qwen1.5 LLM** with an on-device semantic search engine, it provides critical agricultural triage, automated financial tracking, and cultural connection features—fully accessible in both **English and Hausa**.

---

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

---

## 🚀 Key Features

### 🌐 1. System & Language Controls
* **100% Offline Execution**: Runs entirely local with zero cloud dependencies or data consumption.
* **Instant Language Toggle**: Smooth UI dropdown switches the layout instantly between English and Hausa.
* **Auto-Model Provisioning**: Automatically checks local storage on first boot and downloads required model weights from Hugging Face if missing.
* **Zero Storage Bloat**: Built-in runtime cleanup routines purge temp files to maintain a permanent installation footprint under 450 MB.

### 🎙️ 2. Accessibility Core
* **Multilingual Input**: Handles crop descriptions and tracking updates natively in English and Hausa.
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
* **Transaction Accounting**: Automatically extracts transaction categories, items, and currencies from your farm logs.
* **Local Profit Margin Calculators**: Maintains an isolated financial spreadsheet calculating Net Revenue vs Expenses locally.
* **Accessible Visual Dashboards**: Renders intuitive offline charts with strong color coding and shapes for easy reading.
* **Local Desktop Exports**: Features single-button exports generating clean text reports of calendars and ledgers straight to the laptop desktop.

### 🌍 6. Cultural Connection Features
* **Traditional Wisdom Engine**: Displays a rotating dashboard feed of authentic farming proverbs (Yoruba, Ashanti, Hausa, Igbo, Swahili).
* **Motivational Crop Poetry**: Shares short, localized poems designed to uplift and encourage through difficult drought or harvest cycles.
* **Cultural AI Wrap-Ups**: Appends a culturally relevant, encouraging remark to the conclusion of all agricultural advisory text.

---

## 📁 Directory Layout

Your project repository follows this structural format:

```text
├── models/
│   └── qwen1_5-0_5b-chat-q4_k_m.gguf  # Localized LLM Weights
├── knowledge/
│   ├── maize_guide.txt                # Local agricultural text for RAG grounding
│   └── cassava_guide.txt              # Local agricultural text for RAG grounding
├── app.py                             # Main python execution application file
├── metadata.json                      # Challenge Validation Configuration
├── REPORT.md                          # Technical Engineering Report
├── download_model.sh                  # Idempotent Model Provisioning Script
├── requirements.txt                   # Minimalist Python package dependencies
└── README.md                          # Project documentation file
```

---

## 🛠️ Optimization Strategy for 8GB/4GB PCs

To fulfill the strict hackathon constraints of low memory execution and a sub-450MB storage footprint, this application adopts a deep-tech architectural design avoiding heavy ML frameworks (like heavy PyTorch or Transformers):

1. **Model Quantization**: The 0.5B LLM is compressed using **4-bit INT4 quantization (GGUF format)**, lowering runtime RAM to roughly ~350MB.
2. **C++ Inference Bindings**: Utilizing `llama-cpp-python` and native CPU bindings ensures execution speeds are fast without installing multi-gigabyte library environments.
3. **Streamlined UI Framework**: Employs minimalist UI tooling to minimize the final dependency chain and keep installation steps fast and lightweight.

---

## 📝 License
This project is developed for the Africa Deep Tech Challenge 2026. Distributed under the MIT License.
