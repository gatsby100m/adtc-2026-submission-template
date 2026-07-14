# 📊 ADTC Technical Optimization Report

This report outlines the technical and architectural strategies implemented to ensure the Smart Farm Assistant runs efficiently under resource-constrained conditions, specifically targeting laptops with **4GB to 8GB of RAM** and a low storage footprint.

---

## 💾 1. RAM Footprint Breakdown & Benchmarks

To ensure the laptop host machine never freezes, memory consumption is strictly managed. Operating system overhead typically consumes ~2GB to 3GB of RAM. The application is benchmarked to run within **~1 GB of active memory**, ensuring full stability on a standard 4GB PC.

| Software Layer | Target Model / Framework | Memory Usage (RAM) | Optimization Strategy |
| :--- | :--- | :--- | :--- |
| **Language Model Core** | Qwen2.5-0.5B-Instruct | ~350 MB | 4-bit INT4 GGUF Quantization |
| **Speech-to-Text Engine**| Whisper-Tiny | ~75 MB | C++ Compiled Native Inference |
| **Knowledge Engine** | Local Text RAG | <50 MB | Flat-file file streaming (No DB overhead) |
| **Application Layer** | Streamlit + Pandas | ~150 MB | Minimal internal variable caching |
| **System Overhead Cushion**| Linux/Windows Environment | ~400 MB | Strict 1024 Token Context Restriction |
| **TOTAL RUNTIME RAM** | **Smart Farm Assistant Core** | **~1.02 GB** | **Safe for 4GB/8GB deployment hardware** |

---

## 🛠️ 2. Deep-Tech Optimization Methods

### 🧱 INT4 GGUF Model Quantization
Running raw, uncompressed language models requires significant GPU power and VRAM. This project uses a **4-bit quantized version (Q4_K_M)** of the Qwen 0.5B model. Quantization compresses the model weights from 16-bit floating-point parameters to 4-bit integers. 
* **Impact:** Shrinks the model size by over **70%** (down to ~350MB) and allows execution entirely on a standard laptop CPU without speed degradation.

### ⛓️ Native C++ Inference Bindings (`llama.cpp`)
Instead of using heavy ML frameworks like PyTorch or Hugging Face Transformers—which require multi-gigabyte package installations and high CPU overhead—this project utilizes `llama-cpp-python`.
* **Impact:** Runs high-performance C++ matrix arithmetic behind a lightweight Python wrapper, maximizing inference speeds on basic CPU hardware.

### 🛡️ Strict Context Window Clamping
Memory consumption expands dynamically as chat history grows. To prevent memory leaks or out-of-memory crashes during long agricultural advisory sessions, the application engine enforces a hard context limit of **1,024 tokens**. 

---

## 📂 3. Storage Footprint Management (<450MB Goal)

Maintaining a permanent installation size under 450 MB is achieved by stripping away heavy software dependencies:
1. **No External Vector Database:** Vector databases (like ChromaDB or FAISS) require heavy underlying binaries. The RAG architecture uses a custom Python flat-file reading system to parse text files directly from the `knowledge/` directory, consuming **0 MB** of extra package storage.
2. **Audio File Purging:** Dictated audio wav samples are wiped from disk immediate following text transcription, preventing storage creep.
3. **No Torch/PyTorch Dependencies:** By relying strictly on pre-compiled wheels for quantized execution, the setup avoids installing PyTorch libraries, keeping package storage at a minimum.

---

## 🌍 4. Socio-Economic & Local Infrastructure Alignment

In many rural farming communities across Nigeria and West Africa, constant internet connections are unavailable, and mobile data bundles are prohibitively expensive. 

* **100% Offline Capability:** By packing the entire LLM, text guidebooks, and math engines on-device, a farmer can run deep triage diagnostics in the middle of a remote field with zero cellular signal.
* **Literacy & Cultural Inclusivity:** The pairing of local language support (Hausa) with voice transcription opens up advanced AI advisory systems to non-literate demographic pools.

---
*Developed for the Africa Deep Tech Challenge (ADTC) 2026.*
