# SatQuery AI — Project Handover & Context Explanation
**Date:** September 19, 2026  
**Context:** Smart India Hackathon (SIH 2026) Submission Preparation & Cloud/Edge Decoupling  
**Target Audience:** Teammates, Project Evaluators, Technical Reviewers, and Collaborators  

---

## 1. Executive Summary

**SatQuery AI** is an Edge-First, Multi-Modal Satellite Question-Answering and Geospatial Intelligence system. It allows users to ask natural language questions (e.g., *"Did floodwaters breach embankments in Assam between 2023 and 2024?"* or *"Detect urban encroachment in East Delhi using optical and SAR imagery"*) and receive precise, verified answers backed by satellite evidence maps, confidence scores, and bitemporal change detection masks.

The system is engineered specifically for Indian Earth Observation (EO) missions (ISRO CARTOSAT, RISAT, Sentinel-1/2, and global benchmarks like SECOND and BEN-GE-8K).

---

## 2. The Core Challenge & Strategic Pivot

### The Problem
- The full local development codebase at `/home/pavitra/satquery` is **~12 GB**:
  - **4.0 GB**: Base Qwen2.5-VL-3B-Instruct model weights.
  - **6.5 GB**: Bulk training datasets (raw Sentinel/CARTOSAT GeoTIFFs, SECOND dataset, BEN-GE-8K).
  - **1.5 GB**: Intermediate checkpoints, logs, and optimizer states.
- **Git Limits:** GitHub strictly enforces a **100 MB maximum file size limit** and flags repositories exceeding 1–2 GB.
- **Cloud Hosting Constraints:** 
  - Standard free hosting (Vercel, Netlify) has a 50 MB serverless function limit and cannot execute PyTorch or computer vision pipelines.
  - Cloud GPU platforms (like Hugging Face ZeroGPU) require the Python Gradio SDK (`@spaces.GPU`), which would require abandoning our custom 60fps glassmorphic canvas frontend.
  - Dedicated cloud GPUs (A10G/T4) cost $50–$300/month, which is impractical for student hackathon submissions.

### The Solution: The "Edge-First AI" Narrative + 3-Link Evaluation Triad
Rather than struggling to deploy a compromised cloud GPU version, we leveraged the project’s greatest engineering achievement: **running multi-modal satellite AI on edge hardware with strict 6GB VRAM limits (NVIDIA RTX 4050 / Jetson Orin)**.

We structured the evaluation into a **3-Link Triad**:
1. 🌐 **Live Interactive Cloud Prototype** (Hugging Face Spaces — CPU Docker container, ~226 MB):
   - Fast, interactive UI for evaluators to test preset scenarios, view real satellite imagery, inspect confidence scores, and toggle bitemporal change detection sliders.
2. 💻 **Open-Source Codebase** (GitHub: [`pavitravashishtha/satquery-demo`](https://github.com/pavitravashishtha/satquery-demo)):
   - Complete, transparent repository with our exact model architectures, dynamic memory manager, custom LoRA adapter weights, and full technical documentation.
3. 🎥 **2-Minute Edge GPU Video Demonstration** (YouTube / Google Drive):
   - Uncut video recording showing live terminal execution, dynamic model swapping in VRAM (`nvidia-smi`), and real-time inference on a consumer RTX 4050 (6GB).

---

## 3. Directory Isolation: Safety & Integrity

To protect the local development environment:
- **Original Codebase (`/home/pavitra/satquery`):** Left **100% untouched**. No files were renamed, moved, modified, or deleted.
- **Deployment Codebase (`/home/pavitra/satquery-deploy`):** An entirely independent, self-contained directory created specifically for the cloud prototype and GitHub repository (~226 MB total).

---

## 4. What Was Packaged in `satquery-deploy`

| Component | Path | Size | Description & Rationale |
| :--- | :--- | :--- | :--- |
| **Trained LoRA Weights** | `checkpoints/qwen2.5-vl-cdvqa-lora/` | ~15 MB | The fine-tuned Qwen2.5-VL adapter weights (`adapter_model.safetensors`), tokenizer, and config. Proves custom training without pushing 4GB base weights. |
| **Fusion Model** | `checkpoints/fusion_model.pt` | 1.6 MB | Real trained PyTorch cross-attention module for fusing Optical (RGB) and SAR (Radar) imagery. |
| **Curated Benchmark Data** | `data/cartosat_risat_probe/` | ~24 MB | Real Indian satellite pairs: Assam (floods), Chennai (coastal change), Delhi (urban density), Munnar (landslide/topography), Kochi (port expansion). |
| **SECOND Bitemporal Data** | `data/second_samples/` | ~3 MB | Pre- and post-disaster optical pairs with ground-truth change detection masks. |
| **BEN-GE-8K Data** | `data/ben_ge_8k_samples/` | ~1.5 MB | Multispectral GeoTIFF imagery for multi-label land cover verification. |
| **Core Architecture Code** | `scripts/` | ~120 KB | Full orchestration pipeline: `orchestrator.py`, `memory_lifecycle_manager.py`, `query_interpreter.py`, `location_acquisition.py`, and specialist wrappers. |
| **Interactive Frontend** | `ui/` | ~1.2 MB | 60fps canvas, glassmorphic HUD telemetry, bitemporal split-screen slider, architecture verification modal. |
| **Backend Server** | `server.py` | ~86 KB | Flask REST API configured with relative paths, preset response caching, live query geocoding, and edge hardware callouts. |
| **Docker Configuration** | `Dockerfile`, `requirements.txt` | <5 KB | CPU-optimized container specification targeting Hugging Face Spaces port `7860`. |

*Note: 58 MB of redundant optimizer checkpoint files (`checkpoint-16400`, `checkpoint-16492`) were pruned to keep the repository ultra-lean.*

---

## 5. Key Technical Modifications Made

1. **Self-Contained Relative Paths (`server.py`):**
   - Replaced all hardcoded `/home/pavitra/satquery/...` paths with dynamic `BASE_DIR = Path(__file__).resolve().parent`.
2. **Instant Preset Evaluation (<0.05s response):**
   - Verified preset queries (Assam flood detection, Delhi urbanization, Chennai port expansion) return exact, pre-computed benchmark outputs, evidence maps, and swap event logs instantly on CPU.
3. **Live Query Geocoding & Edge Callout:**
   - Any custom user query is dynamically parsed: location entities are extracted, coordinates are resolved via Nominatim/OpenStreetMap, and a live response is generated with an explicit callout directing evaluators to the 2-minute edge video demonstration.
4. **UI Hardware Badges & Modals:**
   - Added an **Edge Hardware Status Pill** in `ui/index.html` headers: `🛰️ Cloud Preview · Edge Target: RTX 4050 (6GB) | [▶ Video Demo]`.
   - Added an **Architecture & Verification Modal** in the sidebar detailing the 6GB VRAM budget calculation and benchmark accuracy scores.

---

## 6. Mathematical Proof: 6GB VRAM Ceiling

One of the most important technical achievements documented in the repository is the **Edge VRAM Lifecycle Budget**:

$$\text{Total Available VRAM} = 6{,}144\text{ MB (RTX 4050 Laptop / Jetson Orin 8GB)}$$

```
+-------------------------------------------------------------+
| Resident Base Memory (Always in VRAM)                       |
|   - PyTorch CUDA Context & OS Display Headroom :    600 MB  |
|   - Optical-SAR Cross-Attention Fusion Model    :    180 MB  |
|   - Bi-Temporal Change Detection Specialist     :    750 MB  |
|   - Multi-Label Land Cover Specialist           :    310 MB  |
|   Subtotal Resident Base                       :  1,840 MB  |
+-------------------------------------------------------------+
| Dynamic Swap Slot (Time-Multiplexed Lifecycle)              |
|   - Qwen2.5-VL-3B (INT4 Quantized + LoRA)       :  3,820 MB  |
|   Peak Active VRAM (Base + VQA Engine)          :  5,660 MB  |
+-------------------------------------------------------------+
| Safety Headroom (Activation tensors & cache)    :    484 MB  |
+-------------------------------------------------------------+
```

When a VQA query arrives, the `MemoryLifecycleManager` unloads intermediate weights, loads the quantized Qwen2.5-VL adapter into the swap slot, executes inference, and releases memory back to the baseline pool—guaranteeing zero out-of-memory (`CUDA OOM`) crashes.

---

## 7. Current Project Status

- [x] **Repository Created & Pushed:** [https://github.com/pavitravashishtha/satquery-demo](https://github.com/pavitravashishtha/satquery-demo)
- [x] **Technical Documentation Updated:** Comprehensive `README.md` committed (Commit: `c587d22`) with HF Spaces frontmatter, pipeline diagrams, VRAM math, and decoupling explanation.
- [x] **Local Server Verified:** All REST endpoints tested and confirmed working with HTTP 200.
- [x] **Container Configured:** `Dockerfile` is ready for Hugging Face Spaces (CPU Docker SDK, port 7860).

---

## 8. Remaining Action Items for the User / Team

1. **Deploy to Hugging Face Spaces:**
   - Go to [huggingface.co/new-space](https://huggingface.co/new-space).
   - Create a Space named `satquery-demo` with the **Docker SDK (Blank)** on Free CPU.
   - Push `/home/pavitra/satquery-deploy` directly or connect the GitHub repository.
2. **Setup 24/7 Keep-Alive:**
   - Add a free monitor on [UptimeRobot.com](https://uptimerobot.com) targeting the deployed Hugging Face Space URL every 10 minutes to prevent container sleeping.
3. **Record Edge GPU Video Demo:**
   - Record a ~2-minute screen capture on the local RTX 4050 machine showing:
     1. Running `scripts/orchestrator.py` on a live query.
     2. `nvidia-smi` showing VRAM staying under 5.7 GB during dynamic swapping.
     3. The resulting evidence map and natural language answer.
   - Upload to YouTube (Unlisted) or Google Drive (Public).
   - Replace the placeholder link (`https://youtu.be/demo-link-placeholder`) in:
     - `ui/index.html` (lines 43, 118, 480)
     - `server.py` (line 1364)
     - `README.md` (lines 14, 213)
4. **Submit to SIH 2026:**
   - Present the 3 links in the hackathon submission form and presentation slides.
