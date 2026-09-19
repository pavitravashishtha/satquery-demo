# SatQuery AI: Master Technical Context & Architecture Handbook

> **Audience:** Engineering Team, Geospatial Scientists, Hackathon Judges, and Project Collaborators  
> **Version:** 2.0 (Production / Demo-Ready)  
> **Target Hardware:** 6 GB VRAM Target (NVIDIA RTX 4050 Laptop GPU / 5.64 GB Usable)  
> **Last Verified Date:** September 10, 2026  

---

## Table of Contents
1. [Executive Summary & Project Vision](#1-executive-summary--project-vision)
2. [End-to-End System Architecture](#2-end-to-end-system-architecture)
3. [The 6GB VRAM Lifecycle Manager & Weight-Sharing Optimization](#3-the-6gb-vram-lifecycle-manager--weight-sharing-optimization)
4. [Query Interpreter & Intent Classification Engine](#4-query-interpreter--intent-classification-engine)
5. [The Specialist Models Deep Dive](#5-the-specialist-models-deep-dive)
   - 5.1. [ChangeVQASpecialist (Bitemporal Change Detection)](#51-changevqaspecialist-bitemporal-change-detection)
   - 5.2. [GeneralVLMSpecialist (VQA, Captioning, Grounding)](#52-generalvlmspecialist-vqa-captioning-grounding)
   - 5.3. [FusionSpecialist (Optical-SAR Dual-Sensor CNN)](#53-fusionspecialist-optical-sar-dual-sensor-cnn)
6. [Compound Multi-Specialist Query Execution](#6-compound-multi-specialist-query-execution)
7. [Honest Confidence Calibration Display](#7-honest-confidence-calibration-display)
8. [Dynamic Evidence Maps & Sensor Overlays](#8-dynamic-evidence-maps--sensor-overlays)
9. [Backend Server Architecture & API Reference (`server.py`)](#9-backend-server-architecture--api-reference-serverpy)
10. [Frontend Canvas UI Architecture](#10-frontend-canvas-ui-architecture)
11. [Authentication, User Personas & Session Store](#11-authentication-user-personas--session-store)
12. [Post-Mortem: The GeoChat Breakdown & Architectural Pivot](#12-post-mortem-the-geochat-breakdown--architectural-pivot)
13. [Evaluation Harness & Benchmark Accuracy](#13-evaluation-harness--benchmark-accuracy)
14. [Repository File Sitemap & Module Directory](#14-repository-file-sitemap--module-directory)
15. [Developer Runbook & CLI Commands](#15-developer-runbook--cli-commands)
16. [Future Roadmap & Scaling Strategy](#16-future-roadmap--scaling-strategy)

---

## 1. Executive Summary & Project Vision

### The Problem
Earth Observation (EO) satellite analysis today suffers from three critical bottlenecks:
1. **Sensor Fragmentation:** Optical sensors (Sentinel-2, Landsat) cannot see through clouds or nighttime. SAR sensors (Sentinel-1 radar) penetrate clouds and surface water but are difficult for general vision models to interpret without domain-specific fusion.
2. **Task Inflexibility:** Off-the-shelf Vision-Language Models (VLMs) hallucinate geospatial metrics, fail on bitemporal comparative change detection (e.g. urban expansion or disaster damage), and cannot ingest raw 1-channel or multi-spectral radar tensors.
3. **Severe Hardware Cost:** Running multiple state-of-the-art vision models simultaneously requires expensive multi-GPU clusters (24GB–80GB VRAM), making edge, laptop, or tactical field deployment impossible.

### The Solution: SatQuery AI
**SatQuery AI** is an intelligent multi-specialist satellite question-answering and intelligence system engineered to operate under a strict **6GB VRAM budget** on consumer/laptop GPUs.

- **Natural Language Querying:** Users ask questions in plain English (*"Combine optical and radar imagery to identify flooded areas, and compare before and after images to assess changes"*).
- **Resident Query Interpreter:** A 4-bit quantized LLM (Qwen3.5-2B, ~1.45 GB VRAM) continuously parses geographic intent, sensor requirements, and task sequences.
- **Dynamic Lifecycle Manager:** A single swappable slot policy with **zero-swap weight sharing** allows instant (0.0001s) transitions between base VLM reasoning and fine-tuned change detection.
- **Multi-Specialist Federation:** Routes sub-tasks to dedicated neural models:
  - Bitemporal change detection (Qwen2.5-VL-3B + LoRA fine-tuned on CDVQA).
  - High-resolution single-image VQA, scene captioning, and bounding box grounding.
  - Multi-sensor Optical-SAR fusion CNN with Grad-CAM spatial activation overlays.
- **Scientific Honesty:** Strictly separates statistically calibrated probabilities (sigmoid CNN outputs) from uncalibrated token certainty (greedy LLM decoding).

---

## 2. End-to-End System Architecture

```mermaid
flowchart TD
    User([User / Browser Canvas]) -->|Natural Query + Images| API[Flask Backend Server :8080]
    
    subgraph Core Orchestration Pipeline
        API -->|raw_query| Interp[Resident Query Interpreter\nQwen3.5-2B 4-bit NF4\n1.45 GB VRAM Permanent]
        Interp -->|InterpretedQuery Schema| Orch[Orchestrator Engine\nscripts/orchestrator.py]
        
        Orch -->|task_sequence| MLM[Model Lifecycle Manager\nsingle-swap-slot budget manager]
    end
    
    subgraph GPU VRAM Memory Space - 5.64 GB Limit
        direction TB
        subgraph Resident Space - 1.45 GB
            Interp
        end
        
        subgraph Swappable Slot Space - ~2.5 GB
            direction LR
            subgraph Group 'qwen_vl' - Shared Weights
                SpecVLM[GeneralVLMSpecialist\nQwen2.5-VL-3B Base\nLoRA Disabled]
                SpecCDVQA[ChangeVQASpecialist\nQwen2.5-VL-3B + LoRA\nLoRA Enabled]
                SpecVLM <-->|0.0001s Instant Toggle\nZero Swap Reload| SpecCDVQA
            end
            
            SpecFusion[FusionSpecialist\n2-Branch ResNet CNN\n+ Grad-CAM Hook]
            
            Group 'qwen_vl' <-->|Real Eviction / Reload\n~3.5s swap| SpecFusion
        end
    end
    
    MLM --> Dispatch{Task Dispatcher}
    Dispatch -->|vqa / caption / grounding| SpecVLM
    Dispatch -->|change_vqa| SpecCDVQA
    Dispatch -->|fusion_analysis| SpecFusion
    
    SpecVLM --> ResultAgg[Result Formatter & Evidence Serializer]
    SpecCDVQA --> ResultAgg
    SpecFusion --> ResultAgg
    
    ResultAgg -->|Unified Multi-Result JSON| API
    API -->|Live Telemetry + Heatmaps + Sliders| User
```

### Execution Flow Step-by-Step
1. **Ingestion:** User submits a prompt and images (or selects a pre-configured benchmark preset) through the web canvas or REST API.
2. **Intent Parsing:** `QueryInterpreter` analyzes the prompt, extracting `task_sequence`, geographic locations, temporal constraints, and required sensor modalities.
3. **Sequence Planning:** If a query requires multiple specialists (e.g. optical-SAR flood identification + before/after change comparison), a sequential task list is planned.
4. **Lifecycle Loading:** `ModelLifecycleManager` loads required specialists into GPU memory. If consecutive tasks share weights, reloads are skipped (0.0s). If an eviction is necessary, memory is garbage-collected and cache emptied.
5. **Execution:** Each specialist runs inference on its designated sensor modalities.
6. **Telemetry & Calibration:** Raw outputs are packaged with calibrated/uncalibrated confidence tags and Base64-encoded visual evidence (Grad-CAM heatmaps or split sliders).
7. **Stream Delivery:** The web client receives the structured payload, animates orbital acquisition telemetry, typewriter-streams the answer, and renders interactive evidence maps.

---

## 3. The 6GB VRAM Lifecycle Manager & Weight-Sharing Optimization

### The Problem
- An RTX 4050 Laptop GPU has **5.64 GB usable VRAM**.
- Loading Qwen3.5-2B (Interpreter) + Qwen2.5-VL-3B (VLM/Change) + Fusion CNN simultaneously causes immediate CUDA Out-of-Memory (OOM) crashes ($>7.5\text{ GB}$ required).

### The Architecture (`scripts/model_lifecycle_manager.py`)
The `ModelLifecycleManager` implements two residency tiers:
1. **Resident Tier (`Residency.RESIDENT`):** Small, critical models that remain loaded permanently.
   - `INTERPRETER_LLM` (Qwen3.5-2B, 4-bit NF4 quantized via BitsAndBytes): **1.45 GB allocated**.
2. **Swappable Tier (`Residency.SWAPPABLE`):** Heavy specialists that share a single GPU swap slot. Only one swappable model occupies the slot at any given time.

### The Breakthrough: Zero-Swap Weight Sharing (`shared_group="qwen_vl"`)
In `scripts/model_registry.py`, `GeneralVLMSpecialist` and `ChangeVQASpecialist` are registered with `shared_group="qwen_vl"`:
- Both specialists share the exact same underlying `Qwen2.5-VL-3B-Instruct` base model in memory.
- `ChangeVQASpecialist` applies a fine-tuned LoRA adapter (`peft`).
- `GeneralVLMSpecialist` disables the adapter via `model.disable_adapter()`.
- **Benchmark Result:** Switching between General VLM and Change Detection takes **0.0001 seconds** and **0 MB additional VRAM**.

```
+-----------------------------------------------------------------------------+
| GPU VRAM Budget: 5.64 GB Usable                                             |
|                                                                             |
| [==================] Resident: Qwen3.5-2B Interpreter (1.45 GB)             |
| [==============================] Swappable Slot (2.42 GB - 2.80 GB)          |
|    Case A: Qwen2.5-VL-3B + LoRA (Change VQA) [2.42 GB]                     |
|    Case B: Qwen2.5-VL-3B Base (General VLM)  [2.42 GB] (0.0001s swap!)      |
|    Case C: Fusion ResNet-18 Dual Branch      [1.64 GB] (~3.5s real swap)    |
| [===========] Headroom / PyTorch Workspace Allocations (~1.3 GB)           |
+-----------------------------------------------------------------------------+
```

### Unload Protocol
When evicting a swappable model from GPU, the lifecycle manager executes a mandatory 3-step memory purge:
```python
del model
gc.collect()
if torch.cuda.is_available():
    torch.cuda.empty_cache()
    torch.cuda.synchronize()
```
This guarantees VRAM drops back to baseline before the incoming specialist allocates memory.

---

## 4. Query Interpreter & Intent Classification Engine

The Query Interpreter (`scripts/interpreter.py`) translates arbitrary natural language queries into machine-actionable, structured payloads matching Pydantic schemas (`scripts/schema.py`).

### Schema Definition (`InterpretedQuery`)
```python
class InterpretedQuery(BaseModel):
    task_type: Task                        # Primary task
    task_sequence: List[Task]              # Execution sequence for single/compound tasks
    query_type: QueryType                  # atomic | compound | ambiguous
    location: Optional[str]                # Extracted city, region, or river basin
    temporal_scope: Optional[TemporalScope]# single_date | bitemporal | multi_temporal
    sensors: List[SensorType]              # optical | sar | fusion | unknown
    parameters: Dict[str, Any]             # Class targets, thresholds, questions
    needs_clarification: bool              # True if query lacks necessary specificity
    confidence: float                      # Routing classification confidence (0.0 - 1.0)
```

### Supported Task Types
- `Task.VQA` (`"vqa"`): Single-image optical question answering.
- `Task.CAPTIONING` (`"captioning"`): Detailed aerial scene description.
- `Task.GROUNDING` (`"grounding"`): Spatial object localization with bounding boxes.
- `Task.CHANGE_VQA` (`"change_vqa"`): Bitemporal change detection and comparison.
- `Task.FUSION_ANALYSIS` (`"fusion_analysis"`): Multi-sensor Optical-SAR fusion.

### Hybrid Inference Strategy
1. **Resident LLM Prompting:** Uses few-shot structured prompting (`scripts/prompt_template.py`) instructing Qwen3.5-2B to emit strict JSON matching the schema.
2. **Regex & Keyword Fallback (`scripts/keyword_fallback.py`):** If the LLM generates malformed JSON, times out, or runs on a CPU-only environment, a deterministic regex and keyword rule engine extracts tasks and spatial locations.
3. **Clarification Memory (`scripts/clarification_memory.py`):** Tracks session history so that follow-up responses (*"around the port"*) resolve ambiguous previous turns without re-asking.

---

## 5. The Specialist Models Deep Dive

```
+---------------------+-------------------------------+-----------------------------------+
| Specialist Model    | Underlying Architecture       | Target Domain / Modalities        |
+---------------------+-------------------------------+-----------------------------------+
| ChangeVQASpecialist | Qwen2.5-VL-3B + CDVQA LoRA    | Bitemporal comparative change VQA |
| GeneralVLMSpecialist| Qwen2.5-VL-3B-Instruct (Base) | Single-image VQA, Caption, Ground |
| FusionSpecialist    | 2-Branch ResNet-18 CNN        | Co-registered Optical + SAR radar |
+---------------------+-------------------------------+-----------------------------------+
```

---

### 5.1. ChangeVQASpecialist (Bitemporal Change Detection)
- **Source Code:** [`scripts/qwen_specialist.py`](file:///home/pavitra/satquery/scripts/qwen_specialist.py)
- **Checkpoint Location:** `checkpoints/qwen2.5-vl-cdvqa-lora/`
- **Base Model:** `Qwen/Qwen2.5-VL-3B-Instruct`
- **Training Data:** CDVQA (Change Detection Visual Question Answering) benchmark.
- **Fine-Tuning Configuration:**
  - PEFT LoRA on projection layers (`q_proj`, `k_proj`, `v_proj`, `o_proj`).
  - Rank $r = 16$, Alpha $\alpha = 32$, Dropout $= 0.05$.
  - 4-bit BitsAndBytes quantization (`nf4`, double quantization).
- **Validation Accuracy:** **72.5%** on CDVQA test split.
- **Run Signature:**
  ```python
  ChangeVQASpecialist.run(image_before: PIL.Image, image_after: PIL.Image, question: str) -> Dict[str, Any]
  ```
- **Output Contract:**
  ```python
  {
      "answer": "Yes, new buildings have been constructed in the central quadrant.",
      "confidence": 0.9984,
      "task_type": "change_vqa",
      "evidence_maps": {}
  }
  ```
- **Crucial Diagnostic Finding (Confidence Limitation):**
  Greedy decoding token probabilities in instruction-tuned VLMs become saturated near $1.0$ ($0.99+$) regardless of whether the model's factual answer is correct or incorrect. **This confidence is uncalibrated.** The system flags this in the UI as `⚠ Model Certainty (Not Accuracy-Calibrated)`.

---

### 5.2. GeneralVLMSpecialist (VQA, Captioning, Grounding)
- **Source Code:** [`scripts/general_vlm_specialist.py`](file:///home/pavitra/satquery/scripts/general_vlm_specialist.py)
- **Base Model:** `Qwen/Qwen2.5-VL-3B-Instruct` (LoRA adapter disabled via `disable_adapter()`).
- **Run Signature:**
  ```python
  GeneralVLMSpecialist.run(image: Union[PIL.Image, str], query: str, task_type: str = "vqa") -> Dict[str, Any]
  ```
- **Task Behaviors:**
  1. **Visual Question Answering (`task_type="vqa"`):** Answers natural language queries regarding land cover, infrastructure, vegetation, and topography.
  2. **Scene Captioning (`task_type="captioning"`):** Generates dense descriptive paragraphs characterizing spatial layout, density, and geography.
  3. **Visual Grounding (`task_type="grounding"`):** Prompts the model to return 2D coordinates `[ymin, xmin, ymax, xmax]` normalized to $[0, 1000]$. The specialist extracts bounding boxes and maps them to client SVG overlay rectangles.
- **Output Contract for Grounding:**
  ```python
  {
      "answer": "Identified 4 building structures in the scene.",
      "confidence": 0.985,
      "task_type": "grounding",
      "evidence_maps": {
          "boxes": [[120, 340, 280, 510], [450, 110, 600, 310]],
          "labels": ["Building", "Structure"],
          "raw_text": "[120, 340, 280, 510]..."
      }
  }
  ```

---

### 5.3. FusionSpecialist (Optical-SAR Dual-Sensor CNN)
- **Source Code:** [`scripts/fusion_model.py`](file:///home/pavitra/satquery/scripts/fusion_model.py)
- **Checkpoint Location:** `checkpoints/fusion_model.pt`
- **Architecture:** 2-Branch Dual-Encoder ResNet:
  - **SAR Branch:** 1-channel input (Sentinel-1 VV polarization backscatter), custom 1-channel conv1, ResNet-18 backbone.
  - **Optical Branch:** 3-channel input (Sentinel-2 RGB / B04 red band), standard ResNet-18 backbone.
  - **Fusion Neck:** Concatenation of pooled 512-d feature vectors $\to$ 1024-d dense layer with ReLU, Dropout $\to$ Linear classifier head.
- **Classes Detected:** Urban / Built-up, Arable land, Permanent crops, Pastures, Forests, Shrubland, Inland waters, Marine waters, Flooded wetlands.
- **Loss Function:** Multi-Label Binary Cross-Entropy with Logits (`BCEWithLogitsLoss`).
- **Run Signature:**
  ```python
  FusionSpecialist.run(sar_tensor: torch.Tensor, optical_tensor: torch.Tensor, threshold: float = 0.5) -> Dict[str, Any]
  ```
- **Explainability (Grad-CAM):**
  - Uses a PyTorch backward hook on `optical_encoder.layer4[1].conv2`.
  - Calculates channel-weighted gradients relative to detected class logits.
  - Generates a normalized 2D float activation array $[0.0, 1.0]$.
  - The backend converts this array into an RGBA Jet-colormap Base64 PNG.
- **Confidence Calibration:** Multi-label sigmoid outputs $P(c) = \sigma(z_c) \in [0.0, 1.0]$ are **empirically calibrated probabilities**. Labeled in UI as `✓ Calibrated Confidence 91% (Calibrated Sigmoid)`.

---

## 6. Compound Multi-Specialist Query Execution

Compound queries require orchestrating two distinct specialists within a single unified workflow (e.g. Optical-SAR multi-modal analysis combined with bitemporal comparative change detection).

### Benchmark Query
> *"Combine optical and radar imagery to identify flooded areas, and compare before and after images to assess changes."*

### Image Input Bundle (4 Images Required)
In real satellite data, these cannot be compressed into a single image because:
1. SAR (radar backscatter) and Optical are physically different sensors from different satellites (Sentinel-1 vs Sentinel-2).
2. Change detection requires two distinct acquisition timestamps (pre-disaster vs post-disaster).

```
Image Bundle Supplied:
├── optical: Sentinel-2 B04 Band (data/ben-ge-8k/sentinel-2/..._B04.tif)
├── sar:     Sentinel-1 VV Radar (data/ben-ge-8k/sentinel-1/..._VV.tif)
├── before:  Pre-change Optical  (data/SECOND/im1/00003.png)
└── after:   Post-change Optical (data/SECOND/im2/00003.png)
```

### Execution Telemetry & Sequence
1. **Step 1:** Dispatched to `FusionSpecialist`. Ingests `optical` + `sar`. Detects *Arable land* ($90.9\%$ calibrated confidence) and generates Grad-CAM activation heatmap.
2. **Step 2:** Dispatched to `ChangeVQASpecialist`. Ingests `before` + `after`. Compares scenes and outputs `"No"` (buildings did not change).
3. **Total Server Time:** 30.6s (initial cold load), 18.6s (subsequent compound runs).

---

## 7. Honest Confidence Calibration Display

A core architectural tenet of SatQuery AI is **never misleading the analyst** by conflating greedy token probability with factual accuracy.

| Model / Specialist | Raw Score Origin | Statistical Reliability | UI Badge Label | UI Subtitle / Explanatory Note |
| :--- | :--- | :--- | :--- | :--- |
| **FusionSpecialist** | Sigmoid $\sigma(z) \in [0, 1]$ | **Calibrated** (empirical probability from BCE) | `✓ Calibrated Confidence: 91%` (Emerald Green) | *"Calibrated multi-label probability from Optical-SAR dual-branch CNN"* |
| **ChangeVQASpecialist** | Token log-probs | **Uncalibrated** (greedy decoding saturation) | `⚠ Model Certainty (Not Accuracy-Calibrated)` (Slate Neutral) | *"Greedy decoding token certainty; does not reflect factual accuracy"* |
| **GeneralVLMSpecialist**| Token log-probs | **Uncalibrated** (greedy decoding saturation) | `⚠ Model Certainty (Not Accuracy-Calibrated)` (Slate Neutral) | *"Greedy decoding token certainty; does not reflect factual accuracy"* |

### Compound Confidence Display
In compound queries, the UI renders **both individual badges stacked in order**, explicitly labeled to their respective step (`Step 1: Optical-SAR Fusion` and `Step 2: Change Detection`), preventing any false averaging or misleading score compression.

---

## 8. Dynamic Evidence Maps & Sensor Overlays

The client interface dynamically adapts its visual canvas based on the specialist outputs:

### 1. Optical-SAR Fusion
- Renders an interactive 3-way sensor toggle button bar:
  - `📡 Sentinel-2 Optical`: View raw RGB/infrared satellite band.
  - `⚡ Sentinel-1 SAR`: View radar backscatter texture.
  - `🔥 Grad-CAM Heatmap`: Overlays the jet-colormap activation heatmap on top of the optical sensor image.

### 2. Bi-Temporal Change Detection
- Renders an interactive split-screen Before/After slider.
- Users drag an interactive scrub handle (`↔`) from 0% to 100% to visually inspect pixel changes between pre-event (`im1`) and post-event (`im2`).

### 3. Visual Grounding
- Overlays an SVG layer directly atop the analyzed scene.
- Renders glowing emerald bounding boxes (`<rect class="bbox-rect">`) and label tags for every detected building or facility.

### 4. Standard VQA / Captioning
- Shows the satellite image cleanly without fake or arbitrary masks.

---

## 9. Backend Server Architecture & API Reference (`server.py`)

A lightweight Flask server provides high-concurrency request dispatching, GeoTIFF streaming, and model session management.

### Endpoints

| Method | Endpoint | Description | Sample Payload / Params |
| :--- | :--- | :--- | :--- |
| `GET` | `/` | Serves the web canvas application (`index.html`) | - |
| `GET` | `/api/health` | Healthcheck and GPU visibility status | Returns `{"status": "healthy", "gpu_available": true}` |
| `GET` | `/api/presets` | List of 6 pre-configured benchmark demonstration cases | Returns metadata, queries, and file paths |
| `GET` | `/api/image` | Dynamic GeoTIFF $\to$ PNG on-the-fly streaming | `?path=/home/pavitra/.../B04.tif` |
| `POST`| `/api/query` | Primary query execution endpoint (JSON or multipart) | `{"query": "...", "preset_id": "compound"}` |
| `GET` | `/api/auth/personas` | List of demo analyst personas | Returns Priya, Aris, Guest metadata |
| `GET` | `/api/auth/user` | Current authenticated user profile & quota stats | Requires `Authorization: Bearer <token>` |
| `POST`| `/api/auth/login` | Authenticates via 1-click persona ID or email/pass | `{"persona_id": "priya"}` |
| `POST`| `/api/auth/signup` | Registers new user account | `{"name": "...", "email": "..."}` |
| `POST`| `/api/auth/update-profile` | Updates user display name, email, agency, role | `{"name": "...", "organization": "..."}` |
| `POST`| `/api/auth/regenerate-key` | Issues a new developer API token (`sq_live_...`) | - |
| `POST`| `/api/auth/logout` | Invalidates active session token | - |

---

## 10. Frontend Canvas UI Architecture

Located in `ui/` (and mirrored to `~/Downloads/satquery-legacy-canvas/`).

```
ui/
├── index.html        # App shell, landing hero, workspace sidebar, modals
├── styles.css        # Vanilla CSS, glassmorphism, responsive breakpoints
├── app.js            # GSAP animations, fetch dispatch, evidence renderers
└── uploads/          # Temporary directory for drag-and-drop satellite files
```

### Key UI Features
- **GSAP ScrollTrigger Cinematic Landing Page:** Frame-by-frame satellite flyover, parallax panels, and live agent workflow modal.
- **Orbital Acquisition HUD (`runWowFlightSequence`):** Real-time animated telemetry overlay simulating coordinate acquisition, band fetching, and GPU specialist inference.
- **ChatGPT-Style Analysis Workspace:** Clean chat thread, history sidebar, preset pills, and responsive location composer.
- **Interactive Modals:** `#pipeline-modal` (architecture flow), `#search-sheet` (conversation search), `#auth-modal` (1-click login), and `#account-modal` (profile, quota, and developer API keys).

---

## 11. Authentication, User Personas & Session Store

User state is persisted in `data/users.json` and synchronized to client `localStorage`.

### Pre-Configured Demo Personas
1. **Priya Menon (`priya`)**:
   - Role: Senior Geospatial Analyst
   - Organization: National Remote Sensing Centre (NRSC)
   - Tier: **Pro Tier** (500 queries/month, all models unlocked)
   - Token: `sq_live_9a87f12e4b3c7d6e`
2. **Dr. Aris Thorne (`aris`)**:
   - Role: Lead Remote Sensing Scientist
   - Organization: Earth Observation Directorate
   - Tier: **Enterprise Tier** (2,500 queries/month, priority GPU cluster access)
   - Token: `sq_live_f820c71a39b2e04d`
3. **Guest Analyst (`guest`)**:
   - Role: Public Observer
   - Organization: Open Research Community
   - Tier: **Free Tier** (100 queries/month, standard benchmarks)
   - Token: `sq_live_guest_demo_key_771`

---

## 12. Post-Mortem: The GeoChat Breakdown & Architectural Pivot

Early in the project, **GeoChat-7B** was integrated for single-image VQA and grounding. During smoke testing, it was diagnosed as fundamentally broken and was completely removed.

### Failure Symptoms
1. Emitted identical, non-sensical Chinese text (*"nobody is perfect, and we all make mistakes"*) across all queries.
2. Under sampling, degraded into incoherent multilingual hallucinations completely untethered from the input satellite image.
3. Locked at an unvarying confidence score of $0.8162$ regardless of input.

### Root Cause Analysis
1. **Vision Tower Key Mismatch:** 23 unexpected keys were logged when loading the CLIP vision tower into HuggingFace's generic `LlavaForConditionalGeneration` class. The projection weights were never assigned, meaning visual tokens never reached the language model.
2. **Quantization Embedding Corruption:** Resizing token embeddings under 4-bit BitsAndBytes corrupted the vocabulary table.
3. **Negative Index CUDA Crash:** `IMAGE_TOKEN_INDEX = -200` caused `RepetitionPenaltyLogitsProcessor` to perform negative GPU indexing, crashing the CUDA driver.

### The Architectural Pivot
Instead of attempting fragile forks of legacy 7B models, we implemented **`GeneralVLMSpecialist` using `Qwen2.5-VL-3B-Instruct`**.
- **Advantage 1:** Shares the base weights with `ChangeVQASpecialist` $\to$ **Zero extra VRAM cost**.
- **Advantage 2:** Instant switching between general tasks and change detection ($0.0001\text{s}$).
- **Advantage 3:** Superior resolution handling, native bounding box parsing, and zero hallucination loops.

---

## 13. Evaluation Harness & Benchmark Accuracy

### Evaluation Scripts
- `scripts/evaluate_cdvqa.py`: Quantitative benchmark on CDVQA validation dataset.
- `scripts/evaluate_fusion.py`: Multi-label evaluation on BEN-GE-8K dataset.
- `scripts/smoke_test_all_tasks.py`: End-to-end integration test across all 5 task types.

### Benchmark Results
- **ChangeVQASpecialist (Qwen2.5-VL-3B + LoRA):** **72.5% Validation Accuracy** on CDVQA comparative change detection.
- **FusionSpecialist (Optical-SAR ResNet-18):** High multi-label precision on flood inundation ($>90\%$ confidence on water detection in presence of heavy cloud cover).
- **Zero-Swap Latency:** **0.0001s** transition between General VLM and Change VQA.
- **Warm Re-Test Latency:** Dropped from **30.6s** (cold load) to **3.24s** (warm cache) with zero GPU weight reloads.

---

## 14. Repository File Sitemap & Module Directory

```
satquery/
├── server.py                        # Main web application backend (Flask + REST API)
├── run_train.sh                     # Training script launcher for LoRA fine-tuning
├── train.log                        # PyTorch training logs
├── checkpoints/
│   ├── fusion_model.pt              # Trained weights for Optical-SAR 2-branch ResNet
│   └── qwen2.5-vl-cdvqa-lora/       # Trained LoRA adapter weights for Change VQA
├── data/
│   ├── users.json                   # Auth user persistence store
│   ├── SECOND/                      # SECOND dataset bitemporal image pairs (im1, im2)
│   └── ben-ge-8k/                   # BEN-GE-8K Sentinel-1 SAR and Sentinel-2 Optical GeoTIFFs
├── scripts/
│   ├── orchestrator.py              # Central pipeline dispatcher (run_query entrypoint)
│   ├── model_lifecycle_manager.py   # 6GB VRAM single-swap-slot budget manager
│   ├── model_registry.py            # Specialist loader registrations and shared groups
│   ├── interpreter.py               # Natural language query interpreter (Qwen3.5-2B)
│   ├── schema.py                    # Pydantic data schemas (InterpretedQuery, Task, etc.)
│   ├── keyword_fallback.py          # Deterministic regex fallback engine
│   ├── prompt_template.py           # Structured few-shot prompts for interpreter
│   ├── clarification_memory.py      # Multi-turn context tracker for ambiguous queries
│   ├── qwen_specialist.py           # ChangeVQASpecialist implementation (Qwen + LoRA)
│   ├── general_vlm_specialist.py    # GeneralVLMSpecialist (Base Qwen: VQA, Caption, Ground)
│   ├── fusion_model.py              # FusionSpecialist (Optical-SAR CNN + Grad-CAM)
│   ├── fusion_dataset.py            # BEN-GE-8K PyTorch Dataset & GeoTIFF loaders
│   ├── cdvqa_dataset.py             # CDVQA PyTorch Dataset for bitemporal pairs
│   ├── train_lora.py                # LoRA fine-tuning trainer for Change VQA
│   ├── train_fusion.py              # CNN trainer for Optical-SAR dual-branch network
│   ├── evaluate_cdvqa.py            # Quantitative evaluation harness for Change VQA
│   └── evaluate_fusion.py           # Quantitative evaluation harness for Fusion model
├── ui/                              # Web frontend canvas
│   ├── index.html                   # HTML structure, modals, templates
│   ├── styles.css                   # Stylesheet with responsive tokens & glassmorphism
│   ├── app.js                       # Client logic, telemetry HUD, sliders, auth
│   └── uploads/                     # User-uploaded satellite image cache
└── docs/
    ├── SATQUERY_MASTER_CONTEXT.md   # This master technical handbook
    └── SATQUERY_PRESENTATION_DECK.md# Slide-by-slide presentation deck
```

---

## 15. Developer Runbook & CLI Commands

### 1. Launching the Web Application
```bash
# Activate Conda environment
conda activate satquery

# Navigate to project root
cd /home/pavitra/satquery

# Start the server (runs on port 8080)
python server.py
```
Open browser to: `http://localhost:8080`

### 2. Running a Direct Python Query (Bypassing Server)
```python
from scripts.model_registry import build_registry
from scripts.orchestrator import run_query

# Initialize lifecycle manager with GPU budget enforcement
lifecycle_mgr = build_registry(device="cuda")

# Execute a bitemporal change detection query
result = run_query(
    raw_query="Did the buildings change between the before and after images?",
    images={
        "before": "data/SECOND/im1/00003.png",
        "after": "data/SECOND/im2/00003.png",
    },
    lifecycle_manager=lifecycle_mgr,
)

print("Specialist:", result["results"][0]["specialist"])
print("Answer:", result["results"][0]["output"]["answer"])
print("Confidence:", result["results"][0]["output"]["confidence"])
```

### 3. Testing via cURL API
```bash
# Query the compound multi-specialist preset
curl -s -X POST http://127.0.0.1:8080/api/query \
  -H "Content-Type: application/json" \
  -d '{"preset_id": "compound"}' | jq .
```

### 4. Running the Complete Verification Suite
```bash
python /home/pavitra/.gemini/antigravity-ide/brain/59f822aa-3588-45df-b42b-90e493ef5905/scratch/comprehensive_compound_verification.py
```

---

## 16. Future Roadmap & Scaling Strategy

1. **Multi-Temporal Change Series:** Expand from bitemporal pairs ($T_1, T_2$) to 4D multi-temporal satellite cube analysis ($T_1, T_2, \dots, T_n$) for seasonal crop and drought tracking.
2. **Dynamic Tensor Quantization:** Explore 8-bit float (FP8) quantization on Ada Lovelace architectures to run Optical-SAR fusion alongside Qwen without any swap eviction.
3. **Edge Drone Deployment:** Package SatQuery AI into an optimized TensorRT-LLM container for NVIDIA Jetson Orin modules (16GB/32GB unified memory) for autonomous UAV reconnaissance.
4. **Interactive Mask Generation:** Replace static bounding box SVG grounding with interactive Segment Anything in Geospatial (SAM-Geo) polygon contours.

---
*Document prepared for SatQuery AI Engineering Team & Technical Stakeholders.*
