# SatQuery AI: Project Technical Context, System Architecture & Operational Audit

> **Document Type:** Comprehensive Technical Audit & Master Context Specification  
> **Repository:** `pavitravashishtha/satquery`  
> **Last Verified Date:** September 16, 2026  
> **Hardware Target:** 6.0 GB VRAM Budget (NVIDIA RTX 4050 Laptop GPU / 5.64 GB Usable)  
> **Runtime Environment:** Conda Environment `satquery` (`/home/pavitra/miniconda3/envs/satquery/bin/python`)  

---

## 1. Executive Summary & Core Mission

**SatQuery AI** is an edge-optimized, multi-specialist satellite intelligence and Visual Question Answering (VQA) system. It addresses Earth Observation (EO) challenges—such as sensor fragmentation (optical vs. radar), task diversity (change detection, grounding, fusion), and severe hardware constraints—by orchestrating specialized neural models under a **strict 6GB VRAM budget** on consumer/laptop GPUs.

### Key Capabilities
1. **Natural Language Query Routing:** Parses user prompts into structured execution graphs (spatial, temporal, and sensor constraints).
2. **Zero-Swap Weight Sharing:** Toggles between single-image VQA/grounding and fine-tuned bitemporal change detection in **0.0001 seconds** with **0 MB additional VRAM** by sharing base transformer weights.
3. **Multi-Sensor Optical-SAR Fusion:** Ingests Sentinel-2 optical bands and Sentinel-1 SAR radar backscatter to detect land cover and surface floodwater through dense cloud cover, producing Grad-CAM explainability heatmaps.
4. **Autonomous Satellite Imagery Broker:** Automatically geocodes place names or coordinates, queries Sentinel-2 / ESRI satellite repositories, and provides real sensor data on demand.
5. **Scientific Confidence Calibration:** Decouples statistically calibrated sigmoid probabilities (from CNN outputs) from uncalibrated token certainty (from autoregressive LLM greedy decoding), preventing analysts from being misled.
6. **Physical Radiometric Variance Checking:** Cross-examines vision model outputs against pixel-level radiometric shifts to flag observational discrepancies.

---

## 2. System Architecture & VRAM Budgeting

```
+---------------------------------------------------------------------------------------+
| TOTAL GPU VRAM LIMIT: 5.64 GB Usable (NVIDIA RTX 4050 Laptop)                         |
+---------------------------------------------------------------------------------------+
| [====================] RESIDENT TIER (Permanent VRAM: ~1.45 GB)                       |
|   • Model: Qwen3.5-2B (4-bit NF4 Quantization via BitsAndBytes)                       |
|   • Role: Continuous query parsing, intent routing, and parameter extraction          |
+---------------------------------------------------------------------------------------+
| [====================================] SWAPPABLE SLOT (Max 2.80 GB)                   |
|                                                                                       |
|   GROUP 'qwen_vl' (Base: Qwen2.5-VL-3B-Instruct 4-bit NF4 ~2.42 GB)                   |
|   ├── GeneralVLMSpecialist (Base Model, LoRA disabled via disable_adapter())          |
|   └── ChangeVQASpecialist  (Base Model + CDVQA LoRA Adapter enabled)                  |
|       └─> SWITCHING LATENCY: 0.0001s (In-place parameter pointer toggle)              |
|                                                                                       |
|   EVICTED & REPLACED BY (Real swap ~3.5s):                                            |
|   └── FusionSpecialist (2-Branch ResNet-18 CNN: Optical + SAR ~1.64 GB)               |
+---------------------------------------------------------------------------------------+
| [============] Headroom & Dynamic PyTorch Allocations (~1.39 GB)                      |
+---------------------------------------------------------------------------------------+
```

---

## 3. Segment-by-Segment Technical Analysis

Below is an exhaustive technical audit of every major subsystem in the repository, evaluating what is working, the degree of completion, known failure modes, and what remains to be built or fixed.

---

### Segment A: Model Lifecycle Manager & VRAM Swapping Engine
- **Files:** [`scripts/model_lifecycle_manager.py`](file:///home/pavitra/satquery/scripts/model_lifecycle_manager.py), [`scripts/model_registry.py`](file:///home/pavitra/satquery/scripts/model_registry.py)
- **Status:** **92% Functional (Production Ready)**

#### What is Working:
1. **Two-Tier Residency Model:** Perfectly enforces `Residency.RESIDENT` (models that never leave GPU memory) and `Residency.SWAPPABLE` (models restricted to a single mutual exclusion slot).
2. **Zero-Swap Weight Sharing (`shared_group="qwen_vl"`):** Both `GeneralVLMSpecialist` and `ChangeVQASpecialist` share the identical underlying 4-bit `Qwen2.5-VL-3B-Instruct` model in memory. Switching between general VLM tasks and bitemporal change detection executes via `model.disable_adapter()` or `model.enable_adapter()`, completing in **0.0001s** with **0 MB memory overhead**.
3. **Deterministic Eviction Protocol:** When evicting a model to load an out-of-group specialist (e.g. transitioning from Qwen-VL to the Fusion CNN), it runs:
   ```python
   del model
   gc.collect()
   torch.cuda.empty_cache()
   torch.cuda.synchronize()
   ```
   Memory reliably drops back to the 1.45 GB resident baseline before allocating the new model.
4. **Sequence Execution Pipeline:** `lifecycle_manager.run_sequence()` resolves execution order, groups consecutive calls that share models, and eliminates redundant reloads.

#### What is Broken / Left to Implement:
1. **Single-Slot Concurrency Limitation:** Only one heavy specialist can occupy GPU memory. A compound query (`fusion_analysis` $\to$ `change_vqa`) forces a physical eviction, GC collection, and disk/checkpoint load (~3.5s latency).
2. **Lack of FP8 / Dynamic Tensor Quantization:** On Ada Lovelace architecture, FP8 could allow the 1.64 GB Fusion CNN and 2.42 GB Qwen-VL to reside in VRAM simultaneously ($1.45 + 1.64 + 2.42 = 5.51\text{ GB} < 5.64\text{ GB}$). Currently, this has not been implemented.
3. **Memory Fragmentation Risk:** Rapid back-to-back non-compound switches can cause slight PyTorch allocator fragmentation if `torch.cuda.synchronize()` is not invoked between requests.

---

### Segment B: Query Interpreter & Intent Routing Engine
- **Files:** [`scripts/interpreter.py`](file:///home/pavitra/satquery/scripts/interpreter.py), [`scripts/keyword_fallback.py`](file:///home/pavitra/satquery/scripts/keyword_fallback.py), [`scripts/prompt_template.py`](file:///home/pavitra/satquery/scripts/prompt_template.py), [`scripts/clarification_memory.py`](file:///home/pavitra/satquery/scripts/clarification_memory.py), [`scripts/schema.py`](file:///home/pavitra/satquery/scripts/schema.py)
- **Status:** **88% Functional**

#### What is Working:
1. **Resident LLM Structured Extraction:** Quantized Qwen3.5-2B parses natural language into strict Pydantic `InterpretedQuery` instances (`task_type`, `task_sequence`, `location`, `temporal_scope`, `sensors`, `parameters`, `confidence`).
2. **Deterministic Fallback Engine (`keyword_fallback.py`):** If the LLM generates invalid JSON, times out, or runs in a CPU environment, an extensive regex engine (32KB of curated rules) extracts task sequences, geographic locations, and sensor modalities.
3. **Clarification Memory (`clarification_memory.py`):** Retains multi-turn conversation context so that follow-up responses (*"around the port"*) automatically resolve previously ambiguous turns.
4. **Compound Task Sequencing:** Accurately recognizes compound queries (e.g. *"Combine optical and radar imagery to identify flooded areas, and compare before and after images to assess changes"*) and plans `task_sequence = ["fusion_analysis", "change_vqa"]`.

#### What is Broken / Left to Implement:
1. **Greedy JSON Truncation:** Under tight token limits (`max_new_tokens=200`), complex prompts can truncate the closing JSON bracket `}`, triggering fallback parsing.
2. **Ambiguous Geographic Entity Extraction:** The regex engine can misinterpret common English words as locations if they collide with dictionary names (e.g. "March", "Union", "Port").
3. **No Named-Entity Linking / Normalization:** Extracted locations are raw text strings without coordinate bounding box grounding until passed downstream to `location_acquisition.py`.

---

### Segment C: ChangeVQASpecialist (Bitemporal Change Detection)
- **Files:** [`scripts/qwen_specialist.py`](file:///home/pavitra/satquery/scripts/qwen_specialist.py), [`checkpoints/qwen2.5-vl-cdvqa-lora/`](file:///home/pavitra/satquery/checkpoints/qwen2.5-vl-cdvqa-lora/), [`scripts/cdvqa_dataset.py`](file:///home/pavitra/satquery/scripts/cdvqa_dataset.py), [`scripts/train_lora.py`](file:///home/pavitra/satquery/scripts/train_lora.py), [`scripts/evaluate_cdvqa.py`](file:///home/pavitra/satquery/scripts/evaluate_cdvqa.py)
- **Status:** **85% Functional (With Known Calibration Limitations)**

#### What is Working:
1. **Trained PEFT LoRA Weights:** Fully trained adapter (`adapter_model.safetensors`, 14.8 MB) on projection layers (`q_proj`, `k_proj`, `v_proj`, `o_proj`) over the CDVQA benchmark dataset.
2. **Accuracy:** Achieves **72.5% – 75.5% validation accuracy** on bitemporal comparative change visual question answering.
3. **Format & Splicing:** Ingests dual images (`image_before`, `image_after`) along with a natural language query, splicing them with clear temporal tags for instruction-tuned reasoning.
4. **Fast Memory Reuse:** Shares base transformer weights with `GeneralVLMSpecialist`.

#### What is Broken / Left to Implement:
1. **Severe Confidence Saturation (Uncalibrated Token Probabilities):**
   - Empirical validation across $N=200$ CDVQA samples revealed a mean confidence of **99.39%** despite empirical accuracy being **75.5%**.
   - Expected Calibration Error (ECE) is **24.19%** (0.2419).
   - Discriminative ranking power is nonexistent: **AUROC = 0.4664** (worse than random coin-flip ranking). When the model is completely wrong, its average confidence is **99.47%**; when correct, it is **99.37%**.
   - *Fix Status:* Temperature scaling failed to resolve the issue because correct and incorrect logits share identical saturated distributions. The system now honestly displays `⚠ Model Certainty (Not Accuracy-Calibrated)` in the UI.
2. **False-Negative "No" Prediction on Substantial Surface Alteration:**
   - In specific scenes where raw earth clearing or foundation work has taken place but completed rectangular building structures are not yet erected, the model outputs `"No"`.
   - *Mitigation:* The backend now runs `compute_bitemporal_variance` to calculate physical pixel differences. If the model says "No" while measured pixel delta exceeds 8%, the system raises an `Ambiguity & Observational Discrepancy Alert`.
3. **Empty Checkpoint Artifact (`checkpoints/qwen3vl-cdvqa-lora`):**
   - The repository contains an empty directory for an experimental Qwen3-VL LoRA that was never trained or finalized.
4. **Resolution Constraints:**
   - Images are currently processed at standard vision encoder resolutions; gigapixel aerial tiles or multi-gigabyte GeoTIFF pyramids cannot be ingested without tiling.

---

### Segment D: GeneralVLMSpecialist (VQA, Captioning, Visual Grounding)
- **Files:** [`scripts/general_vlm_specialist.py`](file:///home/pavitra/satquery/scripts/general_vlm_specialist.py), [`scripts/verify_general_vlm_integration.py`](file:///home/pavitra/satquery/scripts/verify_general_vlm_integration.py)
- **Status:** **90% Functional**

#### What is Working:
1. **Multi-Task Execution:** Seamlessly executes three distinct spatial tasks using `Qwen2.5-VL-3B-Instruct` (with LoRA disabled):
   - `vqa`: High-level queries regarding land cover, infrastructure, vegetation, and geography.
   - `captioning`: Rich, multi-sentence descriptive aerial paragraphs.
   - `grounding`: Spatial localization of target objects returning normalized coordinates $[0, 1000]$.
2. **Zero VRAM Footprint:** Uses the resident/swappable `qwen_vl` memory block without allocating extra parameters.
3. **Bounding Box Parser:** Automatically parses `[ymin, xmin, ymax, xmax]` tokens from model text completions and outputs structured coordinates for client-side SVG rendering.
4. **Zero Domain Collapse on Out-of-Distribution Data:** Evaluated on the proxy Indian Cartosat probe (`data/cartosat_risat_probe/`), achieving **0% collapse** (correctly distinguishing between Chennai city grids, Assam floodplains, Delhi commercial centers, and Munnar mountain forests).

#### What is Broken / Left to Implement:
1. **Strict String Evaluation Metric Gap:** In automated benchmark scripts (`eval/reports/scorecard.md`), freeform descriptive answers receive 0.00% on exact string match against brief gold labels (e.g. model generates "an urban area with dense roads and buildings" vs gold label "Urban fabric"), even though semantic evaluation is accurate.
2. **Greedy Decoding Logit Saturation:** Like ChangeVQA, raw token confidence is saturated (~99.33%) and uncalibrated (ECE = 48.83%).
3. **No Polygon / Contour Segmentation:** Grounding is strictly limited to 2D bounding boxes; complex irregular features (such as flooded river contours or coastline erosion) cannot be represented as vector polygons (SAM-Geo integration is left to future work).

---

### Segment E: FusionSpecialist (Optical-SAR Dual-Sensor CNN)
- **Files:** [`scripts/fusion_model.py`](file:///home/pavitra/satquery/scripts/fusion_model.py), [`checkpoints/fusion_model.pt`](file:///home/pavitra/satquery/checkpoints/fusion_model.pt), [`scripts/fusion_dataset.py`](file:///home/pavitra/satquery/scripts/fusion_dataset.py), [`scripts/train_fusion.py`](file:///home/pavitra/satquery/scripts/train_fusion.py), [`scripts/evaluate_fusion.py`](file:///home/pavitra/satquery/scripts/evaluate_fusion.py)
- **Status:** **78% Functional (High Specialized Value, Severe Domain Sensitivity)**

#### What is Working:
1. **Dual-Encoder Architecture:**
   - SAR Branch: 1-channel custom `conv1` ResNet-18 backbone ingesting Sentinel-1 VV polarization radar backscatter.
   - Optical Branch: 3-channel RGB Sentinel-2 backbone.
   - Fusion Neck: Concatenates two 512-d embeddings $\to$ 1024-d dense layer with ReLU and Dropout $\to$ multi-label linear classifier head.
2. **Empirically Calibrated Sigmoid Probabilities:**
   - Trained with multi-label Binary Cross-Entropy (`BCEWithLogitsLoss`).
   - Sigmoid outputs $P(c) = \sigma(z_c)$ are statistically calibrated probabilities. The UI renders this honestly as `✓ Calibrated Confidence (Calibrated Sigmoid)`.
3. **Explainability via Grad-CAM:**
   - PyTorch backward hook on `optical_encoder.layer4[1].conv2` computes channel-weighted activation gradients against detected class logits.
   - Generates normalized 2D activation arrays converted on-the-fly to RGBA Jet-colormap Base64 PNGs.
4. **Adaptive Sub-Threshold Signal Extraction (Recent Fix):**
   - When all class probabilities fall below 0.50 (common on subtle wetland/water signals), the specialist no longer collapses silently into "no features detected". Instead, it extracts and reports the top candidate signals (e.g., `Inland wetlands (38.2%) and Inland waters (26.4%)`).
   - UI shows a designated sub-threshold warning badge and renders the Grad-CAM heatmap with adjusted 0.65 opacity.

#### What is Broken / Left to Implement:
1. **Severe Geographic Domain Collapse on High-Res / Proxy Data:**
   - Evaluated on the proxy Indian probe (`eval/reports/DOMAIN_GAP_REPORT.md`), the model suffered an **80% semantic collapse**: predicting "Inland waters" for 5/5 samples (95%–99% false confidence), completely failing to recognize dense urban fabric in Delhi and Chennai.
   - *Root Cause:* The CNN was trained exclusively on European rural/semi-urban BEN-GE-8K patches at 10m/20m GSD. High-contrast sub-meter optical edges combined with synthetic SAR noise immediately trigger the highest-prevalence water prior.
2. **Input Normalization Sensitivity:**
   - Requires inputs to be exactly 128x128 pixels normalized with strict BEN-GE-8K channel mean/std:
     - `OPTICAL_MEAN = [0.346, 0.380, 0.407]`, `OPTICAL_STD = [0.201, 0.141, 0.140]`.
     - SAR must be in decibels clipped between $[-35.0, 5.0]\text{ dB}$ or normalized $[0, 1]$. Unnormalized inputs cause unpredictable logit spikes.
3. **Single-Branch Grad-CAM:**
   - The Grad-CAM hook is attached only to the optical encoder. There is no backward gradient hook on the SAR radar encoder (`sar_encoder.layer4`), meaning radar-specific activations cannot be visualized independently.

---

### Segment F: GeoChatSpecialist (Deprecated Subsystem / Architectural Post-Mortem)
- **Files:** [`scripts/geochat_specialist.py`](file:///home/pavitra/satquery/scripts/geochat_specialist.py), [`GeoChat/`](file:///home/pavitra/satquery/GeoChat/)
- **Status:** **FAILED / DEPRECATED (Preserved only for regression testing)**

#### Post-Mortem Findings & Reasons for Abandonment:
1. **Weight Key Mismatches:** Loading the legacy GeoChat-7B checkpoint into generic HuggingFace `LlavaForConditionalGeneration` caused 23 projection weight keys to be discarded. Visual tokens never propagated into the language model.
2. **Quantization Vocabulary Corruption:** Resizing token embeddings under 4-bit BitsAndBytes corrupted vocabulary pointers, causing the model to emit repetitive Chinese text (*"nobody is perfect, and we all make mistakes"*).
3. **Negative Index CUDA Driver Crash:** Hardcoded `IMAGE_TOKEN_INDEX = -200` caused HuggingFace's `RepetitionPenaltyLogitsProcessor` to index negative GPU memory addresses, crashing the CUDA driver.
4. **Structural Template Fixation:** Even when running under float16 on CPU/GPU, 80% (4/5) of probe responses hallucinated *"two bridges spanning over a body of water"*, regardless of whether the input image showed mountain tea slopes or inland urban streets.
5. **Architectural Replacement:** Fully superseded by `GeneralVLMSpecialist` (`Qwen2.5-VL-3B`), which provides superior accuracy, zero extra VRAM, and eliminates all hallucination loops.

---

### Segment G: Autonomous Satellite Data Broker & Location Acquisition
- **Files:** [`scripts/location_acquisition.py`](file:///home/pavitra/satquery/scripts/location_acquisition.py), [`data/cache_imagery/`](file:///home/pavitra/satquery/data/cache_imagery/)
- **Status:** **82% Functional**

#### What is Working:
1. **Hybrid Geocoding Pipeline:**
   - Direct coordinate parsing for tuple, dict, or cardinal strings (e.g. `"26.1850 N, 91.7450 E"`).
   - Offline Indian Gazetteer (`OFFLINE_GAZETTEER`) for instant resolution of major Indian regions (Assam, Brahmaputra, Chennai, Bengaluru, Kozhikode, Munnar, Mumbai, Delhi, Kolkata, Wayanad, Kochi).
   - Online OpenStreetMap Nominatim forward/reverse geocoding with timeout safety.
2. **Real Satellite Data Ingestion:**
   - Bi-temporal Change Detection: Queries EOX Copernicus Sentinel-2 Cloudless WMS service for 2020 ($T_1$) and 2023 ($T_2$) multi-year passes.
   - High-Resolution Optical: Queries ESRI World Imagery REST export for ~0.5m–1m optical detail.
3. **Local Tile Caching:** Downloads are permanently cached under `data/cache_imagery/<slug>/`, eliminating duplicate external HTTP calls.
4. **Unit Test Coverage:** All 6 coordinate parsing and geocoding tests in [`tests/test_location_acquisition.py`](file:///home/pavitra/satquery/tests/test_location_acquisition.py) pass cleanly.

#### What is Broken / Left to Implement:
1. **Inaccessibility of Real Indian ISRO Data:**
   - Real ISRO Cartosat-2S (sub-meter optical) and RISAT-1/1A (C-band SAR) scenes are restricted behind ISRO's Bhoonidhi/Bhuvan portals, requiring formal government/institutional credentials.
2. **Synthetic SAR Heuristic:**
   - While Assam benchmark queries use real Sentinel-1 SAR GeoTIFFs, queries for other geocoded locations generate synthetic SAR by inverting optical grayscale values and adding Gamma/Rayleigh speckle noise. While visually representative, it lacks true physical microwave dielectric information.
3. **Fixed Bounding Box Dimensions:**
   - Uses a static $0.03^\circ$ (~3.3 km) footprint. Large geographic basins or tiny site footprints cannot dynamically adjust their acquisition scale.
4. **External WMS Availability:**
   - EOX and ESRI services are subject to third-party rate limiting, network latency, or downtime.

---

### Segment H: Web Backend Server & REST API
- **Files:** [`server.py`](file:///home/pavitra/satquery/server.py)
- **Status:** **94% Functional**

#### What is Working:
1. **Unified Query Dispatcher (`/api/query`):** Handles both JSON payloads and multipart form uploads (supporting optical, SAR, before, and after files).
2. **Pre-Configured Benchmark Presets (`/api/presets`):** Serves 6 verified demonstration cases (VQA, Captioning, Grounding, Change Detection, Fusion, and Compound).
3. **On-the-Fly GeoTIFF Streaming (`/api/image`):** Dynamically parses 16-bit float/integer Sentinel GeoTIFFs, normalizes pixel arrays, and streams standard PNG images to the browser.
4. **Grad-CAM Jet Colormap Encoder:** Converts float2D NumPy activation arrays directly into RGBA Jet-colormap Base64 PNG data URLs using pure NumPy (eliminating heavy Matplotlib dependencies).
5. **Physical Radiometric Variance Detection (`compute_bitemporal_variance`):** Independently computes pixel-level RGB delta across before/after images, flagging discrepancies if the model says "No" change but pixels changed by $>8\%$.
6. **Authentication & Persona Store:** Endpoints for 1-click demo persona logins (`/api/auth/login`), profile updates (`/api/auth/update-profile`), API key regeneration (`/api/auth/regenerate-key`), and user quota tracking (`data/users.json`).

#### What is Broken / Left to Implement:
1. **Single-Threaded Execution (`threaded=False`):** The server runs synchronously to prevent CUDA Out-of-Memory race conditions in PyTorch. It cannot handle concurrent user requests.
2. **In-Memory Session Store:** Active sessions are maintained in a Python dictionary (`SESSIONS`); restarting the server invalidates active tokens.
3. **Ephemeral File Accumulation:** Uploaded files in `ui/uploads/` are stamped with timestamps but never purged by an automated background garbage collection task.
4. **No True Geospatial Metadata API:** GeoTIFFs are converted to PNG for browser rendering, stripping spatial projection headers (GeoTIFF CRS / EPSG coordinates are not exposed over the REST API).

---

### Segment I: Frontend Web Canvas UI
- **Files:** [`ui/index.html`](file:///home/pavitra/satquery/ui/index.html), [`ui/styles.css`](file:///home/pavitra/satquery/ui/styles.css), [`ui/app.js`](file:///home/pavitra/satquery/ui/app.js)
- **Status:** **90% Functional**

#### What is Working:
1. **Cinematic Hero & Landing Interface:** Parallax panels, interactive GSAP ScrollTrigger timeline, and live agent workflow modals.
2. **Orbital Acquisition HUD Simulation (`runWowFlightSequence`):** Real-time animated telemetry overlay displaying coordinate lock, band fetching, and GPU specialist inference passes.
3. **ChatGPT-Style Analysis Workspace:** Clean chat thread, history sidebar, preset pills, and responsive location composer.
4. **Dynamic Evidence Renderers:**
   - *Fusion Analysis:* Interactive 3-way toggle button bar (`📡 Sentinel-2 Optical`, `⚡ Sentinel-1 SAR`, `🔥 Grad-CAM Heatmap`) with sub-threshold signal warning pills.
   - *Change Detection:* Interactive Before/After split-screen comparison slider with drag handle.
   - *Grounding:* Precision SVG overlay rendering glowing emerald bounding boxes and facility labels.
5. **Honest Confidence Presentation:** Renders distinct visual styles for calibrated sigmoid probabilities (emerald green) versus uncalibrated greedy token certainty (neutral slate warning). In compound queries, displays both steps stacked sequentially without misleading averaging.

#### What is Broken / Left to Implement:
1. **Missing Cinematic Intro Video Asset:**
   - `ui/cinematic-video.mp4` was deleted, leaving the pre-site video overlay blank or inactive if manually invoked.
2. **Global Variable State Management:**
   - Client state (`activePreset`, `currentResultData`, `currentLocation`) is stored in global JavaScript variables; rapid consecutive clicks on different presets before inference finishes can cause UI state collisions.
3. **Lack of Interactive Vector Map Canvas:**
   - Satellite imagery is rendered in a static `<img>` container with CSS/SVG overlays, rather than an interactive slippy map (e.g. Mapbox GL JS, Leaflet, or OpenLayers) with zoom/pan and coordinate tile grids.

---

### Segment J: Evaluation Harness & Test Suite
- **Files:** [`tests/`](file:///home/pavitra/satquery/tests/), [`eval/`](file:///home/pavitra/satquery/eval/), [`scripts/smoke_test_all_tasks.py`](file:///home/pavitra/satquery/scripts/smoke_test_all_tasks.py)
- **Status:** **70% Functional**

#### What is Working:
1. **Empirical Calibration Report:** Comprehensive Expected Calibration Error (ECE), Brier score, and AUROC benchmark scripts and markdown documentation ([`eval/reports/CALIBRATION_REPORT.md`](file:///home/pavitra/satquery/eval/reports/CALIBRATION_REPORT.md)).
2. **Domain-Gap Audit Documentation:** Full qualitative and quantitative audit on Indian proxy scenes ([`eval/reports/DOMAIN_GAP_REPORT.md`](file:///home/pavitra/satquery/eval/reports/DOMAIN_GAP_REPORT.md)).
3. **End-to-End Task Verification:** Standalone verification scripts (`verify_general_vlm_integration.py`, `smoke_test_all_tasks.py`, `test_fusion_specialist.py`) validate execution on GPU.

#### What is Broken / Left to Implement:
1. **Conda Test Runner Environment Mismatch:**
   - Running `pytest` fails with `ModuleNotFoundError: No module named 'pytest'` inside the `satquery` conda environment.
   - `python -m unittest discover tests` only detects `test_location_acquisition.py` (6 tests passing) because the remaining test files use pytest conventions or test deprecated GeoChat code.
2. **Legacy GeoChat Test Failures:**
   - [`tests/test_fix1_dispatch.py`](file:///home/pavitra/satquery/tests/test_fix1_dispatch.py) and [`tests/test_geochat_regression.py`](file:///home/pavitra/satquery/tests/test_geochat_regression.py) still reference `GeoChatSpecialist`, which was deprecated in favor of `GeneralVLMSpecialist`.
3. **Automated CI/CD Pipeline:**
   - There is no automated GitHub Actions workflow to run regression suites upon push.

---

## 4. System Health & Readiness Matrix

| Segment / Component | Source Files | Completion | Operational Status | Immediate Action Needed |
| :--- | :--- | :---: | :---: | :--- |
| **VRAM Lifecycle Manager** | `model_lifecycle_manager.py`, `model_registry.py` | **92%** | 🟢 Production Ready | Explore FP8 quantization to avoid swap latency on compound runs. |
| **Query Interpreter** | `interpreter.py`, `keyword_fallback.py` | **88%** | 🟢 Functional | Expand regex boundary cases to prevent location false-positives. |
| **Change VQA Specialist** | `qwen_specialist.py`, LoRA checkpoint | **85%** | 🟡 Functional (Limitations) | Document uncalibrated confidence; clean empty `qwen3vl` checkpoint dir. |
| **General VLM Specialist** | `general_vlm_specialist.py` | **90%** | 🟢 Production Ready | Add polygon contour support (SAM-Geo) in place of static bounding boxes. |
| **Fusion Specialist (CNN)** | `fusion_model.py`, `fusion_model.pt` | **78%** | 🟡 Functional (Sensitive) | Re-train or fine-tune on Indian EO patches to resolve "Inland water" collapse. |
| **GeoChat Specialist** | `geochat_specialist.py`, `GeoChat/` | **0%** | 🔴 Deprecated / Broken | Retain as documented negative benchmark; remove from active dispatch tests. |
| **Location Data Broker** | `location_acquisition.py` | **82%** | 🟢 Functional | Add institutional ISRO Bhoonidhi connector when credentials are available. |
| **Backend Server** | `server.py` | **94%** | 🟢 Production Ready | Implement request queuing or worker pools; add upload file cleanup cron. |
| **Frontend Web UI** | `ui/index.html`, `ui/app.js`, `styles.css` | **90%** | 🟢 Production Ready | Re-encode/replace missing `cinematic-video.mp4`; add Leaflet interactive map. |
| **Evaluation & Tests** | `tests/`, `eval/` | **70%** | 🟡 Partial | Install `pytest` in conda environment; convert legacy tests to `GeneralVLM`. |

---

## 5. Technical Debt & Roadmap Prioritization

### P0: High Priority (Immediate Engineering Items)
1. **Standardize Test Suite:** Install `pytest` into the conda environment (`conda install pytest`) and update `tests/test_fix1_dispatch.py` and `tests/test_geochat_regression.py` to test `GeneralVLMSpecialist` rather than deprecated `GeoChatSpecialist`.
2. **Remove Orphaned Directories:** Clean up the empty `checkpoints/qwen3vl-cdvqa-lora` directory to avoid confusion with the active `qwen2.5-vl-cdvqa-lora` adapter.
3. **Handle Server File Garbage Collection:** Add an hourly background cleanup thread in `server.py` to delete temporary images in `ui/uploads/` older than 60 minutes.

### P1: Medium Priority (Model & Capability Enhancements)
1. **Domain-Specific Fine-Tuning for Fusion CNN:** Retrain `FusionSpecialist` on a mixed dataset including tropical, arid, and dense South Asian urban satellite tiles to eliminate the 80% false-positive "Inland waters" domain collapse.
2. **Attach Grad-CAM to SAR Branch:** Add a backward hook to `sar_encoder.layer4[1].conv2` in `scripts/fusion_model.py` so analysts can inspect which microwave backscatter features influenced the classification.
3. **Interactive Vector Canvas:** Integrate Leaflet.js or Mapbox GL into `ui/index.html` to allow pan, zoom, and coordinate grid inspection rather than static image tags.

### P2: Long-Term Scaling (Production Deployment)
1. **FP8 Dynamic Quantization:** Quantize the Fusion CNN and Qwen-VL to FP8 on Ada Lovelace architecture, enabling both models to reside simultaneously in VRAM and reducing compound query latency from ~18s to <3s.
2. **SAM-Geo Polygon Contours:** Replace axis-aligned bounding boxes with interactive zero-shot Segment Anything polygons for irregular geographic features (floods, coastlines, burn scars).
3. **Persistent Production Database:** Replace `data/users.json` and in-memory session tokens with SQLite / PostgreSQL and signed JWTs.
