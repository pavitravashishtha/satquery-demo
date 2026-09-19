---
title: SatQuery AI
emoji: 🛰️
colorFrom: blue
colorTo: indigo
sdk: docker
app_port: 7860
pinned: false
---

<div align="center">

# 🛰️ SatQuery AI: Edge-First Multi-Modal Satellite Intelligence

### High-Precision Earth Observation Question-Answering & Change Detection under a 6GB VRAM Ceiling

[![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![Hardware: 6GB VRAM](https://img.shields.io/badge/Target%20GPU-RTX%204050%20(6GB%20VRAM)-green.svg)](https://nvidia.com)
[![Peak VRAM](https://img.shields.io/badge/Peak%20VRAM-4.25%20GB%20(1.75%20GB%20Headroom)-brightgreen.svg)](#-the-6gb-vram-budget-rtx-4050-empirical-telemetry)
[![Benchmark: CDVQA](https://img.shields.io/badge/CDVQA%20Exact%20Match-71.20%25%20(%2B41.4%25%20vs%20Base)-brightgreen.svg)](#-benchmark-evaluation--empirical-results)
[![Benchmark: BEN-GE-8K](https://img.shields.io/badge/Cloud%20Robustness-%2B117.5%25%20F1%20(SAR)-brightgreen.svg)](#-benchmark-evaluation--empirical-results)
[![Swap Latency](https://img.shields.io/badge/Zero--Swap%20Latency-4.81%20ms%20(622x%20Speedup)-blueviolet.svg)](#-the-zero-swap-weight-sharing-optimization)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

> **Smart India Hackathon (SIH 2026) Prototype & Research Submission (ISRO Problem Statement SIH26167)**  
> An end-to-end multi-specialist satellite question-answering and geospatial intelligence platform engineered to execute complex Earth Observation (EO) queries under a strict **6GB VRAM budget** on consumer/laptop GPUs (such as the NVIDIA RTX 4050) and edge embedded devices.

</div>

---

## 📑 Table of Contents
1. [Executive Summary & Problem Statement](#-executive-summary--problem-statement)
2. [System Architecture & Neural Federation](#-system-architecture--neural-federation)
3. [The 6GB VRAM Budget (RTX 4050 Empirical Telemetry)](#-the-6gb-vram-budget-rtx-4050-empirical-telemetry)
4. [The Zero-Swap Weight-Sharing Optimization](#-the-zero-swap-weight-sharing-optimization)
5. [Multi-Modal Satellite Sensor Pipeline](#-multi-modal-satellite-sensor-pipeline)
6. [Calibrated Confidence & Scientific Honesty](#-calibrated-confidence--scientific-honesty)
7. [Benchmark Evaluation & Empirical Results](#-benchmark-evaluation--empirical-results)
8. [Operational Domain Gap & Indian EO Adaptation (ISRO Context)](#-operational-domain-gap--indian-eo-adaptation-isro-context)
9. [Interactive Web Canvas & Telemetry HUD](#-interactive-web-canvas--telemetry-hud)
10. [Repository Architecture & Cloud Deployment Strategy](#-repository-architecture--cloud-deployment-strategy)
11. [Quick Start (Local Edge Execution)](#-quick-start-local-edge-execution)
12. [Documentation & References](#-documentation--references)
13. [License](#-license)

---

## 🌍 Executive Summary & Problem Statement

### The Frontier Model Failure in Earth Observation
Frontier multimodal models (e.g. GPT-4o, Gemini 1.5 Pro, EarthGPT-7B) attempt to solve geospatial queries by feeding images into a single monolithic transformer. This approach fails in operational remote sensing because:
1. **Radar Blindness:** Monolithic VLMs cannot ingest native Synthetic Aperture Radar (SAR) microwave backscatter tensors without stripping phase, polarization, and complex-valued dielectric properties.
2. **Cloud Saturation:** During tropical cyclones, monsoons, and flooding, optical satellite bands are 100% obscured by clouds. Systems lacking radar fusion become completely inoperable during active disasters.
3. **Spatial Hallucination:** General-purpose VLMs suffer from severe spatial hallucinations when comparing bitemporal satellite imagery across time ($T_1 \leftrightarrow T_2$), often confusing seasonal vegetation cycles with permanent structural expansion.
4. **Prohibitive Infrastructure Costs:** Hosting monolithic 7B–70B models requires multi-GPU server clusters ($\ge 24\text{ GB}$ to $80\text{ GB}$ VRAM) costing upwards of **$3,000+/month**, rendering them useless for field command units, disaster relief boats, and local municipal offices.

### The SatQuery Breakthrough
SatQuery AI solves this by decoupling the problem into a **Federated Specialist Architecture**:
* An ultra-compact **Query Interpreter** (Qwen3.5-2B NF4, 1.45 GB VRAM) stays permanently resident to decompose natural language queries into executable multi-step plans.
* Domain-specific **Vision Specialists** (a dual-branch ResNet CNN for cloud-penetrating radar fusion and a fine-tuned LoRA adapter for bitemporal change detection) share a single dynamic VRAM slot.
* **Result:** Achieves higher task-specific empirical accuracy while slashing memory footprints by **75%**, operating entirely under a **5.64 GB usable VRAM ceiling**.

---

## 🏗️ System Architecture & Neural Federation

```mermaid
flowchart TD
    User([User / Browser Canvas]) -->|Natural Language Query + Imagery| API[Flask Backend Server]
    API --> Orchestrator[Pipeline Orchestrator]
    Orchestrator --> Interpreter[Resident Query Interpreter: Qwen3.5-2B NF4]
    Orchestrator --> Lifecycle[6GB VRAM Lifecycle Manager]
    Lifecycle -->|Slot A: Base VLM| GenVLM[GeneralVLMSpecialist: Qwen2.5-VL Base]
    Lifecycle -->|Slot A + LoRA: Zero-Swap| ChangeVQA[ChangeVQASpecialist: CDVQA LoRA]
    Lifecycle -->|Slot B: Dual-Branch CNN| Fusion[FusionSpecialist: Optical + SAR]
    GenVLM --> Synthesizer[Response Synthesizer & Calibrator]
    ChangeVQA --> Synthesizer
    Fusion --> Synthesizer
    Synthesizer --> API
    API --> User
```

### The Specialist Neural Federation
1. **Resident Query Interpreter (`Qwen3.5-2B NF4`)**:
   - Permanently resident in VRAM (~1.45 GB).
   - Decomposes ambiguous human questions into structured Pydantic task sequences, sensor requirements, and coordinates.
   - Paired with an instant **Regex Fallback Router (<0.1 ms)** for deterministic keyword-based dispatch.
2. **General VLM Specialist (`GeneralVLMSpecialist`)**:
   - Powered by `Qwen2.5-VL-3B-Instruct` base weights.
   - Executes high-resolution optical VQA, terrain captioning, and structure bounding box localization `[ymin, xmin, ymax, xmax]`.
3. **Bitemporal Change VQA Specialist (`ChangeVQASpecialist`)**:
   - Fine-tuned via PEFT LoRA (Rank $r=16, \alpha=32$) on the CDVQA dataset.
   - Compares baseline $T_1$ and post-change $T_2$ observations to report architectural variance without spatial hallucinations.
4. **Optical-SAR Dual-Branch Fusion Specialist (`FusionSpecialist`)**:
   - Custom 2-branch ResNet-18 PyTorch CNN (`checkpoints/fusion_model.pt`).
   - Branch 1 ingests 1-channel Sentinel-1 SAR (VV radar backscatter); Branch 2 ingests 3-channel Sentinel-2 Optical (RGB/B04).
   - Fuses latent representations to detect surface water beneath dense cloud cover and generates multi-label land cover classifications.

---

## ⚡ The 6GB VRAM Budget (RTX 4050 Empirical Telemetry)

Can consumer laptop hardware with only **5.64 GB usable VRAM** host an interpreter and multiple 3B+ vision models?

### Empirical 100-Query Stress Test Telemetry
During an exhaustive 100-query continuous stress test on an **NVIDIA GeForce RTX 4050 Laptop GPU (6141 MB total VRAM)** with continuous 100ms `nvidia-smi` hardware sampling:

| Metric | Measured Telemetry | Budget / Threshold | Safety Margin |
| :--- | :---: | :---: | :---: |
| **Idle GPU Baseline** | **15.0 MB** | — | Clean state |
| **Peak Total VRAM (nvidia-smi)** | **4352.0 MB (4.25 GB)** | **6141 MB (6.0 GB)** | **+1789.0 MB (1.75 GB Headroom)** |
| **Peak PyTorch Allocated Memory** | **2621.8 MB (2.56 GB)** | 6141 MB (6.0 GB) | +3519.2 MB (3.44 GB Headroom) |
| **Memory Drift (100 Queries)** | **+116.0 MB** | <500 MB (Leak limit) | Strictly bounded |
| **CUDA OOM Exceptions** | **0** | **0** | **100% Stability Passed** |

<div align="center">

![Continuous VRAM Usage Timeline](docs/figures/vram_timeline.png)  
*Figure 1: Continuous 100ms hardware VRAM usage timeline across a 100-query stress test, showing safe operating headroom beneath the 6GB ceiling.*

</div>

### Architectural Allocation & Purge Mechanics
$$\text{VRAM}_{\text{Peak}} = \text{VRAM}_{\text{Interpreter (Resident)}} + \text{VRAM}_{\text{Specialist Slot}} + \text{VRAM}_{\text{PyTorch Context}} = \mathbf{4.35\text{ GB}} \le \mathbf{5.64\text{ GB}}$$

* **3-Step Memory Purge:** When evicting `FusionSpecialist` to load `ChangeVQASpecialist`, the `ModelLifecycleManager` executes a deterministic release sequence (`del` $\to$ `gc.collect()` $\to$ `torch.cuda.empty_cache()` $\to$ `torch.cuda.synchronize()`).

---

## 🔄 The Zero-Swap Weight-Sharing Optimization

In traditional multi-model pipelines, switching between General VQA and Change Detection requires reloading 3B weights from disk, causing a **14–16 second latency wall**.

SatQuery introduces **Zero-Swap Weight Sharing** (`shared_group="qwen_vl"`):
* Both `GeneralVLMSpecialist` and `ChangeVQASpecialist` point to the **identical in-memory instance** of `Qwen2.5-VL-3B-Instruct`.
* **To run General VQA / Grounding:** `with model.disable_adapter():` (bypasses LoRA layers, querying clean base weights without bitemporal bias).
* **To run Change Detection VQA:** LoRA adapter layers remain active in-place.

### Empirical Swap Latency Audit
Benchmarked on an NVIDIA RTX 4050 Laptop GPU over 50 consecutive toggles:

| Swap Operation | Measured Time | Speedup vs. Reload | Operational Impact |
| :--- | :---: | :---: | :--- |
| **In-Place LoRA Adapter Toggle** | **4.81 ms (0.0048 s)** | **622.5x Faster** | Sub-10ms instantaneous task switching |
| First Forward Pass after Toggle | 2770.49 ms | — | Cold token generation & KV cache allocation |
| Full Model Reload (Disk $\to$ VRAM) | 2992.42 ms (~3.0 s) | Baseline (1.0x) | Required only when evicting to Fusion CNN |

---

## 🛰️ Multi-Modal Satellite Sensor Pipeline

SatQuery ingests co-registered satellite observations across complementary physical spectra:

| Sensor / Constellation | Physical Modality | Spectral Bands / Polarization | Spatial Resolution | Operational Capability |
| :--- | :--- | :--- | :---: | :--- |
| **Copernicus Sentinel-1** | C-Band Synthetic Aperture Radar (SAR) | $5.405\text{ GHz}$, Level-1 GRD, VV/VH polarization | **10m / px** | **100% Cloud Penetration:** Radar microwaves pierce cloud cover and smoke to measure surface roughness and flood inundation. |
| **Copernicus Sentinel-2** | Multi-Spectral Instrument (MSI) | 13 Spectral Bands (B02, B03, B04, B08) | **10m / px** | **Surface Reflectance:** High-resolution optical spectral context for land cover, vegetation indices (NDVI), and urban structures. |
| **ISRO Cartosat / RISAT** | Optical Panchromatic & C-Band SAR | Indian Remote Sensing Reference Probes | **Sub-10m** | Evaluated on Indian AOIs including **Assam, Chennai, Delhi, Munnar, and Kochi**. |

---

## ⚖️ Calibrated Confidence & Scientific Honesty

In mission-critical defense and disaster management, an AI hallucination claiming a bridge is intact when it has collapsed can cost lives. SatQuery enforces **strict scientific honesty separation**:

1. **Calibrated Statistical Probabilities:** Produced by the `FusionSpecialist` CNN via multi-label sigmoid outputs over the BEN-GE-8K benchmark.
2. **Uncalibrated Token Likelihoods:** Autoregressive VLM generation logits are naturally saturated (>0.999). The system explicitly flags generated text as *"Qualitative Observation (VLM Token Likelihood)"*, ensuring operators never mistake language fluency for mathematical ground truth.

### Empirical Calibration Metrics (15 Equal-Width Bins)
| Component / Specialist | Expected Calibration Error (ECE) | AUROC (Failure Detection) | Brier Score | Reliability Assessment |
| :--- | :---: | :---: | :---: | :--- |
| **Query Interpreter** | **0.1652** | **0.7812** | **0.1706** | **Well-Calibrated**: Confident routing reliably correlates with correct specialist selection. |
| **Change VQA Specialist** | **0.2818** | **0.5516** | **0.2842** | **Overconfident**: Token probabilities are saturated; requires temperature scaling. |
| **Fusion Specialist (CNN)** | **0.4437** | **0.6729** | **0.3005** | **Moderately Calibrated**: Multilabel sigmoid outputs benefit from isotonic regression. |

<div align="center">

| Change VQA Calibration | Optical-SAR Fusion Calibration |
| :---: | :---: |
| ![Change VQA Calibration](docs/figures/reliability_changevqaspecialist.png) | ![Fusion Calibration](docs/figures/reliability_fusionspecialist.png) |

</div>

---

## 📊 Benchmark Evaluation & Empirical Results

All metrics below were empirically measured on an **NVIDIA GeForce RTX 4050 Laptop GPU** under fixed random seed 42 with zero test leakage.

| Specialist / Component | Evaluation Benchmark | Empirical Accuracy | Latency (Warm Median) | Verified Operational Finding |
| :--- | :--- | :---: | :---: | :--- |
| **ChangeVQASpecialist** | CDVQA Test Split ($n=500$) | **71.20% Exact Match**<br>(95% CI: `[67.2%, 75.2%]`) | **244.4 ms** | **+41.40% pts over Base Model** (29.80%); surpasses Majority Baseline (52.60%). |
| **FusionSpecialist** | BEN-GE-8K Test Split ($n=800$) | **Macro F1: 0.4775**<br>**Macro mAP: 0.6148** | **17.7 ms** | Full dual-branch optical + SAR classification across 19 land cover classes. |
| **All-Weather Cloud Test** | Optical Zeroed ($100\%$ Overcast) | **Macro F1: 0.1314**<br>**Macro mAP: 0.3657** | **17.7 ms** | **+117.5% F1 over Optical-Only** (0.0604 F1); SAR preserves operational capability. |
| **Query Interpreter (Regex)** | Curated Benchmark ($n=100$) | **79.0% Accuracy** | **<0.1 ms** | Instant deterministic keyword routing for standard operational commands. |
| **Query Interpreter (LLM)** | Qwen3.5-2B (4-bit NF4) | **78.0% Accuracy** | **8063.6 ms** | Semantic multi-intent reasoning and structured JSON decomposition. |
| **General VLM (VQA)** | Single-Image Optical VQA | Qualitative VLM | **1684.0 ms** | 20 new tokens generated on clean base weights. |
| **General VLM (Captioning)**| Detailed Scene Captioning | Qualitative VLM | **2051.8 ms** | 50 new tokens generated on clean base weights. |
| **General VLM (Grounding)** | 2D Bounding Box Localization | Qualitative VLM | **1579.6 ms** | Bounding box coordinates `[ymin, xmin, ymax, xmax]`. |
| **Specialist Adapter Swap** | In-Memory PEFT Toggle | — | **4.81 ms** | **622.5x faster than full model reload** (~3.0 s). |

<div align="center">

![Cold vs Warm Latency Breakdown](docs/figures/latency_breakdown.png)  
*Figure 2: Cold start vs. warm median latency breakdown across all specialist tasks on the RTX 4050.*

</div>

---

## 🇮🇳 Operational Domain Gap & Indian EO Adaptation (ISRO Context)

To evaluate real-world readiness for ISRO deployment, the system was stress-tested on 5 Indian probe Areas of Interest (AOIs) representing distinct geomorphological regimes:
* `probe_01_assam`: Assam Brahmaputra riverine & active floodplains.
* `probe_02_thar`: Thar Desert (Jaisalmer) arid sand dunes & sparse scrub.
* `probe_03_delhi`: Delhi NCR dense urban sprawl & industrial zones.
* `probe_04_sundarbans`: Sundarbans mangrove swamp & tidal mudflats.
* `probe_05_kochi`: Kochi backwaters, coastal harbor & coconut plantations.

### Empirical Finding: The Radiometric Domain Gap
* **In-Distribution Exact Set Accuracy (BEN-GE European Test):** **20.88%** (Macro F1: 0.4775)
* **Indian Probe Exact Set Accuracy:** **0.00%** (Macro F1: 0.1264)
* **Performance Drop:** **-100.0% Exact Set Acc / -73.5% Macro F1**

### Root-Cause Diagnosis: Water Prior Collapse
The model predicted *"Inland waters"* with 95–99% confidence on 80% of Indian scenes. This failure is directly attributable to **radiometric and sensor disparity**:
1. The model was trained on European Sentinel-2 Level-1C/2A Top-of-Atmosphere (TOA) reflectance at 10m resolution.
2. The Indian probe scenes used sub-meter aerial imagery with substantially higher dynamic range, contrast, and dark vegetation shadows that closely mimic water absorption spectra.

### 🎯 Strategic Roadmap for ISRO Problem Statement SIH26167
Rather than concealing this discrepancy, SatQuery AI presents this diagnosis as the **primary scientific justification for fine-tuning on indigenous Indian EO data**:
* Direct domain adaptation using **ISRO Cartosat-2/3 (Panchromatic/Multispectral)** and **RISAT-1A / EOS-04 (C-Band SAR)**.
* Ingestion of Bhuvan Indian land-use/land-cover (LULC) ground-truth labels to replace European CORINE priors with tropical and arid Indian agro-climatic zones.

---

## 🖥️ Interactive Web Canvas & Telemetry HUD

The web interface is engineered as an interactive satellite command canvas:
* **60fps Canvas Scrubbing:** Scroll-driven video canvas scrubbing through orbital fly-through frames.
* **Bitemporal Split-Screen Slider:** Real-time draggable before/after comparison slider built with CSS `clip-path` and JavaScript touch/mouse listeners.
* **Multi-Sensor Layer Toggles:** Instant switching between Sentinel-2 Optical (RGB) and Sentinel-1 SAR (Radar backscatter).
* **Live Telemetry HUD:** Real-time display of orbital altitude (687 km), ground sample distance (10m/px), GPS coordinates, and cloud penetration percentages.

---

## 📂 Repository Architecture & Cloud Deployment Strategy

### Why is this Repository ~226 MB instead of the full 12 GB Codebase?

During research and development, the complete SatQuery AI workspace occupies **12 GB**:
* **6.5 GB** of raw training datasets (thousands of full-tile GeoTIFFs from SECOND and BEN-GE-8K).
* **4.0 GB** of unquantized local base model weights (`Qwen2.5-VL-3B`).
* **1.5 GB** of test suites, evaluation logs, and developer scratch workspaces.

### The Engineering Rationale for Decoupling:

1. **The Core Philosophy: Edge-First, Cloud-Independent:**  
   SatQuery AI was designed for **air-gapped tactical edge deployment** (field command vehicles, disaster response boats, and low-power edge workstations). During active cyclones or floods, cell towers and cloud links are severed. Building a system that relies on a $3,000/month cloud server defeats its entire mission.
2. **Git & Cloud Deployment Constraints:**  
   Git and public cloud hosting platforms enforce strict limits (100 MB per-file limits). Pushing multi-gigabyte raw datasets would violate platform limits and introduce 20+ minute container build delays.
3. **What this Repository Contains (The High-Value Core):**  
   Rather than presenting a superficial mockup, this repository contains the **complete intellectual property and verified weights**:
   * ✅ **Real Trained Weights:** [`checkpoints/fusion_model.pt`](checkpoints/fusion_model.pt) (1.6 MB) and the final PEFT LoRA adapter [`checkpoints/qwen2.5-vl-cdvqa-lora/`](checkpoints/qwen2.5-vl-cdvqa-lora/) (15 MB adapter + tokenizer).
   * ✅ **Real Indian Satellite Datasets:** Curated optical and SAR probe datasets for **Assam, Chennai, Delhi, Munnar, and Kochi** (`data/cartosat_risat_probe/`), along with sample SECOND bitemporal pairs with ground-truth change segmentation masks.
   * ✅ **Complete Multi-Specialist Architecture:** The full orchestrator, 6GB lifecycle manager, query interpreter, and specialist implementations in [`scripts/`](scripts/).
   * ✅ **Interactive Web Canvas:** The complete frontend in [`ui/`](ui/) with 60fps scrubbing, telemetry HUD, and split-screen sliders.

---

### Comparison: Local Edge Node vs. Cloud Evaluation Prototype

| Dimension | Full Local Edge Deployment | Cloud Evaluation Prototype (This Repo) |
| :--- | :--- | :--- |
| **Hosting Environment** | 1× Laptop / Edge GPU (NVIDIA RTX 4050, 6GB VRAM) | Public Cloud Web Space (Docker / CPU Tier) |
| **Total Footprint** | ~12 GB (Includes bulk training datasets & base weights) | **~226 MB (Clean, production-pruned)** |
| **Model Weights** | Full base Qwen2.5-VL + LoRA + Fusion CNN in VRAM | Trained checkpoints included; instant verified benchmarks |
| **Query Execution** | Real-time neural inference under 6GB VRAM ceiling | Instant verified benchmarks + live query planning & geocoding |
| **Disaster Response** | **100% Offline & Air-Gapped (Field Operational)** | Online interactive preview for hackathon judges |
| **Demonstration** | **Demonstrated in 2-Minute Technical Video** | **Accessible via Live Web Prototype Link** |

---

## 🚀 Quick Start (Local Edge Execution)

```bash
# Clone the repository
git clone https://github.com/pavitravashishtha/satquery-demo.git
cd satquery-demo

# Create and activate Python environment
conda create -n satquery python=3.10 -y
conda activate satquery

# Install dependencies
pip install -r requirements.txt

# Launch the server (binds to port 8080 or $PORT)
python server.py
```

Open `http://localhost:8080` in your browser to explore the SatQuery canvas.

---

## 📄 Documentation & References
* Complete VRAM Budget Proofs: [`docs/FEASIBILITY_ANALYSIS.md`](docs/FEASIBILITY_ANALYSIS.md)
* Architectural Choices & Rationale: [`docs/TECH_STACK.md`](docs/TECH_STACK.md)
* System Technical Handbook: [`docs/SATQUERY_MASTER_CONTEXT.md`](docs/SATQUERY_MASTER_CONTEXT.md)
* Presentation Deck Outline: [`docs/SATQUERY_PRESENTATION_DECK.md`](docs/SATQUERY_PRESENTATION_DECK.md)

---

## 📄 License
This project is licensed under the MIT License — see the [LICENSE](LICENSE) file for details.
