# SatQuery AI: Comprehensive Technology Stack Reference

> **Document Type:** System Technology Stack & Architectural Dependencies  
> **Target Audience:** Engineering Team, Technical Reviewers, PPT Builders  
> **Last Updated:** September 2026  

---

## Technology Stack Overview at a Glance

```
+---------------------------------------------------------------------------------------+
| SATQUERY AI TECHNOLOGY STACK                                                          |
+---------------------------------------------------------------------------------------+
| APPLICATION LAYER     | Vanilla HTML5, Vanilla CSS3 (Glassmorphism), Vanilla JS (ES6+) |
| ANIMATION ENGINE      | GSAP 3.15.0 + GSAP ScrollTrigger (3D parallax, cinematic HUD) |
| BACKEND WEB SERVICE   | Python 3.10, Flask, Flask-CORS, REST API                      |
| REASONING & ROUTING   | Qwen3.5-2B (4-bit NF4 Quantized) via HuggingFace Transformers |
| MULTIMODAL VLM CORE   | Qwen2.5-VL-3B-Instruct (Vision-Language Transformer)          |
| ADAPTER FINE-TUNING   | PEFT LoRA (Rank 16, Alpha 32, CDVQA bitemporal tuning)        |
| DUAL-BRANCH RADAR CNN | Custom 2-Branch ResNet-18 (PyTorch) + Native Grad-CAM Hook    |
| QUANTIZATION RUNTIME  | BitsAndBytes (NF4, Double Quantization, FP16 compute)         |
| IMAGE / RASTER MATH   | NumPy, Pillow (PIL), GeoTIFF streaming engine                 |
| DATA VALIDATION       | Pydantic v2 (Strict schema enforcement)                       |
| VRAM LIFECYCLE MGR    | Custom single-swap-slot scheduler with Zero-Swap weight group |
| PERSISTENCE STORE     | JSON flat-file storage (data/users.json) + localStorage       |
| TARGET HARDWARE       | NVIDIA RTX 4050 Laptop GPU (5.64 GB usable VRAM / 6GB budget) |
+---------------------------------------------------------------------------------------+
```

---

## 1. Deep Learning Frameworks & AI Runtimes

| Technology / Library | Version / Spec | Role in SatQuery AI | Why It Was Chosen |
| :--- | :--- | :--- | :--- |
| **PyTorch** | `v2.2+` (CUDA 12.x) | Core deep learning framework, autograd engine, tensor manipulation. | Industry standard; enables native backward hooks for Grad-CAM and granular CUDA memory management (`empty_cache`, `synchronize`). |
| **Hugging Face Transformers** | `v4.39+` | Model loading, tokenization, dynamic chat templating, generation loops. | Direct support for `Qwen2.5-VL` and `Qwen3.5` architecture families with native vision processing. |
| **PEFT (Parameter-Efficient Fine-Tuning)** | Latest | LoRA (Low-Rank Adaptation) injection and adapter toggling. | Allows adapting the 3B vision model to CDVQA without retraining base weights; enables instant adapter toggling (`enable_adapter()` / `disable_adapter()`). |
| **BitsAndBytes (`bitsandbytes`)** | `v0.43+` | 4-bit NormalFloat4 (`nf4`) quantization and double quantization. | Compresses Qwen3.5-2B into **1.45 GB VRAM**, allowing the Query Interpreter to stay permanently resident on GPU alongside vision specialists. |
| **Torchvision** | Latest | Standard ResNet-18 vision backbones, 2D convolution layers. | Provides stable, pre-trained feature extraction backbones for optical and radar sensor branches. |

---

## 2. Neural Model Architectures & Specialists

| Model | Checkpoint / Base | Parameters & Quantization | Function / Modality | Output Contract |
| :--- | :--- | :--- | :--- | :--- |
| **Resident Query Interpreter** | `Qwen/Qwen3.5-2B` | 2.0 Billion params<br>• 4-bit NF4 quantized<br>• ~1.45 GB VRAM | Decomposes raw natural language into structured Pydantic task sequences, sensor requirements, and coordinates. | Structured `InterpretedQuery` JSON (`task_sequence`, `location`, `sensors`). |
| **ChangeVQASpecialist** | `Qwen2.5-VL-3B-Instruct`<br>+ CDVQA LoRA | 3.0 Billion params<br>• 4-bit / 16-bit LoRA<br>• Rank $r=16, \alpha=32$ | Ingests paired pre-change (`im1`) and post-change (`im2`) optical images to answer comparative change questions. | Natural language change answer + token certainty metric (`72.5% Val Acc`). |
| **GeneralVLMSpecialist** | `Qwen2.5-VL-3B-Instruct`<br>(Base weights, LoRA bypassed) | 3.0 Billion params<br>• Shares weights with ChangeVQA<br>• **0 MB extra VRAM** | Handles single-image optical VQA, dense aerial scene captioning, and structure bounding box localization. | Natural language text + normalized coordinates `[ymin, xmin, ymax, xmax]`. |
| **FusionSpecialist** | Custom 2-Branch ResNet-18<br>(`checkpoints/fusion_model.pt`) | ~24 Million params<br>• FP32 PyTorch weights<br>• ~1.64 GB VRAM | Dual-sensor fusion: Branch 1 ingests 1-channel Sentinel-1 SAR (VV); Branch 2 ingests 3-channel Sentinel-2 Optical (RGB/B04). | Multi-label sigmoid probabilities (calibrated) + Grad-CAM 2D float heatmap. |

---

## 3. Remote Sensing Satellites, Sensors & Benchmarks

| Sensor / Dataset | Source Agency | Modality & Specs | Role in System |
| :--- | :--- | :--- | :--- |
| **Sentinel-1 SAR** | European Space Agency (ESA) | C-Band ($5.405\text{ GHz}$) Synthetic Aperture Radar, Level-1 GRD, VV polarization, 10m/px. | Cloud-penetrating radar backscatter sensing surface water roughness and moisture. |
| **Sentinel-2 MSI** | European Space Agency (ESA) | Multi-Spectral Instrument, 13 spectral bands, Level-2A BOA reflectance, 10m/px (B02, B03, B04, B08). | High-resolution optical spectral context for land cover and terrain features. |
| **CDVQA Dataset** | Academic Benchmark | Bitemporal satellite image pairs + comparative question-answer annotations. | Training and evaluation benchmark for `ChangeVQASpecialist` (72.5% validation accuracy). |
| **BEN-GE-8K** | DLR / TU Berlin Benchmark | Co-registered Sentinel-1 SAR and Sentinel-2 Optical GeoTIFF tiles with multi-label land-cover ground truth (BigEarthNet-S2/S1). | Training and evaluation benchmark for `FusionSpecialist`. |
| **SECOND Dataset** | Semantic Change Detection | Aerial bitemporal image pairs (`im1`, `im2`) of urban, agricultural, and natural terrain. | Evaluation imagery for live change detection and interactive slider rendering. |

---

## 4. Backend Web Service & Data Processing

| Technology | Purpose & Implementation |
| :--- | :--- |
| **Python 3.10+** | Foundation language powering the entire server, machine learning pipeline, and evaluation harnesses. |
| **Flask** | Micro-framework providing the RESTful API server (`server.py`), handling CORS, multipart file uploads, and routing. |
| **Pydantic v2** | Strict validation of query schemas, guaranteeing that downstream dispatchers receive type-safe execution trees. |
| **Pillow (PIL)** | Ingestion, dynamic conversion (`RGB`, `I;16`, `F`), and resizing of raw satellite rasters. |
| **NumPy** | High-performance 2D float array math for Grad-CAM gradient pooling, array normalization, and algorithmic colormap generation. |
| **Dynamic GeoTIFF Engine** | Custom on-the-fly streaming endpoint (`/api/image`) converting 16-bit Sentinel GeoTIFFs into 8-bit web-optimized PNG previews. |
| **Pure NumPy Jet Colormap** | Converts 2D activation arrays into RGBA Jet-colormap Base64 data URLs without requiring heavy graphic dependencies like Matplotlib. |

---

## 5. Frontend Canvas & User Experience (UX)

| Technology | Implementation & Function |
| :--- | :--- |
| **HTML5 & Semantic Markup** | Structured shell containing the landing page, chat interface, modal dialogs, and SVG layers. |
| **Vanilla CSS3 (Design System)** | Custom dark-teal glassmorphism palette (`#062629`, `#10b981`), zero CSS framework overhead (no Tailwind/Bootstrap dependency), full responsive flexbox/grid layouts. |
| **Vanilla JavaScript (ES6+)** | Handles asynchronous fetch dispatch, state synchronization, typewriter text streaming, dynamic SVG rectangle rendering, and 4-file upload bundling. |
| **GSAP 3.15.0** | GreenSock Animation Platform powering 3D perspective tilts, smooth modal entrance transitions, and interactive toast alerts. |
| **GSAP ScrollTrigger** | Pin-sharp frame scrubbing on the landing page for interactive satellite flyover animation. |
| **Interactive Before/After Slider** | Built with CSS `clip-path: polygon()` and JavaScript input event listeners for real-time comparative inspection. |

---

## 6. Memory, Lifecycle & Session Management

| System | Architecture & Implementation |
| :--- | :--- |
| **ModelLifecycleManager** | Custom single-swap-slot scheduler enforcing a strict $\le 5.64\text{ GB}$ VRAM ceiling. Implements resident vs swappable policies. |
| **Zero-Swap Weight Grouping** | Uses group identifier `shared_group="qwen_vl"` to bypass model reloads when switching between `GeneralVLMSpecialist` and `ChangeVQASpecialist` ($0.0001\text{s}$ swap latency). |
| **Deterministic VRAM Purge** | Automated 3-step sequence (`del` $\to$ `gc.collect()` $\to$ `torch.cuda.empty_cache()`) executed on every slot eviction to prevent memory fragmentation. |
| **User Persistence Store** | JSON flat-file storage in `data/users.json` tracking user profiles, roles, and query quotas across sessions. |
| **Client Session Sync** | Combines HTTP Authorization headers (`Bearer <token>`) with client-side `localStorage` persistence. |

---

## 7. Development & Deployment Environment

| Layer | Environment Specification |
| :--- | :--- |
| **Operating System** | Linux (Ubuntu 22.04 LTS / Kernel 6.x) |
| **Package Manager** | Miniconda3 (`satquery` conda environment with isolated CUDA 12 runtimes) |
| **Target GPU** | NVIDIA GeForce RTX 4050 Laptop GPU (6 GB VRAM physical, 5.64 GB usable) |
| **Development IDE** | Google Antigravity IDE (Multi-agent pair programming workspace) |
| **Target Edge Platforms** | NVIDIA Jetson AGX Orin / Orin Nano (TensorRT-LLM, ONNX Runtime) |
