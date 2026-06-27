<!-- # Resonant Brane Splatting for Arbitrary-Scale Super-Resolution


# Configurazione ambiente conda
conda create -n rbs_env python=3.11 -y
conda activate rbs_env

installare pytorch (specifica per l'architettura tua)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124
pip install -r requirements.txt
pip install -e .

# Installa rasterizzatore CUDA
cd utils/brane_utils
python setup_branecuda.py install

# Scarica checkpoint
## EDSR
1) scarica checkpoint
gdown 1icbigtNcfQnFqySFXcdpamzOkwywLRZm -O checkpoints/EDSR_rbs_v8.ckpt

## RDN
gdown 1i7bC6Lid1_q-VW9r-XQca4nCIh5OuSqO -O checkpoints/RDN_rbs_v7.ckpt

# Inference 

## Inferenza su singola immagine LR
    1) Metti l'immagine dentro una cartella. C'è già un esempio dentro: data/demo_lr_images/parrot.jpg

    2) Lancia inferenza:
    CUDA_VISIBLE_DEVICES=5 python inference/evaluate_inference.py \
    --path_to_lr_image ./data/demo_lr_images/parrot.jpg \
    --scale_to_evaluate 2 \
    --results-dir ./results \
    --ckpt_path  ./checkpoints/RDN_rbs_v7.ckpt \
    --model_name RBS_v7

## Esempio su DIV2K croppato (centralmente) a 720x720 per testare i tempi di inferenza e consumo gpu

1) Scarica il dataset zippato che è stato caricato su google drive:
    gdown 1k1F2OnCyi_Z0_kJCcocGGAAbJNUW7bBG -O data/benchmarks/DIV2K_720_cropped_benchmark.zip
    unzip data/benchmarks/DIV2K_720_cropped_benchmark.zip -d data/benchmarks/DIV2K_720_cropped_benchmark

2) Lancia l'inferenza sul benchmark (per esempio usando il modello con EDSR (che richiede modello RBS_v8))
    CUDA_VISIBLE_DEVICES=5 python inference/evaluate_inference.py \
    --path_to_image_dataset ./data/benchmarks/DIV2K_720_cropped_benchmark/DIV2K \
    --scale_to_evaluate 4 \
    --results-dir ./results \
    --ckpt_path  ./checkpoints/EDSR_rbs_v8.ckpt \
    --model_name RBS_v8

NB: nel paper i test sono stati eseguiti su una singola H100.

## Valutazione delle metriche di qualità (PSNR, SSIM, LPIPS e DISTS)
NB: usiamo come esempio i risultati ottenuti nella sezione "Esempio su DIV2K croppato (centralmente) a 720x720 per testare i tempi di inferenza e consumo gpu" ma questo NON è un benchmark ufficiale in quanto è servito solo a calcolare i costi computazionali. Scaricare i benchmark ufficiali.

## PSNR and SSIM
python inference/evaluate_metrics.py \
--gt data/benchmarks/DIV2K_720_cropped_benchmark/DIV2K/x4/GT \
--restored results/DIV2K/x4 \
--scale 4

## LPIPS
python inference/evaluate_metrics_lpips.py \
--gt data/benchmarks/DIV2K_720_cropped_benchmark/DIV2K/x4/GT \
--restored results/DIV2K/x4 \
--scale 4

## DISTS
python inference/evaluate_metrics_dists.py \
--gt data/benchmarks/DIV2K_720_cropped_benchmark/DIV2K/x4/GT \
--restored results/DIV2K/x4 \
--scale 4

dove --gt è il path verso le immagini di ground truth, mentre --restored quelle predette. -->



<p align="center">
  <img src="other_files/icon.png" height="90" alt="RBS icon"/>
</p>

<h1 align="center">Resonant Brane Splatting for Arbitrary-Scale Super-Resolution</h1>

<p align="center">
  <b>Arbitrary-Scale Super-Resolution (ASR)</b> reconstructs images at continuous magnification factors.
  Recent methods accelerate inference by replacing computationally heavy implicit neural decoders with
  explicit <b>2D Gaussian Splatting (GS)</b>. However, since standard Gaussians are smooth low-pass
  primitives, modeling edges and fine textures requires multiple overlapping, well-aligned splats,
  creating severe bottlenecks during rasterization.
</p>

<p align="center">
  To address this, we introduce <b>Resonant Brane Splatting (RBS)</b>, a feed-forward ASR framework.
  RBS replaces flat Gaussians with <b>Branes</b> — expressive primitives that emit spatially varying
  colors to natively model local contrast and complex textures within a single footprint.
  We achieve this by augmenting the standard Gaussian envelope with internal
  <b>Gaussian-Hermite modes</b>, assigning a distinct color coefficient to each:
  the zero-order mode recovers standard GS, while higher-order modes capture high frequencies.
</p>

<p align="center">
  Because Branes provide a mathematically richer formulation than simple Gaussians, far fewer
  primitives need to overlap to reconstruct a given target pixel. We exploit this with an
  <b>efficient fully differentiable rasterizer</b> featuring a precise culling strategy based
  on the classical <em>quantum turning point</em>, drastically reducing rendering overhead.
  Experiments on standard ASR benchmarks show RBS improves reconstruction quality over implicit
  and GS baselines, while achieving a superior speed–quality trade-off than prior GS methods.
</p>

---

## 📋 Table of Contents

- [Environment Setup](#-environment-setup)
- [Installation](#-installation)
- [Download Checkpoints](#-download-checkpoints)
- [Inference](#-inference)
  - [Single Image](#single-image-inference)
  - [DIV2K Benchmark](#div2k-cropped-benchmark)
- [Quality Metrics](#-quality-metrics-evaluation)
  - [PSNR & SSIM](#psnr--ssim)
  - [LPIPS](#lpips)
  - [DISTS](#dists)

---

## 🐍 Environment Setup

Create and activate a dedicated Conda environment with **Python 3.11**:

```bash
conda create -n rbs_env python=3.11 -y
conda activate rbs_env
```

---

## ⚙️ Installation

### 1 — Install PyTorch

> Adjust the `--index-url` to match your local CUDA version. The example below targets **CUDA 12.4**:

```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124
```

### 2 — Install project dependencies

```bash
pip install -r requirements.txt
pip install -e .
```

### 3 — Build the CUDA Brane Rasterizer

```bash
cd utils/brane_utils
python setup_branecuda.py install
```

---

## 📦 Download Checkpoints

Use [`gdown`](https://github.com/wkentaro/gdown) to fetch the pre-trained weights from Google Drive.

### EDSR backbone — `RBS_v8`

```bash
gdown 1icbigtNcfQnFqySFXcdpamzOkwywLRZm -O checkpoints/EDSR_rbs_v8.ckpt
```

### RDN backbone — `RBS_v7`

```bash
gdown 1i7bC6Lid1_q-VW9r-XQca4nCIh5OuSqO -O checkpoints/RDN_rbs_v7.ckpt
```

---

## 🚀 Inference

### Single Image Inference

**Step 1** — Place your low-resolution image in a folder.
A sample image is already included at:

```
data/demo_lr_images/parrot.jpg
```

**Step 2** — Run inference (example with `RDN_rbs_v7`, scale ×2):

```bash
CUDA_VISIBLE_DEVICES=5 python inference/evaluate_inference.py \
    --path_to_lr_image   ./data/demo_lr_images/parrot.jpg \
    --scale_to_evaluate  2 \
    --results-dir        ./results \
    --ckpt_path          ./checkpoints/RDN_rbs_v7.ckpt \
    --model_name         RBS_v7
```

---

### DIV2K Cropped Benchmark

> This benchmark uses **720×720 centre-cropped** DIV2K images to measure inference speed and GPU memory usage.
> It is **not** an official benchmark — use the official DIV2K benchmark for paper comparisons.

**Step 1** — Download and extract the benchmark dataset:

```bash
gdown 1k1F2OnCyi_Z0_kJCcocGGAAbJNUW7bBG -O data/benchmarks/DIV2K_720_cropped_benchmark.zip

unzip data/benchmarks/DIV2K_720_cropped_benchmark.zip \
      -d data/benchmarks/DIV2K_720_cropped_benchmark
```

**Step 2** — Run benchmark inference (example with `EDSR_rbs_v8`, scale ×4):

```bash
CUDA_VISIBLE_DEVICES=5 python inference/evaluate_inference.py \
    --path_to_image_dataset ./data/benchmarks/DIV2K_720_cropped_benchmark/DIV2K \
    --scale_to_evaluate     4 \
    --results-dir           ./results \
    --ckpt_path             ./checkpoints/EDSR_rbs_v8.ckpt \
    --model_name            RBS_v8
```

> **📝 Note:** All paper results were obtained on a single **NVIDIA H100** GPU.

---

## 📊 Quality Metrics Evaluation

The examples below use the outputs from the DIV2K cropped benchmark.
`--gt` points to the **ground-truth** images; `--restored` to the **model predictions**.

### PSNR & SSIM

```bash
python inference/evaluate_metrics.py \
    --gt        data/benchmarks/DIV2K_720_cropped_benchmark/DIV2K/x4/GT \
    --restored  results/DIV2K/x4 \
    --scale     4
```

### LPIPS

```bash
python inference/evaluate_metrics_lpips.py \
    --gt        data/benchmarks/DIV2K_720_cropped_benchmark/DIV2K/x4/GT \
    --restored  results/DIV2K/x4 \
    --scale     4
```

### DISTS

```bash
python inference/evaluate_metrics_dists.py \
    --gt        data/benchmarks/DIV2K_720_cropped_benchmark/DIV2K/x4/GT \
    --restored  results/DIV2K/x4 \
    --scale     4
```

| Argument | Description |
|---|---|
| `--gt` | Path to ground-truth HR images |
| `--restored` | Path to super-resolved (predicted) images |
| `--scale` | Upscaling factor used during inference |

---

## 🗂️ Project Structure (quick reference)

```
.
├── checkpoints/          # Pre-trained model weights
├── data/
│   ├── demo_lr_images/   # Sample LR input images
│   └── benchmarks/       # Downloaded benchmark datasets
├── inference/
│   ├── evaluate_inference.py
│   ├── evaluate_metrics.py
│   ├── evaluate_metrics_lpips.py
│   └── evaluate_metrics_dists.py
├── utils/
│   └── brane_utils/      # CUDA rasterizer source & setup
├── requirements.txt
└── setup.py
```

---

## 📌 Quick-Reference: Model & Checkpoint Table

| Model Name | Backbone | Checkpoint File 
|---|---|---|
| `RBS_v7` | RDN | `RDN_rbs_v7.ckpt` |
| `RBS_v8` | EDSR | `EDSR_rbs_v8.ckpt` |
