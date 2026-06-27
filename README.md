# Resonant Brane Splatting for Arbitrary-Scale Super-Resolution


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

dove --gt è il path verso le immagini di ground truth, mentre --restored quelle predette.