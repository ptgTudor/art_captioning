# WikiArt multitask BLIP (server-ready)

This repo contains **server-friendly .py scripts** to train a BLIP-based model on WikiArt with:

- **Primary:** caption generation that includes **Genre + Style + (optional) Title + Description**
- **Auxiliary:** **artist identification** as a classification head (low-weight)
- **Also supported:** genre/style classification heads

The goal is to **improve genre/style understanding while keeping captions long and image-grounded**.

---

## 0) Install

```bash
pip install -r requirements.txt
```

---

## 1) Preprocess the dataset

```bash
python3 preprocess_wikiart.py \
  --output_dir data/wikiart_proc \
  --pseudo_model_name Salesforce/blip-image-captioning-base \
  --pseudo_batch_size 32 \
  --pseudo_max_new_tokens 30 \
```

---

## 2) Train (multitask)

```bash
python3 train_multitask_blip.py \
  --data_dir data/wikiart_proc \
  --output_dir runs/blip_multitask \
  --model_name Salesforce/blip-image-captioning-base \
  --num_train_epochs 10 \
  --per_device_train_batch_size 16 \
  --gradient_accumulation_steps 2 \
  --learning_rate 5e-5 \
  --fp16 \
  --genre_loss_weight 0.5 \
  --style_loss_weight 0.7 \
  --artist_loss_weight 0.05
```

---

## 3) Evaluate

```bash
python3 evaluate_multitask_blip.py \
  --model_dir runs/blip_multitask \
  --data_dir data/wikiart_proc \
  --split validation \
  --compute_sbert
```

## 4) Analyze results

```bash
python3 analyze_eval_results.py \
    --eval_pkl runs/blip_multitask/eval_validation_results.pkl \
    --data_dir data/wikiart_proc \
    --split validation \
    --out_dir runs/blip_multitask/analysis \
    --task style \
    --k 10
```

## 5) Infer from a single image for future use

```bash
python3 infer_single.py \
  --model_dir runs/blip_multitask \
  --image_path /path/to/painting.jpg
```

---
