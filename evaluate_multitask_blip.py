"""
Evaluate a multitask BLIP checkpoint trained with 'train_multitask_blip.py'

Steps
1) parse CLI args
2) load the preprocessed dataset from disk (train or validation split)
3) load processor and multitask BLIP model from model_dir and move to device
4) optionally load an SBERT model for semantic similarity on Description text
5) compute classification-head accuracies only:
   - run the vision model to get pooled features
   - apply linear heads and compare to ground-truth IDs
6) compute generation + parsing metrics:
   - generate text with 'model.generate(...)'
   - parse genre/style via regex patterns
   - measure format validity and exact-match accuracy
7) print results and show a few examples

Usage:
  python3 evaluate_multitask_blip.py \
  --model_dir runs/blip_multitask \
  --data_dir data/wikiart_proc \
  --split validation \
  --compute_sbert
"""

import argparse
import re
import os
import numpy as np
from tqdm import tqdm
import torch
import torch.nn as nn
from datasets import load_from_disk
from transformers import BlipProcessor, BlipConfig, BlipForConditionalGeneration
from sentence_transformers import SentenceTransformer
from sklearn.metrics import f1_score
import pickle
import json


# Model (same as training)

class BlipForConditionalGenerationMultiTask(BlipForConditionalGeneration):
    def __init__(self, config: BlipConfig):
        super().__init__(config)
        
        # storing the number of classes and loss weights in the model config so they
        # are available in config.json when saving/loading checkpoints

        # custom config fields
        self.num_artists = int(getattr(config, "num_artists", 0) or 0)
        self.num_genres  = int(getattr(config, "num_genres", 0) or 0)
        self.num_styles  = int(getattr(config, "num_styles", 0) or 0)

        self.artist_loss_weight = float(getattr(config, "artist_loss_weight", 0.05))
        self.genre_loss_weight  = float(getattr(config, "genre_loss_weight", 0.50))
        self.style_loss_weight  = float(getattr(config, "style_loss_weight", 0.70))

        vision_hidden = getattr(config.vision_config, "hidden_size", None) # get output vector size
        if vision_hidden is None:
            vision_hidden = self.vision_model.config.hidden_size # if nonexistent, get it from the current instance
            
        # classification heads
        self.artist_classifier = nn.Linear(vision_hidden, self.num_artists) if self.num_artists > 0 else None
        self.genre_classifier  = nn.Linear(vision_hidden, self.num_genres) if self.num_genres > 0 else None
        self.style_classifier  = nn.Linear(vision_hidden, self.num_styles) if self.num_styles > 0 else None

    # classification heads run on a pooled vision representation
    # prefer pooler_output if available; otherwise use the CLS token
    def get_image_features(self, pixel_values: torch.Tensor) -> torch.Tensor:
        vision_out = self.vision_model(pixel_values=pixel_values, return_dict=True)
        if hasattr(vision_out, "pooler_output") and vision_out.pooler_output is not None:
            return vision_out.pooler_output
        return vision_out.last_hidden_state[:, 0]


# Parsing helpers

# regex patterns to extract structured fields from generated text
GENRE_RE = re.compile(r"genre\s*:\s*([^\.]+)", re.IGNORECASE)
STYLE_RE = re.compile(r"style\s*:\s*([^\.]+)", re.IGNORECASE)
DESC_RE  = re.compile(r"description\s*:\s*(.+)$", re.IGNORECASE)

# extract a named field from the generated string using a regex
def extract_field(text, pattern):
    if not text:
        return ""
    m = pattern.search(text)
    if not m:
        return ""
    return m.group(1).strip().strip('"').strip()

# normalize strings for more robust exact-match comparisons
def normalize_text(s: str) -> str:
    s = (s or "").strip().lower()
    s = re.sub(r"\s+", " ", s)
    return s


# Main

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_dir", type=str, required=True)
    ap.add_argument("--data_dir", type=str, required=True)
    ap.add_argument("--split", type=str, default="validation", choices=["train", "validation"])
    ap.add_argument("--max_samples", type=int, default=None)

    ap.add_argument("--batch_size", type=int, default=32)
    ap.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")

    # generation
    ap.add_argument("--prompt", type=str, default="Genre:")
    ap.add_argument("--gen_max_new_tokens", type=int, default=80)
    ap.add_argument("--gen_num_beams", type=int, default=5)
    ap.add_argument("--no_repeat_ngram_size", type=int, default=3)
    ap.add_argument("--repetition_penalty", type=float, default=1.2)

    # SBERT
    ap.add_argument("--compute_sbert", action="store_true")
    ap.add_argument("--sbert_model", type=str, default="sentence-transformers/all-MiniLM-L6-v2")

    ap.add_argument("--show_examples", type=int, default=3)

    args = ap.parse_args()

    # load dataset and subset
    ds = load_from_disk(args.data_dir)
    split = ds[args.split]
    label_maps_path = os.path.join(args.data_dir, "label_maps.json")
    with open(label_maps_path, "r") as f:
        label_maps = json.load(f)
    # optional sampling
    if args.max_samples is not None and len(split) > args.max_samples:
        split = split.shuffle(seed=42).select(range(args.max_samples))

    # load processor and model
    processor = BlipProcessor.from_pretrained(args.model_dir)
    model = BlipForConditionalGenerationMultiTask.from_pretrained(args.model_dir)
    model.to(args.device)
    model.eval()

    # SBert
    sbert = None
    if args.compute_sbert:
        sbert = SentenceTransformer(args.sbert_model, device=args.device)

    # dataloader batching
    def batch_iter():
    	bs = args.batch_size
    	n = len(split)
    	for i in range(0, n, bs):
        	idxs = list(range(i, min(i + bs, n)))
        	yield split.select(idxs) # yields a Dataset (iterable over row dicts)


    # collect preds/targets for F1
    a_preds_all, a_true_all = [], []
    g_preds_all, g_true_all = [], []
    s_preds_all, s_true_all = [], []
    a_conf_all, g_conf_all, s_conf_all = [], [], []

    # classification heads accuracies
    correct_artist = 0
    correct_genre  = 0
    correct_style  = 0
    total = 0

    for batch in tqdm(batch_iter(), total=(len(split) + args.batch_size - 1)//args.batch_size, desc="classification"):
        images = [img.convert("RGB") for img in batch["image"]]
        pixel = processor(images=images, return_tensors="pt")["pixel_values"].to(args.device)

        with torch.no_grad():
            feats = model.get_image_features(pixel)
            if model.artist_classifier is not None:
                a_logits = model.artist_classifier(feats)
                a_probs = torch.softmax(a_logits, dim=-1)
                a_pred = a_probs.argmax(dim=-1).cpu().numpy()
                a_conf = a_probs.max(dim=-1).values.cpu().numpy()
            else:
                a_pred = np.zeros(len(images), dtype=np.int64)
                a_conf = np.zeros(len(images), dtype=np.float32)

            if model.genre_classifier is not None:
                g_logits = model.genre_classifier(feats)
                g_probs = torch.softmax(g_logits, dim=-1)
                g_pred = g_probs.argmax(dim=-1).cpu().numpy()
                g_conf = g_probs.max(dim=-1).values.cpu().numpy()
            else:
                g_pred = np.zeros(len(images), dtype=np.int64)
                g_conf = np.zeros(len(images), dtype=np.float32)

            if model.style_classifier is not None:
                s_logits = model.style_classifier(feats)
                s_probs = torch.softmax(s_logits, dim=-1)
                s_pred = s_probs.argmax(dim=-1).cpu().numpy()
                s_conf = s_probs.max(dim=-1).values.cpu().numpy()
            else:
                s_pred = np.zeros(len(images), dtype=np.int64)
                s_conf = np.zeros(len(images), dtype=np.float32)

        a_true = np.array([int(artist_id) for artist_id in batch["artist_id"]], dtype=np.int64)
        g_true = np.array([int(genre_id) for genre_id in batch["genre_id"]], dtype=np.int64)
        s_true = np.array([int(style_id) for style_id in batch["style_id"]], dtype=np.int64)
        
        # store for F1 computation (only if the head exists)
        if model.artist_classifier is not None:
            a_preds_all.append(a_pred)
            a_true_all.append(a_true)
            a_conf_all.append(a_conf)

        if model.genre_classifier is not None:
            g_preds_all.append(g_pred)
            g_true_all.append(g_true)
            g_conf_all.append(g_conf)

        if model.style_classifier is not None:
            s_preds_all.append(s_pred)
            s_true_all.append(s_true)
            s_conf_all.append(s_conf)


        correct_artist += int((a_pred == a_true).sum())
        correct_genre  += int((g_pred == g_true).sum())
        correct_style  += int((s_pred == s_true).sum())
        total += batch.num_rows

    head_artist_acc = correct_artist / total
    head_genre_acc  = correct_genre / total
    head_style_acc  = correct_style / total
    
    def compute_f1s(true_list, pred_list):
        if not true_list:  # head missing
            return None, None
        y_true = np.concatenate(true_list, axis=0)
        y_pred = np.concatenate(pred_list, axis=0)
        f1_macro = f1_score(y_true, y_pred, average="macro", zero_division=0)
        f1_weighted = f1_score(y_true, y_pred, average="weighted", zero_division=0)
        return f1_macro, f1_weighted

    artist_f1_macro, artist_f1_weighted = compute_f1s(a_true_all, a_preds_all)
    genre_f1_macro,  genre_f1_weighted  = compute_f1s(g_true_all, g_preds_all)
    style_f1_macro,  style_f1_weighted  = compute_f1s(s_true_all, s_preds_all)

    # generation
    valid = 0
    gen_genre_correct = 0
    gen_style_correct = 0
    gen_total = 0
    lengths = []

    sbert_sims = []  # description only
    # save generation outputs for later analysis
    gen_records = []  # list of dicts, one per example
    global_idx = 0    # running index over the evaluated split

    for batch in tqdm(batch_iter(), total=(len(split) + args.batch_size - 1)//args.batch_size, desc="generation"):
        images = [img.convert("RGB") for img in batch["image"]]
        inputs = processor(images=images, text=[args.prompt]*len(images), return_tensors="pt", padding=True).to(args.device)

        with torch.no_grad():
            out_ids = model.generate(
                **inputs,
                max_new_tokens=args.gen_max_new_tokens,
                num_beams=args.gen_num_beams,
                do_sample=False,
                no_repeat_ngram_size=args.no_repeat_ngram_size,
                repetition_penalty=args.repetition_penalty,
            )
        texts = processor.batch_decode(out_ids, skip_special_tokens=True)
        # convert generated token ids back to strings.
        texts = [t.strip() for t in texts]

        for ex, pred in zip(batch, texts):
        # compare predicted vs reference genre/style using normalized exact match
            if args.prompt and pred.lower().startswith(args.prompt.lower()):
                pred = pred[len(args.prompt):].strip()
                pred = args.prompt + " " + pred

            ref_genre = normalize_text(ex["genre"])
            ref_style = normalize_text(ex["style"])

            pred_genre = normalize_text(extract_field(pred, GENRE_RE))
            pred_style = normalize_text(extract_field(pred, STYLE_RE))

            is_valid = bool(pred_genre) and bool(pred_style)
            valid += int(is_valid)
            gen_total += 1
            lengths.append(len(pred.split()))

            if is_valid:
                gen_genre_correct += int(pred_genre == ref_genre)
                gen_style_correct += int(pred_style == ref_style)
                
            # save per-example generation outputs
            gen_records.append({
                "idx": global_idx,
                "prompt": args.prompt,
                "pred_text": pred,

                "ref_genre": ex.get("genre", ""),
                "ref_style": ex.get("style", ""),
                "ref_artist": ex.get("artist", ""),

                "pred_genre": extract_field(pred, GENRE_RE),
                "pred_style": extract_field(pred, STYLE_RE),
                "pred_desc": extract_field(pred, DESC_RE),

                "valid": bool(is_valid),
                "genre_exact": bool(is_valid) and (pred_genre == ref_genre),
                "style_exact": bool(is_valid) and (pred_style == ref_style),
            })
            global_idx += 1

            # SBERT similarity on Description only (optional)
            if sbert is not None:
                ref_desc = (ex.get("description_ref") or "").strip()
                pred_desc = extract_field(pred, DESC_RE)
                if ref_desc and pred_desc:
                    with torch.no_grad():
                        emb = sbert.encode([ref_desc, pred_desc], convert_to_tensor=True, normalize_embeddings=True)
                        sim = float((emb[0] * emb[1]).sum().cpu().item())
                    sbert_sims.append(sim)

    format_valid_rate = valid / gen_total
    gen_genre_acc = gen_genre_correct / gen_total # counts invalid as incorrect
    gen_style_acc = gen_style_correct / gen_total
    avg_len = float(np.mean(lengths)) if lengths else 0.0

    # results
    print("\n================= RESULTS =================")
    print(f"Split: {args.split} | N={gen_total}")
    print("\n[Classification heads]")
    print(f"  genre_acc : {head_genre_acc:.3f}")
    print(f"  style_acc : {head_style_acc:.3f}")
    print(f"  artist_acc: {head_artist_acc:.3f}")
    
    print("  artist_f1_macro   :", "n/a" if artist_f1_macro is None else f"{artist_f1_macro:.3f}")
    print("  artist_f1_weighted:", "n/a" if artist_f1_weighted is None else f"{artist_f1_weighted:.3f}")
    print("  genre_f1_macro    :", "n/a" if genre_f1_macro is None else f"{genre_f1_macro:.3f}")
    print("  genre_f1_weighted :", "n/a" if genre_f1_weighted is None else f"{genre_f1_weighted:.3f}")
    print("  style_f1_macro    :", "n/a" if style_f1_macro is None else f"{style_f1_macro:.3f}")
    print("  style_f1_weighted :", "n/a" if style_f1_weighted is None else f"{style_f1_weighted:.3f}")

    print("\n[Generated text parsing]")
    print(f"  valid_format_rate: {format_valid_rate:.3f}")
    print(f"  genre_acc_gen     : {gen_genre_acc:.3f}")
    print(f"  style_acc_gen     : {gen_style_acc:.3f}")
    print(f"  avg_pred_len_words: {avg_len:.1f}")

    if sbert is not None:
        if sbert_sims:
            print(f"\n[SBERT cosine similarity on Description] mean={float(np.mean(sbert_sims)):.3f} | N={len(sbert_sims)}")
        else:
            print("\n[SBERT] No pairs had both REF and PRED descriptions.")

    # show a few examples
    k = max(0, int(args.show_examples))
    if k > 0:
        print("\n================= EXAMPLES =================")
        for i in range(min(k, len(split))):
            ex = split[i]
            img = ex["image"]
            inputs = processor(images=img.convert("RGB"), text=args.prompt, return_tensors="pt").to(args.device)
            with torch.no_grad():
                out_ids = model.generate(
                    **inputs,
                    max_new_tokens=args.gen_max_new_tokens,
                    num_beams=args.gen_num_beams,
                    do_sample=False,
                    no_repeat_ngram_size=args.no_repeat_ngram_size,
                    repetition_penalty=args.repetition_penalty,
                )
            pred = processor.decode(out_ids[0], skip_special_tokens=True).strip()
            print("\n---")
            print("REF genre:", ex["genre"])
            print("REF style:", ex["style"])
            print("REF target:", ex["target_text"][:250])
            print("PRED:", pred[:400])

    # save predictions for later visualization
    head_preds = {
        "artist_pred": np.concatenate(a_preds_all) if a_preds_all else None,
        "artist_true": np.concatenate(a_true_all) if a_true_all else None,
        "genre_pred":  np.concatenate(g_preds_all) if g_preds_all else None,
        "genre_true":  np.concatenate(g_true_all) if g_true_all else None,
        "style_pred":  np.concatenate(s_preds_all) if s_preds_all else None,
        "style_true":  np.concatenate(s_true_all) if s_true_all else None,
    }
    
    head_confs = {
        "artist_conf": np.concatenate(a_conf_all) if a_conf_all else None,
        "genre_conf":  np.concatenate(g_conf_all) if g_conf_all else None,
        "style_conf":  np.concatenate(s_conf_all) if s_conf_all else None,
    }

    metrics = {
        "split": args.split,
        "N": int(gen_total),

        "heads": {
            "artist_acc": float(head_artist_acc),
            "genre_acc":  float(head_genre_acc),
            "style_acc":  float(head_style_acc),

            "artist_f1_macro":    None if artist_f1_macro is None else float(artist_f1_macro),
            "artist_f1_weighted": None if artist_f1_weighted is None else float(artist_f1_weighted),

            "genre_f1_macro":     None if genre_f1_macro is None else float(genre_f1_macro),
            "genre_f1_weighted":  None if genre_f1_weighted is None else float(genre_f1_weighted),

            "style_f1_macro":     None if style_f1_macro is None else float(style_f1_macro),
            "style_f1_weighted":  None if style_f1_weighted is None else float(style_f1_weighted),
        },

        "generation": {
            "valid_format_rate": float(format_valid_rate),
            "genre_acc_gen":     float(gen_genre_acc),
            "style_acc_gen":     float(gen_style_acc),
            "avg_pred_len_words": float(avg_len),
            "sbert_mean": (float(np.mean(sbert_sims)) if (sbert is not None and sbert_sims) else None),
            "sbert_N": int(len(sbert_sims)) if sbert is not None else 0,
        },

        "gen_params": {
            "prompt": args.prompt,
            "gen_max_new_tokens": int(args.gen_max_new_tokens),
            "gen_num_beams": int(args.gen_num_beams),
            "no_repeat_ngram_size": int(args.no_repeat_ngram_size),
            "repetition_penalty": float(args.repetition_penalty),
        }
    }

    # Save everything in one pickle for convenience
    out_path = os.path.join(args.model_dir, f"eval_{args.split}_results.pkl")
    payload = {
        "label_maps": label_maps,      # from label_maps.json
        "metrics": metrics,            # metrics
        "head_preds": head_preds,      # numpy arrays (or None)
        "head_confs": head_confs,      # confidence values
        "gen_records": gen_records,    # list of dicts
    }
    with open(out_path, "wb") as f:
        pickle.dump(payload, f)

    print(f"Saved evaluation payload to: {out_path}")
    print("\nDone.")

if __name__ == "__main__":
    main()
