"""
Generates:
1) top confusions (head predictions) for artist/genre/style + mean confidence
2) best vs worst grid using head confidence
3) head vs generation comparison report (2x2 table per task, plus agreement rate)

Usage:
  python3 analyze_eval_results.py \
    --eval_pkl runs/blip_multitask/eval_validation_results.pkl \
    --data_dir data/wikiart_proc \
    --split validation \
    --out_dir runs/blip_multitask/analysis \
    --task style \
    --k 10
"""

import argparse
import os
import csv
import pickle
from collections import defaultdict
import numpy as np
import matplotlib.pyplot as plt
from datasets import load_from_disk


def normalize_text(s: str) -> str:
    s = (s or "").strip().lower()
    s = " ".join(s.split())
    return s


def build_label_lookup(id2labels):
    norm_to_id = {}
    for i, name in enumerate(id2labels):
        key = normalize_text(name)
        norm_to_id.setdefault(key, i)
    return id2labels, norm_to_id


def gen_label_to_id(label_str: str, norm_to_id: dict):
    key = normalize_text(label_str)
    return norm_to_id.get(key, None)


def top_confusions(true_ids, pred_ids, id2labels, conf, topn=30):
    pair_counts = defaultdict(int)
    pair_confs = defaultdict(float)
    pair_conf_n = defaultdict(int)

    for t, p, c in zip(true_ids, pred_ids, conf):
        if int(t) == int(p):
            continue
        key = (int(t), int(p))
        pair_counts[key] += 1
        pair_confs[key] += float(c)
        pair_conf_n[key] += 1

    rows = []
    for (t, p), cnt in pair_counts.items():
        mean_conf = None
        if pair_conf_n[(t, p)] > 0:
            mean_conf = pair_confs[(t, p)] / pair_conf_n[(t, p)]
        rows.append((
            cnt, t, p,
            id2labels[t] if t < len(id2labels) else str(t),
            id2labels[p] if p < len(id2labels) else str(p),
            mean_conf
        ))

    rows.sort(key=lambda x: x[0], reverse=True)
    return rows[:topn]


def compute_head_vs_gen_agreement(records, head_pred, head_true, id2labels, norm_to_id, field_pred, field_ref):
    # map reference strings to ids for generation correctness
    ref_ids = []
    gen_ids = []
    idxs = []

    for r in records:
        idx = int(r["idx"])
        idxs.append(idx)
        ref_str = r.get(field_ref, "")
        gen_str = r.get(field_pred, "")
        ref_id = gen_label_to_id(ref_str, norm_to_id)
        gen_id = gen_label_to_id(gen_str, norm_to_id)
        ref_ids.append(ref_id)
        gen_ids.append(gen_id)

    idxs = np.array(idxs, dtype=np.int64)
    head_p = head_pred[idxs]
    head_t = head_true[idxs]

    head_correct = (head_p == head_t)

    gen_valid = np.array([g is not None and r is not None for g, r in zip(gen_ids, ref_ids)], dtype=bool)
    gen_correct = np.array([(g == r) if (g is not None and r is not None) else False for g, r in zip(gen_ids, ref_ids)], dtype=bool)

    agreement = np.array([(g == int(h)) if g is not None else False for g, h in zip(gen_ids, head_p)], dtype=bool)

    # 2x2 counts: head_correct vs gen_correct
    hc_gc = int(np.sum(head_correct & gen_correct))
    hc_gw = int(np.sum(head_correct & ~gen_correct))
    hw_gc = int(np.sum(~head_correct & gen_correct))
    hw_gw = int(np.sum(~head_correct & ~gen_correct))

    summary = {
        "N": int(len(records)),
        "gen_valid_rate": float(np.mean(gen_valid)) if len(records) else 0.0,
        "head_acc_on_records": float(np.mean(head_correct)) if len(records) else 0.0,
        "gen_acc_on_records": float(np.mean(gen_correct)) if len(records) else 0.0,
        "agreement_rate": float(np.mean(agreement)) if len(records) else 0.0,
        "2x2": {
            "head_correct__gen_correct": hc_gc,
            "head_correct__gen_wrong":   hc_gw,
            "head_wrong__gen_correct":   hw_gc,
            "head_wrong__gen_wrong":     hw_gw,
        }
    }
    return summary


def make_best_worst_grid(ds_split, head_pred, head_true, head_conf, gen_records_by_idx, id2labels, out_path, k=10, title=""):
    conf = head_conf

    correct = (head_pred == head_true)
    correct_idxs = np.where(correct)[0]
    wrong_idxs = np.where(~correct)[0]

    # Sort by confidence descending
    best_idxs = correct_idxs[np.argsort(conf[correct_idxs])[::-1]][:k]
    worst_idxs = wrong_idxs[np.argsort(conf[wrong_idxs])[::-1]][:k]

    # grid size
    cols = k
    rows = 2
    fig_w = max(12, cols * 1.2)
    fig_h = 8.5
    fig = plt.figure(figsize=(fig_w, fig_h))
    
    def clip(s, n=28):
        s = "" if s is None else str(s)
        s = " ".join(s.split())
        return (s[: n - 1] + "…") if len(s) > n else s


    def render_cell(ax, idx, tag):
        ex = ds_split[int(idx)]
        img = ex["image"].convert("RGB")
        ax.imshow(img)
        ax.axis("off")

        true_id = int(head_true[idx])
        pred_id = int(head_pred[idx])
        c = float(conf[idx])

        painting_id = idx
        title = ex.get("title", ex.get("name", ""))

        gen = gen_records_by_idx.get(int(idx), None)
        gen_style = gen.get("pred_style", "") if gen else ""
        gen_genre = gen.get("pred_genre", "") if gen else ""

        true_name = id2labels[true_id] if true_id < len(id2labels) else str(true_id)
        pred_name = id2labels[pred_id] if pred_id < len(id2labels) else str(pred_id)

        ax.set_title(f"{tag} | p={c:.2f}", fontsize=10, pad=2)

        info = (
            f"id: {clip(painting_id, 22)}\n"
            f"title: {clip(title, 34)}\n"
            f"true: {clip(true_name, 30)}\n"
            f"pred: {clip(pred_name, 30)}\n"
            f"gen_style: {clip(gen_style, 26)}\n"
            f"gen_genre: {clip(gen_genre, 26)}"
        )

        ax.text(
            0.5, -0.22, info,
            transform=ax.transAxes,
            ha="center", va="top",
            fontsize=7,
            wrap=True
        )

    # row 1: best
    for j, idx in enumerate(best_idxs):
        ax = fig.add_subplot(rows, cols, 1 + j)
        render_cell(ax, idx, "BEST")

    # row 2: worst
    for j, idx in enumerate(worst_idxs):
        ax = fig.add_subplot(rows, cols, cols + 1 + j)
        render_cell(ax, idx, "WORST")

    if title:
        fig.suptitle(title, fontsize=14)

    fig.tight_layout(rect=[0, 0.06, 1, 0.95])
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--eval_pkl", type=str, required=True)
    ap.add_argument("--data_dir", type=str, required=True)
    ap.add_argument("--split", type=str, default="validation", choices=["train", "validation"])
    ap.add_argument("--out_dir", type=str, required=True)
    ap.add_argument("--task", type=str, default="style", choices=["artist", "genre", "style"])
    ap.add_argument("--k", type=int, default=12, help="number of examples for best/worst grid")
    ap.add_argument("--topn", type=int, default=30, help="top-N confusions to save")
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    with open(args.eval_pkl, "rb") as f:
        payload = pickle.load(f)

    label_maps = payload["label_maps"]
    head_preds = payload["head_preds"]
    head_confs = payload.get("head_confs", None)
    gen_records = payload.get("gen_records", [])

    # load dataset (for images)
    ds = load_from_disk(args.data_dir)
    split = ds[args.split]

    # build id2labels and normalization lookup for mapping generation strings to IDs
    if args.task == "artist":
        id2 = label_maps["id2artist"]
        pred = head_preds["artist_pred"]
        true = head_preds["artist_true"]
        conf = head_confs.get("artist_conf")
        gen_field_pred = "pred_artist"
        gen_field_ref = "ref_artist"
    elif args.task == "genre":
        id2 = label_maps["id2genre"]
        pred = head_preds["genre_pred"]
        true = head_preds["genre_true"]
        conf = head_confs.get("genre_conf")
        gen_field_pred = "pred_genre"
        gen_field_ref = "ref_genre"
    else:
        id2 = label_maps["id2style"]
        pred = head_preds["style_pred"]
        true = head_preds["style_true"]
        conf = head_confs.get("style_conf")
        gen_field_pred = "pred_style"
        gen_field_ref = "ref_style"

    if pred is None or true is None:
        raise RuntimeError(f"Missing preds/targets for task={args.task} in eval pickle.")

    id2labels, norm_to_id = build_label_lookup(id2)

    # top confusions
    confusions = top_confusions(true, pred, id2labels, conf=conf, topn=args.topn)

    conf_csv = os.path.join(args.out_dir, f"top_confusions_{args.task}.csv")
    with open(conf_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["count", "true_id", "pred_id", "true_name", "pred_name", "mean_confidence"])
        for row in confusions:
            w.writerow([
                row[0], row[1], row[2], row[3], row[4],
                "" if row[5] is None else f"{row[5]:.6f}"
            ])
    print(f"[OK] Saved top confusions: {conf_csv}")

    # short summary
    print("\nTop confusions (heads):")
    for cnt, t, p, tn, pn, mc in confusions[:10]:
        mc_s = "" if mc is None else f" | mean_conf={mc:.2f}"
        print(f"  {cnt:4d}  {tn}  ->  {pn}{mc_s}")

    # build idx to gen_record lookup for grid
    gen_by_idx = {int(r["idx"]): r for r in gen_records}

    # best vs worst grid
    grid_path = os.path.join(args.out_dir, f"best_worst_{args.task}.png")
    title = f"{args.task.upper()} best vs worst (by head confidence) | split={args.split}"
    make_best_worst_grid(
        split, pred, true, conf,
        gen_by_idx, id2labels,
        out_path=grid_path,
        k=args.k,
        title=title
    )
    print(f"[OK] Saved best/worst grid: {grid_path}")

    # head vs generation comparison
    if args.task == "artist":
        print("[INFO] Head-vs-generation comparison for artist requires you to parse/save pred_artist in gen_records.")
    else:
        agreement = compute_head_vs_gen_agreement(
            records=gen_records,
            head_pred=pred,
            head_true=true,
            id2labels=id2labels,
            norm_to_id=norm_to_id,
            field_pred=gen_field_pred,
            field_ref=gen_field_ref
        )

        rep_path = os.path.join(args.out_dir, f"head_vs_gen_{args.task}.txt")
        with open(rep_path, "w", encoding="utf-8") as f:
            f.write(f"Task: {args.task}\n")
            f.write(f"Split: {args.split}\n")
            f.write(f"N records: {agreement['N']}\n\n")
            f.write(f"Gen valid rate: {agreement['gen_valid_rate']:.4f}\n")
            f.write(f"Head acc on records: {agreement['head_acc_on_records']:.4f}\n")
            f.write(f"Gen acc on records:  {agreement['gen_acc_on_records']:.4f}\n")
            f.write(f"Head/Gen agreement:  {agreement['agreement_rate']:.4f}\n\n")
            f.write("2x2 (counts):\n")
            f.write(f"  head_correct & gen_correct: {agreement['2x2']['head_correct__gen_correct']}\n")
            f.write(f"  head_correct & gen_wrong:   {agreement['2x2']['head_correct__gen_wrong']}\n")
            f.write(f"  head_wrong   & gen_correct: {agreement['2x2']['head_wrong__gen_correct']}\n")
            f.write(f"  head_wrong   & gen_wrong:   {agreement['2x2']['head_wrong__gen_wrong']}\n")

        print(f"[OK] Saved head-vs-generation agreement report: {rep_path}")
        print("\nHead vs Gen agreement summary:")
        print(f"  gen_valid_rate     : {agreement['gen_valid_rate']:.3f}")
        print(f"  head_acc_on_records: {agreement['head_acc_on_records']:.3f}")
        print(f"  gen_acc_on_records : {agreement['gen_acc_on_records']:.3f}")
        print(f"  agreement_rate     : {agreement['agreement_rate']:.3f}")
        print("  2x2:", agreement["2x2"])


if __name__ == "__main__":
    main()
