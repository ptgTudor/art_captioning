"""
Preprocess WikiArt dataset for training

Outputs a HuggingFace dataset directory with:
- image (PIL)
- title, artist, genre, style (strings)
- artist_id, genre_id, style_id (int)
- description_ref (string, optional)
- target_text (string) # the caption target used for training
- has_description (bool)

And also writes label_maps.json (id2label lists)

Steps:
1) parse CLI arguments and set RNG seeds for reproducibility
2) load the WikiArt dataset from HuggingFace (images + metadata)
3) ensure train/validation splits exist (create them if missing)
4) optionally downsample splits for quick experiments.
5) keep only the columns we need (image/title/artist/genre/style/description)
6) build label->id mappings from the training split (artist/genre/style)
7) generate pseudo-captions using a BLIP captioning model
8) for each row trim to a word budget and construct target_text
9) save the processed Dataset to disk and export label_maps.json
10) print a short report and a few example rows for sanity checking
    
Usage:
  python3 preprocess_wikiart.py \
  --output_dir data/wikiart_proc \
  --pseudo_model_name Salesforce/blip-image-captioning-base \
  --pseudo_batch_size 32 \
  --pseudo_max_new_tokens 30 \
"""

import argparse
import json
import os
import random
import re
from collections import Counter
from dataclasses import dataclass
from typing import Dict, List
import numpy as np
from datasets import load_dataset, DatasetDict
import torch
from transformers import BlipProcessor, BlipForConditionalGeneration


# Utils

def set_seed(seed):
    # set the same seed for all experiments so that they can be reproduced
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def clean_label(x):
    if x is None: 
        return "" # converting None to empty strings
    s = str(x).strip() # strip leading and trailing whitespaces
    s = s.replace("_", " ") # replace underscores with whitespaces
    s = re.sub(r"\s+", " ", s) # multiple whitespaces are replaced with a single one
    return s


def trim_words(text, max_words):
    text = (text or "").strip() # strip leading and trailing whitespaces, and replace with empty string if None
    if not text:
        return "" # return empty string if empty after cleaning
    words = text.split() # split the text into words based on whitespaces
    if len(words) <= max_words:
        return text # if text is within word limit, return it
    return " ".join(words[:max_words]).rstrip(",;: ") + "..." # take only the first max_words words,
                                                              # joining them with whitespaces after stripping punctuation marks

# Label mapping

@dataclass
class LabelMaps:
    artist2id: Dict[str, int]
    genre2id: Dict[str, int]
    style2id: Dict[str, int]
    id2artist: List[str]
    id2genre: List[str]
    id2style: List[str]

    def to_json(self) -> Dict: # to return the mappings in JSON format and save them for later use
        return {
            "id2artist": self.id2artist,
            "id2genre": self.id2genre,
            "id2style": self.id2style,
        }

    # build the mappings only inside the training set to avoid data leakage
    @staticmethod
    def from_train(train_split, artist_col = "artist", genre_col = "genre", style_col = "style"):

        # extract raw labels
        artists = [clean_label(x) for x in train_split[artist_col]]
        genres  = [clean_label(x) for x in train_split[genre_col]]
        styles  = [clean_label(x) for x in train_split[style_col]]

        # count the frequency of each label
        a_counts = Counter([a for a in artists if a])
        g_counts = Counter([g for g in genres if g])
        s_counts = Counter([s for s in styles if s])

        def top_labels(counts):
            items = sorted(counts.items(), key=lambda x: (-x[1], x[0]))
            labels = [k for k, _ in items]
            return labels

        id2artist = ["<UNK_ARTIST>"] + top_labels(a_counts, None)
        id2genre  = ["<UNK_GENRE>"]  + top_labels(g_counts, None)
        id2style  = ["<UNK_STYLE>"]  + top_labels(s_counts, None)

        artist2id = {lab: i for i, lab in enumerate(id2artist)}
        genre2id  = {lab: i for i, lab in enumerate(id2genre)}
        style2id  = {lab: i for i, lab in enumerate(id2style)}

        return LabelMaps(
            artist2id=artist2id,
            genre2id=genre2id,
            style2id=style2id,
            id2artist=id2artist,
            id2genre=id2genre,
            id2style=id2style,
        )


# Pseudo caption generation

def add_pseudo_captions_to_split(ds_split, model_name, batch_size, max_new_tokens, num_beams, device):
    
    # generate descriptions using the BLIP captioning model
    processor = BlipProcessor.from_pretrained(model_name)
    model = BlipForConditionalGeneration.from_pretrained(model_name)
    model.to(device)
    model.eval()

    def batched(batch):
        images = [img.convert("RGB") for img in batch["image"]]
        inputs = processor(images=images, return_tensors="pt").to(device)
        with torch.no_grad():
            out_ids = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                num_beams=num_beams,
                do_sample=False,
            )
        caps = processor.batch_decode(out_ids, skip_special_tokens=True)
        caps = [c.strip() for c in caps]
        return {"pseudo_caption": caps}

    ds_split = ds_split.map(batched, batched=True, batch_size=batch_size)
    return ds_split


# Main

def main():

    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset_name", type=str, default="Artificio/WikiArt")
    ap.add_argument("--output_dir", type=str, required=True)
    ap.add_argument("--seed", type=int, default=42)

    ap.add_argument("--val_split", type=float, default=0.1)
    ap.add_argument("--max_train_samples", type=int, default=None)
    ap.add_argument("--max_val_samples", type=int, default=None)

    # description settings
    ap.add_argument("--desc_max_words", type=int, default=35)
    ap.add_argument("--include_title", action="store_true", help="Include Title: ... in target_text")

    # pseudo caption options
    ap.add_argument("--pseudo_model_name", type=str, default="Salesforce/blip-image-captioning-base")
    ap.add_argument("--pseudo_batch_size", type=int, default=32)
    ap.add_argument("--pseudo_max_new_tokens", type=int, default=30)
    ap.add_argument("--pseudo_num_beams", type=int, default=3)

    args = ap.parse_args()

    # ensure reproducibility
    set_seed(args.seed)

    # create output directory
    os.makedirs(args.output_dir, exist_ok=True)

    # load dataset from HuggingFace
    # image (PIL), title, artist, genre, style, description
    ds = load_dataset(args.dataset_name)

    # ensure existence of validation/test split
    if "validation" not in ds:
        tmp = ds["train"].train_test_split(test_size=args.val_split, seed=args.seed)
        ds = DatasetDict({"train": tmp["train"], "validation": tmp["test"]})

    # optional downsample for preprocessing smaller parts of a dataset
    if args.max_train_samples is not None and len(ds["train"]) > args.max_train_samples:
        ds["train"] = ds["train"].shuffle(seed=args.seed).select(range(args.max_train_samples))
    if args.max_val_samples is not None and len(ds["validation"]) > args.max_val_samples:
        ds["validation"] = ds["validation"].shuffle(seed=args.seed).select(range(args.max_val_samples))

    # keep only needed columns
    keep_cols = ["image", "title", "artist", "genre", "style", "description"]
    for split in ds.keys():
        cols = ds[split].column_names
        drop_cols = [c for c in cols if c not in keep_cols]
        if drop_cols:
            ds[split] = ds[split].remove_columns(drop_cols)

    # build label maps from the training split:
    maps = LabelMaps.from_train(ds["train"])

    def build_rows(batch, indices=None):
        # get source columns with fallbacks
        titles = batch.get("title", [""] * len(batch["image"]))
        artists = batch.get("artist", [""] * len(batch["image"]))
        genres = batch.get("genre", [""] * len(batch["image"]))
        styles = batch.get("style", [""] * len(batch["image"]))
        pseudo_caps = batch.get("pseudo_caption", [""] * len(batch["image"]))

        out_artist_id = []
        out_genre_id = []
        out_style_id = []
        out_target_text = []
        out_desc_ref = []
        out_desc_source = []
        out_has_desc = []

        for i in range(len(batch["image"])):
            # clean entries
            title = clean_label(titles[i])
            artist = clean_label(artists[i]) or "Unknown Artist"
            genre = clean_label(genres[i]) or "Unknown Genre"
            style = clean_label(styles[i]) or "Unknown Style"

            # IDs (unknown ones get assigned 0)
            out_artist_id.append(maps.artist2id.get(artist, 0))
            out_genre_id.append(maps.genre2id.get(genre, 0))
            out_style_id.append(maps.style2id.get(style, 0))

            # clean description
            desc = trim_words((pseudo_caps[i] or "").strip(), args.desc_max_words)
            src = "pseudo" if desc else "none"

            # building the target text (without artist since it confuses the model)
            parts = [f"Genre: {genre}.", f"Style: {style}."]
            if args.include_title and title:
                parts.append(f"Title: {title}.")
            if desc:
                parts.append(f"Description: {desc}")
            target_text = " ".join(parts)

            # save outputs to their lists
            out_target_text.append(target_text)
            out_desc_ref.append(desc)
            out_desc_source.append(src)
            out_has_desc.append(bool(desc))

        return {
            "artist_id": out_artist_id,
            "genre_id": out_genre_id,
            "style_id": out_style_id,
            "target_text": out_target_text,
            "description_ref": out_desc_ref,
            "desc_source": out_desc_source,
            "has_description": out_has_desc,
        }

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[preprocess] Generating pseudo captions with {args.pseudo_model_name} on {device} ...")
    ds["train"] = add_pseudo_captions_to_split(
        ds["train"], model_name=args.pseudo_model_name, batch_size=args.pseudo_batch_size,
        max_new_tokens=args.pseudo_max_new_tokens, num_beams=args.pseudo_num_beams, device=device
    )
    ds["validation"] = add_pseudo_captions_to_split(
        ds["validation"], model_name=args.pseudo_model_name, batch_size=args.pseudo_batch_size,
        max_new_tokens=args.pseudo_max_new_tokens, num_beams=args.pseudo_num_beams, device=device
    )

    # build targets and IDs

    ds["train"] = ds["train"].map(build_rows, with_indices=True, batched=True, batch_size=256)
    ds["validation"] = ds["validation"].map(build_rows, with_indices=True, batched=True, batch_size=256)

    # save dataset
    ds.save_to_disk(args.output_dir)
    print(f"[preprocess] Saved dataset to: {args.output_dir}")

    # save label maps
    label_maps_path = os.path.join(args.output_dir, "label_maps.json")
    with open(label_maps_path, "w", encoding="utf-8") as f:
        json.dump(maps.to_json(), f, ensure_ascii=False, indent=2)
    print(f"[preprocess] Saved label maps to: {label_maps_path}")

    # check if all samples have description
    for split in ["train", "validation"]:
        has = float(np.mean(ds[split]["has_description"]))
        print(f"[preprocess] {split}: {has*100:.1f}% have descriptions")

    # show a few examples
    for idx in range(3):
        ex = ds["validation"][idx]
        print("\n--- example ---")
        print("artist:", ex["artist"])
        print("genre :", ex["genre"])
        print("style :", ex["style"])
        print("title :", ex.get("title",""))
        print("desc_source:", ex.get("desc_source"))
        print("target_text:", ex["target_text"])


if __name__ == "__main__":
    main()
