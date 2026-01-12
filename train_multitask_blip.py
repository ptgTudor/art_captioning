"""
Train a multitask BLIP model on a preprocessed WikiArt dataset

Steps:
1) parse CLI args
2) set seeds for reproducibility
3) load the HuggingFace dataset created by `preprocess_wikiart.py` (expecting train/validation splits)
4) load `label_maps.json` to determine the number of classes for each auxiliary head (artist_id, genre_id, style_id)
5) create a BLIP processor (image preprocessing + tokenizer) from a base checkpoint
6) extend BLIP with simple linear classification heads on pooled vision features:
   - the base BLIP loss is the captioning / conditional generation loss
   - extra losses are cross-entropy for genre/style/artist, weighted and added to caption loss
7) build a data collator that:
   - converts PIL images to pixel_values
   - tokenizes target_text into input_ids/attention_mask
   - creates labels (pad tokens masked as -100 for loss)
   - bundles artist/genre/style labels as tensors
8) configure training arguments and start the training process
9) save the fine-tuned model, processor and a copy of label_maps.json to output_dir

Usage:
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
"""

import argparse
import json
import os
import random
from dataclasses import dataclass
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from datasets import load_from_disk
from transformers import BlipProcessor, BlipConfig, BlipForConditionalGeneration, Trainer, TrainingArguments


# Utils

def set_seed(seed: int):
    # set the same seed for all experiments so that they can be reproduced
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


# Model

class BlipForConditionalGenerationMultiTask(BlipForConditionalGeneration):
    def __init__(self, config):
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
    def get_image_features(self, pixel_values):
        vision_out = self.vision_model(pixel_values=pixel_values, return_dict=True)
        if hasattr(vision_out, "pooler_output") and vision_out.pooler_output is not None:
            feats = vision_out.pooler_output
        else:
            feats = vision_out.last_hidden_state[:, 0]  # CLS token
        return feats


    # forward computes the standard BLIP captioning loss
    # then adds auxiliary classification losses for genre/style/artist.
    def forward(
        self,
        pixel_values = None,
        input_ids = None,
        attention_mask = None,
        labels = None,
        artist_labels = None,
        genre_labels = None,
        style_labels = None,
        **kwargs,
    ):
        # HF Trainer may pass this kwarg for some models
        kwargs.pop("num_items_in_batch", None)

        # Caption loss (standard BLIP)
        outputs = super().forward(
            pixel_values=pixel_values,
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
            return_dict=True,
            **kwargs,
        )

        total_loss = outputs.loss if outputs.loss is not None else None

        # compute pooled vision features
        feats = None
        if pixel_values is not None and (self.artist_classifier or self.genre_classifier or self.style_classifier):
            feats = self.get_image_features(pixel_values)

        # classification losses
        if feats is not None and total_loss is not None:
            if self.genre_classifier is not None and genre_labels is not None:
                glogits = self.genre_classifier(feats)
                loss_g = F.cross_entropy(glogits, genre_labels)
                total_loss = total_loss + self.genre_loss_weight * loss_g
                outputs.genre_logits = glogits

            if self.style_classifier is not None and style_labels is not None:
                slogits = self.style_classifier(feats)
                loss_s = F.cross_entropy(slogits, style_labels)
                total_loss = total_loss + self.style_loss_weight * loss_s
                outputs.style_logits = slogits

            if self.artist_classifier is not None and artist_labels is not None:
                alogits = self.artist_classifier(feats)
                loss_a = F.cross_entropy(alogits, artist_labels)
                total_loss = total_loss + self.artist_loss_weight * loss_a
                outputs.artist_logits = alogits

        outputs.loss = total_loss
        return outputs


# Collator

# turn a list of dataset rows into model inputs
# keep remove_unused_columns=False in TrainingArguments so Trainer doesn't drop extra label tensors (artist_labels/genre_labels/style_labels)
@dataclass
class DataCollatorBlipMultiTask:
    processor: BlipProcessor
    max_text_len: int = 128

    def __call__(self, batch):
        # collect PIL images and target strings for generation
        images = [ex["image"].convert("RGB") for ex in batch]
        texts  = [ex["target_text"] for ex in batch]

        # image preprocessing and text tokenization
        enc = self.processor(
            images=images,
            text=texts,
            padding=True,
            truncation=True,
            max_length=self.max_text_len,
            return_tensors="pt",
        )

        labels = enc["input_ids"].clone()
        # mask padding tokens with -100 so they don't contribute to the seq2seq loss
        labels[labels == self.processor.tokenizer.pad_token_id] = -100
        enc["labels"] = labels

        enc["artist_labels"] = torch.tensor([int(ex["artist_id"]) for ex in batch], dtype=torch.long)
        enc["genre_labels"]  = torch.tensor([int(ex["genre_id"]) for ex in batch], dtype=torch.long)
        enc["style_labels"]  = torch.tensor([int(ex["style_id"]) for ex in batch], dtype=torch.long)

        return enc


# Main

def main():
    ap = argparse.ArgumentParser()

    ap.add_argument("--data_dir", type=str, required=True, help="Directory created by preprocess_wikiart.py")
    ap.add_argument("--output_dir", type=str, required=True)
    ap.add_argument("--model_name", type=str, default="Salesforce/blip-image-captioning-base")

    ap.add_argument("--seed", type=int, default=42)

    ap.add_argument("--max_text_len", type=int, default=128)

    ap.add_argument("--per_device_train_batch_size", type=int, default=8)
    ap.add_argument("--per_device_eval_batch_size", type=int, default=8)
    ap.add_argument("--gradient_accumulation_steps", type=int, default=1)

    ap.add_argument("--num_train_epochs", type=float, default=5.0)
    ap.add_argument("--learning_rate", type=float, default=5e-5)
    ap.add_argument("--weight_decay", type=float, default=0.01)
    ap.add_argument("--warmup_ratio", type=float, default=0.03)
    ap.add_argument("--lr_scheduler_type", type=str, default="cosine")

    ap.add_argument("--logging_steps", type=int, default=50)
    ap.add_argument("--save_strategy", type=str, default="epoch", choices=["no", "steps", "epoch"])
    ap.add_argument("--save_steps", type=int, default=1000)
    ap.add_argument("--evaluation_strategy", type=str, default="epoch", choices=["no", "steps", "epoch"])
    ap.add_argument("--eval_steps", type=int, default=1000)
    ap.add_argument("--save_total_limit", type=int, default=3)

    ap.add_argument("--fp16", action="store_true")
    ap.add_argument("--bf16", action="store_true")
    ap.add_argument("--gradient_checkpointing", action="store_true")

    ap.add_argument("--dataloader_num_workers", type=int, default=4)

    # loss weights
    ap.add_argument("--artist_loss_weight", type=float, default=0.05)
    ap.add_argument("--genre_loss_weight", type=float, default=0.50)
    ap.add_argument("--style_loss_weight", type=float, default=0.70)

    args = ap.parse_args()
    
    # ensure reproducibility
    set_seed(args.seed)

    os.makedirs(args.output_dir, exist_ok=True)

    # load processed dataset
    ds = load_from_disk(args.data_dir)
    if "train" not in ds or "validation" not in ds:
        raise ValueError("Processed dataset must contain train and validation splits.")

    # load label maps
    label_maps_path = os.path.join(args.data_dir, "label_maps.json")
    if not os.path.exists(label_maps_path):
        raise FileNotFoundError(f"Missing label_maps.json at: {label_maps_path}")
    with open(label_maps_path, "r", encoding="utf-8") as f:
        lm = json.load(f)

    num_artists = len(lm["id2artist"])
    num_genres  = len(lm["id2genre"])
    num_styles  = len(lm["id2style"])

    # load processor from base checkpoint
    processor = BlipProcessor.from_pretrained(args.model_name)

    # build config with extra fields
    config = BlipConfig.from_pretrained(args.model_name)
    config.num_artists = num_artists
    config.num_genres  = num_genres
    config.num_styles  = num_styles
    config.artist_loss_weight = args.artist_loss_weight
    config.genre_loss_weight  = args.genre_loss_weight
    config.style_loss_weight  = args.style_loss_weight

    # model instance
    model = BlipForConditionalGenerationMultiTask.from_pretrained(args.model_name, config=config)

    if args.gradient_checkpointing:
        model.gradient_checkpointing_enable()
        model.config.use_cache = False

    # collator
    collator = DataCollatorBlipMultiTask(processor=processor, max_text_len=args.max_text_len)

    # training arguments
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        per_device_train_batch_size=args.per_device_train_batch_size,
        per_device_eval_batch_size=args.per_device_eval_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        num_train_epochs=args.num_train_epochs,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        warmup_ratio=args.warmup_ratio,
        lr_scheduler_type=args.lr_scheduler_type,
        fp16=args.fp16,
        bf16=args.bf16,
        logging_steps=args.logging_steps,
        save_strategy=args.save_strategy,
        save_steps=args.save_steps,
        evaluation_strategy=args.evaluation_strategy,
        eval_steps=args.eval_steps,
        save_total_limit=args.save_total_limit,
        remove_unused_columns=False,
        dataloader_num_workers=args.dataloader_num_workers,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=ds["train"],
        eval_dataset=ds["validation"],
        data_collator=collator,
    )

    trainer.train()

    # save model, processor and label maps
    trainer.save_model(args.output_dir)
    processor.save_pretrained(args.output_dir)

    # copy label maps
    with open(os.path.join(args.output_dir, "label_maps.json"), "w", encoding="utf-8") as f:
        json.dump(lm, f, ensure_ascii=False, indent=2)

    print(f"[train] Saved model + processor to: {args.output_dir}")


if __name__ == "__main__":
    main()
