"""
Run inference on a single image

Steps
1) parse CLI args (checkpoint dir, image path, generation settings).
2) load label_maps.json
3) load processor and multitask BLIP model from model_dir and move to device
4) load the input image with PIL and convert to RGB
5) classification-head predictions:
   - preprocess image -> pixel_values
   - run vision encoder -> pooled features
   - apply linear heads -> argmax id -> map to string labels
6) text generation (slower):
   - run `model.generate()` with a simple prompt (default "Genre:")
   - decode to text and print

Usage:
    python3 infer_single.py \
  --model_dir runs/blip_multitask \
  --image_path /path/to/painting.jpg
"""

import argparse
import json
import os
import torch
import torch.nn as nn
from PIL import Image
from transformers import BlipProcessor, BlipConfig, BlipForConditionalGeneration


# Model

class BlipForConditionalGenerationMultiTask(BlipForConditionalGeneration):
    def __init__(self, config: BlipConfig):
        super().__init__(config)

        # custom config fields
        self.num_artists = int(getattr(config, "num_artists", 0) or 0)
        self.num_genres  = int(getattr(config, "num_genres", 0) or 0)
        self.num_styles  = int(getattr(config, "num_styles", 0) or 0)

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


# Main

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_dir", type=str, required=True)
    ap.add_argument("--image_path", type=str, required=True)
    ap.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--prompt", type=str, default="Genre:")
    ap.add_argument("--max_new_tokens", type=int, default=90)
    ap.add_argument("--num_beams", type=int, default=5)
    ap.add_argument("--no_repeat_ngram_size", type=int, default=3)
    ap.add_argument("--repetition_penalty", type=float, default=1.2)
    args = ap.parse_args()

    # label maps
    lm_path = os.path.join(args.model_dir, "label_maps.json")
    if not os.path.exists(lm_path):
        raise FileNotFoundError(f"Missing label_maps.json in {args.model_dir}")
    with open(lm_path, "r", encoding="utf-8") as f:
        lm = json.load(f)

    id2artist = lm["id2artist"]
    id2genre  = lm["id2genre"]
    id2style  = lm["id2style"]

    # load processor and model
    processor = BlipProcessor.from_pretrained(args.model_dir)
    model = BlipForConditionalGenerationMultiTask.from_pretrained(args.model_dir)
    model.to(args.device)
    model.eval()

    # load image
    img = Image.open(args.image_path).convert("RGB")

    # classification
    with torch.no_grad():
        pixel = processor(images=img, return_tensors="pt")["pixel_values"].to(args.device)
        feats = model.get_image_features(pixel)

        artist = "<no-head>"
        genre  = "<no-head>"
        style  = "<no-head>"

        if model.artist_classifier is not None:
            a = int(model.artist_classifier(feats).argmax(dim=-1).item())
            artist = id2artist[a] if a < len(id2artist) else str(a)

        if model.genre_classifier is not None:
            g = int(model.genre_classifier(feats).argmax(dim=-1).item())
            genre = id2genre[g] if g < len(id2genre) else str(g)

        if model.style_classifier is not None:
            s = int(model.style_classifier(feats).argmax(dim=-1).item())
            style = id2style[s] if s < len(id2style) else str(s)


    # generation
    inputs = processor(images=img, text=args.prompt, return_tensors="pt").to(args.device)
    with torch.no_grad():
        out_ids = model.generate(
            **inputs,
            max_new_tokens=args.max_new_tokens,
            num_beams=args.num_beams,
            do_sample=False,
            no_repeat_ngram_size=args.no_repeat_ngram_size,
            repetition_penalty=args.repetition_penalty,
        )
    gen = processor.decode(out_ids[0], skip_special_tokens=True).strip()

    print("=== HEAD PREDICTIONS ===")
    print("Artist:", artist)
    print("Genre :", genre)
    print("Style :", style)
    print("\n=== GENERATED TEXT ===")
    print(gen)


if __name__ == "__main__":
    main()
