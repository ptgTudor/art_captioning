"""
Extract first n images of the validation dataset
    
Usage:
  python3 extract_validation_images.py
"""


from datasets import load_from_disk
import sys

data_dir = "data/wikiart_proc"
ds = load_from_disk(data_dir)
validation = ds["validation"]

# Save the first 3 images
for i in range(3):
    img = validation[i]["image"]
    img.save(f"validation_example_{i}.jpg")
    print(f"Saved example {i}:")
    print(f"  Title: {validation[i]['title']}")
    print(f"  Genre: {validation[i]['genre']}")
    print(f"  Style: {validation[i]['style']}")
    print(f"  Artist: {validation[i]['artist']}")
    print()