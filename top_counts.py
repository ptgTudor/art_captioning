from datasets import load_from_disk
from collections import Counter

ds = load_from_disk("data/wikiart_proc")
train = ds["train"]

# genre
genre_counts = Counter(train["genre"])
# style
style_counts = Counter(train["style"])
# artist
artist_counts = Counter(train["artist"])

print("Top genres:")
for k, v in genre_counts.most_common(10):
    print(f"{k:25s} {v}")

print("\nTop styles:")
for k, v in style_counts.most_common(10):
    print(f"{k:25s} {v}")

print("\nTop artists:")
for k, v in artist_counts.most_common(10):
    print(f"{k:25s} {v}")