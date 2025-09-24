import pandas as pd
from datasets import Dataset, load_dataset

# 🔹 Hugging Face dataset (text + metadata)
HF_DATASET_NAME = "chcaa/chr-book-ads-articles"
texts = load_dataset(HF_DATASET_NAME, split="train").to_pandas()

# 🔹 Local embeddings dataset
LOCAL_EMBS_PATH = "../path_to_embeddings_created_with_mean_pooling.py"
embs = Dataset.load_from_disk(LOCAL_EMBS_PATH).to_pandas()

# 🔹 Merge on article_id
merged = embs.merge(
    texts[["article_id", "text", "clean_category", "article_length", "characters"]],
    on="article_id"
)

# 🔹 Save merged dataset
OUTPUT_PATH = "../directory_where_this_is_saved"
dataset = Dataset.from_pandas(merged)
dataset.save_to_disk(OUTPUT_PATH)

print(f"✅ Saved merged dataset to {OUTPUT_PATH}")