import re
import hashlib
from pathlib import Path

import typer
from loguru import logger
from tqdm import tqdm
from datasets import load_dataset, Dataset
from sentence_transformers import SentenceTransformer

app = typer.Typer()
logger.add("embeddings.log", format="{time} {message}")


def hash_prompt(prompt: str) -> str:
    return hashlib.sha256(prompt.encode()).hexdigest()[:8]


def clean_whitespace(text: str) -> str:
    text = text.replace('\n', ' ')
    text = re.sub(r'\s+', ' ', text)
    text = re.sub(r'\s+([.,!?;:])', r'\1', text)
    text = re.sub(r'([.,!?;:])\s+', r'\1 ', text)
    return text.strip()


def simple_sentencize(text: str) -> list:
    sentences = re.findall(r'[^.!?]*[.!?]', text)
    return [sentence.strip() for sentence in sentences if sentence.strip()]


def chunk_sentences(sentences: list, max_tokens: int, model: SentenceTransformer) -> list:
    output = []
    current_chunk = []
    chunk_len = 0

    for sentence in sentences:
        tokens = model.tokenize(sentence)
        seq_len = len(tokens["input_ids"])

        if chunk_len + seq_len > max_tokens:
            if len(current_chunk) == 0:
                parts = split_long_sentence(sentence, max_tokens=max_tokens, model=model)
                output.extend(parts)
            else:
                output.append(" ".join(current_chunk))
                current_chunk = []
                chunk_len = 0

        current_chunk.append(sentence)
        chunk_len += seq_len

    if current_chunk:
        output.append(" ".join(current_chunk))

    return output


def split_long_sentence(sentence: str, max_tokens: int, model: SentenceTransformer) -> list:
    words = sentence.split()
    parts = []
    current_part = []
    current_len = 0

    for word in words:
        tokens = model.tokenize(word)
        seq_len = len(tokens["input_ids"])

        if current_len + seq_len > max_tokens:
            parts.append(" ".join(current_part))
            current_part = []
            current_len = 0

        current_part.append(word)
        current_len += seq_len

    if current_part:
        parts.append(" ".join(current_part))

    return parts


@app.command()
def main(
    dataset_name: str = typer.Option(..., help="Hugging Face dataset name, e.g. 'chcaa/chr-book-ads-articles'"),
    split: str = typer.Option("train", help="Which split of the dataset to use"),
    output_dir: Path = typer.Option(..., help="Directory where the processed dataset will be saved, should be in embeddings"),
    model_name: str = typer.Option("intfloat/multilingual-e5-large", help="SentenceTransformer model name for inference"),
    max_tokens: int = typer.Option(510, help="Maximum number of tokens per chunk"),
    prefix: str = typer.Option('Query: ', help="Optional prefix/instruction to add to each chunk before encoding"),
    prefix_description: str = typer.Option(None, help="Short description of the prefix (used in the output directory name)"),
):
    """
    This script loads a Hugging Face dataset, preprocesses and chunks texts,
    computes embeddings for each chunk, and saves the output dataset to disk.
    """
    model = SentenceTransformer(model_name)

    # Build output path based on model name and optional prefix
    mname = model_name.replace("/", "__")
    if prefix:
        if prefix_description:
            output_path = output_dir / f"emb__{mname}_{prefix_description}"
        else:
            prefix_hash = hash_prompt(prefix)
            output_path = output_dir / f"emb__{mname}_{prefix_hash}"
            logger.info(f"Hashing prefix: {prefix} == {prefix_hash}")
    else:
        output_path = output_dir / f"emb__{mname}"

    # ✅ Load HF dataset
    ds = load_dataset(dataset_name, split=split)

    processed_articles = []

    for row in tqdm(ds, total=len(ds), desc="Processing articles"):
        article_id = row['article_id']
        text = row['text']
        cat = row['clean_category']
        date = row['date']

        try:
            text_clean = clean_whitespace(text)
            sentences = simple_sentencize(text_clean)
            chunks = chunk_sentences(sentences, max_tokens=max_tokens, model=model)
        except Exception as e:
            logger.error(f"Preprocessing error for article_id {article_id}: {e}")
            continue

        try:
            embeddings = []
            for chunk in chunks:
                chunk_input = f"{prefix} {chunk}" if prefix else chunk
                emb = model.encode(chunk_input)
                embeddings.append(emb)
        except Exception as e:
            logger.error(f"Inference error for article_id {article_id}: {e}")
            continue

        processed_articles.append({
            "article_id": article_id,
            "date": date,
            "chunk": chunks,
            "embedding": embeddings,
            "clean_category": cat
        })

    dataset = Dataset.from_list(processed_articles)
    dataset.save_to_disk(output_path)
    print(f"✅ Saved processed dataset to {output_path}")


if __name__ == "__main__":
    app()
