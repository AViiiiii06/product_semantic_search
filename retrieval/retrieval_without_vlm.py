"""
Retriever without VLM metadata.

Searches the FAISS index using only FashionCLIP text/image similarity. This
version does not require metadata.json and does not rerank with Groq output.

Example:
    python retrieval/retrieval_without_vlm.py --query "A red tie and a white shirt in a formal setting" --top-k 5
"""

from __future__ import annotations

import argparse
import json
import os
import re
from pathlib import Path
from typing import Iterable, List, Tuple

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import faiss
import numpy as np
import torch
from transformers import CLIPModel, CLIPProcessor


DEFAULT_ARTIFACTS_DIR = Path("artifacts")
DEFAULT_CANDIDATE_K = 250

STOPWORDS = {
    "a", "an", "and", "are", "as", "at", "by", "for", "from", "in", "into",
    "is", "it", "of", "on", "or", "person", "photo", "picture", "showing",
    "the", "to", "wearing", "with",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Retrieve fashion images without VLM metadata reranking.")
    parser.add_argument("--query", required=True)
    parser.add_argument("--artifacts-dir", type=Path, default=DEFAULT_ARTIFACTS_DIR)
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--candidate-k", type=int, default=DEFAULT_CANDIDATE_K)
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    parser.add_argument("--show-prompts", action="store_true")
    parser.add_argument(
        "--allow-download",
        action="store_true",
        help="Allow Hugging Face downloads if the model is not already cached locally.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    config = read_json(args.artifacts_dir / "index_config.json")
    image_paths = read_json(args.artifacts_dir / "image_paths.json")

    model_name = config["model_name"]
    device = resolve_device(args.device)

    processor = CLIPProcessor.from_pretrained(model_name, local_files_only=not args.allow_download)
    model = CLIPModel.from_pretrained(model_name, local_files_only=not args.allow_download).to(device)
    model.eval()

    index = faiss.read_index(str(args.artifacts_dir / "fashion.index"))
    query_embedding = encode_texts([args.query], processor, model, device)

    candidate_k = min(max(args.top_k, args.candidate_k), len(image_paths))
    scores, indices = index.search(query_embedding, candidate_k)
    rerank_prompts = build_rerank_prompts(args.query)
    rerank_embeddings = encode_texts(rerank_prompts, processor, model, device)

    results = []
    for initial_rank, (idx, score) in enumerate(zip(indices[0], scores[0]), start=1):
        image_vector = index.reconstruct(int(idx)).reshape(1, -1)
        prompt_scores = (image_vector @ rerank_embeddings.T)[0]
        final_score = combine_scores(float(score), prompt_scores)
        results.append(
            {
                "initial_rank": initial_rank,
                "path": image_paths[int(idx)],
                "clip_score": float(score),
                "rerank_score": float(np.mean(prompt_scores)),
                "coverage_score": float(np.percentile(prompt_scores, 25)),
                "final_score": final_score,
            }
        )

    results.sort(key=lambda item: item["final_score"], reverse=True)

    if args.show_prompts:
        print("Rerank prompts:")
        for prompt in rerank_prompts:
            print(f"- {prompt}")
        print()
    print_results(results[: args.top_k])


def resolve_device(device: str) -> str:
    if device == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but torch.cuda.is_available() is False.")
    return device


@torch.inference_mode()
def encode_texts(
    texts: List[str],
    processor: CLIPProcessor,
    model: CLIPModel,
    device: str,
) -> np.ndarray:
    inputs = processor(text=texts, return_tensors="pt", padding=True, truncation=True, max_length=77)
    inputs = {key: value.to(device) for key, value in inputs.items()}
    features = model.get_text_features(**inputs)
    features = torch.nn.functional.normalize(features, dim=-1)
    return features.cpu().numpy().astype("float32")


def build_rerank_prompts(query: str) -> List[str]:
    phrases = extract_query_phrases(query)
    prompts = [
        query,
        f"a fashion photo showing {query}",
        f"a person wearing {query}",
    ]

    for phrase, role in phrases:
        prompts.extend(
            [
                phrase,
                f"a fashion photo showing {phrase}",
                f"a close view of {phrase}",
            ]
        )
        if role == "scene":
            prompts.append(f"a fashion photo in {phrase}")
        else:
            prompts.append(f"a person wearing {phrase}")

    return dedupe(prompts)


def extract_query_phrases(query: str) -> List[Tuple[str, str]]:
    text = normalize_text(query)
    parts = re.split(r"\b(and|with|in|on|at|for|near|beside|plus)\b|,", text)
    phrases: List[Tuple[str, str]] = []
    next_role = "wearable"

    for part in parts:
        if not part:
            continue
        connector = part.strip()
        if connector in {"in", "on", "at", "near", "beside"}:
            next_role = "scene"
            continue
        if connector in {"and", "with", "for", "plus"}:
            next_role = "wearable"
            continue

        chunk = part.strip()
        tokens = content_tokens(chunk)
        if not tokens:
            continue
        if len(tokens) >= 2:
            phrases.append((" ".join(tokens), next_role))
        for phrase in make_ngrams(tokens, max_n=3):
            phrases.append((phrase, next_role))
        next_role = "wearable"

    return dedupe_phrases(phrase for phrase in phrases if len(phrase[0]) > 2)[:18]


def normalize_text(text: str) -> str:
    return re.sub(r"\s+", " ", re.sub(r"[^a-zA-Z0-9\s-]", " ", text.lower())).strip()


def content_tokens(text: str) -> List[str]:
    return [token for token in re.findall(r"[a-zA-Z0-9-]+", text.lower()) if token not in STOPWORDS]


def make_ngrams(tokens: List[str], max_n: int) -> List[str]:
    phrases = []
    for size in range(min(max_n, len(tokens)), 1, -1):
        for start in range(0, len(tokens) - size + 1):
            phrases.append(" ".join(tokens[start : start + size]))
    return phrases


def combine_scores(base_score: float, prompt_scores: np.ndarray) -> float:
    mean_prompt_score = float(np.mean(prompt_scores))
    coverage_score = float(np.percentile(prompt_scores, 25))
    best_prompt_score = float(np.max(prompt_scores))
    return (
        0.40 * base_score
        + 0.35 * mean_prompt_score
        + 0.15 * coverage_score
        + 0.10 * best_prompt_score
    )


def dedupe(values: Iterable[str]) -> List[str]:
    seen = set()
    unique = []
    for value in values:
        normalized = " ".join(value.split())
        key = normalized.lower()
        if key and key not in seen:
            seen.add(key)
            unique.append(normalized)
    return unique


def dedupe_phrases(values: Iterable[Tuple[str, str]]) -> List[Tuple[str, str]]:
    seen = set()
    unique = []
    for text, role in values:
        normalized = " ".join(text.split())
        key = (normalized.lower(), role)
        if normalized and key not in seen:
            seen.add(key)
            unique.append((normalized, role))
    return unique


def print_results(results: List[dict]) -> None:
    for rank, item in enumerate(results, start=1):
        print(
            f"{rank}. {item['path']} | "
            f"clip={item['clip_score']:.4f} | "
            f"rerank={item['rerank_score']:.4f} | "
            f"coverage={item['coverage_score']:.4f} | "
            f"final={item['final_score']:.4f} | "
            f"initial_rank={item['initial_rank']}"
        )


def read_json(path: Path):
    if not path.exists():
        raise FileNotFoundError(f"Missing required file: {path}")
    with path.open("r", encoding="utf-8") as file:
        return json.load(file)


if __name__ == "__main__":
    main()
