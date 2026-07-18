"""
Part B: Retriever

Accepts a natural language query and returns the top-k matching images.

Core workflow:
1. Embed the query with the same FashionCLIP model used by the indexer.
2. Retrieve candidate images from the FAISS index.
3. Parse fashion attributes from the query.
4. Rerank candidates using cached Groq VLM metadata.

Example:
    python retrieval/retriever.py --query "A red tie and a white shirt in a formal setting" --top-k 5
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import faiss
import numpy as np
import torch
from dotenv import load_dotenv
from transformers import CLIPModel, CLIPProcessor


DEFAULT_ARTIFACTS_DIR = Path("artifacts")
DEFAULT_CANDIDATE_K = 50

COLORS = {
    "black", "white", "red", "blue", "green", "yellow", "pink", "purple",
    "orange", "brown", "gray", "grey", "beige", "cream", "gold", "silver",
    "navy", "maroon", "teal", "bright yellow", "light blue", "dark blue",
}

GARMENTS = {
    "shirt", "t-shirt", "tee", "blazer", "jacket", "coat", "raincoat",
    "hoodie", "dress", "skirt", "pants", "trousers", "jeans", "shorts",
    "tie", "suit", "sweater", "top", "outerwear", "button-down",
}

ENVIRONMENTS = {
    "office", "modern office", "street", "city", "urban", "park", "bench",
    "home", "runway", "fashion show", "formal setting", "beach", "outdoor",
    "indoor",
}

STYLES = {
    "formal", "business", "professional", "casual", "weekend", "streetwear",
    "party", "minimal", "sporty", "elegant",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Retrieve fashion images from a built index.")
    parser.add_argument("--query", required=True)
    parser.add_argument("--artifacts-dir", type=Path, default=DEFAULT_ARTIFACTS_DIR)
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--candidate-k", type=int, default=DEFAULT_CANDIDATE_K)
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    parser.add_argument("--metadata-weight", type=float, default=0.12)
    parser.add_argument("--show-metadata", action="store_true")
    return parser.parse_args()


def main() -> None:
    load_dotenv()
    args = parse_args()

    config = read_json(args.artifacts_dir / "index_config.json")
    image_paths = read_json(args.artifacts_dir / "image_paths.json")
    metadata = read_json(args.artifacts_dir / "metadata.json")
    if not metadata:
        raise RuntimeError(
            "metadata.json is empty. Run indexing/indexer.py with GROQ_API_KEY set "
            "so Groq metadata is generated before retrieval."
        )

    model_name = config["model_name"]
    device = resolve_device(args.device)

    processor = CLIPProcessor.from_pretrained(model_name)
    model = CLIPModel.from_pretrained(model_name).to(device)
    model.eval()

    index = faiss.read_index(str(args.artifacts_dir / "fashion.index"))
    query_embedding = encode_query(args.query, processor, model, device)

    candidate_k = min(max(args.top_k, args.candidate_k), len(image_paths))
    scores, indices = index.search(query_embedding, candidate_k)

    query_attributes = parse_query(args.query)
    candidates = []
    for rank, (idx, score) in enumerate(zip(indices[0], scores[0]), start=1):
        path = image_paths[int(idx)]
        meta = metadata.get(path, {})
        metadata_score = attribute_match_score(query_attributes, meta)
        final_score = float(score) + args.metadata_weight * metadata_score
        candidates.append(
            {
                "initial_rank": rank,
                "path": path,
                "clip_score": float(score),
                "metadata_score": metadata_score,
                "final_score": final_score,
                "metadata": meta,
            }
        )

    candidates.sort(key=lambda item: item["final_score"], reverse=True)
    print_results(candidates[: args.top_k], args.show_metadata)


def resolve_device(device: str) -> str:
    if device == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but torch.cuda.is_available() is False.")
    return device


@torch.inference_mode()
def encode_query(
    query: str,
    processor: CLIPProcessor,
    model: CLIPModel,
    device: str,
) -> np.ndarray:
    inputs = processor(text=[query], return_tensors="pt", padding=True, truncation=True)
    inputs = {key: value.to(device) for key, value in inputs.items()}
    features = model.get_text_features(**inputs)
    features = torch.nn.functional.normalize(features, dim=-1)
    return features.cpu().numpy().astype("float32")


def parse_query(query: str) -> Dict[str, List[str]]:
    text = query.lower()
    return {
        "colors": sorted(find_terms(text, COLORS)),
        "garments": sorted(find_terms(text, GARMENTS)),
        "environments": sorted(find_terms(text, ENVIRONMENTS)),
        "styles": sorted(find_terms(text, STYLES)),
    }


def find_terms(text: str, vocabulary: Iterable[str]) -> set:
    matches = set()
    for term in vocabulary:
        pattern = r"(?<!\w)" + re.escape(term) + r"(?!\w)"
        if re.search(pattern, text):
            matches.add(term)
    return matches


def attribute_match_score(query_attributes: Dict[str, List[str]], metadata: dict) -> float:
    if not metadata:
        return 0.0

    flattened = flatten(metadata).lower()
    total = 0
    matched = 0

    for terms in query_attributes.values():
        for term in terms:
            total += 1
            if term.lower() in flattened:
                matched += 1

    if total == 0:
        return 0.0
    return matched / total


def flatten(value) -> str:
    if isinstance(value, dict):
        return " ".join(flatten(item) for item in value.values())
    if isinstance(value, list):
        return " ".join(flatten(item) for item in value)
    return str(value)


def print_results(results: List[dict], show_metadata: bool) -> None:
    for rank, item in enumerate(results, start=1):
        print(
            f"{rank}. {item['path']} | "
            f"clip={item['clip_score']:.4f} | "
            f"metadata={item['metadata_score']:.4f} | "
            f"final={item['final_score']:.4f}"
        )
        if show_metadata and item["metadata"]:
            print(json.dumps(item["metadata"], indent=2, ensure_ascii=False))


def read_json(path: Path, default=None):
    if not path.exists():
        if default is not None:
            return default
        raise FileNotFoundError(f"Missing required file: {path}")
    with path.open("r", encoding="utf-8") as file:
        return json.load(file)


if __name__ == "__main__":
    main()
