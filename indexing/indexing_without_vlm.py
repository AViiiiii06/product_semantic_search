"""
Indexer without VLM metadata.

Builds a searchable fashion image index using only FashionCLIP image embeddings
and FAISS. This version does not call Groq or any vision-language model, so it
is safe to run when API rate limits are hit.

Example:
    python indexing/indexing_without_vlm.py --image-dir test --output-dir artifacts --batch-size 16
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Iterable, List

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import faiss
import numpy as np
import torch
from PIL import Image
from tqdm import tqdm
from transformers import CLIPModel, CLIPProcessor


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp"}
DEFAULT_MODEL = "patrickjohncyh/fashion-clip"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build a FashionCLIP + FAISS image index without VLM metadata.")
    parser.add_argument("--image-dir", type=Path, default=Path("test"))
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts"))
    parser.add_argument("--model-name", default=DEFAULT_MODEL)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    parser.add_argument(
        "--allow-download",
        action="store_true",
        help="Allow Hugging Face downloads if the model is not already cached locally.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    device = resolve_device(args.device)

    image_paths = list(iter_images(args.image_dir))
    if not image_paths:
        raise RuntimeError(f"No images found in {args.image_dir}")

    args.output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Images found: {len(image_paths)}")
    print(f"Embedding model: {args.model_name}")
    print(f"Device: {device}")
    print("VLM metadata: disabled")

    processor = CLIPProcessor.from_pretrained(args.model_name, local_files_only=not args.allow_download)
    model = CLIPModel.from_pretrained(args.model_name, local_files_only=not args.allow_download).to(device)
    model.eval()

    embeddings = encode_all_images(
        image_paths=image_paths,
        processor=processor,
        model=model,
        device=device,
        batch_size=args.batch_size,
    )

    index = faiss.IndexFlatIP(embeddings.shape[1])
    index.add(embeddings)

    faiss.write_index(index, str(args.output_dir / "fashion.index"))
    write_json(args.output_dir / "image_paths.json", [str(path) for path in image_paths])
    write_json(
        args.output_dir / "index_config.json",
        {
            "model_name": args.model_name,
            "embedding_dim": int(embeddings.shape[1]),
            "num_images": len(image_paths),
            "index_type": "faiss.IndexFlatIP",
            "similarity": "cosine via normalized embeddings",
            "vlm_metadata": False,
        },
    )

    print(f"Saved FAISS index: {args.output_dir / 'fashion.index'}")
    print(f"Saved image path manifest: {args.output_dir / 'image_paths.json'}")
    print(f"Saved index config: {args.output_dir / 'index_config.json'}")


def resolve_device(device: str) -> str:
    if device == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but torch.cuda.is_available() is False.")
    return device


def iter_images(image_dir: Path) -> Iterable[Path]:
    for path in sorted(image_dir.rglob("*")):
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS:
            yield path


def batched(items: List[Path], batch_size: int) -> Iterable[List[Path]]:
    for start in range(0, len(items), batch_size):
        yield items[start : start + batch_size]


@torch.inference_mode()
def encode_all_images(
    image_paths: List[Path],
    processor: CLIPProcessor,
    model: CLIPModel,
    device: str,
    batch_size: int,
) -> np.ndarray:
    batches: List[np.ndarray] = []
    for batch_paths in tqdm(list(batched(image_paths, batch_size)), desc="Embedding images"):
        images = [Image.open(path).convert("RGB") for path in batch_paths]
        inputs = processor(images=images, return_tensors="pt", padding=True)
        inputs = {key: value.to(device) for key, value in inputs.items()}
        features = model.get_image_features(**inputs)
        features = torch.nn.functional.normalize(features, dim=-1)
        batches.append(features.cpu().numpy().astype("float32"))
    return np.vstack(batches)


def write_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file:
        json.dump(value, file, indent=2, ensure_ascii=False)


if __name__ == "__main__":
    main()
