"""
Part A: Indexer

Builds a searchable fashion image index from a folder of raw images.

Core workflow:
1. Load images from --image-dir.
2. Extract FashionCLIP image embeddings.
3. Store vectors in a FAISS inner-product index.
4. Call Groq vision models once per image to cache structured fashion metadata.

Example:
    set GROQ_API_KEY=your_key_here
    python indexing/indexer.py --image-dir test --output-dir artifacts --batch-size 16
"""

from __future__ import annotations

import argparse
import base64
import json
import os
import re
from pathlib import Path
from typing import Dict, Iterable, List, Optional

import faiss
import numpy as np
import torch
from dotenv import load_dotenv
from groq import Groq
from PIL import Image
from tqdm import tqdm
from transformers import CLIPModel, CLIPProcessor


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp"}
DEFAULT_MODEL = "patrickjohncyh/fashion-clip"
DEFAULT_GROQ_VISION_MODEL = "qwen/qwen3.6-27b"

GROQ_METADATA_PROMPT = """
Analyze this fashion image for a retrieval system.
Return only valid JSON with this schema:
{
  "caption": "short fashion-focused caption",
  "clothing_items": [
    {"type": "shirt|dress|jacket|pants|tie|coat|etc", "color": "visible color", "body_region": "upper|lower|full|accessory"}
  ],
  "style": ["formal", "casual", "business", "runway", "streetwear", "..."],
  "environment": "office|street|park|home|runway|beach|indoor|outdoor|unknown",
  "objects": ["bench", "audience", "bag", "..."],
  "weather": "sunny|rainy|cloudy|indoor|unknown"
}
Do not invent details. If uncertain, use "unknown" or an empty list.
""".strip()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build a FashionCLIP + FAISS image index.")
    parser.add_argument("--image-dir", type=Path, default=Path("test"))
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts"))
    parser.add_argument("--model-name", default=DEFAULT_MODEL)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    parser.add_argument("--groq-model", default=os.getenv("GROQ_VISION_MODEL", DEFAULT_GROQ_VISION_MODEL))
    parser.add_argument(
        "--metadata-limit",
        type=int,
        default=0,
        help="For quick testing Groq metadata calls. 0 means process all images.",
    )
    return parser.parse_args()


def main() -> None:
    load_dotenv()
    args = parse_args()
    device = resolve_device(args.device)

    image_paths = list(iter_images(args.image_dir))
    if not image_paths:
        raise RuntimeError(f"No images found in {args.image_dir}")

    args.output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Images found: {len(image_paths)}")
    print(f"Embedding model: {args.model_name}")
    print(f"Device: {device}")

    processor = CLIPProcessor.from_pretrained(args.model_name)
    model = CLIPModel.from_pretrained(args.model_name).to(device)
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
        },
    )

    metadata = build_groq_metadata(
        image_paths=image_paths,
        output_dir=args.output_dir,
        groq_model=args.groq_model,
        limit=args.metadata_limit,
    )

    print(f"Saved FAISS index: {args.output_dir / 'fashion.index'}")
    print(f"Saved image path manifest: {args.output_dir / 'image_paths.json'}")
    print(f"Saved Groq metadata cache: {args.output_dir / 'metadata.json'}")
    print(f"Metadata records: {len(metadata)}")


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


def build_groq_metadata(
    image_paths: List[Path],
    output_dir: Path,
    groq_model: str,
    limit: int,
) -> Dict[str, dict]:
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        raise RuntimeError("Set GROQ_API_KEY before running the indexer.")

    metadata_path = output_dir / "metadata.json"
    metadata: Dict[str, dict] = read_json(metadata_path, default={})
    client = Groq(api_key=api_key)

    selected_paths = image_paths if limit <= 0 else image_paths[:limit]
    for path in tqdm(selected_paths, desc="Generating Groq metadata"):
        key = str(path)
        if key in metadata:
            continue
        try:
            metadata[key] = describe_image_with_groq(client, path, groq_model)
        except Exception as exc:
            metadata[key] = {"error": str(exc)}
        write_json(metadata_path, metadata)

    return metadata


def describe_image_with_groq(client: Groq, image_path: Path, model_name: str) -> dict:
    data_url = image_to_data_url(image_path)
    response = client.chat.completions.create(
        model=model_name,
        messages=[
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": GROQ_METADATA_PROMPT},
                    {"type": "image_url", "image_url": {"url": data_url}},
                ],
            }
        ],
        temperature=0,
        response_format={"type": "json_object"},
    )
    text = response.choices[0].message.content or "{}"
    return parse_json_object(text)


def image_to_data_url(image_path: Path) -> str:
    suffix = image_path.suffix.lower()
    mime = "image/png" if suffix == ".png" else "image/jpeg"
    encoded = base64.b64encode(image_path.read_bytes()).decode("utf-8")
    return f"data:{mime};base64,{encoded}"


def parse_json_object(text: str) -> dict:
    text = text.strip()
    text = re.sub(r"^```json\s*", "", text)
    text = re.sub(r"\s*```$", "", text)
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", text, flags=re.DOTALL)
        if match:
            return json.loads(match.group(0))
        return {"raw_response": text}


def write_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file:
        json.dump(value, file, indent=2, ensure_ascii=False)


def read_json(path: Path, default):
    if not path.exists():
        return default
    with path.open("r", encoding="utf-8") as file:
        return json.load(file)


if __name__ == "__main__":
    main()
