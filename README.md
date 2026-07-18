# Product Semantic Search

Fashion image retrieval using FashionCLIP embeddings, FAISS vector search, and optional Groq vision metadata.

The repository contains two working retrieval paths:

- **VLM-enhanced path**: builds FashionCLIP embeddings and uses Groq vision metadata for reranking.
- **VLM-free path**: builds and searches the same FAISS index without Groq calls, then reranks candidates with dynamic CLIP text prompts.

The VLM-free path is useful when API rate limits block metadata generation. The VLM-enhanced path is more accurate for detailed attribute queries when metadata is available.

## Project Structure

```text
.
├── artifacts/
│   ├── fashion.index
│   ├── image_paths.json
│   ├── index_config.json
│   └── metadata.json
├── indexing/
│   ├── indexer.py
│   ├── indexing_without_vlm.py
│   └── README.md
├── retrieval/
│   ├── retriever.py
│   ├── retrieval_without_vlm.py
│   └── README.md
├── requirements.txt
└── README.md
```

## Installation

Create and activate a virtual environment, then install dependencies:

```powershell
python -m venv myenv
myenv\Scripts\activate
pip install -r requirements.txt
```

The code uses `patrickjohncyh/fashion-clip` by default. The first model load may require internet access so Hugging Face can download the model. After that, the no-VLM scripts load cached model files by default.

## Artifacts

The retrieval scripts expect these files in `artifacts/`:

```text
fashion.index
image_paths.json
index_config.json
```

`metadata.json` is required only for `retrieval/retriever.py`, the VLM-enhanced retriever.

## VLM-Free Indexing

Use this when you want to avoid Groq/API calls:

```powershell
myenv\Scripts\python.exe indexing\indexing_without_vlm.py --image-dir test --output-dir artifacts --batch-size 16
```

This script:

1. Finds images in `--image-dir`.
2. Encodes images with FashionCLIP.
3. Normalizes embeddings.
4. Saves a FAISS `IndexFlatIP` index to `artifacts/fashion.index`.
5. Saves `image_paths.json` and `index_config.json`.

It does not create or require `metadata.json`.

If the model is not cached and you want to allow download:

```powershell
myenv\Scripts\python.exe indexing\indexing_without_vlm.py --image-dir test --output-dir artifacts --allow-download
```

## VLM-Free Retrieval

Run:

```powershell
myenv\Scripts\python.exe retrieval\retrieval_without_vlm.py --query "A red tie and a white shirt in a formal setting" --top-k 5
```

Useful options:

```powershell
myenv\Scripts\python.exe retrieval\retrieval_without_vlm.py `
  --query "blue dress outdoor fashion" `
  --top-k 5 `
  --candidate-k 250 `
  --show-prompts
```

This script:

1. Embeds the full query with FashionCLIP.
2. Searches FAISS for a candidate pool.
3. Dynamically extracts query phrases from the user query.
4. Builds reranking prompts from those phrases.
5. Scores each candidate against all prompts.
6. Combines base CLIP score, mean prompt score, coverage score, and best prompt score.

Current scoring formula:

```text
final = 0.40 * clip_score
      + 0.35 * mean_prompt_score
      + 0.15 * coverage_score
      + 0.10 * best_prompt_score
```

The VLM-free retriever is dynamic, but it is still embedding-only. It can struggle with fine-grained attribute binding, such as making sure "red" belongs specifically to "tie" and not another red object in the image.

## VLM-Enhanced Indexing

Use this when you have a Groq API key and want structured metadata:

```powershell
$env:GROQ_API_KEY="your_key_here"
myenv\Scripts\python.exe indexing\indexer.py --image-dir test --output-dir artifacts --batch-size 16
```

For a small metadata test:

```powershell
$env:GROQ_API_KEY="your_key_here"
myenv\Scripts\python.exe indexing\indexer.py --image-dir test --output-dir artifacts --metadata-limit 20
```

This script:

1. Builds the same FashionCLIP + FAISS index.
2. Calls Groq vision once per selected image.
3. Saves structured metadata in `artifacts/metadata.json`.

The metadata schema includes:

```json
{
  "caption": "short fashion-focused caption",
  "clothing_items": [
    {
      "type": "shirt",
      "color": "white",
      "body_region": "upper"
    }
  ],
  "style": ["formal"],
  "environment": "office",
  "objects": [],
  "weather": "indoor"
}
```

## VLM-Enhanced Retrieval

Run this only after `metadata.json` has useful records:

```powershell
myenv\Scripts\python.exe retrieval\retriever.py --query "A red tie and a white shirt in a formal setting" --top-k 5
```

With metadata output:

```powershell
myenv\Scripts\python.exe retrieval\retriever.py `
  --query "A red tie and a white shirt in a formal setting" `
  --top-k 5 `
  --candidate-k 50 `
  --metadata-weight 0.12 `
  --show-metadata
```

This script:

1. Embeds the query.
2. Searches FAISS for candidates.
3. Parses colors, garments, environments, and styles from the query.
4. Matches those terms against cached VLM metadata.
5. Reranks candidates using CLIP score plus metadata score.

Current scoring formula:

```text
final = clip_score + metadata_weight * metadata_score
```

The default `metadata_weight` is `0.12`.

## Choosing an Approach

Use `indexing_without_vlm.py` and `retrieval_without_vlm.py` when:

- Groq rate limits are a problem.
- You need offline retrieval after model download.
- You want fast indexing.
- Approximate semantic results are acceptable.

Use `indexer.py` and `retriever.py` when:

- You need better matching for detailed fashion attributes.
- You have a valid `GROQ_API_KEY`.
- You can tolerate slower indexing and API rate limits.
- `metadata.json` has been generated successfully.

## Known Limitations

- Embedding-only retrieval cannot reliably guarantee object-level attribute binding.
- The no-VLM retriever may confuse a query like "red tie" with another red accessory.
- The VLM-enhanced retriever depends on metadata quality and Groq availability.
- `metadata.json` may be empty if indexing was interrupted by API rate limits.

## Possible Improvements

- Use a stronger fashion embedding model, such as Marqo FashionCLIP or FashionSigLIP.
- Add top-k lazy VLM reranking instead of generating metadata for every image during indexing.
- Cache VLM checks incrementally during retrieval.
- Add a small evaluation script with manually labeled ground truth for project-specific metrics.

## Security Notes

Do not commit `.env` or API keys. This repository's `.gitignore` excludes `.env`, virtual environments, caches, and local test images.
