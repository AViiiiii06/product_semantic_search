# Indexing Workflow

This folder contains Part A of the assignment.

`indexer.py` converts raw images into a searchable index:

1. Reads images from `test/`.
2. Encodes each image with FashionCLIP.
3. Normalizes embeddings so inner product behaves like cosine similarity.
4. Stores vectors in `artifacts/fashion.index` using FAISS.
5. Stores image paths in `artifacts/image_paths.json`.
6. Stores Groq VLM metadata in `artifacts/metadata.json`.

Run:

```powershell
set GROQ_API_KEY=your_key_here
python indexing/indexer.py --image-dir test --output-dir artifacts --batch-size 16
```

Quick test on a small subset:

```powershell
set GROQ_API_KEY=your_key_here
python indexing/indexer.py --image-dir test --output-dir artifacts --metadata-limit 20
```
