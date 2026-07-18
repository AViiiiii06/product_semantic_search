# Retrieval Workflow

This folder contains Part B of the assignment.

`retriever.py` searches the index:

1. Loads `artifacts/fashion.index`.
2. Loads the same FashionCLIP model used during indexing.
3. Embeds the natural language query.
4. Retrieves candidate images using FAISS.
5. Parses query attributes like color, garment, style, and environment.
6. Reranks candidates using cached Groq VLM metadata.

Retrieval expects `artifacts/metadata.json` to exist. Generate it by running the indexer with `GROQ_API_KEY` set.

Run:

```powershell
python retrieval/retriever.py --query "A red tie and a white shirt in a formal setting" --top-k 5
```
