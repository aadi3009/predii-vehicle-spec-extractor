import os
import json
import argparse
import numpy as np
from sentence_transformers import SentenceTransformer


def l2_normalize(mat: np.ndarray) -> np.ndarray:
    """
    Normalize vectors so cosine similarity becomes dot-product.
    """
    norms = np.linalg.norm(mat, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return mat / norms


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--chunks", default="outputs/chunks.json", help="Chunks JSON from Step 2")
    parser.add_argument("--out", default="outputs/embeddings.npy", help="Where to save embeddings")
    parser.add_argument("--model", default="sentence-transformers/all-MiniLM-L6-v2", help="Embedding model name")
    parser.add_argument("--batch_size", type=int, default=64, help="Batch size for embedding")
    args = parser.parse_args()

    if not os.path.exists(args.chunks):
        raise FileNotFoundError(f"Missing {args.chunks}. Run Step 2 first.")

    os.makedirs("outputs", exist_ok=True)

    with open(args.chunks, "r", encoding="utf-8") as f:
        chunks = json.load(f)

    texts = [c["text"] for c in chunks]
    print(f"✅ Loaded {len(texts)} chunks")

    # Load embedding model
    print(f"🔧 Loading embedding model: {args.model}")
    model = SentenceTransformer(args.model)

    # Encode to embeddings (float32 for compactness)
    print("⚙️ Creating embeddings...")
    emb = model.encode(
        texts,
        batch_size=args.batch_size,
        show_progress_bar=True,
        convert_to_numpy=True
    ).astype(np.float32)

    # Normalize so later retrieval is just dot-product
    emb = l2_normalize(emb)

    # Save embeddings
    np.save(args.out, emb)

    # Save small info file (helps reproducibility)
    info = {
        "chunks_file": args.chunks,
        "embeddings_file": args.out,
        "embedding_model": args.model,
        "num_chunks": len(chunks),
        "embedding_dim": int(emb.shape[1]),
        "normalized": True
    }
    with open("outputs/embedding_info.json", "w", encoding="utf-8") as f:
        json.dump(info, f, indent=2)

    print(f"✅ Saved embeddings to: {args.out}")
    print("✅ Saved info to: outputs/embedding_info.json")


if __name__ == "__main__":
    main()
