import os
import json
import argparse
import numpy as np
from sentence_transformers import SentenceTransformer


def l2_normalize(vec: np.ndarray) -> np.ndarray:
    norm = np.linalg.norm(vec)
    if norm == 0:
        return vec
    return vec / norm


def retrieve_top_k(chunks, embeddings: np.ndarray, query: str, model_name: str, top_k: int):
    model = SentenceTransformer(model_name)

    q_emb = model.encode([query], convert_to_numpy=True).astype(np.float32)[0]
    q_emb = l2_normalize(q_emb)

    # embeddings are already normalized => cosine similarity = dot product
    scores = embeddings @ q_emb  # shape: (num_chunks,)

    # Top-k indices by score
    k = min(top_k, len(scores))
    top_idx = np.argpartition(-scores, kth=k-1)[:k]
    top_idx = top_idx[np.argsort(-scores[top_idx])]  # sort top-k properly

    results = []
    for rank, idx in enumerate(top_idx, start=1):
        c = chunks[int(idx)]
        results.append({
            "rank": rank,
            "score": float(scores[int(idx)]),
            "chunk_id": c["chunk_id"],
            "page_number": c["page_number"],
            "type": c.get("type", "page_text"),
            "text": c["text"]
        })

    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--query", default="", help="Your query text")
    parser.add_argument("--top_k", type=int, default=8, help="How many chunks to retrieve")
    parser.add_argument("--model", default="sentence-transformers/all-MiniLM-L6-v2", help="Embedding model used in Step 3")
    parser.add_argument("--chunks", default="outputs/chunks.json", help="Chunks JSON")
    parser.add_argument("--emb", default="outputs/embeddings.npy", help="Embeddings file")
    parser.add_argument("--out", default="outputs/retrieved_chunks.json", help="Save retrieved results here")
    args = parser.parse_args()

    if not args.query.strip():
        # simple interactive fallback
        args.query = input("Enter your query: ").strip()

    if not os.path.exists(args.chunks):
        raise FileNotFoundError(f"Missing {args.chunks}. Run Step 2 first.")
    if not os.path.exists(args.emb):
        raise FileNotFoundError(f"Missing {args.emb}. Run Step 3 first.")

    with open(args.chunks, "r", encoding="utf-8") as f:
        chunks = json.load(f)

    embeddings = np.load(args.emb)

    if len(chunks) != embeddings.shape[0]:
        raise ValueError(
            f"Mismatch: chunks={len(chunks)} but embeddings={embeddings.shape[0]}. "
            "Re-run Step 3 after regenerating chunks."
        )

    results = retrieve_top_k(
        chunks=chunks,
        embeddings=embeddings,
        query=args.query,
        model_name=args.model,
        top_k=args.top_k
    )

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    print("\n====================")
    print(f"QUERY: {args.query}")
    print("====================\n")

    for r in results:
        print(f"[{r['rank']}] score={r['score']:.4f}  page={r['page_number']}  chunk={r['chunk_id']}")
        preview = r["text"].replace("\n", " ")
        print("    " + preview[:220] + ("..." if len(preview) > 220 else ""))
        print()

    print(f"✅ Saved retrieved chunks to: {args.out}")


if __name__ == "__main__":
    main()
