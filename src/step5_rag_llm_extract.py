import os
import json
import csv
import argparse
import re
import numpy as np
import requests
from sentence_transformers import SentenceTransformer


# ----------------------------
# Retrieval (same logic as Step 4)
# ----------------------------

def l2_normalize(vec: np.ndarray) -> np.ndarray:
    norm = np.linalg.norm(vec)
    if norm == 0:
        return vec
    return vec / norm


def retrieve_top_k(chunks, embeddings: np.ndarray, query: str, model_name: str, top_k: int):
    model = SentenceTransformer(model_name)

    q_emb = model.encode([query], convert_to_numpy=True).astype(np.float32)[0]
    q_emb = l2_normalize(q_emb)

    scores = embeddings @ q_emb
    k = min(top_k, len(scores))

    top_idx = np.argpartition(-scores, kth=k - 1)[:k]
    top_idx = top_idx[np.argsort(-scores[top_idx])]

    results = []
    for rank, idx in enumerate(top_idx, start=1):
        c = chunks[int(idx)]
        results.append({
            "rank": rank,
            "score": float(scores[int(idx)]),
            "chunk_id": c["chunk_id"],
            "page_number": c["page_number"],
            "text": c["text"]
        })
    return results


# ----------------------------
# LLM call (OpenAI-compatible)
# ----------------------------

def call_chat_completions(base_url: str, api_key: str, model: str, messages, temperature: float = 0.0) -> str:
    url = base_url.rstrip("/") + "/v1/chat/completions"
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    payload = {
        "model": model,
        "messages": messages,
        "temperature": temperature,
    }
    r = requests.post(url, headers=headers, json=payload, timeout=90)
    r.raise_for_status()
    data = r.json()
    return data["choices"][0]["message"]["content"]


def build_messages(query: str, retrieved_chunks, max_context_chars: int = 12000):
    """
    Create a strict extraction prompt.
    - Uses only retrieved text chunks (text-only requirement).
    - Forces JSON array output.
    """
    context_parts = []
    for ch in retrieved_chunks:
        context_parts.append(
            f"[Page {ch['page_number']} | Chunk {ch['chunk_id']}]\n{ch['text']}"
        )

    context = "\n\n---\n\n".join(context_parts)

    # Hard cap context if needed (avoid huge prompts)
    if len(context) > max_context_chars:
        context = context[:max_context_chars] + "\n\n[TRUNCATED]\n"

    system = (
        "You extract vehicle specifications from service manual text.\n"
        "Rules:\n"
        "1) Use ONLY the provided context. Do NOT guess.\n"
        "2) If the spec is not explicitly in the text, omit it.\n"
        "3) Output MUST be valid JSON only (no markdown, no extra text).\n"
        "4) Output format: a JSON array of objects.\n\n"
        "Each object must have:\n"
        '  "component": string,\n'
        '  "spec_type": string,   // e.g., Torque, Capacity, PartNumber\n'
        '  "value": string,\n'
        '  "unit": string,\n'
        '  "source_page": number\n'
    )

    user = (
        f"Query: {query}\n\n"
        f"Context:\n{context}\n\n"
        "Extract the specifications that answer the query."
    )

    return [
        {"role": "system", "content": system},
        {"role": "user", "content": user},
    ]


def parse_json_array(s: str):
    """
    Robust JSON extraction:
    - Try direct json.loads
    - Else pull the first [...] block and parse
    """
    s = s.strip()
    try:
        obj = json.loads(s)
        return obj if isinstance(obj, list) else []
    except Exception:
        pass

    m = re.search(r"\[.*\]", s, flags=re.DOTALL)
    if not m:
        return []
    try:
        obj = json.loads(m.group(0))
        return obj if isinstance(obj, list) else []
    except Exception:
        return []


# ----------------------------
# Save outputs (JSON + CSV)
# ----------------------------

def save_json(path: str, data):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def save_csv(path: str, rows):
    fieldnames = ["component", "spec_type", "value", "unit", "source_page"]
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, "") for k in fieldnames})


# ----------------------------
# Main
# ----------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--query", default="", help="Example: Torque for brake caliper bolts")
    parser.add_argument("--top_k", type=int, default=8, help="How many chunks to retrieve")
    parser.add_argument("--embed_model", default="sentence-transformers/all-MiniLM-L6-v2",
                        help="Same embedding model used in Step 3/4")
    parser.add_argument("--chunks", default="outputs/chunks.json", help="Chunks from Step 2")
    parser.add_argument("--emb", default="outputs/embeddings.npy", help="Embeddings from Step 3")
    parser.add_argument("--out_json", default="outputs/results.json", help="Final JSON output")
    parser.add_argument("--out_csv", default="outputs/results.csv", help="Final CSV output")
    args = parser.parse_args()

    if not args.query.strip():
        args.query = input("Enter your query: ").strip()

    # Check inputs
    if not os.path.exists(args.chunks):
        raise FileNotFoundError(f"Missing {args.chunks}. Run Step 2 first.")
    if not os.path.exists(args.emb):
        raise FileNotFoundError(f"Missing {args.emb}. Run Step 3 first.")

    # LLM config
    api_key = os.getenv("OPENAI_API_KEY", "").strip()
    base_url = os.getenv("OPENAI_BASE_URL", "https://api.openai.com").strip()
    llm_model = os.getenv("OPENAI_MODEL", "gpt-4o-mini").strip()

    if not api_key:
        raise ValueError("Set OPENAI_API_KEY in your environment before running Step 5.")

    # Load data
    with open(args.chunks, "r", encoding="utf-8") as f:
        chunks = json.load(f)
    embeddings = np.load(args.emb)

    if len(chunks) != embeddings.shape[0]:
        raise ValueError("Chunks/embeddings mismatch. Re-run Step 3 after Step 2.")

    # Retrieve
    retrieved = retrieve_top_k(chunks, embeddings, args.query, args.embed_model, args.top_k)

    # LLM extract
    messages = build_messages(args.query, retrieved)
    raw = call_chat_completions(base_url, api_key, llm_model, messages, temperature=0.0)
    extracted = parse_json_array(raw)

    # Save outputs
    os.makedirs("outputs", exist_ok=True)
    save_json(args.out_json, extracted)
    save_csv(args.out_csv, extracted)

    print(f"\n✅ Retrieved {len(retrieved)} chunks, extracted {len(extracted)} specs.")
    print(f"✅ JSON saved: {args.out_json}")
    print(f"✅ CSV saved:  {args.out_csv}")

    # Print a small preview
    if extracted:
        print("\nPreview (first 3):")
        for r in extracted[:3]:
            print(r)
    else:
        print("\n⚠️ No specs extracted. Try increasing --top_k or changing query wording.")


if __name__ == "__main__":
    main()
