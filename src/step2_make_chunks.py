import os
import json
import re
import argparse


def clean_text_keep_table_structure(text: str) -> str:
    """
    Conservative cleaning:
    - Keep line breaks (important for tables)
    - Normalize spaces
    - Remove repeated empty lines
    """
    if not text:
        return ""

    # Normalize line endings
    text = text.replace("\r\n", "\n").replace("\r", "\n")

    # Replace tabs with spaces
    text = text.replace("\t", " ")

    # Normalize multiple spaces BUT keep newlines intact
    # (do this line-by-line to preserve table-like rows)
    lines = []
    for line in text.split("\n"):
        line = re.sub(r"[ ]{2,}", " ", line).strip()
        lines.append(line)

    text = "\n".join(lines)

    # Collapse many blank lines to max 2
    text = re.sub(r"\n{3,}", "\n\n", text)

    return text.strip()


def chunk_text(text: str, chunk_chars: int, overlap: int):
    """
    Simple sliding window chunker.
    Keeps the raw text order.
    """
    chunks = []
    start = 0
    n = len(text)

    while start < n:
        end = min(n, start + chunk_chars)
        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)

        if end == n:
            break
        start = max(0, end - overlap)

    return chunks


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--infile", default="outputs/pages_raw.json", help="Step 1 output")
    parser.add_argument("--pages_out", default="outputs/pages_cleaned.json", help="Cleaned pages output")
    parser.add_argument("--chunks_out", default="outputs/chunks.json", help="Chunks output")
    parser.add_argument("--chunk_chars", type=int, default=1400, help="Chunk size in characters")
    parser.add_argument("--overlap", type=int, default=200, help="Overlap between chunks")
    args = parser.parse_args()

    if not os.path.exists(args.infile):
        raise FileNotFoundError(f"Missing {args.infile}. Run Step 1 first.")

    os.makedirs("outputs", exist_ok=True)

    with open(args.infile, "r", encoding="utf-8") as f:
        pages = json.load(f)

    # Clean pages
    cleaned_pages = []
    for p in pages:
        cleaned_text = clean_text_keep_table_structure(p.get("text", ""))
        cleaned_pages.append({
            "page_number": p["page_number"],
            "text": cleaned_text,
            "char_count": len(cleaned_text),
            "num_images": p.get("num_images", 0),
            "has_text": len(cleaned_text) > 0
        })

    with open(args.pages_out, "w", encoding="utf-8") as f:
        json.dump(cleaned_pages, f, ensure_ascii=False, indent=2)

    # Chunk pages
    all_chunks = []
    chunk_counter = 0

    for p in cleaned_pages:
        page_num = p["page_number"]
        text = p["text"]

        if not text:
            continue

        page_chunks = chunk_text(text, args.chunk_chars, args.overlap)
        for idx, ch in enumerate(page_chunks, start=1):
            chunk_counter += 1
            all_chunks.append({
                "chunk_id": f"p{page_num:04d}_c{idx:02d}",
                "type": "page_text",
                "page_number": page_num,
                "text": ch
            })

    with open(args.chunks_out, "w", encoding="utf-8") as f:
        json.dump(all_chunks, f, ensure_ascii=False, indent=2)

    print(f"✅ Cleaned pages saved to: {args.pages_out}")
    print(f"✅ Total chunks created: {len(all_chunks)}")
    print(f"✅ Chunks saved to: {args.chunks_out}")


if __name__ == "__main__":
    main()
