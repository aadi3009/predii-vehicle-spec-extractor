import os
import json
import fitz  # PyMuPDF
import argparse


def extract_pages(pdf_path: str):
    """
    Extract plain text from each page + basic page stats.
    Returns list of dicts:
      {
        "page_number": 1,
        "text": "...",
        "char_count": 1234,
        "num_images": 2,
        "has_text": true
      }
    """
    doc = fitz.open(pdf_path)
    pages = []

    for i in range(len(doc)):
        page = doc.load_page(i)

        text = page.get_text("text") or ""
        text = text.strip()

        # Count images on the page (diagrams, scanned pages often have many)
        images = page.get_images(full=True)
        num_images = len(images)

        pages.append({
            "page_number": i + 1,
            "text": text,
            "char_count": len(text),
            "num_images": num_images,
            "has_text": len(text) > 0
        })

    doc.close()
    return pages


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pdf", default="data/sample-service-manual 1.pdf", help="Path to the manual PDF")
    parser.add_argument("--out", default="outputs/pages_raw.json", help="Output JSON file path")
    args = parser.parse_args()

    if not os.path.exists(args.pdf):
        raise FileNotFoundError(f"PDF not found at: {args.pdf}")

    os.makedirs(os.path.dirname(args.out), exist_ok=True)

    pages = extract_pages(args.pdf)

    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(pages, f, ensure_ascii=False, indent=2)

    print(f"✅ Extracted {len(pages)} pages")
    print(f"✅ Saved to: {args.out}")


if __name__ == "__main__":
    main()
