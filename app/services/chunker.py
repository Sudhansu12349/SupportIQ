
def chunk_pages(
    pages: list[dict],
    chunk_size: int = 1000,
    overlap: int = 150
) -> list[dict]:
    """Split page text into chunks and preserve page metadata."""

    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")

    if overlap < 0 or overlap >= chunk_size:
        raise ValueError(
            "overlap must be non-negative and smaller than chunk_size"
        )

    chunks = []

    for page in pages:
        text = page["text"]
        start = 0

        while start < len(text):
            end = min(start + chunk_size, len(text))
            content = text[start:end].strip()

            if content:
                chunks.append({
                    "chunk_id": f"{page['source']}_page_{page['page_number']}_chunk_{len(chunks) + 1}",
                    "content": content,
                    "source": page["source"],
                    "page_number": page["page_number"],
                })

            if end == len(text):
                break

            start = end - overlap

    return chunks
