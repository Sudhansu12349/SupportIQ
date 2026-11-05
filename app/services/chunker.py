
from langchain_text_splitters import RecursiveCharacterTextSplitter


def chunk_pages(
    pages: list[dict],
    chunk_size: int = 1000,
    overlap: int = 150
) -> list[dict]:
    """Split page text using LangChain and preserve page metadata."""

    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")

    if overlap < 0 or overlap >= chunk_size:
        raise ValueError(
            "overlap must be non-negative and smaller than chunk_size"
        )

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=overlap,
        separators=["\n\n", "\n", " ", ""]
    )

    chunks = []

    for page in pages:
        text = page["text"]

        split_texts = splitter.split_text(text)

        for content in split_texts:
            chunks.append({
                "chunk_id": (
                    f"{page['source']}_page_"
                    f"{page['page_number']}_chunk_{len(chunks) + 1}"
                ),
                "content": content,
                "source": page["source"],
                "page_number": page["page_number"],
            })

    return chunks
