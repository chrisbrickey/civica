"""
Deterministic character-window chunker for corpus ingestion.

Splits a section's text into fixed-size character windows with a fixed
overlap between consecutive windows. Prevents sending an oversized
block of text to the downstream embedding/retrieval modules.
"""

CHUNK_SIZE = 800
OVERLAP = 100

def chunk_section(text: str) -> list[str]:
    """Split text into roughly CHUNK_SIZE-character chunks with specified OVERLAP.

    If input is empty or whitespace-only, it returns an empty list.
    If input is shorter than CHUNK_SIZE, it returns a single chunk equal to the input.
    """
    if not text.strip():
        return []

    total_length: int = len(text)
    chunks: list[str] = []

    start: int = 0
    while start < total_length:
        end = min(start + CHUNK_SIZE, total_length)
        chunks.append(text[start:end])
        if end == total_length:
            break
        start += (CHUNK_SIZE - OVERLAP)

    return chunks
