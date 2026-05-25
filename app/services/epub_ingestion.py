from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional


@dataclass
class EpubChapter:
    chapter: int
    text: str


@dataclass
class ExtractedEpub:
    title: Optional[str]
    author: Optional[str]
    chapters: list[EpubChapter]


@dataclass
class Chunk:
    chapter: int
    chunk_index: int
    text: str


def extract_epub(path: str | Path) -> ExtractedEpub:
    import ebooklib
    from bs4 import BeautifulSoup
    from ebooklib import epub

    book = epub.read_epub(str(path))
    title = _metadata_first(book, "title")
    author = _metadata_first(book, "creator")

    id_to_item = {item.get_id(): item for item in book.get_items()}
    ordered_items = []
    for spine_entry in book.spine:
        idref = spine_entry[0] if isinstance(spine_entry, tuple) else spine_entry
        item = id_to_item.get(idref)
        if item is not None:
            ordered_items.append(item)

    if not ordered_items:
        ordered_items = list(book.get_items_of_type(ebooklib.ITEM_DOCUMENT))

    chapters: list[EpubChapter] = []
    for index, item in enumerate(ordered_items, start=1):
        if item.get_type() != ebooklib.ITEM_DOCUMENT:
            continue
        soup = BeautifulSoup(item.get_content(), "html.parser")
        text = soup.get_text("\n", strip=True)
        if text:
            chapters.append(EpubChapter(chapter=index, text=text))

    return ExtractedEpub(title=title, author=author, chapters=chapters)


def chunk_chapters(
    chapters: list[EpubChapter],
    *,
    chunk_size: int,
    chunk_overlap: int,
) -> list[Chunk]:
    if chunk_size <= 0:
        raise ValueError("chunk_size must be greater than zero")
    if chunk_overlap < 0:
        raise ValueError("chunk_overlap cannot be negative")
    if chunk_overlap >= chunk_size:
        raise ValueError("chunk_overlap must be smaller than chunk_size")

    chunks: list[Chunk] = []
    for chapter in chapters:
        normalized = "\n".join(line.strip() for line in chapter.text.splitlines() if line.strip())
        start = 0
        chunk_index = 0
        while start < len(normalized):
            end = min(start + chunk_size, len(normalized))
            text = normalized[start:end].strip()
            if text:
                chunks.append(
                    Chunk(
                        chapter=chapter.chapter,
                        chunk_index=chunk_index,
                        text=text,
                    )
                )
                chunk_index += 1
            if end >= len(normalized):
                break
            start = end - chunk_overlap
    return chunks


def _metadata_first(book, key: str) -> Optional[str]:
    metadata = book.get_metadata("DC", key)
    if not metadata:
        return None
    value = metadata[0][0]
    return str(value).strip() or None
