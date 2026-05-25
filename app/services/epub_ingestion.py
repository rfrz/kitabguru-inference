from __future__ import annotations

import re
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
    heading: str = ""
    heading_number: Optional[int] = None


ARABIC_ORDINALS: dict[str, int] = {
    "الأول": 1,
    "الأولى": 1,
    "الاول": 1,
    "الاولى": 1,
    "الثاني": 2,
    "الثانية": 2,
    "الثالث": 3,
    "الثالثة": 3,
    "الرابع": 4,
    "الرابعة": 4,
    "الخامس": 5,
    "الخامسة": 5,
    "السادس": 6,
    "السادسة": 6,
    "السابع": 7,
    "السابعة": 7,
    "الثامن": 8,
    "الثامنة": 8,
    "التاسع": 9,
    "التاسعة": 9,
    "العاشر": 10,
    "العاشرة": 10,
}

NUMBERED_HEADING_RE = re.compile(
    r"^\s*[\(\[]?\s*([0-9٠-٩۰-۹]{1,3})\s*[\)\].:\-–—/،]\s*(.+?)\s*$"
)
ORDINAL_HEADING_RE = re.compile(
    r"^\s*(?:الطريقة|الوسيلة|الأمر|السبب)?\s*"
    r"(الأول|الأولى|الاول|الاولى|الثاني|الثانية|الثالث|الثالثة|الرابع|الرابعة|"
    r"الخامس|الخامسة|السادس|السادسة|السابع|السابعة|الثامن|الثامنة|التاسع|التاسعة|"
    r"العاشر|العاشرة)\s*[:\-–—/،]\s*(.+?)\s*$"
)


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
        chapter_chunks = _chunk_chapter_text(
            chapter,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
        )
        chunks.extend(chapter_chunks)
    return chunks


def _chunk_chapter_text(
    chapter: EpubChapter,
    *,
    chunk_size: int,
    chunk_overlap: int,
) -> list[Chunk]:
    lines = [line.strip() for line in chapter.text.splitlines() if line.strip()]
    sections = _split_numbered_sections(lines)

    chunks: list[Chunk] = []
    chunk_index = 0
    for section in sections:
        for text in _split_section_text(
            section["lines"],
            heading=section["heading"],
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
        ):
            chunks.append(
                Chunk(
                    chapter=chapter.chapter,
                    chunk_index=chunk_index,
                    text=text,
                    heading=section["heading"],
                    heading_number=section["heading_number"],
                )
            )
            chunk_index += 1
    return chunks


def _split_numbered_sections(lines: list[str]) -> list[dict[str, object]]:
    sections: list[dict[str, object]] = []
    current_lines: list[str] = []
    current_heading = ""
    current_heading_number: Optional[int] = None

    for line in lines:
        heading_number, heading = parse_numbered_heading(line)
        if heading_number is not None and current_lines:
            sections.append(
                {
                    "lines": current_lines,
                    "heading": current_heading,
                    "heading_number": current_heading_number,
                }
            )
            current_lines = []

        if heading_number is not None:
            current_heading = heading
            current_heading_number = heading_number
        current_lines.append(line)

    if current_lines:
        sections.append(
            {
                "lines": current_lines,
                "heading": current_heading,
                "heading_number": current_heading_number,
            }
        )
    return sections


def _split_section_text(
    lines: list[str],
    *,
    heading: str,
    chunk_size: int,
    chunk_overlap: int,
) -> list[str]:
    chunks: list[str] = []
    current: list[str] = []

    for line in lines:
        proposed = "\n".join([*current, line]).strip()
        if current and len(proposed) > chunk_size:
            chunks.append("\n".join(current).strip())
            current = _continuation_seed(current, heading, chunk_overlap)

        if len(line) > chunk_size:
            if current:
                chunks.append("\n".join(current).strip())
                current = _continuation_seed(current, heading, chunk_overlap)
            chunks.extend(_split_long_line(line, heading=heading, chunk_size=chunk_size, chunk_overlap=chunk_overlap))
            current = []
            continue

        current.append(line)

    if current:
        chunks.append("\n".join(current).strip())

    return [chunk for chunk in chunks if chunk]


def _continuation_seed(lines: list[str], heading: str, chunk_overlap: int) -> list[str]:
    if chunk_overlap <= 0:
        return [heading] if heading else []

    seed: list[str] = []
    total = 0
    for line in reversed(lines):
        if heading and line == heading:
            continue
        next_total = total + len(line)
        if seed and next_total > chunk_overlap:
            break
        seed.insert(0, line)
        total = next_total

    if heading and (not seed or seed[0] != heading):
        seed.insert(0, heading)
    return seed


def _split_long_line(
    line: str,
    *,
    heading: str,
    chunk_size: int,
    chunk_overlap: int,
) -> list[str]:
    chunks: list[str] = []
    prefix = f"{heading}\n" if heading and not line.startswith(heading) else ""
    available = max(1, chunk_size - len(prefix))
    start = 0
    while start < len(line):
        end = min(start + available, len(line))
        chunks.append(f"{prefix}{line[start:end]}".strip())
        if end >= len(line):
            break
        start = max(start + 1, end - chunk_overlap)
    return chunks


def parse_numbered_heading(line: str) -> tuple[Optional[int], str]:
    numbered = NUMBERED_HEADING_RE.match(line)
    if numbered:
        return _parse_int(numbered.group(1)), numbered.group(2).strip()

    ordinal = ORDINAL_HEADING_RE.match(line)
    if ordinal:
        return ARABIC_ORDINALS[ordinal.group(1)], ordinal.group(2).strip()

    return None, ""


def _parse_int(value: str) -> int:
    translation = str.maketrans("٠١٢٣٤٥٦٧٨٩۰۱۲۳۴۵۶۷۸۹", "01234567890123456789")
    return int(value.translate(translation))


def _metadata_first(book, key: str) -> Optional[str]:
    metadata = book.get_metadata("DC", key)
    if not metadata:
        return None
    value = metadata[0][0]
    return str(value).strip() or None
