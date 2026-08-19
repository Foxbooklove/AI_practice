"""Docling으로 PDF를 구조 보존 마크다운으로 변환"""
from pathlib import Path
from docling.document_converter import DocumentConverter

SRC = Path(__file__).parent / "paper.pdf"

conv = DocumentConverter()
doc = conv.convert(SRC).document

md = doc.export_to_markdown()          # 표·제목 계층이 마크다운으로 보존된다
SRC.with_suffix(".md").write_text(md, encoding="utf-8")

print(f"{len(doc.tables)}개 표, {len(md)}자 -> {SRC.with_suffix('.md').name}")
