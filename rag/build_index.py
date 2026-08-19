"""docling이 만든 마크다운을 청크로 쪼개 임베딩 인덱스를 만든다.

parsing/docling_pdf.py -> *.md -> (여기) -> index.npz + index.json -> ask.py
"""
from pathlib import Path
import json
import numpy as np
import ollama

HERE = Path(__file__).parent
SRC = HERE.parent / "parsing"        # docling_pdf.py가 .md를 떨구는 곳
INDEX = HERE / "index.npz"
META = HERE / "index.json"

EMBED_MODEL = "bge-m3"
MAX_CHARS = 800                      # 청크 최대 길이
OVERLAP = 100                        # 문장이 경계에서 잘려도 문맥이 남도록 겹친다
BATCH = 32

def split_markdown(md: str) -> list[str]:
    """제목 경계를 우선 지키고, 한 절이 너무 길면 길이로 다시 자른다."""
    sections, buf = [], []
    for line in md.splitlines():
        if line.startswith("#") and buf:
            sections.append("\n".join(buf))
            buf = []
        buf.append(line)
    sections.append("\n".join(buf))

    chunks = []
    for sec in sections:
        sec = sec.strip()
        if not sec:
            continue
        if len(sec) <= MAX_CHARS:
            chunks.append(sec)
            continue
        for i in range(0, len(sec), MAX_CHARS - OVERLAP):
            piece = sec[i:i + MAX_CHARS].strip()
            if piece:
                chunks.append(piece)
    return chunks

def embed(texts: list[str]) -> np.ndarray:
    """L2 정규화까지 해서 반환한다. 이후 내적이 곧 코사인 유사도가 된다."""
    out = []
    for i in range(0, len(texts), BATCH):
        resp = ollama.embed(model=EMBED_MODEL, input=texts[i:i + BATCH])
        out.extend(resp.embeddings)
        print(f"  임베딩 {min(i + BATCH, len(texts))}/{len(texts)}")
    v = np.array(out, dtype=np.float32)
    return v / np.linalg.norm(v, axis=1, keepdims=True)

if __name__ == "__main__":
    docs = sorted(SRC.glob("*.md"))
    if not docs:
        raise SystemExit(f"{SRC}에 .md가 없다 — parsing/docling_pdf.py를 먼저 실행해라")

    chunks, meta = [], []
    for doc in docs:
        for i, c in enumerate(split_markdown(doc.read_text(encoding="utf-8"))):
            chunks.append(c)
            meta.append({"source": doc.name, "chunk": i, "text": c})

    vecs = embed(chunks)
    np.savez(INDEX, vecs=vecs)
    META.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"문서 {len(docs)}개 -> 청크 {len(chunks)}개, 차원 {vecs.shape[1]}")
