"""인덱스에서 관련 청크를 찾아 근거로 넘기고 답을 받는다."""
from pathlib import Path
import json
import sys
import numpy as np
import ollama

HERE = Path(__file__).parent
INDEX = HERE / "index.npz"
META = HERE / "index.json"

EMBED_MODEL = "bge-m3"
CHAT_MODEL = "qwen3:8b"
TOP_K = 3

if not INDEX.is_file():
    raise SystemExit("인덱스가 없다 — build_index.py를 먼저 실행해라")

VECS = np.load(INDEX)["vecs"]
META_ROWS = json.loads(META.read_text(encoding="utf-8"))

def search(question: str, k: int = TOP_K) -> list[tuple[float, dict]]:
    q = np.array(ollama.embed(model=EMBED_MODEL, input=[question]).embeddings[0], dtype=np.float32)
    q /= np.linalg.norm(q)
    scores = VECS @ q                       # 둘 다 정규화돼 있어 내적 = 코사인 유사도
    top = np.argsort(-scores)[:k]
    return [(float(scores[i]), META_ROWS[i]) for i in top]

def ask(question: str) -> tuple[str, list[tuple[float, dict]]]:
    hits = search(question)
    context = "\n\n---\n\n".join(
        f"[{i}] {m['source']}\n{m['text']}" for i, (_, m) in enumerate(hits, 1)
    )
    prompt = f"""아래 근거만 사용해 질문에 답해라.

주의:
- 근거에 없는 내용은 지어내지 말고 '문서에 없음'이라고 답해라.
- 사용한 근거 번호를 [1] 형태로 문장 끝에 표시해라.

근거:
{context}

질문: {question}"""
    resp = ollama.chat(
        model=CHAT_MODEL,
        messages=[{"role": "user", "content": prompt}],
        options={"temperature": 0},
    )
    return resp.message.content, hits

if __name__ == "__main__":
    question = " ".join(sys.argv[1:]) or "이 문서의 핵심 결론은 무엇인가?"
    answer, hits = ask(question)
    print(answer)
    print("\n--- 근거 ---")
    for i, (score, m) in enumerate(hits, 1):
        preview = " ".join(m["text"].split())[:70]
        print(f"[{i}] {m['source']}#{m['chunk']} (유사도 {score:.3f}) {preview}...")
