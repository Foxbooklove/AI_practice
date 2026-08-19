# AI_practice

로컬에서 돌려보는 AI 연습 스크립트 모음. 각 폴더는 독립이고 서로 의존하지 않는다.

| 폴더 | 내용 | 주요 패키지 |
|---|---|---|
| [ollama_practice](ollama_practice) | 로컬 LLM 구조화 출력과 도구 호출 | `ollama`, `pydantic` |
| [image_detect](image_detect) | 사전학습 YOLO로 객체 검출 | `ultralytics` |
| [yolo_finetune](yolo_finetune) | 커스텀 클래스로 YOLO 재학습 | `ultralytics` |
| [anomaly_detection](anomaly_detection) | 정상 이미지만으로 불량 탐지 | `anomalib` |
| [sound_to_text](sound_to_text) | 음성 전사 + 타임스탬프 | `faster-whisper` |
| [parsing](parsing) | PDF를 구조 보존 마크다운으로 | `docling` |
| [rag](rag) | 파싱한 문서를 임베딩해 근거 기반 질의응답 | `ollama`, `numpy` |

```
pip install -r requirements.txt
```

## ollama_practice

- `text.py` — 문장에서 실험 조건을 여러 건 추출. Pydantic 스키마를 `format=`으로 넘겨 JSON을 강제한다.
- `image.py` — 영수증 이미지에서 결제 정보 추출. VLM에 스키마를 함께 준다.
- `tool.py` — 로컬 함수 3개를 도구로 노출하는 최소 에이전트 루프. `data/`에 샘플 csv가 들어 있어 그대로 실행된다.

## 입력 파일

이미지·PDF·오디오 같은 입력 샘플은 커밋하지 않는다(`.gitignore`). 스크립트가 참조하는 이름으로 각 폴더에 직접 넣으면 된다.

| 스크립트 | 필요한 파일 |
|---|---|
| `image_detect/YOLO_practice.py` | `photo.jpg` |
| `ollama_practice/image.py` | `receipt.png` |
| `sound_to_text/transcribe.py` | `meeting.m4a` |
| `parsing/docling_pdf.py` | `paper.pdf` |
| `rag/build_index.py` | `parsing/*.md` — `docling_pdf.py`의 출력 |
| `anomaly_detection/detect.py` | `data/good/*`, `data/bad/*` |
| `yolo_finetune/train.py` | `dataset/` — [README](yolo_finetune/README.md) 참고 |

## rag

`parsing`이 만든 마크다운을 실제로 써먹는 부분. 벡터 DB 없이 numpy 배열 하나로 끝낸다.

```
python parsing/docling_pdf.py     # paper.pdf -> paper.md
python rag/build_index.py         # 청크 분할 + 임베딩 -> index.npz
python rag/ask.py "증착 온도가 수율에 미치는 영향은?"
```

- 청크는 마크다운 제목 경계를 먼저 지키고, 한 절이 800자를 넘으면 100자씩 겹쳐 자른다. 겹침이 없으면 경계에 걸린 문장이 양쪽 모두에서 의미를 잃는다.
- 임베딩을 L2 정규화해 저장하므로 검색은 내적 한 번(`VECS @ q`)이면 끝난다.
- 답변에는 근거 번호와 유사도가 함께 출력된다. 엉뚱한 청크를 물어왔는지 바로 보여야 청킹 전략을 고칠 수 있다.
