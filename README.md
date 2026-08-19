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
| `anomaly_detection/detect.py` | `data/good/*`, `data/bad/*` |
| `yolo_finetune/train.py` | `dataset/` — [README](yolo_finetune/README.md) 참고 |
