"""재학습한 가중치로 추론"""
from pathlib import Path
from ultralytics import YOLO

HERE = Path(__file__).parent
WEIGHTS = HERE / 'runs' / 'defect' / 'weights' / 'best.pt'

if not WEIGHTS.is_file():
    raise SystemExit(f'가중치 없음: {WEIGHTS} — train.py를 먼저 실행해라')

model = YOLO(str(WEIGHTS))
results = model.predict(
    source=str(HERE / 'dataset' / 'images' / 'val'),
    conf=0.25,
    save=True,                      # 박스 그린 이미지를 runs/predict에 저장
)

for r in results:
    for b in r.boxes:
        x1, y1, x2, y2 = (round(v) for v in b.xyxy[0].tolist())
        print(f'{Path(r.path).name} {model.names[int(b.cls)]} {float(b.conf):.2f} ({x1},{y1},{x2},{y2})')
