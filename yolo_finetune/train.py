"""YOLO 커스텀 클래스 재학습 최소 예제

data.yaml의 names에 맞춰 검출 헤드가 새로 만들어지고,
백본은 COCO 사전학습 가중치에서 이어받아 미세조정된다.
"""
from pathlib import Path
import torch
from ultralytics import YOLO

HERE = Path(__file__).parent
DEVICE = 0 if torch.cuda.is_available() else 'cpu'

model = YOLO('yolo11n.pt')          # 없으면 자동 다운로드. 정확도 필요하면 yolo11s/m

model.train(
    data=str(HERE / 'data.yaml'),
    epochs=50,
    imgsz=640,
    batch=8,                        # OOM이면 줄인다
    device=DEVICE,
    project=str(HERE / 'runs'),
    name='defect',
    patience=10,                    # val 지표가 10에폭 정체하면 조기 종료
    freeze=10,                      # 데이터가 적을 때 백본 앞단 동결
    seed=0,
)

m = model.val().box                 # 학습 종료 시점의 best 가중치로 평가
print(f'mAP50={m.map50:.3f} mAP50-95={m.map:.3f}')
for i, name in model.names.items():
    print(f'  {name}: mAP50={m.ap50[i]:.3f}')
