# YOLO 커스텀 클래스 재학습

## 설치
```
pip install ultralytics
```
현재 환경의 torch는 CPU 빌드다. GPU로 돌리려면 CUDA 빌드 torch를 먼저 깔아야 한다.

## 데이터 채우기
`dataset/`은 지금 빈 placeholder다. 아래 규칙대로 파일만 넣으면 그대로 돌아간다.

```
dataset/
  images/train/img001.jpg
  images/val/img900.jpg
  labels/train/img001.txt     # 이미지와 확장자만 다른 같은 이름
  labels/val/img900.txt
```

라벨은 한 줄에 박스 하나, 값 5개를 공백으로 구분한다.

```
<class_id> <cx> <cy> <w> <h>
```
- `class_id`: `data.yaml`의 names 인덱스 (0=scratch, 1=dent, 2=crack)
- 나머지 넷: 이미지 크기로 나눈 0~1 정규화 값. 좌상단 좌표가 아니라 **중심** 좌표다.

예 (640x480 이미지에서 (100,80)~(220,200) 박스가 scratch):
```
0 0.250 0.292 0.188 0.250
```
객체가 없는 이미지는 빈 .txt를 두면 배경 샘플로 쓰인다.

## 실행
```
python train.py       # runs/defect/weights/best.pt 생성
python predict.py     # val 이미지에 추론, 결과 이미지 저장
```

클래스를 바꾸려면 `data.yaml`의 names만 고치면 된다. 헤드는 클래스 수에 맞춰 자동으로 다시 만들어진다.
