from anomalib.data import Folder
from anomalib.models import Patchcore
from anomalib.engine import Engine

datamodule = Folder(
    name="scratch",
    root="./data",
    normal_dir="good",
    abnormal_dir="bad",
)

model = Patchcore()
engine = Engine()

engine.fit(datamodule=datamodule, model=model)     # 정상 특징을 메모리 뱅크에 저장
engine.test(datamodule=datamodule, model=model)    # 점수 매기고 지표 출력