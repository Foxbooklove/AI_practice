from pydantic import BaseModel
import ollama

class Condition(BaseModel):
    material: str
    temperature_c: float
    pressure_torr: float
    result: str

text = """
샘플 A는 850도, 10 Torr에서 SiGe를 증착했고 표면 거칠기가 0.8nm였다.
샘플 B는 동일 조건에서 압력만 50 Torr로 올렸더니 거칠기가 2.1nm로 악화됐다.
"""

resp = ollama.chat(
    model='gemma4:12b',
    messages=[{
        'role': 'user',
        'content': f'다음에서 실험 조건을 추출해. 스키마: {Condition.model_json_schema()}\n\n{text}'
    }],
    format=Condition.model_json_schema(),
    options={'temperature': 0, 'num_ctx': 8192},
)

print(Condition.model_validate_json(resp.message.content))