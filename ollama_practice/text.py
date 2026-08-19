from pydantic import BaseModel
import ollama

class Condition(BaseModel):
    sample: str
    material: str
    temperature_c: float
    pressure_torr: float
    roughness_nm: float
    result: str

class Conditions(BaseModel):
    """텍스트에 조건이 여러 개면 하나만 뽑히지 않도록 리스트로 받는다."""
    items: list[Condition]

text = """
샘플 A는 850도, 10 Torr에서 SiGe를 증착했고 표면 거칠기가 0.8nm였다.
샘플 B는 동일 조건에서 압력만 50 Torr로 올렸더니 거칠기가 2.1nm로 악화됐다.
"""

PROMPT = f"""다음에서 실험 조건을 모두 추출해.

주의:
- 샘플마다 항목을 하나씩 만들어라. 언급된 샘플을 빠뜨리지 마라.
- '동일 조건'처럼 생략된 값은 앞 샘플에서 이어받아 채워라.

스키마: {Conditions.model_json_schema()}

{text}"""

resp = ollama.chat(
    model='gemma4:12b',
    messages=[{'role': 'user', 'content': PROMPT}],
    format=Conditions.model_json_schema(),
    options={'temperature': 0, 'num_ctx': 8192},
)

for c in Conditions.model_validate_json(resp.message.content).items:
    print(f'{c.sample}: {c.material} {c.temperature_c}C {c.pressure_torr}Torr -> {c.roughness_nm}nm')
