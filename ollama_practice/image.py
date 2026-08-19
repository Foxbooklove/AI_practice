from pydantic import BaseModel
from typing import Literal
import ollama

class Receipt(BaseModel):
    date: str                    # YYYY-MM-DD
    merchant: str
    amount: int                  # 실제 총 결제 금액
    payment_method: str          # 없으면 빈 문자열
    category: Literal['소모품', '도서', '식비', '장비', '기타']
    item_count: int
    note: str

PROMPT = f"""영수증 이미지에서 값을 추출해.

주의:
- 거래명세표는 배송비가 별도 행이다. 맨 아래 '실제 총 결제 금액'이 정답이다.
- 카드 매출전표는 '합계금액'이 정답이다. '과세금액'이나 '부가세'가 아니다.
- 결제수단 정보가 없으면 추측하지 말고 빈 문자열로 둬라.

스키마: {Receipt.model_json_schema()}"""

resp = ollama.chat(
    model='qwen3-vl:8b',
    messages=[{'role': 'user', 'content': PROMPT, 'images': ['receipt.png']}],
    format=Receipt.model_json_schema(),
    options={'temperature': 0},
)

r = Receipt.model_validate_json(resp.message.content)
print(r.amount, r.merchant)