"""Ollama 도구 호출 에이전트 최소 예제"""
from pathlib import Path
import ollama

BASE = Path(__file__).parent / "data"

def list_files(pattern: str = "*") -> str:
    """data 폴더의 파일 목록을 반환한다. pattern은 glob 패턴 (예: '*.csv')"""
    return "\n".join(p.name for p in sorted(BASE.glob(pattern))) or "(없음)"

def read_head(filename: str, n: int = 5) -> str:
    """파일의 앞 n줄을 반환한다."""
    path = BASE / filename
    if not path.is_file():
        return f"error: {filename} 없음"
    with path.open(encoding="utf-8") as f:
        return "".join(line for _, line in zip(range(n), f))

def count_rows(filename: str) -> str:
    """csv 파일의 헤더를 제외한 데이터 행 수를 반환한다."""
    path = BASE / filename
    if not path.is_file():
        return f"error: {filename} 없음"
    with path.open(encoding="utf-8") as f:
        rows = sum(1 for line in f if line.strip())   # 빈 줄은 세지 않는다
    return str(max(0, rows - 1))

TOOLS = {f.__name__: f for f in (list_files, read_head, count_rows)}

def run(question: str, model: str = "qwen3.5:4b", max_steps: int = 10):
    messages = [{"role": "user", "content": question}]
    for step in range(max_steps):
        resp = ollama.chat(model=model, messages=messages, tools=list(TOOLS.values()))
        messages.append(resp.message)

        if not resp.message.tool_calls:
            return resp.message.content

        for call in resp.message.tool_calls:
            name, args = call.function.name, call.function.arguments
            fn = TOOLS.get(name)
            if fn is None:
                result = f"error: 없는 도구 {name}"
            else:
                try:
                    result = fn(**args)
                except Exception as e:
                    result = f"error: {e}"
            print(f"  [{step}] {name}({args}) -> {str(result)[:60]!r}")
            messages.append({"role": "tool", "content": str(result), "tool_name": name})
    return "중단: 최대 스텝 초과"

if __name__ == "__main__":
    print(run("data 폴더에 csv 파일이 몇 개 있고 각각 몇 행인지 알려줘"))