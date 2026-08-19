from faster_whisper import WhisperModel

model = WhisperModel("large-v3-turbo", device="cuda", compute_type="float16")

segments, info = model.transcribe(
    "meeting.m4a",
    language="ko",
    vad_filter=True,          # 무음 구간 제거
    word_timestamps=True,
)

for seg in segments:
    print(f"[{seg.start:6.1f}-{seg.end:6.1f}] {seg.text}")