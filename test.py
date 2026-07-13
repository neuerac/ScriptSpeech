import json
import re
from pathlib import Path

import soundfile as sf
from funasr import AutoModel

AUDIO_PATH = "test000.mp3"
OUTPUT_DIR = Path("outputs/test")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

model = AutoModel(
    model="paraformer-zh",
    vad_model="fsmn-vad",
    vad_kwargs={"max_single_segment_time": 60000},
    punc_model="ct-punc",
    device="cuda:0",
    disable_update=True,
)

result = model.generate(
    input=AUDIO_PATH,
    batch_size_s=300,
)[0]

text = result["text"]
timestamps = result["timestamp"]  # 毫秒

print("识别文本：", text)
print("时间戳数量：", len(timestamps))

# 标点和空白没有时间戳
punctuation = set(
    "，。！？、；：,.!?;:…—（）()[]【】“”\"'《》<> \t\r\n"
)

timed_positions = [
    i for i, char in enumerate(text)
    if char not in punctuation
]

if len(timed_positions) != len(timestamps):
    raise RuntimeError(
        f"文本字符与时间戳无法一一对应："
        f"有效字符={len(timed_positions)}, 时间戳={len(timestamps)}。"
        "如果包含英文、数字或特殊符号，需要增加 token 对齐。"
    )

# 建立“文本字符位置 -> 时间戳序号”的映射
char_to_timestamp = {
    char_pos: timestamp_index
    for timestamp_index, char_pos in enumerate(timed_positions)
}

# 句号、问号、感叹号和分号视为句末
matches = list(
    re.finditer(r"[^。！？!?；;\n]+(?:[。！？!?；;]+|$)", text)
)

sentences = []

for match in matches:
    sentence_text = match.group().strip()
    timestamp_ids = [
        char_to_timestamp[pos]
        for pos in range(match.start(), match.end())
        if pos in char_to_timestamp
    ]

    if not sentence_text or not timestamp_ids:
        continue

    sentences.append({
        "text": sentence_text,
        "first_timestamp": timestamp_ids[0],
        "last_timestamp": timestamp_ids[-1],
    })

if not sentences:
    raise RuntimeError("没有找到可切分的句子")

audio, sample_rate = sf.read(AUDIO_PATH, always_2d=False)
audio_duration_ms = round(len(audio) * 1000 / sample_rate)

# 相邻句子共用一个切点，避免重叠或丢音频
boundaries = []

for current, following in zip(sentences, sentences[1:]):
    current_end = timestamps[current["last_timestamp"]][1]
    following_start = timestamps[following["first_timestamp"]][0]
    boundaries.append(round((current_end + following_start) / 2))

starts = [max(0, timestamps[sentences[0]["first_timestamp"]][0] - 100)]
starts.extend(boundaries)

ends = boundaries.copy()
ends.append(
    min(
        audio_duration_ms,
        timestamps[sentences[-1]["last_timestamp"]][1] + 100,
    )
)

manifest = []

for index, (sentence, start_ms, end_ms) in enumerate(
    zip(sentences, starts, ends), start=1
):
    start_sample = round(start_ms * sample_rate / 1000)
    end_sample = round(end_ms * sample_rate / 1000)

    output_path = OUTPUT_DIR / f"{index:04d}.wav"

    sf.write(
        output_path,
        audio[start_sample:end_sample],
        sample_rate,
        subtype="PCM_16",
    )

    item = {
        "sentence_id": index,
        "text": sentence["text"],
        "start_ms": start_ms,
        "end_ms": end_ms,
        "duration_ms": end_ms - start_ms,
        "audio_path": str(output_path),
    }
    manifest.append(item)
    print(item)

with open(OUTPUT_DIR / "manifest.jsonl", "w", encoding="utf-8") as file:
    for item in manifest:
        file.write(json.dumps(item, ensure_ascii=False) + "\n")
