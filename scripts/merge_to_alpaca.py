import os
import json

INPUT_DIR = "C:/Users/seana/maze/outputs/real_auto/real_25x25/"
OUTPUT_PATH = "C:/Users/seana/maze/outputs/alpaca_25x25.jsonl"

INSTRUCTION_TEMPLATE = (
    "You are a memory-based navigation agent. Below is a full trajectory of your successful exploration "
    "that collected all goals. Based on the trajectory, facing directions, and step-by-step observations, "
    "infer the shortest path that collects all goals. Diagonal moves are allowed. Just return the sequence "
    "of positions in the optimal path. The agent starts at position {start}."
)

merged = []
file_count = 0
missing = 0

for filename in sorted(os.listdir(INPUT_DIR)):
    if not filename.endswith(".json"):
        continue
    path = os.path.join(INPUT_DIR, filename)
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)

        start_pos = data["start_pos"]
        input_data = {
            "goals": data["goals"],
            "trajectory": data["trajectory"],
            "facing": data["facing"],
            "view": data["view"]
        }

        merged.append({
            "instruction": INSTRUCTION_TEMPLATE.format(start=start_pos),
            "input": json.dumps(input_data, ensure_ascii=False),
            "output": ""
        })
        file_count += 1
    except Exception as e:
        print(f"⚠️ Failed to read {filename}: {e}")
        missing += 1

with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
    for item in merged:
        f.write(json.dumps(item, ensure_ascii=False) + "\n")

print(f"✅ Merged {file_count} files into Alpaca format. Skipped {missing} files.")