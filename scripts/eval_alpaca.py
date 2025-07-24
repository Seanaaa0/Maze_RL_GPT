import json
import glob
import os

instruction_text = (
    "這是一個 15x15 的非決定性迷宮環境，起點是[0, 0]，每次行動有 30% 機率會偏離預期方向，走向相反的方向。請根據 seed 和實際執行的動作序列，推測 agent 最終到達的位置。"

)

output_data = []

# 調整為你實際儲存的位置
input_folder = "C:/Users/seana/maze/outputs/nondeter2"
file_paths = sorted(glob.glob(os.path.join(
    input_folder, "nondeter_mem_*.jsonl")))

for path in file_paths:
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            item = json.loads(line)
            seed = item["seed"]
            actions = item["intended_actions"]
            goal = item["goal_pos"]

            output_data.append({
                "instruction": instruction_text,
                "input": f"seed = {seed}, intended_actions = {actions}",
                "target": f"{goal}"
            })

# 輸出成 alpaca 格式 jsonl
output_path = "C:/llm/inputs/eval/alpaca_eval_101to120.jsonl"
with open(output_path, 'w', encoding='utf-8') as f:
    for record in output_data:
        json.dump(record, f, ensure_ascii=False)
        f.write('\n')

print(f"✅ Alpaca 格式已儲存至 {output_path}，共 {len(output_data)} 筆")
