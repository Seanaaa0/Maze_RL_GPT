import json
import glob
import os

instruction_text = (
    "你是一個探索迷宮的專家，你的工作是用我給你的 agent 的 intended action 推測出出口座標。\n"
    "現在有一個 agent 從入口 (0, 0) 開始移動，直到抵達出口結束。\n"
    "這個迷宮是 15x15，具有 30% 非決定性（例如：左（2）有 30% 機率會變成右（3），上（0）可能變成下（1）等等）。\n"
    "視野限制為 1x1，agent 使用 BFS 搜索法，請根據動作推測 goal_pos。"
)

output_data = []

# 調整為你實際儲存的位置
input_folder = "C:/Users/seana/maze/outputs/nondeter2/"
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
                "output": f"{goal}"
            })

# 輸出成 alpaca 格式 jsonl
output_path = "C:/Users/seana/maze/outputs/alpaca_non_eval_1to500.jsonl"
with open(output_path, 'w', encoding='utf-8') as f:
    for record in output_data:
        json.dump(record, f, ensure_ascii=False)
        f.write('\n')

print(f"✅ Alpaca 格式已儲存至 {output_path}，共 {len(output_data)} 筆")
