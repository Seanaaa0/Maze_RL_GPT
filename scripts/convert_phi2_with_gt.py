
import json
import argparse
import numpy as np

def load_ground_truth(gt_path):
    gt = np.load(gt_path, allow_pickle=True).item()
    return {
        "goal": gt["goal"],
        "wall_map": gt["wall_map"].tolist()
    }

def convert(input_path, output_path, gt_path, mode="trajectory"):
    gt_data = load_ground_truth(gt_path)

    data = []
    with open(input_path, "r", encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)

            prompt = (
                f"起點：{tuple(r['start_pos'])}，終點：{tuple(r['goal_pos'])}。\n"
                f"你在有2%反方向機率的迷宮中探索，軌跡如下：\n"
                f"{r['trajectory']}\n"
                f"你實際執行的動作序列是：{r['actual_actions']}"
            )

            if mode == "trajectory":
                confidence_path = [[int(p[0]), int(p[1])] for p in r["trajectory"]]
            elif mode == "unique":
                confidence_path = list({(int(p[0]), int(p[1])) for p in r["trajectory"]})
            else:
                raise ValueError("Unknown mode")

            data.append({
                "prompt": prompt,
                "response": {
                    "confidence_path": confidence_path,
                    "success": r.get("success", True),
                    "ground_truth_map": gt_data["wall_map"],
                    "goal": gt_data["goal"]
                }
            })

    with open(output_path, "w", encoding="utf-8") as f:
        for item in data:
            json.dump(item, f, ensure_ascii=False)
            f.write("\n")

    print(f"✅ 已儲存轉換後資料到 {output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str, required=True, help="輸入 JSONL 路徑")
    parser.add_argument("--output", type=str, required=True, help="輸出 JSONL 路徑")
    parser.add_argument("--gt", type=str, required=True, help="Ground Truth .npy 檔案路徑")
    parser.add_argument("--mode", type=str, default="trajectory", choices=["trajectory", "unique"])
    args = parser.parse_args()

    convert(args.input, args.output, args.gt, args.mode)
