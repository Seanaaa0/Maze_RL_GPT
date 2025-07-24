# 🧠 MazeRL: Custom Maze Environment for Reinforcement Learning

This project implements a customizable maze environment for reinforcement learning experiments.  
Agents can explore, observe partial environments, and learn to reach random goals using tabular or heuristic policies.

---

## 🚀 Features

- ✅ Procedurally generated mazes (DFS, Prim, Growing Tree)
- ✅ Supports multi-goal, traps, non-deterministic moves
- ✅ Fully and partially observable variants
- ✅ Directional agents with limited field of view
- ✅ CLI-controlled map size and random seeds
- ✅ Output trajectory, view, facing, actions, success logs

---

## 📂 Folder Structure

```
maze/
├── env/           # Custom Gym-style environments
├── train/         # Exploration and trajectory generation
├── run/           # Run trained policies or visualize agent paths
├── outputs/       # Saved .npy, .jsonl, .json for training
├── visual/        # Rendering utilities
├── requirements.txt
└── .gitignore
```

---

## 💻 Quick Start

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Run exploration script

```bash
python train/train_real_auto.py --seed 101 --size 15
```

> Outputs will be saved in the `outputs/` directory.

---

## 🔮 Future Integration

This environment supports data generation for GPT-based fine-tuning (e.g., predicting goals or optimal actions from partial observations).

---

## 👤 Author

Developed by [@Seanaaa0](https://github.com/Seanaaa0)  
Focus: Reinforcement Learning, LLM data generation, intelligent agent systems.
