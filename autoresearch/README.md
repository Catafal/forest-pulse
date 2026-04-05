# Auto-Research Harness

Karpathy-style overnight fine-tuning optimization for tree crown detection.

Based on [karpathy/autoresearch](https://github.com/karpathy/autoresearch) — a system where an AI coding agent runs 100+ ML experiments overnight with zero human babysitting.

## How It Works

Three files, strict roles:

| File | Who controls | Editable? |
|------|-------------|-----------|
| `eval.py` | Human | **NEVER** — defines the ground truth metric |
| `train.py` | AI agent | **YES** — the only file the agent modifies |
| `program.md` | Human | **NO** — defines agent behavior |

## The Loop

```
1. Agent reads train.py and results.tsv (what's been tried)
2. Agent edits train.py (ONE conceptual change)
3. Agent runs: python train.py > run.log 2>&1
4. Agent runs: python eval.py >> run.log 2>&1
5. Agent extracts: grep "val_map50:" run.log
6. If improved → git commit (keep)
7. If not → git reset --hard (discard)
8. Log to results.tsv → repeat
```

## Running the Harness

```bash
# Prerequisites: download training data first
python scripts/download_data.py --dataset oam-tcd

# Start the harness (runs until interrupted)
# Use Claude Code or any AI coding agent with these instructions:
# "Read autoresearch/program.md and follow the instructions exactly."
```

## Why This Approach

The academic literature on DeepForest explicitly states that "optimal fine-tuning strategies for specific forest types remain unclear." This harness answers that question empirically:

- 15-minute wall-clock budget per experiment
- ~32 experiments per overnight run
- Only winning configurations survive in git history
- `results.tsv` keeps the full experiment log (including failures)

After one night, you have an empirically validated model configuration — not a guess.

## Files

- **eval.py** — Locked evaluation metric (mAP50 on validation shard)
- **train.py** — Agent-editable training configuration
- **program.md** — Agent operating instructions
- **results.tsv** — Experiment log (git-ignored, survives resets)
- **run.log** — Latest experiment output (git-ignored, overwritten each run)
