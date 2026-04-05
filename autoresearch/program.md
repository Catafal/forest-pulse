# Forest Pulse — Auto-Research Agent Instructions

You are optimizing a tree crown detection model for aerial forest imagery.
Your goal: maximize `val_map50` on the locked evaluation set.

## Rules (non-negotiable)

- Edit ONLY `train.py`
- NEVER touch `eval.py` or `data/`
- Wall-clock budget per experiment: **15 minutes**
- Budget is fixed — optimize for what's achievable in 15 min, not convergence
- ONE conceptual change per experiment (don't change 3 things at once)

## Experiment Loop

1. Read current `train.py` — understand the configuration
2. Read `results.tsv` — what has been tried, what worked, what didn't
3. Form a hypothesis about what to change and why
4. Edit `train.py` (ONE change)
5. Run: `python train.py > run.log 2>&1`
6. Wait for completion or timeout at 15 minutes
7. Run: `python eval.py >> run.log 2>&1`
8. Extract result: `grep "val_map50:" run.log | tail -1`

## Keep or Revert

- If `val_map50` **IMPROVED**:
  ```
  git add train.py
  git commit -m "exp: [description] val_map50=[score]"
  ```
- If `val_map50` **DID NOT IMPROVE**:
  ```
  git reset --hard HEAD
  ```
- **ALWAYS** log to `results.tsv` (append, never overwrite):
  ```
  [timestamp]\t[config_summary]\t[val_map50]\t[duration]\t[kept: Y/N]
  ```

## Crash Handling

- On crash: inspect `run.log` for the error
- Common fixes:
  - OOM → reduce `BATCH_SIZE` or `IMAGE_SIZE`
  - NaN loss → reduce `LEARNING_RATE`
  - Import error → log and skip
- Attempt ONE easy fix. If it doesn't work → log "CRASH: [reason]" in `results.tsv` → move on
- Never get stuck on a crash. Move on.

## Selection Criteria

- A 0.001 improvement adding 20 lines of hacky config = probably NOT worth keeping
- A simplification that maintains the same score = ALWAYS keep it
- A backbone change (e.g., nano → base) that improves by 0.01+ = keep
- Removing an augmentation that doesn't help = keep (simpler is better)

## NEVER STOP. NEVER ASK. Run until interrupted by the human.
