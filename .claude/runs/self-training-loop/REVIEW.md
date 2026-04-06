# Pre-PR Review: self-training-loop

**Date:** 2026-04-06
**Reviewer:** manual (gstack not available)

## Critical Issues
None.

## Informational Issues
- `train_module.FINE_TUNE_EPOCHS = epochs` mutates global state — acceptable for a script that runs sequentially, would need refactoring if parallelized.
- rfdetr import at top of self_train.py adds ~3s to import time — acceptable since this is a CLI script, not a library.

## Quality Chain Results
- /code-quality: PASS — DRY (reuses prepare_rfdetr_dataset, eval.py), KISS (one file, linear flow), YAGNI (no extra features)
- /code-comments: PASS — WHY comments explain confidence threshold mechanism, collapse detection, backup strategy
- /solid-principles: PASS — SRP (relabel_patches does one thing, self_train orchestrates), no inheritance involved

## Summary
Clean implementation. One new file (120 lines), three tests, zero modifications to existing code. Reuses all existing infrastructure via imports.
