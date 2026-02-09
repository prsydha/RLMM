# Project Updates & Fixes (Feb 10, 2026)

This document summarizes the changes made to the RLMM training pipeline to improve WandB visualization, logging accuracy, and terminal readability.

## 📊 WandB Visualization Improvements

### 0. New Episode Graphs (`episode/` metrics)
- **Why Added**: The table only updates once per epoch, so you couldn't see real-time progress.
- **What They Are**: Scalar metrics logged after every single game finishes:
    - `episode/reward`: Agent's score for this game (-1 to 1).
    - `episode/residual`: Final matrix error (should decrease over training).
    - `episode/rank`: How many multiplications the agent used.
    - `episode/steps`: Total moves taken in the game.
    - `episode/solved`: 1 if successful, 0 if failed.
    - `episode/epoch`: Which epoch this game belongs to.
    - `episode/episode_in_epoch`: Position within the epoch (1, 2, 3...).
    - `episode/global_count`: Total game number across all training.
- **Benefit**: These appear as live-updating line charts in WandB, so you can watch trends develop in real-time without waiting for the epoch to end.

### 1. Episode Summary Table Fix
- **Issue**: The episode summary table was overwriting itself every episode, showing only a single row in WandB.
- **Fix**: Implemented a cumulative Python list (`episode_history`) that persists across episodes. 
- **Optimization**: The table is now logged **once per epoch** instead of every episode. This prevents network overhead and avoids WandB's "immutable artifact" errors.

### 2. Step Metrics & Monotonicity
- **Issue**: Warnings were appearing because `epoch` and `global_step` were being used interchangeably as the step index, leading to non-monotonic step errors.
- **Fix**:
    - Standardized all episode logging to use `step=global_step`.
    - Defined a custom `Epoch` metric in WandB using `wandb.define_metric`.
    - Associated epoch-level metrics (`epoch_loss`, etc.) with the `Epoch` x-axis to allow clean plotting without step conflicts.

### 3. Data Accuracy Improvements
- **Issue**: The `residual_norm` captured in the table was sometimes stale (from a previous step).
- **Fix**: Updated `train.py` to use the explicitly calculated `res_norm` from the current observation at the exact moment a game finishes.

## 🖥 Terminal & Console Enhancements

### 1. Log Decluttering
- **Issue**: The terminal was flooded with "Game X | Reward: ... | Rank: ..." logs for every single move (10+ lines per game).
- **Fix**: Removed per-step console logging. The terminal now only shows the start and end of each game for a cleaner overview.

### 2. Consistency & Numbering
- **Issue**: Console logs used 1-based episode counts within an epoch (1, 2, 3), while WandB used global counts (1, 2, 3... 10... 20).
- **Fix**: Updated all console prints to use the `global_episode_count` so the numbers in your terminal match exactly with the rows in the WandB table.

## ⚙️ Experimental Tuning

- **Hyperparameter Fast-Start**: Modified `config.py` to allow for rapid experimental cycles:
    - Reduced `MCTS_SIMS` (50) for faster search.
    - Reduced `EPISODES_PER_EPOCH` (3-5) for more frequent table updates.
    - Reduced `WARM_START_COPIES` and `PRE_TRAIN_STEPS` for near-instant training startup.

---
**Status**: All fixes verified. The pipeline is now stable and provides a clean, accurate visualization of the agent's progress.
