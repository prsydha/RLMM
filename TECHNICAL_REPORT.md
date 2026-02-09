# Technical Report: RLMM Training Pipeline Refactor
**Date:** February 10, 2026

## 1. WandB Logging Architecture Overhaul

### Problem: Table Data Loss
**Issue:** The `wandb.Table` was being initialized and logged inside the episode loop. In WandB, logging a table with the same key overwrites the previous version. This meant only the results of the *last* episode were visible.

**Technical Solution:** 
*   Initialized a persistent Python list `episode_history = []` at the start of the `train()` function.
*   Implemented `episode_history.append([...])` at the end of every episode to capture: `Global Episode`, `Epoch`, `Episode in Epoch`, `Reward`, `Residual`, `Rank`, `Steps`, and `Solved (bool)`.
*   Moved `wandb.log` for the table outside the episode loop, triggering only once per **Epoch**. This creates a new table object `wandb.Table(data=episode_history)` that includes all historical rows, ensuring no data loss.

### Problem: Monotonic Step Violations
**Issue:** WandB requires the `step` parameter in `wandb.log()` to be strictly increasing. The previous code sometimes logged with `epoch` or `global_episode_count` as the step, causing warnings and data rejection when the internal WandB counter was already higher.

**Technical Solution:**
*   Established `global_step = 0` as the master clock.
*   Updated `pretrain_on_expert()` to accept and return `global_step`, incrementing it for every weight update.
*   Ensured every `wandb.log()` call throughout the script explicitly uses `step=global_step`.
*   Defined `Epoch` as a custom x-axis using `wandb.define_metric("epoch_*", step_metric="Epoch")`. This allows plotting epoch-level data against the current epoch count without violating the global step monotonicity.

## 2. Real-Time Performance Tracking

### Feature: Scalar Metric Streams
**Implementation:** Added a new stream of scalar logs prefixed with `episode/` (e.g., `episode/reward`, `episode/residual`).
**Rationale:** Unlike tables, scalar line charts update instantly in WandB. This provides the developer with immediate visual feedback on whether the agent is improving during the self-play phase, rather than having to wait 5-10 minutes for an epoch to complete.

### Feature: Pre-training Visibility
**Implementation:** Moved the `init_logger()` call before the `pretrain_on_expert()` phase.
**Rationale:** This captures the loss reduction during the expert demonstration phase, allowing us to verify if the model is correctly "learning" the standard matrix multiplication algorithm before the reinforcement learning loop begins.

## 3. Terminal Interface & Debugging

### Improvement: Local Step Monitoring
**Change:** Restored the per-step console logging but optimized the string formatting.
**Details:** Each step now displays `Step XX [Ep YY] | Reward: ... | Residual: ...`. 
**Purpose:** This allows for "micro-debugging" of the agent's behavior. For example, developers can see exactly which rank-1 tensor addition caused the `Residual` to spike or drop, providing intuition that a single summary line cannot offer.

## 4. Experimental Configuration (config.py)

### Optimization for Development
To enable faster testing of these logging features, the følgende parameters were adjusted:
*   `BATCH_SIZE`: 128 → 16 (Allows training to start earlier and update more frequently).
*   `MCTS_SIMS`: 350 → 50 (Reduces CPU overhead for logic verification).
*   `EPISODES_PER_EPOCH`: 10 → 3-5 (Increases table update frequency).
*   `PRE_TRAIN_STEPS`: 100 → 10 (Reduces wait time for initial run start).

---
**Summary of File State:**
*   `training/train.py`: Contains the updated logging state-machine and I/O logic.
*   `config.py`: Tuned for rapid validation.
*   `CHANGELOG.md`: Detailed developer-level log of all commits.
