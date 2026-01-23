# Changes Made to train.py

This document summarizes the modifications made to `training/train.py` to fix issues and add features.

## 1. Fixed Import for Checkpoint Saving
- **Issue**: The import `from mlo.checkpoints.checkpoint import save_checkpoint` was incorrect because the module path didn't exist.
- **Change**: Updated to `from mlo.checkpoint import save_checkpoint`.
- **Reason**: The actual file is `mlo/checkpoint.py`, not in a `checkpoints` subdirectory.
- **Impact**: Checkpoints can now be saved without import errors.

## 2. Added Error Handling for WandB Logging
- **Issue**: `wandb.log()` was called directly, causing crashes if WandB wasn't initialized (e.g., due to login issues).
- **Change**: Wrapped the WandB table creation and logging in a `try-except` block.
- **Code Added**:
  ```python
  try:
      import wandb
      table = wandb.Table(columns=["Episode", "Reward", "Residual", "Rank"])
      table.add_data(episode, episode_reward, info["residual_norm"], info["rank_used"])
      wandb.log({"episode_summary": table})
  except Exception as e:
      print(f"WandB logging failed: {e}. Skipping episode summary log.")
  ```
- **Reason**: Prevents the script from crashing if WandB fails to initialize.
- **Impact**: Training continues even without WandB, with a console message for failed logs.

## 3. Added Live Plotting with Matplotlib
- **Issue**: No visual feedback during training; graphs were only available post-training or via WandB.
- **Change**: Integrated Matplotlib for real-time plotting of episode rewards and residual norms.
- **Code Added**:
  - Imports: `import matplotlib.pyplot as plt` and `plt.ion()` for interactive mode.
  - Data collection: Lists `episodes_list`, `episode_rewards`, `residual_norms` to store data.
  - Plotting logic after each episode:
    ```python
    episodes_list.append(episode)
    episode_rewards.append(episode_reward)
    residual_norms.append(info["residual_norm"])
    plt.figure(1)
    plt.clf()
    plt.subplot(1, 2, 1)
    plt.plot(episodes_list, episode_rewards, 'b-')
    plt.title('Episode Rewards')
    plt.xlabel('Episode')
    plt.ylabel('Reward')
    plt.subplot(1, 2, 2)
    plt.plot(episodes_list, residual_norms, 'r-')
    plt.title('Residual Norms')
    plt.xlabel('Episode')
    plt.ylabel('Residual Norm')
    plt.tight_layout()
    plt.pause(0.1)
    ```
- **Reason**: Provides immediate visual feedback on training progress.
- **Impact**: A plot window appears and updates live, showing rewards and residuals as the script runs.

## Summary of Benefits
- **Stability**: Script no longer crashes on WandB or import issues.
- **Usability**: Live plots allow monitoring without waiting for training to finish.
- **Logging**: Checkpoints save correctly, and metrics are logged robustly.

If you run into issues or need further modifications, refer to this doc or ask for help!

## 4. Enhanced Benchmark Visualization Script
- **Feature**: Updated `visualize_benchmark.py` to link benchmark results with the training run.
- **Details**:
  - Automatically extracts `run_id` and environment info (GPU, CuPy version) from the benchmark JSON.
  - Logs these as WandB config tokens for easy filtering.
  - Adds a "Source Run ID" column to the WandB table to track which training run produced which benchmark results.
- **Reason**: Prevents benchmark results from appearing "static" or disconnected from the actual agent training history.

## 5. Enhanced Checkpoint System with Detailed Logging
- **Feature**: Improved checkpoint saving to include timestamps, run duration, and a JSON log for traceability.
- **Changes**:
  - Modified `mlo/checkpoint.py` to save additional metadata (timestamp, episode, run_duration) in checkpoints.
  - Added `checkpoints/checkpoint_log.json` that records each checkpoint with details for error tracing.
  - Updated `training/train.py` to pass `start_time` and `episode` to `save_checkpoint`.
  - Added WandB logging for checkpoint events (step, episode, run duration).
  - Added cumulative WandB table `checkpoints_summary` with columns: Episode, Step, Run Duration (s), File.
- **Details**:
  - Checkpoints now include run duration so far.
  - JSON log lists all checkpoints with timestamps, episodes, and durations.
  - WandB logs checkpoint metrics and a summary table for clear visibility.
- **Reason**: Provides a "checkpoint area" to see run duration, which checkpoints were made, and trace back in case of errors.
- **Location**: Check `checkpoints/checkpoint_log.json` for the log; checkpoints are in `checkpoints/` directory; WandB shows checkpoint events and table.

## 6. Added Cumulative Episode Summary Table
- **Feature**: Episode summaries now accumulate in a single WandB table instead of overwriting per episode.
- **Changes**:
  - Modified `training/train.py` to collect episode data in `episode_summaries` list.
  - Logs the full table to WandB after each episode with all previous episodes.
- **Columns**: Episode, Reward, Residual, Rank.
- **Reason**: Provides a summary table after each episode with status, as requested.

## 7. Updated Logger for Team Support
- **Feature**: Added support for WandB teams by allowing `entity` in config.
- **Changes**:
  - Modified `project/logger.py` to check for `entity` in config and pass to `wandb.init`.
- **Usage**: Add `"entity": "your-team-name"` to the config dict in `training/train.py`.
- **Reason**: Fixes logging issues for users in WandB teams.

## 8. Made Agent Benchmark Dynamic and Traceable
- **Feature**: Benchmarks now use the trained agent's discovered algorithm and include unique run metadata.
- **Changes**:
  - `update_benchmark.py`: Generates a unique `run_id` for every benchmark session and records the GPU hardware details.
  - `training/train.py`: Automatically triggers the update and visualization suite at the end of training.
- **Details**:
  - Dynamic steps/multiplications based on agent's rank.
  - Real-time latency measurement from GPU.
- **Reason**: Ensures the benchmark reflects actual agent performance and provides a clear audit trail back to the training session.

## 9. Enabled Actual Checkpoint File Upload to WandB
- **Issue**: Checkpoints were mentioned in a WandB table, but the actual `.pt` files were not uploaded to the "Files" or "Artifacts" tabs.
- **Change**: Updated `mlo/checkpoint.py` to use `wandb.save()` and `wandb.log_artifact()`.
- **Details**:
  - `wandb.save()`: Makes the checkpoint file visible in the **Files** tab of the WandB run.
  - `wandb.log_artifact()`: Creates a versioned **Artifact** in WandB, which is the recommended way to store models.
- **Impact**: You can now see and download the actual model weights directly from the Weights & Biases dashboard.

## Summary of All Benefits
- **Dynamic Benchmarks**: Agent performance updates based on training, showing real steps and latency with full metadata traceability (Run ID, GPU info).
- **Full Model Traceability**: Download actual model weight files (.pt) directly from WandB via Files and Artifacts.
- **Checkpoint Tracking**: Detailed logs in `checkpoints/checkpoint_log.json` for local traceability.
- **Team Logging**: Supports WandB teams.
- **Stability**: Robust error handling across all scripts.

For checkpoint details, check `checkpoints/checkpoint_log.json` after running training. It includes run duration, timestamps, and episode info for each checkpoint. Files are also available in the 'Files' and 'Artifacts' tabs on your WandB run page.
