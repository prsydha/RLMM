import torch
import os
import logging  # Added for logging integration
import json
import time

def save_checkpoint(model, step, path="checkpoints", optimizer=None, scheduler=None, logger=None, start_time=None, episode=None):
    os.makedirs(path, exist_ok=True)
    checkpoint = {
        "step": step,
        "model_state": model.state_dict(),
        "timestamp": time.time(),
        "episode": episode,
    }
    if optimizer:
        checkpoint["optimizer_state"] = optimizer.state_dict()
    if scheduler:
        checkpoint["scheduler_state"] = scheduler.state_dict()
    if start_time:
        checkpoint["run_duration"] = time.time() - start_time
    torch.save(checkpoint, f"{path}/model_step_{step}.pt")
    
    # Log checkpoint info to a JSON file for tracking
    log_path = os.path.join(path, "checkpoint_log.json")
    log_entry = {
        "step": step,
        "episode": episode,
        "timestamp": checkpoint["timestamp"],
        "run_duration": checkpoint.get("run_duration", None),
        "file": f"model_step_{step}.pt"
    }
    if os.path.exists(log_path):
        with open(log_path, "r") as f:
            logs = json.load(f)
    else:
        logs = []
    logs.append(log_entry)
    with open(log_path, "w") as f:
        json.dump(logs, f, indent=2)
    
    print(f"📁 Checkpoint log updated: {log_path}")
    
    if logger:
        logger.info(f"Checkpoint saved at step {step}, episode {episode}")
    else:
        print(f"Checkpoint saved at step {step}, episode {episode}")
    
    # Log to wandb if available
    try:
        import wandb
        if wandb.run is not None:
            checkpoint_file = f"{path}/model_step_{step}.pt"
            
            # Log metrics
            wandb.log({
                "checkpoint_step": step,
                "checkpoint_episode": episode,
                "run_duration_at_checkpoint": checkpoint.get("run_duration", None)
            }, step=step)
            
            # 1. Save file to wandb (makes it appear in 'Files' tab)
            wandb.save(checkpoint_file, base_path=os.path.dirname(checkpoint_file))
            
            # 2. Log as Artifact (makes it appear in 'Artifacts' tab - recommended)
            artifact = wandb.Artifact(
                name=f"model-checkpoint-{wandb.run.id}", 
                type="model",
                description=f"Checkpoint at step {step}, episode {episode}",
                metadata={
                    "step": step,
                    "episode": episode,
                    "run_duration": checkpoint.get("run_duration", None)
                }
            )
            artifact.add_file(checkpoint_file)
            wandb.log_artifact(artifact)
            
            print(f"🚀 Checkpoint uploaded to WandB: {checkpoint_file}")
    except Exception as e:
        print(f"WandB checkpoint upload failed: {e}")

def load_checkpoint(model, ckpt_path, optimizer=None, scheduler=None, logger=None):
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(f"Checkpoint file not found: {ckpt_path}")
    ckpt = torch.load(ckpt_path)
    model.load_state_dict(ckpt["model_state"])
    if optimizer and "optimizer_state" in ckpt:
        optimizer.load_state_dict(ckpt["optimizer_state"])
    if scheduler and "scheduler_state" in ckpt:
        scheduler.load_state_dict(ckpt["scheduler_state"])
    step = ckpt.get("step", 0)
    if logger:
        logger.info(f"Checkpoint loaded from {ckpt_path}, resuming at step {step}")
    else:
        print(f"Checkpoint loaded from {ckpt_path}, resuming at step {step}")
    return step