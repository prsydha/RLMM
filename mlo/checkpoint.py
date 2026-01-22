import torch
import os
import logging  # Added for logging integration

def save_checkpoint(model, step, path=".", optimizer=None, scheduler=None, logger=None):
    os.makedirs(path, exist_ok=True)
    checkpoint = {
        "step": step,
        "model_state": model.state_dict(),
    }
    if optimizer:
        checkpoint["optimizer_state"] = optimizer.state_dict()
    if scheduler:
        checkpoint["scheduler_state"] = scheduler.state_dict()
    torch.save(checkpoint, f"{path}/model_step_{step}.pt")
    if logger:
        logger.info(f"Checkpoint saved at step {step}")
    else:
        print(f"Checkpoint saved at step {step}")

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