import wandb
import logging
import os

# Global flag to track if wandb was successfully initialized
_wandb_enabled = False

def init_logger(config, offline=False):
    global _wandb_enabled
    
    logging.basicConfig(
        filename="training.log",
        level=logging.INFO,
        format="%(asctime)s - %(message)s",
        force=True  # Override existing basicConfig
    )
    
    try:
        mode = "offline" if offline else "online"
        init_kwargs = {
            "project": "alpha_tensor_rl",
            "name": config.get("run_name"),
            "config": config,
            "mode": mode,
            "reinit": True  # Allow reinit in case of previous runs
        }
        if "entity" in config:
            init_kwargs["entity"] = config["entity"]
        
        # Support resuming a specific run if ID is provided
        if config.get("resume_id"):
            init_kwargs["id"] = config["resume_id"]
            init_kwargs["resume"] = "allow"
            
        run = wandb.init(**init_kwargs)
        _wandb_enabled = True
        return run.id if run else None
    except Exception as e:
        _wandb_enabled = False
        logging.warning(f"WandB init failed: {e}. Falling back to file logging only.")
        return None


def log_metrics(metrics, step=None):
    global _wandb_enabled
    
    if _wandb_enabled:
        try:
            wandb.log(metrics, step=step)
        except Exception as e:
            logging.warning(f"WandB log failed: {e}. Logging to console.")
            for key, value in metrics.items():
                logging.info(f"Step {step}: {key} = {value}")
    else:
        for key, value in metrics.items():
            logging.info(f"Step {step}: {key} = {value}")

def finish_logger():
    global _wandb_enabled
    
    if _wandb_enabled:
        try:
            wandb.finish()
        except Exception as e:
            logging.warning(f"WandB finish failed: {e}")
