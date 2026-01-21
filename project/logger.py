import wandb
import logging
import os

def init_logger(config, offline=False):
    try:
        mode = "offline" if offline else "online"
        wandb.init(
            project="alpha_tensor_rl",
            name=config.get("run_name"),
            config=config,
            mode=mode
        )
    except Exception as e:
        logging.warning(f"WandB init failed: {e}. Falling back to console logging.")
        logging.basicConfig(
            filename="training.log",
            level=logging.INFO,
            format="%(asctime)s - %(message)s"
        )


def log_metrics(metrics, step=None):
    try:
        wandb.log(metrics, step=step)
    except Exception as e:
        logging.warning(f"WandB log failed: {e}. Logging to console.")
        for key, value in metrics.items():
            logging.info(f"Step {step}: {key} = {value}")

def finish_logger():
    try:
        wandb.finish()
    except Exception as e:
        logging.warning(f"WandB finish failed: {e}")
        
from logger import finish_logger

finish_logger()
