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
        force=True,  # Override existing basicConfig
        encoding="utf-8"
    )
    
    try:
        mode = "offline" if offline else "online"
        init_kwargs = {
            "project": "alpha_tensor_rl",
            "entity": "prsydha",
            "name": config.get("run_name"),
            "config": config,
            "mode": mode,
            "reinit": "finish_previous"  # Allow reinit in case of previous runs
        }
        if "entity" in config:
            init_kwargs["entity"] = config["entity"]
        wandb.init(**init_kwargs)

        wandb.define_metric("step/*", step_metric="global_step")
        wandb.define_metric("episode/*", step_metric="episode_step")
        wandb.define_metric("epoch/*", step_metric="epoch_step")
        wandb.define_metric("eval/*", step_metric="eval_step")

        _wandb_enabled = True
        logging.info("WandB initialized successfully")
    except Exception as e:
        _wandb_enabled = False
        logging.warning(f"WandB init failed: {e}. Falling back to file logging only.")
        logging.info("Tip: Run 'wandb login' to authenticate with your W&B account.")


def log_metrics(metrics, step=None):
    # if step is None:
    #     raise ValueError("wandb.log called without a step -- this breaks step ordering")
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
