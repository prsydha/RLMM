import wandb

#function to initilaize logger

def init_logger(config):
    wandb.init(
        project="alpha_tensor_rl",
        name = config['run_name'],
        config= config
    )
    
def log_metrics(metrics, step=None):
    wandb.log(metrics,step=step)