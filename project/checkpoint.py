import torch
import os

def save_checkpoint(model,step,path="checkpoints"):
    os.makedirs(path, exist_ok= True)
    torch.save(
        {
        
            "step": step,
            "model_state": model.state_dict()
        },
        f"{path}/model_step_{step}.pt"
    )


def load_checkpoint(model,ckpt_path):
    ckpt = torch.load(ckpt_path)
    model.load_state_dict(ckpt["model_state"])
    return ckpt["step"]