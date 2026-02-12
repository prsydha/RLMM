import numpy as np

# Scaled Final Residual Norm reward
def calc_reward(obs):
    norm = np.linalg.norm(obs)
    sqrt27 = 27 ** 0.5
    val = 0
    if norm <= sqrt27:
        val = 1 - norm / sqrt27
    elif norm <= 10:
        val = (sqrt27 - norm) / (10 - sqrt27)
    else:
        val = -1
    return val

def completion_reward(terminated):
    return 1.0 if terminated else -1.0

reward_func = completion_reward