import numpy as np

# Scaled Progress reward
def calc_reward(obs):
    norm = np.linalg.norm(obs)
    sqrt8 = 8 ** 0.5
    val = 0
    if norm <= sqrt8:
        val = 1 - norm / sqrt8
    elif norm <= 6:
        val = (sqrt8 - norm) / (6 - sqrt8)
    else:
        val = -1
    return val

def completion_reward(terminated):
    return 1.0 if terminated else -1.0

reward_func = completion_reward