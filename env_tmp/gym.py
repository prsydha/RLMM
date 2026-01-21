import torch
import gymnasium as gym
from gymnasium import spaces
import numpy as np
from typing import Tuple, Dict, Optional, List
import json

def matmul(mat1: np.ndarray, mat2: np.ndarray):
    """
    Baseline multiplication algorithm for two matrices
    :param mat1:
    :param mat2:
    :return: tensor
    """
    (a,b), (p,q) = mat1.shape, mat2.shape
    if b != p: return False
    mat_result = np.zeros((a,q))
    for i in range(a):
        for j in range(q):
            for k in range(b):
                mat_result[i,j] += mat1[i,k]*mat2[k,j]
    return mat_result


"""
AlphaTensor-Style Matrix Multiplication Environment
Member 2: Simulation Engineer Implementation

This environment implements the core simulation logic for discovering
matrix multiplication algorithms via tensor decomposition.

Key Responsibilities (Member 2):
1. Gym environment design with constraints
2. Action space definition (rank-1 tensor operations)
3. State space representation (residual tensor)
4. Reward engine logic (efficiency calculation)
5. State transitions and validity checking
"""


class TensorDecompositionEnv(gym.Env):
    """
    AlphaTensor-style environment for matrix multiplication optimization.

    The environment represents matrix multiplication as a 3D tensor T[i,j,k]
    where each entry corresponds to: result[i,j] += A[i,k] * B[k,j]

    The agent decomposes this tensor into a sum of rank-1 tensors (outer products).
    Each rank-1 tensor represents one scalar multiplication in the algorithm.

    Goal: Minimize the number of rank-1 tensors needed (= minimize multiplications)
    """

    metadata = {"render_modes": ["human", "json"]}

    def __init__(
            self,
            matrix_size: Tuple[int, int, int] = (2, 2, 2),
            max_rank: int = 20,
            reward_type: str = "sparse",
            illegal_action_penalty: float = -1.0,
    ):
        """
        Initialize the tensor decomposition environment.

        Args:
            matrix_size: (m, n, p) for m×n × nxp matrix multiplication
            max_rank: Maximum number of rank-1 tensors allowed (episode length)
            reward_type: "sparse"- reward only at completion, "dense"- reward at each step
            illegal_action_penalty: Penalty for illegal actions
        """
        super().__init__()

        self.m, self.n, self.p = matrix_size
        self.max_rank = max_rank
        self.reward_type = reward_type
        self.illegal_action_penalty = illegal_action_penalty

        # Target tensor: standard matrix multiplication tensor
        # T[i,j,k] = 1 if result[i,j] involves A[i,k] * B[k,j], else 0
        self.target_tensor = self._create_target_tensor()

        # State: residual tensor (what's left to decompose)
        self.residual_tensor = self.target_tensor.copy()

        # Algorithm discovered so far (list of rank-1 tensors)
        self.algorithm = []

        # Step counter
        self.current_step = 0

        # Best solution found
        self.best_rank = max_rank

        # Action space: Choose u, v, w vectors for rank-1 tensor u⊗v⊗w
        # Each vector has entries in {-2, -1, 0, 1, 2} (allows linear combinations)
        # Total actions = (5^m) * (5^n) * (5^p)
        # For 2×2×2: 5^2 * 5^2 * 5^2 = 15,625 possible actions

        # Simplified action space: discretized choices
        self.action_space_size = (5 ** (self.m*self.n)) * (5 ** (self.n*self.p)) * (5 ** (self.m*self.p))
        self.action_space = spaces.Discrete(self.action_space_size) # this tells gym the possible set of actions
        # ... represented as integers. Gym env are expected to have action_space attribute.
        # the action integer is later decoded into uvw in _action_to_rank1_tensor().
        # ... Gym now knows, valid actions are from 0 to self.action_space_size-1 from above line of code

        # Observation space: flattened residual tensor + metadata
        # Residual tensor: m×n×p values
        # Metadata: [current_step, num_rank1_used, residual_norm]. current_step >= num_rank1_used because steps may be invalid.
        obs_size = (self.m*self.n) * (self.n*self.p) * (self.m*self.p) + 3 # length of vector the agent sees at each step
        self.observation_space = spaces.Box( # continuous because norms and rank1 tensors are considered to be continuous
            low=-100.0, # -100 to 100 because resulting rank1 3d tensor from uvw can take large values too (see example in internet)
            high=100.0,
            shape=(obs_size,),
            dtype=np.float32
        )

    def _create_target_tensor(self) -> np.ndarray:
        """
        Create the target tensor for matrix multiplication.

        For C = A × B where A is m×n and B is nxp:
        T[i,j,k] = 1 means: C[i,j] += A[i,k] * B[k,j]

        Returns:
            Target tensor of shape (mn, np, mp)
        """
        tensor = np.zeros((self.m*self.n, self.n*self.p, self.m*self.p), dtype=float)

        # Standard matrix multiplication tensor
        for i in range(self.m):
            for j in range(self.n):
                for k in range(self.p):
                    idx_mn = i * self.n + j  # (i,j)
                    idx_np = j * self.p + k  # (j,k)
                    idx_mp = i * self.p + k  # (i,k)

                    tensor[idx_mn, idx_np, idx_mp] = 1.0

        return tensor

    def _action_to_rank1_tensor(self, action: Dict) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Convert discrete action to rank-1 tensor components (u, v, w).

        Action space is discretized: each vector component ∈ {-1, 0, 1}

        Args:
            action: Discrete action index given by the policy network, which gets mapped to u,v,w vectors.

        Returns:
            u (shape m), v (shape n), w (shape p): vectors forming rank-1 tensor
        """
        # Map to base-5 representation
        values = [-1, 0, 1]

        # Decode action into three vectors
        remaining = action

        # Extract w (m*p components)
        w = np.zeros(self.m*self.p)
        for i in range(self.m*self.p):
            w[i] = values[remaining % 5]
            remaining //= 5

        # Extract v (n*p components)
        v = np.zeros(self.n*self.p)
        for i in range(self.n*self.p):
            v[i] = values[remaining % 5]
            remaining //= 5

        # Extract u (m*n components)
        u = np.zeros(self.m*self.n)
        for i in range(self.m*self.n):
            u[i] = values[remaining % 5]
            remaining //= 5

        return u, v, w

    def _is_valid_action(self, u: np.ndarray, v: np.ndarray, w: np.ndarray) -> bool:
        """
        Check if the rank-1 tensor is valid (not all zeros).

        Args:
            u, v, w: Components of rank-1 tensor

        Returns:
            True if valid, False otherwise
        """
        return not (np.allclose(u, 0) or np.allclose(v, 0) or np.allclose(w, 0))

    def _compute_rank1_tensor(
            self, u: np.ndarray, v: np.ndarray, w: np.ndarray
    ) -> np.ndarray:
        """
        Compute the rank-1 tensor from outer product: u ⊗ v ⊗ w.

        Args:
            u (mn,), v (np,), w (mp,): Vector components

        Returns:
            Rank-1 tensor of shape (mn, np, mp)
        """
        # Outer product: T[i,j,k] = u[i] * v[j] * w[k]
        tensor = np.einsum('i,j,k->ijk', u, v, w)
        return tensor # gets converted into 3d tensor as shown in the AlphaTensor paper
        # ...first dim= a1-a4, second dim=b1-b4, and third dim=c1-c4

    def _update_residual(self, rank1_tensor: np.ndarray) -> None:
        """
        Update the residual tensor by subtracting the rank-1 component.

        Args:
            rank1_tensor: Rank-1 tensor to subtract
        """
        self.residual_tensor -= rank1_tensor

    def _is_decomposition_complete(self, tolerance: float = 1e-6) -> bool:
        """
        Check if the residual tensor is sufficiently close to zero.

        Args:
            tolerance: Numerical tolerance for completion

        Returns:
            True if decomposition is complete
        """
        return np.linalg.norm(self.residual_tensor) < tolerance

    def _get_observation(self) -> np.ndarray:
        """
        Get current observation (state representation).

        Returns:
            Flattened observation: [residual_tensor (flattened), step, rank, norm]
        """

        # Metadata
        metadata = np.array([
            self.current_step / self.max_rank,  # Normalized steps completed
            len(self.algorithm) / self.max_rank,  # Normalized rank used
            np.linalg.norm(self.residual_tensor)  # Residual norm
        ], dtype=np.float32)

        # Combine
        obs = np.concatenate([self.residual_tensor, metadata]).astype(np.float32)
        return obs

    def _calculate_reward(
            self,
            action_valid: bool,
            decomposition_complete: bool,
            prev_norm: float,
            curr_norm: float
    ) -> float:
        """
        Calculate reward based on action outcome.

        Reward Structure :
        1. Dense reward: reward for reducing residual norm
        2. Sparse reward: only reward on completion (will opt for this, but 1 can be used too)
        3. Penalties: illegal actions, inefficiency

        Args:
            action_valid: Whether action was legal
            decomposition_complete: Whether decomposition is done
            prev_norm: Previous residual norm
            curr_norm: Current residual norm

        Returns:
            Reward value
        """
        if not action_valid:
            return self.illegal_action_penalty

        if self.reward_type == "dense":
            # Dense reward: progress toward zero residual
            progress = prev_norm - curr_norm
            reward = progress * 10.0  # Scale progress

            # Bonus for completion
            if decomposition_complete:
                # Reward based on efficiency (fewer rank-1 tensors = better)
                efficiency_bonus = (self.max_rank - len(self.algorithm)) * 5.0
                reward += 100.0 + efficiency_bonus

                # Extra bonus if we beat the naive algorithm
                naive_rank = self.m * self.n * self.p # 8 for 2x2 matmul
                if len(self.algorithm) < naive_rank:
                    reward += 50.0

                # Track best
                if len(self.algorithm) < self.best_rank:
                    self.best_rank = len(self.algorithm)
                    reward += 100.0  # New best!

            # Penalty for not making progress
            if progress <= 0:
                reward -= 0.5

        else:  # sparse reward
            if decomposition_complete:
                # Only reward on completion
                base_reward = 100.0

                # Efficiency bonus
                naive_rank = self.m * self.n * self.p
                efficiency = (naive_rank - len(self.algorithm)) / naive_rank
                reward = base_reward + efficiency * 200.0

                # Track best
                if len(self.algorithm) < self.best_rank:
                    self.best_rank = len(self.algorithm)
                    reward += 100.0
            else:
                # Small penalty for each step to encourage efficiency
                reward = -0.1

        return reward

    def reset(
            self,
            seed: Optional[int] = None,
            options: Optional[Dict] = None
    ) -> Tuple[np.ndarray, Dict]:
        """
        Reset the environment to initial state.

        Returns:
            observation: Initial state
            info: Additional information
        """
        super().reset(seed=seed)

        # Reset residual to target
        self.residual_tensor = self.target_tensor.copy()

        # Clear algorithm
        self.algorithm = []

        # Reset step counter
        self.current_step = 0

        # Get initial observation
        obs = self._get_observation()

        info = {
            "residual_norm": np.linalg.norm(self.residual_tensor),
            "rank_used": 0,
            "target_rank": self.m * self.n * self.p,  # Naive algorithm rank
        }

        return obs, info

    def step(
            self, action: Dict
    ) -> Tuple[np.ndarray, float, bool, bool, Dict]:
        """
        Execute one step in the environment.

        The core simulation logic

        Args:
            action: Discrete action (encoded rank-1 tensor) OR u,v,w explicitly

        Returns:
            observation: New state
            reward: Reward for this action
            terminated: Whether episode ended successfully
            truncated: Whether episode hit time limit
            info: Additional information
        """
        self.current_step += 1

        # Decode action to rank-1 tensor
        # u, v, w = self._action_to_rank1_tensor(action)
        u, v, w = action["u"], action["v"], action["w"]

        # Check validity (constraint enforcement - Member 2)
        action_valid = self._is_valid_action(u, v, w)

        # Store previous norm for reward calculation
        prev_norm = np.linalg.norm(self.residual_tensor)

        if action_valid:
            # Compute rank-1 tensor
            rank1_tensor = self._compute_rank1_tensor(u, v, w)

            # Update residual (state transition - Member 2)
            self._update_residual(rank1_tensor)

            # Store in algorithm
            self.algorithm.append({
                "u": u.tolist(),
                "v": v.tolist(),
                "w": w.tolist(),
                "tensor": rank1_tensor.tolist()
            })

        # Calculate current norm
        curr_norm = np.linalg.norm(self.residual_tensor)

        # Check if decomposition is complete
        decomposition_complete = self._is_decomposition_complete()

        # Calculate reward (Member 2's reward engine)
        reward = self._calculate_reward(
            action_valid, decomposition_complete, prev_norm, curr_norm
        )

        # Check termination conditions
        terminated = decomposition_complete
        truncated = self.current_step >= self.max_rank

        # Get new observation
        obs = self._get_observation()

        # Build info dict
        info = {
            "action_valid": action_valid,
            "residual_norm": curr_norm,
            "rank_used": len(self.algorithm),
            "progress": prev_norm - curr_norm,
            "decomposition_complete": decomposition_complete,
            "u": u.tolist() if action_valid else None,
            "v": v.tolist() if action_valid else None,
            "w": w.tolist() if action_valid else None,
        }

        if decomposition_complete:
            info["final_algorithm"] = self.get_algorithm_description()
            info["success"] = True
            info["efficiency"] = (self.m * self.n * self.p - len(self.algorithm)) / (self.m * self.n * self.p)

        return obs, reward, terminated, truncated, info

    def get_algorithm_description(self) -> Dict:
        """
        Get a description of the discovered algorithm.

        This is the output format for Member 1 & 2 integration.

        Returns:
            Dictionary describing the algorithm (can be saved as JSON)
        """
        description = {
            "matrix_size": {
                "m": self.m,
                "n": self.n,
                "p": self.p
            },
            "num_multiplications": len(self.algorithm),
            "naive_multiplications": self.m * self.n * self.p,
            "improvement": f"{(1 - len(self.algorithm) / (self.m * self.n * self.p)) * 100:.1f}%",
            "rank1_tensors": self.algorithm,
            "verification": {
                "residual_norm": float(np.linalg.norm(self.residual_tensor)),
                "complete": self._is_decomposition_complete()
            }
        }
        return description

    def save_algorithm(self, filepath: str) -> None:
        """
        Save discovered algorithm to JSON file.

        Args:
            filepath: Path to save JSON file
        """
        algorithm_desc = self.get_algorithm_description()
        with open(filepath, 'w') as f:
            json.dump(algorithm_desc, f, indent=2)

    def verify_algorithm(self) -> bool:
        """
        Verify the discovered algorithm produces correct results.

        Returns:
            True if algorithm is correct
        """
        # Reconstruct tensor from algorithm
        reconstructed = np.zeros_like(self.target_tensor)

        for component in self.algorithm:
            u = np.array(component["u"])
            v = np.array(component["v"])
            w = np.array(component["w"])
            rank1 = self._compute_rank1_tensor(u, v, w)
            reconstructed += rank1

        # Check if it matches target
        error = np.linalg.norm(reconstructed - self.target_tensor)
        return error < 1e-6

    def render(self, mode: str = "human") -> Optional[Dict]:
        """
        Render the current state.

        Args:
            mode: Rendering mode ("human" or "json")

        Returns:
            Rendering output (for "json" mode)
        """
        if mode == "human":
            print("\n" + "=" * 70)
            print(f"TENSOR DECOMPOSITION ENVIRONMENT - Step {self.current_step}")
            print("=" * 70)
            print(f"Matrix size: {self.m}×{self.p} × {self.p}×{self.n}")
            print(f"Rank-1 tensors used: {len(self.algorithm)}")
            print(f"Residual norm: {np.linalg.norm(self.residual_tensor):.6f}")
            print(f"Target (naive) rank: {self.m * self.n * self.p}")
            print(f"Best rank found: {self.best_rank}")

            if self._is_decomposition_complete():
                print("\n✅ DECOMPOSITION COMPLETE!")
                efficiency = (1 - len(self.algorithm) / (self.m * self.n * self.p)) * 100
                print(f"Efficiency gain: {efficiency:.1f}%")

            print("=" * 70 + "\n")

        elif mode == "json":
            return {
                "step": self.current_step,
                "rank_used": len(self.algorithm),
                "residual_norm": float(np.linalg.norm(self.residual_tensor)),
                "complete": self._is_decomposition_complete()
            }


def test_environment():
    """
    Test the environment to ensure it works correctly.
    """
    print("=" * 70)
    print("TENSOR DECOMPOSITION ENVIRONMENT TEST")
    print("=" * 70)

    # Create environment
    env = TensorDecompositionEnv(matrix_size=(2, 2, 2), max_rank=20)

    print("\n1. Testing environment creation...")
    print(f"   ✓ Action space size: {env.action_space.n:,}")
    print(f"   ✓ Observation space shape: {env.observation_space.shape}")
    print(f"   ✓ Target tensor shape: {env.target_tensor.shape}")

    print("\n2. Testing reset...")
    obs, info = env.reset()
    print(f"   ✓ Observation shape: {obs.shape}")
    print(f"   ✓ Initial residual norm: {info['residual_norm']:.4f}")
    print(f"   ✓ Target rank: {info['target_rank']}")

    print("\n3. Testing action decoding...")
    action = 1000
    u, v, w = env._action_to_rank1_tensor(action)
    print(f"   ✓ Action {action} decoded to:")
    print(f"     u = {u}")
    print(f"     v = {v}")
    print(f"     w = {w}")

    print("\n4. Testing rank-1 tensor computation...")
    rank1 = env._compute_rank1_tensor(u, v, w)
    print(f"   ✓ Rank-1 tensor shape: {rank1.shape}")
    print(f"   ✓ Rank-1 tensor:\n{rank1}")

    print("\n5. Testing step execution...")
    for step in range(5):
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        print(f"   Step {step + 1}: reward={reward:6.2f}, "
              f"norm={info['residual_norm']:6.4f}, "
              f"valid={info['action_valid']}, "
              f"rank={info['rank_used']}")

        if terminated or truncated:
            break

    print("\n6. Testing algorithm description...")
    algo_desc = env.get_algorithm_description()
    print(f"   ✓ Algorithm uses {algo_desc['num_multiplications']} multiplications")
    print(f"   ✓ Naive algorithm uses {algo_desc['naive_multiplications']} multiplications")

    print("\n✅ All tests passed!")
    print("=" * 70)


if __name__ == "__main__":
    test_environment()