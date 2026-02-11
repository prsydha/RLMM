
import sys
import json
import numpy as np
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

try:
    from kernels.gpu.agent_to_kernel import run_all_benchmarks
except ImportError as e:
    print(f"Error importing kernels.gpu.agent_to_kernel: {e}")
    sys.exit(1)

def verify_agent():
    # Actions from user request (Episode 8/9)
    actions = [
        {'u': [1, 0, 0, 0], 'v': [1, 0, 0, 1], 'w': [1, 0, 0, 0]},
        {'u': [0, 1, 0, 0], 'v': [0, 0, 1, 0], 'w': [1, 0, 0, 0]},
        {'u': [1, 0, 0, 0], 'v': [0, 1, 0, -1], 'w': [0, 1, 0, 0]},
        {'u': [1, 0, 0, 0], 'v': [0, 0, 0, 1], 'w': [-1, 1, 0, 0]},
        {'u': [0, 1, 0, 0], 'v': [0, 0, 0, 1], 'w': [0, 1, 0, 0]},
        {'u': [0, 0, 1, 0], 'v': [1, 0, 0, 0], 'w': [0, 0, 1, 0]},
        {'u': [0, 0, 0, 1], 'v': [0, 0, 1, 0], 'w': [0, 0, 1, 0]},
        {'u': [0, 0, 1, 0], 'v': [0, 1, 0, 0], 'w': [0, 0, 0, 1]},
        {'u': [0, 0, 0, 1], 'v': [0, 0, 0, 1], 'w': [0, 0, 0, 1]}
    ]
    
    # Convert to uvw_list for the benchmark
    uvw_list = []
    for action in actions:
        uvw_list.append({
            'u': np.array(action['u'], dtype=np.float32),
            'v': np.array(action['v'], dtype=np.float32),
            'w': np.array(action['w'], dtype=np.float32)
        })
    
    print("=" * 70)
    print("Verifying Agent Actions from Episode 8/9")
    print("Rank: 9, Algorithm: Discovered Decomposition")
    print("=" * 70)
    
    try:
        results = run_all_benchmarks(uvw_list)
        
        # Save results
        output_file = "agent_episode_9_verification.json"
        with open(output_file, "w") as f:
            json.dump(results, f, indent=2)
            
        print(f"\n✅ Verification results saved to {output_file}")
        
        # Print summary
        print("\nSummary:")
        for impl in results["implementations"]:
            status = "✅" if impl["correctness"]["passed"] else "❌"
            print(f"  {status} {impl['name']:10s}: {impl['performance']['latency_us']:8.3f} μs "
                  f"({impl['performance']['num_multiplications']} mults)")
                  
        if results["summary"]["all_correct"]:
            print("\nSUCCESS: All implementations (including agent's) passed correctness check.")
        else:
            print("\nFAILURE: Some implementations failed correctness check.")
            
    except Exception as e:
        print(f"Error during benchmark: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    verify_agent()
