import json
import wandb
import os

def visualize_results():
    # 1. Load Benchmark Results
    file_path = "agent_benchmark_results.json"
    if not os.path.exists(file_path):
        print(f"Error: {file_path} not found.")
        return

    with open(file_path, "r") as f:
        data = json.load(f)

    # 2. Initialize WandB
    # We use the same project "alpha_tensor_rl" but a distinct run name
    try:
        run = wandb.init(
            project="alpha_tensor_rl",
            name="benchmark_results_visualization",
            job_type="benchmark",
            config=data.get("problem", {})
        )
    except Exception as e:
        print(f"Failed to initialize WandB: {e}")
        print("Please ensure you are logged in using 'wandb login'.")
        return

    print("WandB initialized successfully.")

    # 3. Extract Data
    implementations = data.get("implementations", [])
    
    # Prepare data for Table
    # Columns: Implementation, Latency (us), Multiplications, Op Count
    table_data = []
    
    # Prepare data for Custom Charts (lists)
    names = []
    latencies = []
    multiplications = []
    
    for impl in implementations:
        name = impl.get("name", "Unknown")
        perf = impl.get("performance", {})
        
        latency = perf.get("latency_us", 0)
        mults = perf.get("num_multiplications", 0)
        ops = perf.get("op_count", 0)
        
        # Table row
        table_data.append([name, latency, mults, ops])
        
        # Chart lists
        names.append(name)
        latencies.append(latency)
        multiplications.append(mults)

    # 4. Log Table
    columns = ["Implementation", "Latency (us)", "Multiplications", "Op Count"]
    table = wandb.Table(data=table_data, columns=columns)
    wandb.log({"benchmark_table": table})

    # 5. Log Charts
    # We can use wandb.plot.bar explicitly for better control
    # Chart 1: Latency Comparison
    latency_table = wandb.Table(data=[[n, l] for n, l in zip(names, latencies)], columns=["implementation", "latency_us"])
    wandb.log({"latency_comparison": wandb.plot.bar(latency_table, "implementation", "latency_us", title="Benchmark Latency (us)")})

    # Chart 2: Multiplication Count Comparison
    mult_table = wandb.Table(data=[[n, m] for n, m in zip(names, multiplications)], columns=["implementation", "multiplications"])
    wandb.log({"multiplications_comparison": wandb.plot.bar(mult_table, "implementation", "multiplications", title="Benchmark Multiplications")})

    print("Logged table and charts to WandB.")
    
    # Finish run
    wandb.finish()
    print("Run finished.")

if __name__ == "__main__":
    visualize_results()
