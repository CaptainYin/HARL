import os
import concurrent.futures
import itertools
import time
import multiprocessing

# Updated paths with specific model files
MRPP_file = "/project/HARL-main/examples/results/lasercar/mix/mappo/SRLORCA_mappo_mix6_binary_Ct/seed-00001-2025-09-21-08-56-26/models/actor_agent0_10000.pt"
AMARL_file = "/project/MinMax-MTSP-master/partition/savemodel_homo_seq/AMARL_test5_topk4.pt"

def run_scenario(goal_num, num_agents, run_id):
    """Run a single scenario with given parameters"""
    log_file = f"python_goal{goal_num}_agents{num_agents}_{run_id}.log"
    exp_name = f"eval_goal{goal_num}_agents{num_agents}_{run_id}"
    
    # Use absolute path to train.py and disable CUDA for parallel runs
    cmd = f"cd /project/HARL-main/examples && python3 train.py --algo mappo --env lasercar --scenario Warehouse --goal_num {goal_num} --num_agents {num_agents} --Car_resettype random_target --save_history True --use_render True --cuda False --task SRLORCA --reachgoalset False --assignment AMARL --reward_type binary --model_dir {MRPP_file} --render_episodes 1 --fixed_theta True --modelfile {AMARL_file} --exp_name {exp_name} > {log_file} 2>&1"
    
    print(f"Starting: goal_num={goal_num}, num_agents={num_agents}, run_id={run_id}")
    result = os.system(cmd)
    
    if result == 0:
        return f"Completed successfully: goal_num={goal_num}, num_agents={num_agents}, run_id={run_id}"
    else:
        return f"Failed: goal_num={goal_num}, num_agents={num_agents}, run_id={run_id} (exit code: {result})"

def run_sequential():
    """Run scenarios sequentially"""
    goal_nums = [4, 6, 8]
    agent_nums = [2, 4, 6]
    
    print("Running scenarios sequentially...")
    
    for goal_num, num_agents in itertools.product(goal_nums, agent_nums):
        run_id = f"{goal_num}_{num_agents}"
        print(f"\n=== Running goal_num={goal_num}, num_agents={num_agents} ===")
        result = run_scenario(goal_num, num_agents, run_id)
        print(result)

def run_parallel_no_cuda():
    """Run in parallel but disable CUDA"""
    goal_nums = [4, 6, 8]
    agent_nums = [2, 4, 6]
    MAX_WORKERS = 3
    
    tasks = []
    run_id = 0
    
    for goal_num, num_agents in itertools.product(goal_nums, agent_nums):
        run_id += 1
        tasks.append((goal_num, num_agents, run_id))
    
    print(f"Running {len(tasks)} tasks in parallel (CUDA disabled)")
    
    with concurrent.futures.ProcessPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = [executor.submit(run_scenario, *task) for task in tasks]
        
        completed = 0
        for future in concurrent.futures.as_completed(futures):
            try:
                result = future.result()
                completed += 1
                print(f"[{completed}/{len(tasks)}] {result}")
            except Exception as exc:
                completed += 1
                print(f"[{completed}/{len(tasks)}] Task failed: {exc}")

def run_parallel_spawn():
    """Run in parallel using spawn method"""
    # Set spawn method for CUDA compatibility
    try:
        multiprocessing.set_start_method('spawn', force=True)
    except RuntimeError:
        pass
    
    goal_nums = [4, 6, 8]
    agent_nums = [2, 4, 6]
    MAX_WORKERS = 2  # Reduced for stability
    
    tasks = []
    run_id = 0
    
    for goal_num, num_agents in itertools.product(goal_nums, agent_nums):
        run_id += 1
        tasks.append((goal_num, num_agents, run_id))
    
    print(f"Running {len(tasks)} tasks in parallel (spawn method)")
    
    with concurrent.futures.ProcessPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = [executor.submit(run_scenario, *task) for task in tasks]
        
        completed = 0
        for future in concurrent.futures.as_completed(futures):
            try:
                result = future.result()
                completed += 1
                print(f"[{completed}/{len(tasks)}] {result}")
            except Exception as exc:
                completed += 1
                print(f"[{completed}/{len(tasks)}] Task failed: {exc}")

def main():
    # Check if model files exist
    if not os.path.exists(MRPP_file):
        print(f"Error: MRPP model file {MRPP_file} does not exist")
        return
    if not os.path.exists(AMARL_file):
        print(f"Error: AMARL model file {AMARL_file} does not exist")
        return
    
    print("Model files found:")
    print(f"  MRPP: {MRPP_file}")
    print(f"  AMARL: {AMARL_file}")
    print()
    
    print("Choose execution mode:")
    print("1. Sequential execution (safer, slower)")
    print("2. Parallel execution without CUDA (faster)")
    print("3. Parallel execution with spawn method (experimental)")
    
    choice = input("Enter choice (1/2/3) or press Enter for sequential: ").strip()
    
    if choice == "2":
        # Parallel execution without CUDA
        run_parallel_no_cuda()
    elif choice == "3":
        # Parallel execution with spawn
        run_parallel_spawn()
    else:
        # Default: Sequential execution
        run_sequential()

if __name__ == "__main__":
    main()
