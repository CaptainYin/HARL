import os
import concurrent.futures

file = "/project/HARL-main/examples/results/lasercar/mix/mappo/SRLORCA_mappo_mix6_binary_Ct/seed-00001-2025-09-21-08-56-26/models"
# "/project/HARL-main/examples/results/lasercar/mix/mappo/SRLORCA_mappo_mix7_binary_Ct/seed-00001-2025-09-20-05-15-01/models"
# "/project/HARL-main/examples/results/lasercar/mix/mappo/SRLORCA_mappo_mix7_binary/seed-00001-2025-09-20-05-07-25/models"
# "/project/HARL-main/examples/results/lasercar/mix/mappo/SRLORCA_mappo_mix6_binary_Ct/seed-00001-2025-09-20-05-17-22/models"
# "/project/HARL-main/examples/results/lasercar/mix/mappo/SRLORCA_mappo_mix6_binary/seed-00001-2025-09-20-05-04-29/models"
# "/project/HARL-main/examples/results/lasercar/Cube/ippo/SRLORCA_Cube_ippo_icm001/seed-00001-2025-09-20-03-33-44/models"
# "/project/HARL-main/examples/results/lasercar/Cube/mappo/Mtsp1_Cube_mappohpn_selfatt/seed-00001-2025-09-19-13-57-34/models"
# "/project/HARL-main/examples/results/lasercar/Cube/mappo/SRLORCA_Cube_mappohpn_selfatt/seed-00001-2025-09-19-13-58-39/models"

# "/project/HARL-main/examples/results/lasercar/Cube/ippo/Mtsp1_Cube_ippo_selfatt/seed-00001-2025-09-19-02-05-17/models"
# "/project/HARL-main/examples/results/lasercar/mix/ippo/Mtsp1_ippo_mix5_delfatt/seed-00001-2025-09-16-14-39-51/models"
# "/project/HARL-main/examples/results/lasercar/Cube/ippo/SRLORCA_Cube_ippo_icm005_10v10/seed-00001-2025-09-15-07-50-57/models"
# "/project/HARL-main/examples/results/lasercar/mix/ippo/SRLORCA_mix5_ippo_icm01_10v10/seed-00001-2025-09-16-07-42-46/models"
# "/project/HARL-main/examples/results/lasercar/Warehouse/mappo/SRLORCA_Warehouse_10v10/seed-00001-2025-09-14-07-38-50/models"
# "/project/HARL-main/examples/results/lasercar/Cube/mappo/SRLORCA_10v10/seed-00001-2025-08-25-13-01-23/models"

# list all the file start with "actor_agent"
def list_actor_files(directory):
    actor_files = [f for f in os.listdir(directory) if f.startswith("actor_agent")]
    return actor_files

def run_scenario(model_dir, scenario, goal_num, num_agents):
    """Run a single scenario with given parameters"""
    # cmd = f"nohup python3 train.py --algo ippo  --share_reward False --assignment AMARL1 --reachgoalset False --env lasercar --save_history True --use_render True --cuda False --task SRLORCA --scenario {scenario} --goal_num {goal_num} --num_agents {num_agents} --Car_resettype diag --use_curiosity True --model_dir {model_dir} --csvfile renderSRLORCA_Cube_ippo_icm001.csv --exp_name HPNCritic > python.log 2>&1 &"
    cmd = f"nohup python3 train.py --algo mappo  --assignment AMARL1 --reachgoalset False  --share_reward True --reward_type binary --env lasercar --save_history True --use_render True --cuda False --task SRLORCA --scenario {scenario} --goal_num {goal_num} --num_agents {num_agents} --Car_resettype diag --model_dir {model_dir} --csvfile SRLORCA_mappo_mix6_binary_Ct1.csv --exp_name HPNCritic > python.log 2>&1 &"
    # cmd = f"nohup python3 train.py --algo ippo  --assignment AMARL1 --reachgoalset False  --share_reward False --env lasercar --save_history True --use_render True --cuda False --task hybrid --scenario {scenario} --goal_num {goal_num} --num_agents {num_agents} --Car_resettype diag --model_dir {model_dir} --csvfile renderhybrid_Cube_ippo_Long2018.csv --exp_name HPNCritic > python.log 2>&1 &"
    
    # --add_sensor_noise True --fixed_theta True  --use_curiosity True --use_self_attention True --task SRLORCA
    os.system(cmd)
    return f"Completed: {scenario} with {model_dir}"

scenarios = {
    "Unmaze": [4, 4],
    "CA": [4, 4],
    "Cube": [8, 8],
    "Warehouse": [8, 8],
    "scenario1": [4, 4],
    "scenario3": [8, 8],
    "scenario6": [8, 8]
}
if __name__ == "__main__":
    # Set maximum number of parallel processes (adjust based on your system)
    MAX_WORKERS = 256  # Change this to your desired limit
    
    actor_files = list_actor_files(file)
    # actor_files=['actor_agent0_200.pt']
    
    
    print("Actor files in directory:")
    
    # Prepare all tasks
    tasks = []
    for f in actor_files:
        model_dir = os.path.join(file, f)
        print(f"Using model: {model_dir}")
        for scenario in ["Unmaze", "CA", "Cube" ,"Warehouse", "scenario1", "scenario3", "scenario6"]:
            # Get goal_num and num_agents for this scenario
            goal_num = scenarios[scenario][0]
            num_agents = scenarios[scenario][1]
            tasks.append((model_dir, scenario, goal_num, num_agents))
    
    print(f"Total tasks to run: {len(tasks)}")
    print(f"Running with maximum {MAX_WORKERS} parallel processes")
    
    # Run tasks in parallel with limited workers
    with concurrent.futures.ProcessPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = [executor.submit(run_scenario, *task) for task in tasks]
        
        # Wait for all tasks to complete and print results
        for future in concurrent.futures.as_completed(futures):
            try:
                result = future.result()
                print(result)
            except Exception as exc:
                print(f'Task generated an exception: {exc}')



