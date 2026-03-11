import os
import concurrent.futures

MRPP_file = "/project/HARL-main/examples/results/lasercar/Warehouse/mappo/SRLORCA_Warehouse_10v10/seed-00001-2025-09-14-07-38-50/models/actor_agent0_4100.pt"
# "/project/HARL-main/examples/results/lasercar/Cube/mappo/HPN2H_Cube_10v10/seed-00001-2025-06-13-09-21-58/models/actor_agent0_4950.pt"
# list all the file start with "actor_agent"
# AMARL_file1 = "/project/MinMax-MTSP-master/partition/savemodel_homo_seq/AMARL_test5_topk4.pt"
# AMARL_file1 = "/project/MinMax-MTSP-master/partition/savemodel_homo_seq/AMARL_topk10_Cube_4_13.pt"
# AMARL_file2 = "/project/MinMax-MTSP-master/partition/savemodel_homo_seq/AMARL_topk1_no_warehouse.pt"
AMARL_file2 = "/project/MinMax-MTSP-master/partition/savemodel_homo_seq/AMARL_woscenario_4_13.pt"
AMARL_file2_1 = "/project/MinMax-MTSP-master/partition/savemodel_homo_seq/AMARLEAW_Dec_Cube_4_13_trinorm_ft.pt"
# AMARL_file3 = "/project/MinMax-MTSP-master/partition/savemodel_homo_seq/CMPNN_topk10_Cube_new4.pt"
# AMARL_file3 = "/project/MinMax-MTSP-master/partition/savemodel_homo_seq/CMPNN_topk10_Cube_4_13.pt"
# AMARL_file4 = "/project/MinMax-MTSP-master/partition/savemodel_homo_seq/DisPn_topk1_no_warehouse.pt"
# AMARL_file4 = "/project/MinMax-MTSP-master/partition/savemodel_homo_seq/DisPn_topk10_Cube_4_13.pt"
AMARL_file4 = "/project/MinMax-MTSP-master/partition/savemodel_homo_seq/DisPn_woscenario_4_13.pt"
# AMARL_file5 = "/project/MinMax-MTSP-master/partition/savemodel_homo_seq/CMPNNGAT_topk10_Cube_4_13_1.pt"

eval_episode=100
eval_thread=1
def list_actor_files(directory):
    actor_files = [f for f in os.listdir(directory) if f.startswith("actor_agent")]
    return actor_files

def run_scenario(MRPP_file, scenario, goal_num, num_agents,rank,assignment):
    """Run a single scenario with given parameters"""
    # cmd = f"nohup python3 train.py --algo mappo --env lasercar --scenario {scenario} --goal_num {goal_num} --num_agents {num_agents} --Car_resettype random_target --save_history True --use_render True --use_parallel_render True --n_eval_rollout_threads {eval_thread} --eval_episodes {eval_episode} --cuda False --gpu_id 0 --task SRLORCA --reachgoalset False --assignment AMARL  --reward_type binary --model_dir {MRPP_file} --render_episodes 10 --fixed_theta True --modelfile {AMARL_file1} --exp_name tt > python.log 2>&1 &"
    # CUDA_LAUNCH_BLOCKING=1 
    # cmd = f"python3 train.py --algo mappo --env lasercar --scenario {scenario} --goal_num {goal_num} --num_agents {num_agents} --Car_resettype random_target --save_history True --use_render True --use_parallel_render True --n_eval_rollout_threads {eval_thread} --eval_episodes {eval_episode} --cuda True --gpu_id 0 --task SRLORCA --reachgoalset False --assignment AMARL  --reward_type binary --model_dir {MRPP_file} --render_episodes 10 --fixed_theta True --modelfile {AMARL_file1} --exp_name tt "
    # os.system(cmd)
 
    # cmd = f"nohup python3 train.py --algo mappo --env lasercar --scenario {scenario} --goal_num {goal_num} --num_agents {num_agents} --Car_resettype random_target --save_history True --use_render True --use_parallel_render True --n_eval_rollout_threads {eval_thread} --eval_episodes {eval_episode} --cuda False --gpu_id 0 --task SRLORCA --reachgoalset False --assignment AMARL  --reward_type binary --model_dir {MRPP_file} --render_episodes 10 --fixed_theta True --modelfile {AMARL_file2} --exp_name tt > python.log 2>&1 &"
    if assignment=="EAGAT":
        cmd = f"python3 train.py --algo mappo --env lasercar --scenario {scenario} --goal_num {goal_num} --num_agents {num_agents} --Car_resettype random_target --save_history True --use_render True --use_parallel_render True --n_eval_rollout_threads {eval_thread} --eval_episodes {eval_episode} --cuda True --gpu_id {rank%3} --task SRLORCA --reachgoalset False --assignment EAGAT --reward_type binary --model_dir {MRPP_file} --render_episodes 10 --fixed_theta True --modelfile {AMARL_file2_1} --reborn False --fixed_maxsteps False --csvfile render_10v20.csv --exp_name tt "
        os.system(cmd)
    elif assignment=="AMARL":
        cmd = f"python3 train.py --algo mappo --env lasercar --scenario {scenario} --goal_num {goal_num} --num_agents {num_agents} --Car_resettype random_target --save_history True --use_render True --use_parallel_render True --n_eval_rollout_threads {eval_thread} --eval_episodes {eval_episode} --cuda True --gpu_id {rank%3} --task SRLORCA --reachgoalset False --assignment AMARL  --reward_type binary --model_dir {MRPP_file} --render_episodes 10 --fixed_theta True --modelfile {AMARL_file2} --reborn False --fixed_maxsteps False --csvfile render_10v20.csv --exp_name tt "
        os.system(cmd)
    # python3 train.py --algo mappo --env lasercar --scenario Warehouse --goal_num 10 --num_agents 5 --Car_resettype random_target --save_history True --use_render True --use_parallel_render True --n_eval_rollout_threads 1 --eval_episodes 5 --cuda True --gpu_id 0 --task SRLORCA --reachgoalset False --assignment AMARL  --reward_type binary --model_dir "/project/HARL-main/examples/results/lasercar/Warehouse/mappo/SRLORCA_Warehouse_10v10/seed-00001-2025-09-14-07-38-50/models/actor_agent0_4100.pt" --render_episodes 10 --fixed_theta True --modelfile "/project/MinMax-MTSP-master/partition/savemodel_homo_seq/AMARL_woscenario_4_13.pt" --reborn False --fixed_maxsteps False --csvfile render3.csv --exp_name tt 
    
    
    # cmd = f"nohup python3 train.py --algo mappo --env lasercar --scenario {scenario} --goal_num {goal_num} --num_agents {num_agents} --Car_resettype random_target --save_history True --use_render True --use_parallel_render True --n_eval_rollout_threads {eval_thread} --eval_episodes {eval_episode} --cuda False --gpu_id 0 --task SRLORCA --reachgoalset False --assignment CMPNN --reward_type binary --model_dir {MRPP_file} --render_episodes 10 --fixed_theta True --modelfile {AMARL_file3} --exp_name tt > python.log 2>&1 &"
    # cmd = f"python3 train.py --algo mappo --env lasercar --scenario {scenario} --goal_num {goal_num} --num_agents {num_agents} --Car_resettype random_target --save_history True --use_render True --use_parallel_render True --n_eval_rollout_threads {eval_thread} --eval_episodes {eval_episode} --cuda True --gpu_id {rank%4} --task SRLORCA --reachgoalset False --assignment CMPNN --reward_type binary --model_dir {MRPP_file} --render_episodes 10 --fixed_theta True --modelfile {AMARL_file3} --exp_name tt "
    # os.system(cmd)
    
    # python3 train.py --algo mappo --env lasercar --scenario Warehouse --goal_num 100 --num_agents 40 --Car_resettype random_target --save_history True --use_render True --use_parallel_render True --n_eval_rollout_threads 10 --eval_episodes 100 --cuda True --gpu_id 0 --task SRLORCA --reachgoalset False --assignment CMPNN --reward_type binary --model_dir "/project/HARL-main/examples/results/lasercar/Warehouse/mappo/SRLORCA_Warehouse_10v10/seed-00001-2025-09-14-07-38-50/models/actor_agent0_4100.pt" --render_episodes 10 --fixed_theta True --modelfile "/project/MinMax-MTSP-master/partition/savemodel_homo_seq/AMARL_woscenario_4_13.pt" --exp_name tt
    
    
    # cmd = f"nohup python3 train.py --algo mappo --env lasercar --scenario {scenario} --goal_num {goal_num} --num_agents {num_agents} --Car_resettype random_target --save_history True --use_render True --use_parallel_render True --n_eval_rollout_threads {eval_thread} --eval_episodes {eval_episode} --cuda False --gpu_id 0 --task SRLORCA --reachgoalset False --assignment DisPn --reward_type binary --model_dir {MRPP_file} --render_episodes 10 --fixed_theta True --modelfile {AMARL_file4} --exp_name tt > python.log 2>&1 &"
    elif assignment=="DisPn":
        cmd = f"python3 train.py --algo mappo --env lasercar --scenario {scenario} --goal_num {goal_num} --num_agents {num_agents} --Car_resettype random_target --save_history True --use_render True --use_parallel_render True --n_eval_rollout_threads {eval_thread} --eval_episodes {eval_episode} --cuda True --gpu_id {rank%3} --task SRLORCA --reachgoalset False --assignment DisPn --reward_type binary --model_dir {MRPP_file} --render_episodes 10 --fixed_theta True --modelfile {AMARL_file4} --reborn False --fixed_maxsteps False --csvfile render_10v20.csv --exp_name tt "
        os.system(cmd)

    # cmd = f"nohup python3 train.py --algo mappo --env lasercar --scenario {scenario} --goal_num {goal_num} --num_agents {num_agents} --Car_resettype random_target --save_history True --use_render True --use_parallel_render True --n_eval_rollout_threads {eval_thread} --eval_episodes {eval_episode} --cuda False --gpu_id 0 --task SRLORCA --reachgoalset False --assignment DisPn --reward_type binary --model_dir {MRPP_file} --render_episodes 10 --fixed_theta True --modelfile {AMARL_file5} --exp_name tt > python.log 2>&1 &"
    # cmd = f"python3 train.py --algo mappo --env lasercar --scenario {scenario} --goal_num {goal_num} --num_agents {num_agents} --Car_resettype random_target --save_history True --use_render True --use_parallel_render True --n_eval_rollout_threads {eval_thread} --eval_episodes {eval_episode} --cuda True --gpu_id {rank%4} --task SRLORCA --reachgoalset False --assignment CMPNNGAT --reward_type binary --model_dir {MRPP_file} --render_episodes 10 --fixed_theta True --modelfile {AMARL_file5} --exp_name tt "
    # os.system(cmd)
    elif assignment=="GoalSet":
        ff="/project/HARL-main/examples/results/lasercar/Warehouse/ippo/Mtsp_Warehouse4V10_ippo_woEq12/seed-00001-2025-09-19-01-38-47/models/actor_agent0_5000.pt"
        # cmd =f"python3 train.py --algo ippo --env lasercar --scenario {scenario} --goal_num {goal_num} --num_agents {num_agents} --Car_resettype random_target --save_history True --use_render True --use_parallel_render True --n_eval_rollout_threads {eval_thread} --eval_episodes {eval_episode} --cuda False --gpu_id 0 --task MTSP1 --assignment AMARL1 --reachgoalset True  --share_reward True  --model_dir {ff} --exp_name tt > python.log 2>&1 &"   
        cmd =f"python3 train.py --algo ippo --env lasercar --scenario {scenario} --goal_num {goal_num} --num_agents {num_agents} --Car_resettype random_target --save_history True --use_render True --use_parallel_render True --n_eval_rollout_threads {eval_thread} --eval_episodes {eval_episode} --cuda True --gpu_id {rank%3} --task MTSP1 --assignment GoalSet --reachgoalset True  --share_reward True  --model_dir {ff} --reborn False --fixed_maxsteps False --csvfile render_10v20.csv --exp_name tt "  
        # python3 train.py --algo ippo --env lasercar --scenario Warehouse --goal_num 10 --num_agents 5 --Car_resettype random_target --save_history True --use_render True --use_parallel_render True --n_eval_rollout_threads 1 --eval_episodes 5 --cuda True --gpu_id 0 --task MTSP1 --assignment GoalSet --reachgoalset True  --share_reward True  --model_dir "/project/HARL-main/examples/results/lasercar/Warehouse/ippo/Mtsp_Warehouse4V10_ippo_woEq12/seed-00001-2025-09-19-01-38-47/models/actor_agent0_5000.pt" --exp_name tt --reborn False
        os.system(cmd)   
     
    #  python3 train.py --algo mappo --env lasercar --scenario Warehouse --goal_num 10 --num_agents 5 --Car_resettype random_target --save_history True --use_render True --use_parallel_render True --n_eval_rollout_threads 1 --eval_episodes 5 --cuda True --gpu_id 0 --task SRLORCA --reachgoalset False --assignment lapjv --metric dijstra --reward_type binary --model_dir "/project/HARL-main/examples/results/lasercar/Warehouse/mappo/SRLORCA_Warehouse_10v10/seed-00001-2025-09-14-07-38-50/models/actor_agent0_4100.pt" --fixed_theta True --exp_name tt --reborn False
    elif assignment=="lapjv":
        cmd = f"python3 train.py --algo mappo --env lasercar --scenario {scenario} --goal_num {goal_num} --num_agents {num_agents} --Car_resettype random_target --save_history True --use_render True --use_parallel_render True --n_eval_rollout_threads {eval_thread} --eval_episodes {eval_episode} --cuda True --gpu_id {rank%3} --task SRLORCA --reachgoalset False --assignment lapjv --metric dijstra --reward_type binary --model_dir {MRPP_file}  --fixed_theta True --reborn False --fixed_maxsteps False --csvfile render_10v20.csv --exp_name tt "
        os.system(cmd)
    elif assignment=="render":
        cmd = f"python3 train.py --algo mappo --env lasercar --scenario Warehouse --goal_num {goal_num} --num_agents {num_agents} --Car_resettype random_target --save_history True --use_render True --use_parallel_render False --cuda True --task SRLORCA --reachgoalset False --assignment EAGAT --reward_type binary --model_dir {MRPP_file} --render_episodes 500 --fixed_theta True --modelfile {AMARL_file2_1}  --exp_name tt"
        os.system(cmd)
    
    return
        

if __name__ == "__main__":
    # Set maximum number of parallel processes (adjust based on your system)
    MAX_WORKERS = 48  # Change this to your desired limit
    scenario_list = ['CA','Cube','Warehouse','scenario1','scenario3','scenario6']#
    # Prepare all tasks
    tasks = []
    rank=0
    goal_nums = [80,90,100]#
    # goal_nums = [20]
    agent_nums = [5]
    # assignment=["AMARL","EAGAT","DisPn","lapjv","GoalSet"]#
    assignment=["render"]
    for goal_num in goal_nums:
        for num_agents in agent_nums:
            for assign in assignment:
                if goal_num >= num_agents:
                    tasks.append((MRPP_file, 'Warehouse', goal_num, num_agents,rank,assign))
                    # for scenario in scenario_list:
                    #     tasks.append((MRPP_file, scenario, goal_num, num_agents,rank,assign))
                    rank+=1
                
    # goal_nums = [100]
    # agent_nums = [10,20,30,40,50,60,70,80,90,100]#
    # # agent_nums = [20,30,40,60,70,80,100]
    # for goal_num in goal_nums:
    #     for num_agents in agent_nums:
    #         for assign in assignment:
    #             if goal_num >= num_agents:
    #                 tasks.append((MRPP_file, 'Warehouse', goal_num, num_agents,rank,assign))
    #                 rank+=1
    print(f"Total tasks to run: {len(tasks)}")
    print(f"Running with maximum {MAX_WORKERS} parallel processes")
    
    # Run tasks in parallel with limited workers
    with concurrent.futures.ProcessPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = [executor.submit(run_scenario, *task) for task in tasks]
        # Wait for all tasks to complete and print results
        for future in concurrent.futures.as_completed(futures):
            try:
                result = future.result()
                # print(result)
            except Exception as exc:
                print(f'Task generated an exception: {exc}')
