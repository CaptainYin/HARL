 
# MAPPOLNHPN training with for loops
# Define model directories
# declare -A model_dirs=(
    # ["4H"]="/project/HARL-main/examples/results/lasercar/Cube/mappo/HPN4H_Cube_4v4/seed-00001-2025-06-10-13-46-12/models"
    # ["8H"]="/project/HARL-main/examples/results/lasercar/Cube/mappo/HPN8H_Cube_4v4/seed-00001-2025-06-10-13-50-05/models"
    # ["16H"]="/project/HARL-main/examples/results/lasercar/Cube/mappo/HPN16H_Cube_4v4/seed-00001-2025-06-10-13-52-25/models"
    # ["4H"]="/project/HARL-main/examples/results/lasercar/Cube/mappo/HPN4H_Cube_10v10/seed-00001-2025-06-11-01-41-34/models"
    # ["8H"]="/project/HARL-main/examples/results/lasercar/Cube/mappo/HPN8H_Cube_10v10/seed-00001-2025-06-11-01-42-25/models"
    # ["16H"]="/project/HARL-main/examples/results/lasercar/Cube/mappo/HPN16H_Cube_10v10/seed-00001-2025-06-11-01-42-43/models"
    # ["2H"]="/project/HARL-main/examples/results/lasercar/Cube/mappo/MAPPOHPN_Cube_4v4_randompos/seed-00001-2025-05-14-03-25-27/models"
    # ["16H32"]="/project/HARL-main/examples/results/lasercar/Cube/mappo/HPN16H32_Cube_4v4/seed-00001-2025-06-11-11-10-41/models"
    # ["16H128"]="/project/HARL-main/examples/results/lasercar/Cube/mappo/HPN16H128_Cube_4v4/seed-00001-2025-06-11-11-11-20/models/"
#     ["MLP_4v4"]="/project/HARL-main/examples/results/lasercar/Cube/mappo/MAPPOMLP_Cube_4v4_randompos1/seed-00001-2025-05-14-15-43-07/models"
#     ["MLP_10v10"]="/project/HARL-main/examples/results/lasercar/Cube/mappo/MAPPOMLP_Cube_10v10_randompos1/seed-00001-2025-05-14-18-06-45/models"
# )

# Define scenarios with their corresponding goal_num and num_agents
declare -A scenarios=(
    ["Unmaze"]="4 4"
    ["CA"]="4 4"
    ["Cube"]="8 8"
    ["Warehouse"]="8 8"
    ["scenario1"]="4 4"
    ["scenario3"]="8 8"
    ["scenario6"]="8 8"
)

# # Loop through each model configuration
# for config in "MLP_10v10" "MLP_4v4"; do
#     echo "${config}"
#     model_dir="${model_dirs[$config]}"
    
#     # Loop through each scenario --critic_hpn_structure "[0, [4, 7], [0, 3], 4, 64]"
#     for scenario in "Unmaze" "CA" "Cube" "Warehouse" "scenario1" "scenario3" "scenario6"; do
#         # Get goal_num and num_agents for this scenario
#         scenario_params=(${scenarios[$scenario]})
#         goal_num=${scenario_params[0]}
#         num_agents=${scenario_params[1]}
        
#         python3 train.py --algo mappo --env lasercar --scenario "$scenario" \
#             --goal_num "$goal_num" --num_agents "$num_agents" --Car_resettype diag \
#             --model_dir "$model_dir"  --exp_name HPNCritic
#     done
# done

# for PID_calaction_dis in 1.5 2 2.5 3 3.5; do
#     for emergency_dis in 0.3 0.4 0.5 0.6 0.7 0.8 1 1.2; do
#         for scenario in "Unmaze" "CA" "Cube" "Warehouse" "scenario1" "scenario3" "scenario6"; do
#             # Get goal_num and num_agents for this scenario
#             scenario_params=(${scenarios[$scenario]})
#             goal_num=${scenario_params[0]}
#             num_agents=${scenario_params[1]}
#             python3 train.py --algo mappo --env lasercar --scenario "$scenario" \
#                 --goal_num "$goal_num" --num_agents "$num_agents" --Car_resettype diag \
#                 --save_history True --use_render True --task hybrid --emergency_dis "$emergency_dis" --PID_calaction_dis "$PID_calaction_dis"
#         done
#         # echo "Running training for scenario: $scenario with emergency_dis: $emergency_dis"

#     done
# done


# for emergency_dis in 0.3 0.4 0.5 0.6 0.7 0.8 1 1.2; do
#     for scenario in "Unmaze" "CA" "Cube" "Warehouse" "scenario1" "scenario3" "scenario6"; do
#         # Get goal_num and num_agents for this scenario
#         scenario_params=(${scenarios[$scenario]})
#         goal_num=${scenario_params[0]}
#         num_agents=${scenario_params[1]}
#         python3 train.py --algo mappo --env lasercar --scenario "$scenario" \
#             --goal_num "$goal_num" --num_agents "$num_agents" --Car_resettype diag \
#             --save_history True --use_render True --task hybridorca --emergency_dis "$emergency_dis" 
#     done
#     # echo "Running training for scenario: $scenario with emergency_dis: $emergency_dis"

# done

for scenario in "Unmaze" "CA" "Cube" "Warehouse" "scenario1" "scenario3" "scenario6"; do
    # Get goal_num and num_agents for this scenario
    scenario_params=(${scenarios[$scenario]})
    goal_num=${scenario_params[0]}
    num_agents=${scenario_params[1]}
    python3 train.py --algo mappo --env lasercar --scenario "$scenario" \
        --goal_num "$goal_num" --num_agents "$num_agents" --Car_resettype diag \
        --save_history True --use_render True --task SRLORCA #NHORCA   
    # echo "Running training for scenario: $scenario with emergency_dis: $emergency_dis"
done

#  python3 train.py --algo mappo --env lasercar --scenario CA --goal_num 4 --num_agents 4 --Car_resettype diag --save_history True --use_render True --task NHORCA 
# python3 train.py --algo mappo --env lasercar --scenario Cube --goal_num 8 --num_agents 8 --Car_resettype random_target --save_history False --use_render False --task MTSP1 
# python3 train.py --algo mappo --env lasercar --scenario Cube --goal_num 8 --num_agents 8 --Car_resettype random_target --save_history False --use_render False --task Hybridorca
# python3 train.py --algo mappo --env lasercar --scenario Cube --goal_num 8 --num_agents 8 --Car_resettype random_target --save_history False --use_render False --task Hybrid --exp_name Hybrid
python3 train.py --algo mappo --env lasercar --scenario Cube --goal_num 10 --num_agents 10 --Car_resettype random_target --save_history False --use_render False --task SRLORCA --exp_name SRLORCA_10v10

python3 train.py --algo mappo --env lasercar --scenario Warehouse --goal_num 10 --num_agents 10 --Car_resettype random_target --save_history False --use_render False --task SRLORCA --exp_name SRLORCA_Warehouse_10v10

python3 train.py --algo mappo --env lasercar --scenario mix --goal_num 10 --num_agents 10 --Car_resettype random_target --save_history False --use_render False --task SRLORCA --exp_name SRLORCA_mix_10v10
python3 train.py --algo mappo --env lasercar --scenario mix --goal_num 10 --num_agents 10 --Car_resettype random_target --save_history False --use_render False --task SRLORCA --exp_name SRLORCA_mix_10v10_5s_rt

python3 train.py --algo mappo --env lasercar --scenario Cube --goal_num 10 --num_agents 10 --Car_resettype random_target --save_history False --use_render False --task SRLORCA --exp_name SRLORCA_10v10_rt

python3 train.py --algo mappo --env lasercar --scenario Warehouse --goal_num 100 --num_agents 10 --Car_resettype random_target --save_history True --use_render True --task SRLORCA --assignment AMARL --exp_name SRLORCA_mix_10v10

python3 train.py --algo mappo --env lasercar --scenario Warehouse --goal_num 100 --num_agents 10 --Car_resettype random_target --save_history False --use_render False --task MTSP --reachgoalset True --exp_name reachgoalset10V100

python3 train.py --algo mappo --env lasercar --scenario Warehouse --goal_num 8 --num_agents 4 --Car_resettype random_target --save_history True --use_render True --task MTSP --assignment AMARL1 --reachgoalset True --critic_hpn False --exp_name reachgoalset10V100

python3 train.py --algo ippo --env lasercar --scenario Cube --goal_num 10 --num_agents 10 --Car_resettype random_target --save_history False --use_render False --task SRLORCA --use_curiosity True --exp_name SRLORCA_Cube_ippo_icm_10v10

python3 train.py --algo ippo --env lasercar --scenario Cube --goal_num 10 --num_agents 10  --Car_resettype random_target --save_history False --use_render False --task mtsp1 --cuda True --exp_name IPPO_cube_10v10_encoder

python3 train.py --algo ippo --env lasercar --scenario Warehouse --goal_num 8 --num_agents 8 --Car_resettype random_target --save_history False --use_render False --cuda False --task MTSP1 --assignment AMARL1 --reachgoalset False --reward_type long2018 --exp_name Mtsp1_ippo_Warehouse_Long2018

python3 train.py --algo mappo --env lasercar --scenario mix --goal_num 10 --num_agents 10 --Car_resettype random_target --save_history False --use_render False --cuda True --task SRLORCA --assignment AMARL1 --reachgoalset False --reward_type binary --model_dir "/project/HARL-main/examples/results/lasercar/mix/mappo/SRLORCA_mappo_mix6_binary/seed-00001-2025-09-20-05-04-29/models" --exp_name SRLORCA_mappo_mix6_binary_Ct

python3 train.py --algo mappo --env lasercar --scenario CA --goal_num 4 --num_agents 4 --Car_resettype diag --save_history True --use_render True --render_episodes 200 --model_dir "/project/HARL-main/examples/results/lasercar/mix/mappo/SRLORCA_mix_10v10_5s_rt/seed-00001-2025-09-13-14-01-35/models/actor_agent0_3400.pt" --task SRLORCA --cuda True --gpu_id 1 --exp_name HPNCritic

python3 train.py --algo mappo --env lasercar --scenario CA --goal_num 4 --num_agents 4 --Car_resettype diag --save_history True --use_render True --render_episodes 1000 --model_dir "/project/HARL-main/examples/results/lasercar/Cube/mappo/MAPPOMLP_Cube_10v10_randompos1/seed-00001-2025-05-14-18-06-45/models/actor_agent0.pt" --critic_hpn False --task MTSP1 --exp_name HPNCritic

python3 train.py --algo mappo --env lasercar --scenario scenario1 --goal_num 4 --num_agents 4 --Car_resettype diag --save_history True --use_render True --render_episodes 1000 --model_dir "/project/HARL-main/examples/results/lasercar/Cube/mappo/MAPPOMLP_Cube_10v10_randompos1/seed-00001-2025-05-14-18-06-45/models/actor_agent0.pt" --critic_hpn False --task MTSP1 --exp_name HPNCritic

python3 train.py --algo mappo --env lasercar --scenario scenario3 --goal_num 8 --num_agents 8 --Car_resettype diag --save_history True --use_render True --render_episodes 1000 --model_dir "/project/HARL-main/examples/results/lasercar/Cube/mappo/MAPPOMLP_Cube_10v10_randompos1/seed-00001-2025-05-14-18-06-45/models/actor_agent0.pt" --critic_hpn False --task MTSP1 --exp_name HPNCritic

python3 train.py --algo mappo --env lasercar --scenario scenario6 --goal_num 8 --num_agents 8 --Car_resettype diag --save_history True --use_render True --render_episodes 1000 --model_dir "/project/HARL-main/examples/results/lasercar/Cube/mappo/MAPPOMLP_Cube_10v10_randompos1/seed-00001-2025-05-14-18-06-45/models/actor_agent0.pt" --critic_hpn False --task MTSP1 --exp_name HPNCritic

python3 train.py --algo ippo --env lasercar --scenario Cube --goal_num 10 --num_agents 10 --Car_resettype random_target --save_history False --use_render False --cuda False --task MTSP1 --assignment AMARL1 --reachgoalset False --reward_type long2018 --share_reward False --exp_name Mtsp1_Cube_ippo_Long2018

python3 train.py --algo ippo --env lasercar --scenario Cube --goal_num 10 --num_agents 10 --Car_resettype random_target --save_history False --use_render False --cuda False --task MTSP1 --assignment AMARL1 --reachgoalset False --share_reward False --exp_name Mtsp1_Cube_ippo

python3 train.py --algo ippo --env lasercar --scenario Cube --goal_num 10 --num_agents 10 --Car_resettype random_target --save_history False --use_render False --cuda True --task MTSP1 --assignment AMARL1 --reachgoalset False  --share_reward False --use_self_attention True --exp_name Mtsp1_Cube_ippo_selfatt

python3 train.py --algo mappo --env lasercar --scenario Cube --goal_num 10 --num_agents 10 --Car_resettype random_target --save_history False --use_render False --cuda True --task MTSP1 --assignment AMARL1 --reachgoalset False  --share_reward True --use_self_attention True --exp_name Mtsp1_Cube_mappohpn_selfatt

python3 train.py --algo mappo --env lasercar --scenario Cube --goal_num 10 --num_agents 10 --Car_resettype random_target --save_history False --use_render False --cuda True  --gpu_id 1 --task SRLORCA --assignment AMARL1 --reachgoalset False  --share_reward True --use_curiosity True --exp_name SRLORCA_Cube_hpn_icm

python3 train.py --algo ippo --env lasercar --scenario Cube --goal_num 10 --num_agents 10 --Car_resettype random_target --save_history False --use_render False --cuda False --task MTSP1 --assignment AMARL1 --reachgoalset False  --share_reward False --use_curiosity True --exp_name Mtsp1_Cube_ippo_icm
python3 train.py --algo ippo --env lasercar --scenario Cube --goal_num 10 --num_agents 10 --Car_resettype random_target --save_history False --use_render False --cuda False --task SRLORCA --assignment AMARL1 --reachgoalset False  --share_reward False --use_curiosity True --exp_name SRLORCA_Cube_ippo_icm

python3 train.py --algo ippo --env lasercar --scenario Cube --goal_num 10 --num_agents 10 --Car_resettype random_target --save_history False --use_render False --cuda True --task SRLORCA --assignment AMARL1 --reachgoalset False  --share_reward False --use_curiosity True --exp_name SRLORCA_Cube_ippo_icm001

python3 train.py --algo ippo --env lasercar --scenario Warehouse --goal_num 10 --num_agents 4 --Car_resettype random_target --save_history False --use_render False --cuda False --task MTSP --assignment AMARL1 --reachgoalset True  --share_reward False  --exp_name Mtsp_Warehouse4V10_ippo_woEq12

python3 train.py --algo ippo --env lasercar --scenario Cube --goal_num 10 --num_agents 4 --Car_resettype random_target --save_history False --use_render False --cuda True --task MTSP --assignment AMARL1 --reachgoalset True  --share_reward False  --exp_name Mtsp_Cube4V10_ippo_woEq12

python3 train.py --algo mappo --env lasercar --scenario Cube --goal_num 10 --num_agents 10 --Car_resettype random_target --save_history False --use_render False --cuda True  --gpu_id 1 --task SRLORCA --assignment AMARL1 --reachgoalset False  --share_reward True --reward_type binary --use_curiosity True --exp_name SRLORCA_Cube_hpn_binary_icm

python3 train.py --algo mappo --env lasercar --scenario Cube --goal_num 10 --num_agents 10 --Car_resettype random_target --save_history False --use_render False --cuda True  --gpu_id 1 --task SRLORCA --assignment AMARL1 --reachgoalset False  --share_reward True --reward_type binary --use_curiosity True --exp_name SRLORCA_Cube_hpn_binary_icm_clc001

python3 train.py --algo mappo --env lasercar --scenario Warehouse --goal_num 100 --num_agents 10 --Car_resettype random_target --save_history True --use_render True --cuda False --task MTSP --reachgoalset True --assignment AMARL1 --reachgoalset False --reward_type binary --model_dir "/project/HARL-main/examples/results/lasercar/mix/mappo/SRLORCA_mappo_mix6_binary/seed-00001-2025-09-20-05-04-29/models" --exp_name reachgoalset10V100

python3 train.py --algo mappo --env lasercar --scenario Warehouse --goal_num 20 --num_agents 5 --Car_resettype random_target --save_history True --use_render True --use_parallel_render True --n_eval_rollout_threads 2 --eval_episodes 2 --cuda True --gpu_id 1 --task SRLORCA --reachgoalset False --assignment AMARL  --reward_type binary --model_dir "/project/HARL-main/examples/results/lasercar/Warehouse/mappo/SRLORCA_Warehouse_10v10/seed-00001-2025-09-14-07-38-50/models/actor_agent0_4100.pt" --render_episodes 10 --fixed_theta True --modelfile "/project/MinMax-MTSP-master/partition/savemodel_homo_seq/AMARL_test5_topk4.pt" --exp_name tt

python3 train.py --algo mappo --env lasercar --scenario Warehouse --goal_num 20 --num_agents 5 --Car_resettype random_target --save_history True --use_render True --use_parallel_render True --n_eval_rollout_threads 2 --eval_episodes 2 --cuda True --gpu_id 1 --task SRLORCA --reachgoalset False --assignment AMARL  --reward_type binary --model_dir "/project/HARL-main/examples/results/lasercar/Warehouse/mappo/SRLORCA_Warehouse_10v10/seed-00001-2025-09-14-07-38-50/models/actor_agent0_4100.pt" --render_episodes 1 --fixed_theta True --modelfile "/project/MinMax-MTSP-master/partition/savemodel_homo_seq/AMARL_topk1_no_warehouse.pt" --exp_name tt

python3 train.py --algo mappo --env lasercar --scenario Warehouse --goal_num 20 --num_agents 5 --Car_resettype random_target --save_history True --use_render True --use_parallel_render True --n_eval_rollout_threads 100 --eval_episodes 25 --cuda True --gpu_id 1 --task SRLORCA --reachgoalset False --assignment CMPNN  --reward_type binary --model_dir "/project/HARL-main/examples/results/lasercar/Warehouse/mappo/SRLORCA_Warehouse_10v10/seed-00001-2025-09-14-07-38-50/models/actor_agent0_4100.pt" --render_episodes 1 --fixed_theta True --modelfile "/project/MinMax-MTSP-master/partition/savemodel_homo_seq/CMPNN_test1_topk4_agent1.pt" --exp_name tt

python3 train.py --algo mappo --env lasercar --scenario Warehouse --goal_num 20 --num_agents 5 --Car_resettype random_target --save_history True --use_render True --use_parallel_render True --n_eval_rollout_threads 2 --eval_episodes 2 --cuda True --gpu_id 1 --task SRLORCA --reachgoalset False --assignment DisPn  --reward_type binary --model_dir "/project/HARL-main/examples/results/lasercar/Warehouse/mappo/SRLORCA_Warehouse_10v10/seed-00001-2025-09-14-07-38-50/models/actor_agent0_4100.pt" --render_episodes 1 --fixed_theta True --modelfile "/project/MinMax-MTSP-master/partition/savemodel_homo_seq/DisPn_topk1_no_warehouse.pt" --exp_name tt

python3 train.py --algo mappo --env lasercar --scenario Warehouse --goal_num 20 --num_agents 5 --Car_resettype random_target --save_history True --use_render True --use_parallel_render True --n_eval_rollout_threads 2 --eval_episodes 2 --cuda True --gpu_id 1 --task SRLORCA --reachgoalset False --assignment DisPn  --reward_type binary --model_dir "/project/HARL-main/examples/results/lasercar/Warehouse/mappo/SRLORCA_Warehouse_10v10/seed-00001-2025-09-14-07-38-50/models/actor_agent0_4100.pt" --render_episodes 1 --fixed_theta True --modelfile "/project/MinMax-MTSP-master/partition/savemodel_homo_seq/DisPn_topk1_warehouse.pt" --exp_name tt

        # self.model_name ="AMARL_topk1_no_warehouse"#with out warehouse data,gap:  2.09
        # self.model_name ="DisPn_topk1_no_warehouse"#with out warehouse data:gap:  6.1
        self.model_name ="DisPn_topk1_warehouse"#with warehouse data,gap:  3.78
        # self.model_name ="AMARL_test5_topk4" #AMARL topk=4:gap:  1.124
        # self.model_name ="CMPNN_test1_topk4_agent1"#with CMPNN:1.182
python3 train.py --algo ippo --env lasercar --scenario Warehouse --goal_num 20 --num_agents 5 --Car_resettype random_target --save_history True --use_render True --use_parallel_render True --n_eval_rollout_threads 2 --eval_episodes 2 --cuda True --gpu_id 1 --task MTSP1 --assignment AMARL1 --reachgoalset True  --share_reward True  --model_dir "/project/HARL-main/examples/results/lasercar/Warehouse/ippo/Mtsp_Warehouse4V10_ippo_woEq12/seed-00001-2025-09-19-01-38-47/models/actor_agent0_5000.pt" --exp_name tt

