
# python3 train.py --algo mappo --env lasercar --exp_name Unmaze_wonderingpenalty  --scenario Unmaze --reward_type wonderingpenalty
# python3 train.py --algo mappo --env lasercar --exp_name Unmaze_reachall  --scenario Unmaze --reward_type reachall
# python3 train.py --algo mappo --env lasercar --exp_name Unmaze_reachallandwonderingpenalty --scenario Unmaze --reward_type reachallandwonderingpenalty
# python3 train.py --algo mappo --env lasercar --exp_name Unmaze_binary --scenario Unmaze --reward_type binary
# python3 train.py --algo mappo --env lasercar --exp_name Unmaze_binaryreachall --scenario Unmaze --reward_type binaryreachall

# python3 train.py --algo mappo --env lasercar --exp_name Unmaze_rollout2 --n_rollout_threads 2
# python3 train.py --algo mappo --env lasercar --exp_name Unmaze_rollout5 --n_rollout_threads 5
# python3 train.py --algo mappo --env lasercar --exp_name Unmaze_rollout10 --n_rollout_threads 10

# python3 train.py --algo mappo --env lasercar --exp_name Unmaze_naivernn --use_naive_recurrent_policy True
# python3 train.py --algo mappo --env lasercar --exp_name Unmaze_rnn --use_recurrent_policy True

# #MAPPOHPN_Cube_4v4
# python3 train.py --algo mappo --env lasercar --scenario Unmaze --goal_num 4 --num_agents 4 --Car_resettype diag --model_dir "/project/HARL-main/examples/results/lasercar/Cube/mappo/MAPPOHPN_Cube_4v4_randompos/seed-00001-2025-05-14-03-25-27/models" --exp_name HPNCritic
# python3 train.py --algo mappo --env lasercar --scenario CA --goal_num 4 --num_agents 4 --Car_resettype diag --model_dir "/project/HARL-main/examples/results/lasercar/Cube/mappo/MAPPOHPN_Cube_4v4_randompos/seed-00001-2025-05-14-03-25-27/models" --exp_name HPNCritic
# python3 train.py --algo mappo --env lasercar --scenario Cube --goal_num 8 --num_agents 8 --Car_resettype diag --model_dir "/project/HARL-main/examples/results/lasercar/Cube/mappo/MAPPOHPN_Cube_4v4_randompos/seed-00001-2025-05-14-03-25-27/models" --exp_name HPNCritic
# python3 train.py --algo mappo --env lasercar --scenario Warehouse --goal_num 8 --num_agents 8 --Car_resettype diag --model_dir "/project/HARL-main/examples/results/lasercar/Cube/mappo/MAPPOHPN_Cube_4v4_randompos/seed-00001-2025-05-14-03-25-27/models" --exp_name HPNCritic
# python3 train.py --algo mappo --env lasercar --scenario scenario1 --goal_num 4 --num_agents 4 --Car_resettype diag --model_dir "/project/HARL-main/examples/results/lasercar/Cube/mappo/MAPPOHPN_Cube_4v4_randompos/seed-00001-2025-05-14-03-25-27/models" --exp_name HPNCritic
# python3 train.py --algo mappo --env lasercar --scenario scenario3 --goal_num 8 --num_agents 8 --Car_resettype diag --model_dir "/project/HARL-main/examples/results/lasercar/Cube/mappo/MAPPOHPN_Cube_4v4_randompos/seed-00001-2025-05-14-03-25-27/models" --exp_name HPNCritic
# python3 train.py --algo mappo --env lasercar --scenario scenario6 --goal_num 8 --num_agents 8 --Car_resettype diag --model_dir "/project/HARL-main/examples/results/lasercar/Cube/mappo/MAPPOHPN_Cube_4v4_randompos/seed-00001-2025-05-14-03-25-27/models" --exp_name HPNCritic

# #MAPPOMLP_Cube_4v4
# python3 train.py --algo mappo --env lasercar --scenario Unmaze --goal_num 4 --num_agents 4 --Car_resettype diag --model_dir "/project/HARL-main/examples/results/lasercar/Cube/mappo/MAPPOMLP_Cube_4v4_randompos1/seed-00001-2025-05-14-15-43-07/models" --exp_name HPNCritic
# python3 train.py --algo mappo --env lasercar --scenario CA --goal_num 4 --num_agents 4 --Car_resettype diag  --model_dir "/project/HARL-main/examples/results/lasercar/Cube/mappo/MAPPOMLP_Cube_4v4_randompos1/seed-00001-2025-05-14-15-43-07/models" --exp_name HPNCritic
# python3 train.py --algo mappo --env lasercar --scenario Cube --goal_num 8 --num_agents 8 --Car_resettype diag  --model_dir "/project/HARL-main/examples/results/lasercar/Cube/mappo/MAPPOMLP_Cube_4v4_randompos1/seed-00001-2025-05-14-15-43-07/models" --exp_name HPNCritic
# python3 train.py --algo mappo --env lasercar --scenario Warehouse --goal_num 8 --num_agents 8 --Car_resettype diag  --model_dir "/project/HARL-main/examples/results/lasercar/Cube/mappo/MAPPOMLP_Cube_4v4_randompos1/seed-00001-2025-05-14-15-43-07/models" --exp_name HPNCritic
# python3 train.py --algo mappo --env lasercar --scenario scenario1 --goal_num 4 --num_agents 4 --Car_resettype diag  --model_dir "/project/HARL-main/examples/results/lasercar/Cube/mappo/MAPPOMLP_Cube_4v4_randompos1/seed-00001-2025-05-14-15-43-07/models" --exp_name HPNCritic
# python3 train.py --algo mappo --env lasercar --scenario scenario3 --goal_num 8 --num_agents 8 --Car_resettype diag  --model_dir "/project/HARL-main/examples/results/lasercar/Cube/mappo/MAPPOMLP_Cube_4v4_randompos1/seed-00001-2025-05-14-15-43-07/models" --exp_name HPNCritic
# python3 train.py --algo mappo --env lasercar --scenario scenario6 --goal_num 8 --num_agents 8 --Car_resettype diag  --model_dir "/project/HARL-main/examples/results/lasercar/Cube/mappo/MAPPOMLP_Cube_4v4_randompos1/seed-00001-2025-05-14-15-43-07/models" --exp_name HPNCritic

# #MAPPOHPN_Cube_5v5
# python3 train.py --algo mappo --env lasercar --scenario Unmaze --goal_num 4 --num_agents 4 --Car_resettype diag --model_dir  "/project/HARL-main/examples/results/lasercar/Cube/mappo/MAPPOHPN_Cube_5v5_randompos1/seed-00001-2025-05-14-15-39-29/models" --exp_name HPNCritic
# python3 train.py --algo mappo --env lasercar --scenario CA --goal_num 4 --num_agents 4 --Car_resettype diag --model_dir  "/project/HARL-main/examples/results/lasercar/Cube/mappo/MAPPOHPN_Cube_5v5_randompos1/seed-00001-2025-05-14-15-39-29/models" --exp_name HPNCritic
# python3 train.py --algo mappo --env lasercar --scenario Cube --goal_num 8 --num_agents 8 --Car_resettype diag --model_dir  "/project/HARL-main/examples/results/lasercar/Cube/mappo/MAPPOHPN_Cube_5v5_randompos1/seed-00001-2025-05-14-15-39-29/models" --exp_name HPNCritic
# python3 train.py --algo mappo --env lasercar --scenario Warehouse --goal_num 8 --num_agents 8 --Car_resettype diag --model_dir  "/project/HARL-main/examples/results/lasercar/Cube/mappo/MAPPOHPN_Cube_5v5_randompos1/seed-00001-2025-05-14-15-39-29/models" --exp_name HPNCritic
# python3 train.py --algo mappo --env lasercar --scenario scenario1 --goal_num 4 --num_agents 4 --Car_resettype diag --model_dir  "/project/HARL-main/examples/results/lasercar/Cube/mappo/MAPPOHPN_Cube_5v5_randompos1/seed-00001-2025-05-14-15-39-29/models" --exp_name HPNCritic
# python3 train.py --algo mappo --env lasercar --scenario scenario3 --goal_num 8 --num_agents 8 --Car_resettype diag --model_dir  "/project/HARL-main/examples/results/lasercar/Cube/mappo/MAPPOHPN_Cube_5v5_randompos1/seed-00001-2025-05-14-15-39-29/models" --exp_name HPNCritic
# python3 train.py --algo mappo --env lasercar --scenario scenario6 --goal_num 8 --num_agents 8 --Car_resettype diag --model_dir  "/project/HARL-main/examples/results/lasercar/Cube/mappo/MAPPOHPN_Cube_5v5_randompos1/seed-00001-2025-05-14-15-39-29/models" --exp_name HPNCritic

# #MAPPOHPN_Cube_10v10
# python3 train.py --algo mappo --env lasercar --scenario Unmaze --goal_num 4 --num_agents 4 --Car_resettype diag --model_dir  "/project/HARL-main/examples/results/lasercar/Cube/mappo/MAPPOHPN_Cube_10v10_randompos1/seed-00001-2025-05-14-15-40-08/models" --exp_name HPNCritic
# python3 train.py --algo mappo --env lasercar --scenario CA --goal_num 4 --num_agents 4 --Car_resettype diag --model_dir  "/project/HARL-main/examples/results/lasercar/Cube/mappo/MAPPOHPN_Cube_10v10_randompos1/seed-00001-2025-05-14-15-40-08/models" --exp_name HPNCritic
# python3 train.py --algo mappo --env lasercar --scenario Cube --goal_num 8 --num_agents 8 --Car_resettype diag --model_dir  "/project/HARL-main/examples/results/lasercar/Cube/mappo/MAPPOHPN_Cube_10v10_randompos1/seed-00001-2025-05-14-15-40-08/models" --exp_name HPNCritic
# python3 train.py --algo mappo --env lasercar --scenario Warehouse --goal_num 8 --num_agents 8 --Car_resettype diag --model_dir  "/project/HARL-main/examples/results/lasercar/Cube/mappo/MAPPOHPN_Cube_10v10_randompos1/seed-00001-2025-05-14-15-40-08/models" --exp_name HPNCritic
# python3 train.py --algo mappo --env lasercar --scenario scenario1 --goal_num 4 --num_agents 4 --Car_resettype diag --model_dir  "/project/HARL-main/examples/results/lasercar/Cube/mappo/MAPPOHPN_Cube_10v10_randompos1/seed-00001-2025-05-14-15-40-08/models" --exp_name HPNCritic
# python3 train.py --algo mappo --env lasercar --scenario scenario3 --goal_num 8 --num_agents 8 --Car_resettype diag --model_dir  "/project/HARL-main/examples/results/lasercar/Cube/mappo/MAPPOHPN_Cube_10v10_randompos1/seed-00001-2025-05-14-15-40-08/models" --exp_name HPNCritic
# python3 train.py --algo mappo --env lasercar --scenario scenario6 --goal_num 8 --num_agents 8 --Car_resettype diag --model_dir  "/project/HARL-main/examples/results/lasercar/Cube/mappo/MAPPOHPN_Cube_10v10_randompos1/seed-00001-2025-05-14-15-40-08/models" --exp_name HPNCritic

# #NHORCA
# python3 train.py --algo mappo --env lasercar --scenario Unmaze --goal_num 4 --num_agents 4 --Car_resettype diag --task NHORCA --exp_name HPNCritic
# python3 train.py --algo mappo --env lasercar --scenario CA --goal_num 4 --num_agents 4 --Car_resettype diag --task NHORCA --exp_name HPNCritic
# python3 train.py --algo mappo --env lasercar --scenario Cube --goal_num 8 --num_agents 8 --Car_resettype diag --task NHORCA --exp_name HPNCritic
# python3 train.py --algo mappo --env lasercar --scenario Warehouse --goal_num 8 --num_agents 8 --Car_resettype diag --task NHORCA --exp_name HPNCritic
# python3 train.py --algo mappo --env lasercar --scenario scenario1 --goal_num 4 --num_agents 4 --Car_resettype diag --task NHORCA --exp_name HPNCritic
# python3 train.py --algo mappo --env lasercar --scenario scenario3 --goal_num 8 --num_agents 8 --Car_resettype diag --task NHORCA --exp_name HPNCritic
# python3 train.py --algo mappo --env lasercar --scenario scenario6 --goal_num 8 --num_agents 8 --Car_resettype diag --task NHORCA --exp_name HPNCritic

# #MAPPOMLP_Cube_5v5
# python3 train.py --algo mappo --env lasercar --scenario Unmaze --goal_num 4 --num_agents 4 --Car_resettype diag --model_dir "/project/HARL-main/examples/results/lasercar/Cube/mappo/MAPPOMLP_Cube_5v5_randompos1/seed-00001-2025-05-14-17-02-14/models" --exp_name HPNCritic
# python3 train.py --algo mappo --env lasercar --scenario CA --goal_num 4 --num_agents 4 --Car_resettype diag  --model_dir "/project/HARL-main/examples/results/lasercar/Cube/mappo/MAPPOMLP_Cube_5v5_randompos1/seed-00001-2025-05-14-17-02-14/models" --exp_name HPNCritic
# python3 train.py --algo mappo --env lasercar --scenario Cube --goal_num 8 --num_agents 8 --Car_resettype diag  --model_dir "/project/HARL-main/examples/results/lasercar/Cube/mappo/MAPPOMLP_Cube_5v5_randompos1/seed-00001-2025-05-14-17-02-14/models" --exp_name HPNCritic
# python3 train.py --algo mappo --env lasercar --scenario Warehouse --goal_num 8 --num_agents 8 --Car_resettype diag  --model_dir "/project/HARL-main/examples/results/lasercar/Cube/mappo/MAPPOMLP_Cube_5v5_randompos1/seed-00001-2025-05-14-17-02-14/models" --exp_name HPNCritic
# python3 train.py --algo mappo --env lasercar --scenario scenario1 --goal_num 4 --num_agents 4 --Car_resettype diag  --model_dir "/project/HARL-main/examples/results/lasercar/Cube/mappo/MAPPOMLP_Cube_5v5_randompos1/seed-00001-2025-05-14-17-02-14/models" --exp_name HPNCritic
# python3 train.py --algo mappo --env lasercar --scenario scenario3 --goal_num 8 --num_agents 8 --Car_resettype diag  --model_dir "/project/HARL-main/examples/results/lasercar/Cube/mappo/MAPPOMLP_Cube_5v5_randompos1/seed-00001-2025-05-14-17-02-14/models" --exp_name HPNCritic
# python3 train.py --algo mappo --env lasercar --scenario scenario6 --goal_num 8 --num_agents 8 --Car_resettype diag  --model_dir "/project/HARL-main/examples/results/lasercar/Cube/mappo/MAPPOMLP_Cube_5v5_randompos1/seed-00001-2025-05-14-17-02-14/models" --exp_name HPNCritic

# #MAPPOMLP_Cube_10v10
# python3 train.py --algo mappo --env lasercar --scenario Unmaze --goal_num 4 --num_agents 4 --Car_resettype diag --model_dir "/project/HARL-main/examples/results/lasercar/Cube/mappo/MAPPOMLP_Cube_10v10_randompos1/seed-00001-2025-05-14-18-06-45/models" --exp_name HPNCritic
# python3 train.py --algo mappo --env lasercar --scenario CA --goal_num 4 --num_agents 4 --Car_resettype diag  --model_dir "/project/HARL-main/examples/results/lasercar/Cube/mappo/MAPPOMLP_Cube_10v10_randompos1/seed-00001-2025-05-14-18-06-45/models" --exp_name HPNCritic
# python3 train.py --algo mappo --env lasercar --scenario Cube --goal_num 8 --num_agents 8 --Car_resettype diag  --model_dir "/project/HARL-main/examples/results/lasercar/Cube/mappo/MAPPOMLP_Cube_10v10_randompos1/seed-00001-2025-05-14-18-06-45/models" --exp_name HPNCritic
# python3 train.py --algo mappo --env lasercar --scenario Warehouse --goal_num 8 --num_agents 8 --Car_resettype diag  --model_dir "/project/HARL-main/examples/results/lasercar/Cube/mappo/MAPPOMLP_Cube_10v10_randompos1/seed-00001-2025-05-14-18-06-45/models" --exp_name HPNCritic
# python3 train.py --algo mappo --env lasercar --scenario scenario1 --goal_num 4 --num_agents 4 --Car_resettype diag  --model_dir "/project/HARL-main/examples/results/lasercar/Cube/mappo/MAPPOMLP_Cube_10v10_randompos1/seed-00001-2025-05-14-18-06-45/models" --exp_name HPNCritic
# python3 train.py --algo mappo --env lasercar --scenario scenario3 --goal_num 8 --num_agents 8 --Car_resettype diag  --model_dir "/project/HARL-main/examples/results/lasercar/Cube/mappo/MAPPOMLP_Cube_10v10_randompos1/seed-00001-2025-05-14-18-06-45/models" --exp_name HPNCritic
# python3 train.py --algo mappo --env lasercar --scenario scenario6 --goal_num 8 --num_agents 8 --Car_resettype diag  --model_dir "/project/HARL-main/examples/results/lasercar/Cube/mappo/MAPPOMLP_Cube_10v10_randompos1/seed-00001-2025-05-14-18-06-45/models" --exp_name HPNCritic

# MAPPOLNHPN training with for loops
# Define model directories
declare -A model_dirs=(
    # ["4H"]="/project/HARL-main/examples/results/lasercar/Cube/mappo/HPN4H_Cube_4v4/seed-00001-2025-06-10-13-46-12/models"
    # ["8H"]="/project/HARL-main/examples/results/lasercar/Cube/mappo/HPN8H_Cube_4v4/seed-00001-2025-06-10-13-50-05/models"
    #Warehouse scenario1
    ["16H"]="/project/HARL-main/examples/results/lasercar/Cube/mappo/HPN16H_Cube_4v4/seed-00001-2025-06-10-13-52-25/models/actor_agent0.pt"
    # ["4H"]="/project/HARL-main/examples/results/lasercar/Cube/mappo/HPN4H_Cube_10v10/seed-00001-2025-06-11-01-41-34/models"
    # ["8H"]="/project/HARL-main/examples/results/lasercar/Cube/mappo/HPN8H_Cube_10v10/seed-00001-2025-06-11-01-42-25/models"
    # ["16H"]="/project/HARL-main/examples/results/lasercar/Cube/mappo/HPN16H_Cube_10v10/seed-00001-2025-06-11-01-42-43/models"
    #unmaze CA Cube scenario6
    ["2H"]="/project/HARL-main/examples/results/lasercar/Cube/mappo/HPN2H_Cube_4v4/seed-00001-2025-06-11-10-54-48/models/actor_agent0_4650.pt"
    # ["16H32"]="/project/HARL-main/examples/results/lasercar/Cube/mappo/HPN16H32_Cube_4v4/seed-00001-2025-06-11-11-10-41/models"
    # ["16H128"]="/project/HARL-main/examples/results/lasercar/Cube/mappo/HPN16H128_Cube_4v4/seed-00001-2025-06-11-11-11-20/models/"
    # ["MLP_4v4"]="/project/HARL-main/examples/results/lasercar/Cube/mappo/MAPPOMLP_Cube_4v4_randompos1/seed-00001-2025-05-14-15-43-07/models/actor_agent0.pt"
    # ["MLP_10v10"]="/project/HARL-main/examples/results/lasercar/Cube/mappo/MAPPOMLP_Cube_10v10_randompos1/seed-00001-2025-05-14-18-06-45/models"
    # TAMP scenario3
    # ["2H"]="/project/HARL-main/examples/results/lasercar/Cube/mappo/HPN2H_Cube_10v10/seed-00001-2025-06-13-09-21-58/models/actor_agent0_4950.pt"
)

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

# Loop through each model configuration
for config in "2H" ; do
    echo "${config}"
    model_dir="${model_dirs[$config]}"
    
    # Loop through each scenario --critic_hpn_structure "[0, [4, 7], [0, 3], 4, 64]"
    #"scenario1" "scenario3"   "Warehouse" "CA" "Cube"  "scenario6" "Unmaze"
    for scenario in "scenario3" ; do
        # Get goal_num and num_agents for this scenario
        scenario_params=(${scenarios[$scenario]})
        goal_num=${scenario_params[0]}
        num_agents=${scenario_params[1]}
        
        python3 train.py --algo mappo --env lasercar --scenario "$scenario" \
            --goal_num "$goal_num" --num_agents "$num_agents" --Car_resettype diag \
            --model_dir "$model_dir" --save_history True --cuda False --use_render True --exp_name HPNCritic
    done
done
