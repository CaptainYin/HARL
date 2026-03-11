#!/usr/bin/env bash
#python3 train.py --algo happo --env dexhands --exp_name test
# python3 train.py  --exp_name render --load_config "/project/HARL-main/tuned_configs/pettingzoo_mpe/simple_reference_v2-discrete/mappo/config_render.json"
# python3 train.py  --exp_name happo --load_config "/project/HARL-main/tuned_configs/pettingzoo_mpe/simple_reference_v2-discrete/happo/config.json"
# python3 train.py  --exp_name hatrpo --load_config "/project/HARL-main/tuned_configs/pettingzoo_mpe/simple_reference_v2-discrete/hatrpo/config.json"
# python3 train.py  --exp_name haa2c --load_config "/project/HARL-main/tuned_configs/pettingzoo_mpe/simple_reference_v2-discrete/haa2c/config.json"


# python3 train.py  --exp_name mappo_sparse --load_config "/project/HARL-main/tuned_configs/pettingzoo_mpe/simple_reference_sparserew_v1-discrete/mappo/config.json"
# python3 train.py  --exp_name happo_sparse --load_config "/project/HARL-main/tuned_configs/pettingzoo_mpe/simple_reference_sparserew_v1-discrete/happo/config.json"
# python3 train.py  --exp_name hasac_sparse --load_config "/project/HARL-main/tuned_configs/pettingzoo_mpe/simple_reference_sparserew_v1-discrete/hasac/config.json"


# python3 train.py  --exp_name test --load_config "/project/HARL-main/tuned_configs/smac/3s5z/happo/config.json"
# python3 train.py  --exp_name test --load_config "/project/HARL-main/tuned_configs/smacv2/protoss_5_vs_5/happo/config.json"
# python3 train.py  --exp_name test --load_config "/project/HARL-main/tuned_configs/football/academy_3_vs_1_with_keeper/happo/config.json"
# python3 train.py  --exp_name test --load_config "/project/HARL-main/tuned_configs/mamujoco/Ant-v2-2x4/hasac/config.json"
# python3 train.py  --exp_name test --load_config  "/project/HARL-main/tuned_configs/mamujoco/Ant-v2-4x2/happo/config.json"
# python3 train.py  --exp_name test --load_config  "/project/HARL-main/tuned_configs/mamujoco/Ant-v2-4x2/mappo/config.json"
# python3 train.py  --exp_name test --load_config "/project/HARL-main/tuned_configs/football/academy_3_vs_1_with_keeper/happo/config.json"
# python3 train.py --algo mappo --env lasercar --exp_name test
# python3 train.py --algo happo --env lasercar --exp_name test



# # not installed
# python3 train.py  --exp_name test --load_config "/project/HARL-main/tuned_configs/dexhands/ShadowHandCatchAbreast/hasac/config.json"


# python3 train.py --algo matd3 --env lasercar --exp_name test
# python3 train.py --algo hatd3 --env lasercar --exp_name test
# python3 train.py --algo hasac --env lasercar --exp_name test
# python3 train.py --algo haa2c --env lasercar --exp_name test
# python3 train.py --algo hatrpo --env lasercar --exp_name test

# python3 train.py --load_config "/project/HARL-main/tuned_configs/lasercar/CA/mappo/config.json" --exp_name wonoisev2wpi_elen100 
# python3 train.py --load_config "/project/HARL-main/tuned_configs/lasercar/CA/mappo/config1.json"
# python3 train.py --load_config "/project/HARL-main/tuned_configs/lasercar/CA/happo/config.json"
# python3 train.py --load_config "/project/HARL-main/tuned_configs/lasercar/CA/happo/config1.json"
# python3 train.py --load_config "/project/HARL-main/tuned_configs/lasercar/CA/happo/config2.json"
# python3 train.py --load_config "/project/HARL-main/tuned_configs/lasercar/CA/happo/config3.json"

# python3 train.py --algo mappo --env lasercar --exp_name test --ppo_epoch 5 --scenario Unmaze --reward_type wonderingpenalty


# python3 train.py --algo mappo --env lasercar --exp_name Unmaze_ppoepoch5 --ppo_epoch 5 
# python3 train.py --algo mappo --env lasercar --exp_name Unmaze_ppoepoch7 --ppo_epoch 7 
# python3 train.py --algo mappo --env lasercar --exp_name Unmaze_ppoepoch10 --ppo_epoch 10 
# python3 train.py --algo mappo --env lasercar --exp_name Unmaze_ppoepoch12 --ppo_epoch 12 
# python3 train.py --algo mappo --env lasercar --exp_name Unmaze_ppoepoch15 --ppo_epoch 15 

# python3 train.py --exp_name mtspGS_ds_actor --load_config "/project/HARL-main/examples/results/lasercar/Unmaze/mappo/test1/seed-00001-2025-04-16-01-57-40/config3.json"
# python3 train.py --exp_name mtspGS_ds_critic --load_config "/project/HARL-main/examples/results/lasercar/Unmaze/mappo/test1/seed-00001-2025-04-16-01-57-40/config2.json"

# python3 train.py --exp_name mtspGS_ds_actor_critic --load_config "/project/HARL-main/examples/results/lasercar/Unmaze/mappo/test1/seed-00001-2025-04-16-01-57-40/config.json"
# python3 train.py --algo mappo --env lasercar --num_agents 10 --goal_num 10 --exp_name MAPPOLNHPN_Cube_10v10
# python3 train.py --algo mappo --env lasercar --exp_name MAPPOMLP_Cube_4v4_randompos
# python3 train.py --algo mappo --env lasercar --exp_name DeepsetCritic
# python3 train.py --algo mappo --env lasercar --exp_name HPNCritic


# python3 train.py --algo mappo --env lasercar --exp_name eval --model_name MAPPOHPN_unmaze_10v10 --model_dir \
#  "/project/HARL-main/examples/results/lasercar/Unmaze/mappo/nnhpn_scale10_rt_anum10_cnum10/seed-00001-2025-05-10-14-24-00/models" \
#  --assignment AMARL --num_agents 5 --goal_num 15 --scenario Unmaze 


#
  # "/project/HARL-main/examples/results/lasercar/CA/mappo/nnhpnCA_scale10_rt_anum4_cnum4/seed-00001-2025-05-12-10-11-36/models"
  # "/project/HARL-main/examples/results/lasercar/Unmaze/mappo/nnhpn_scale10_rt_anum30_cnum30/seed-00001-2025-05-11-13-51-27/models"
  # "/project/HARL-main/examples/results/lasercar/Unmaze/mappo/nnhpn_scale10_rt_anum20_cnum20/seed-00001-2025-05-11-13-50-25/models"

# anums="se eeee"

# for anum in $anums; do
#     for cnum in $(seq 4 4); do
#         for dnum in $(seq 732 750); do
#             echo "python3 main.py 'test' $anum $cnum $dnum 2 > python.log 2>&1 &"
#         done
#     done
# done

# A="foo bar baz"
# B="one two three"

# # build arrays
# read -r -a a_arr <<<"$A"
# read -r -a b_arr <<<"$B"

# for i in "${!a_arr[@]}"; do
#   echo "${a_arr[i]} ${b_arr[i]}"
# done
# A=(foo bar baz)
# B=(one two three)

# loop over indices
MAX_PROCS=${MAX_PROCS:-$(nproc)}
model_name=(HPN2H_Cube_10v10)
#  MAPPOHPN_Cube_4v4 MAPPOHPN_Cube_5v5 MAPPOHPN_Cube_10v10 MAPPOHPN_unmaze_5v5 MAPPOHPN_unmaze_10v10 MAPPOHPN_unmaze_15v15 MAPPOHPN_CA_5v5 MAPPOHPN_CA_10v10 MAPPOHPN_CA_15v15
model_dir=(""/project/HARL-main/examples/results/lasercar/Cube/mappo/HPN2H_Cube_10v10/seed-00001-2025-06-13-09-21-58/models/actor_agent0_4950.pt""\
          "/project/HARL-main/examples/results/lasercar/Cube/mappo/MAPPOHPN_Cube_4v4_randompos/seed-00001-2025-05-14-03-25-27/models"\
            "/project/HARL-main/examples/results/lasercar/Cube/mappo/MAPPOHPN_Cube_5v5_randompos/seed-00001-2025-05-14-05-27-10/models"\
            "/project/HARL-main/examples/results/lasercar/Cube/mappo/MAPPOHPN_Cube_10v10_randompos/seed-00001-2025-05-14-05-25-12/models"\
            "/project/HARL-main/examples/results/lasercar/Unmaze/mappo/nnhpn_scale10_rt_anum5_cnum5/seed-00001-2025-05-12-23-49-00/models" \
            "/project/HARL-main/examples/results/lasercar/Unmaze/mappo/nnhpn_scale10_rt_anum10_cnum10/seed-00001-2025-05-10-14-24-00/models" \
            "/project/HARL-main/examples/results/lasercar/Unmaze/mappo/nnhpn_scale10_rt_anum15_cnum15/seed-00001-2025-05-10-14-27-04/models" \
            "/project/HARL-main/examples/results/lasercar/CA/mappo/nnhpnCA_scale10_rt_anum5_cnum5/seed-00001-2025-05-12-10-12-39/models" \
            "/project/HARL-main/examples/results/lasercar/CA/mappo/nnhpnCA_scale10_rt_anum10_cnum10/seed-00001-2025-05-12-10-13-17/models" \
           "/project/HARL-main/examples/results/lasercar/CA/mappo/nnhpnCA_scale10_rt_anum15_cnum15/seed-00001-2025-05-12-10-14-16/models")
assignment=(lapjv)
# AMARL lapjv greedy
metric=(dijstra)
#euclidean dijstra
#  10 20 30 40 50 60 70 80 90 100
num_agents=(5 )
goal_num=(10 20 30 40 50) 
# (15 30 50 100)
# 'CA' 'Cube' 'Warehouse' 'Unmaze' 'scenario2' 'scenario3' 'scenario6'
scenario=('Warehouse')
count=0
for i in "${!model_name[@]}"; do
  for j in "${!assignment[@]}"; do
    for k in "${!scenario[@]}"; do
      for l in "${!num_agents[@]}"; do
        for m in "${!goal_num[@]}"; do
          # skip case: scenario is Unmaze or CA AND agents=15 AND goals=100
          if { [ "${scenario[k]}" = "Unmaze" ] || [ "${scenario[k]}" = "CA" ]; } \
             && [ "${goal_num[m]}" -eq 100 ] && [ "${num_agents[l]}" -ge 10 ]; then
              # echo "Skipping Unmaze and CA with 15 agents and 100 goals"
              sleep 0.01
              continue
          fi
          # require num_agents not less than goal_num
          if [ "${num_agents[l]}" -gt "${goal_num[m]}" ]; then
            continue
          fi
          if [ "${assignment[j]}" == "AMARL" ]; then

              if [ -f "evalres/trainAt_${model_name[i]}_evalAt_${scenario[k]}_${assignment[j]}_dijstra_${num_agents[l]}_${goal_num[m]}.pkl" ]; then
                  # echo "trainAt_${model_name[i]}_evalAt_${scenario[k]}_${assignment[j]}_dijstra_${num_agents[l]}_${goal_num[m]}.pkl"
                  # echo "File already exists, skipping..."
                  sleep 0.01
              else
                  count=$((count+1))
                  echo $count
                  # nohup python3 test.py 2>&1 &
                  nohup python3 train.py --algo mappo --env lasercar --exp_name eval --save_history True --cuda False --use_render True --model_name ${model_name[i]} --model_dir ${model_dir[i]} --assignment ${assignment[j]} --num_agents ${num_agents[l]} --goal_num ${goal_num[m]} --scenario ${scenario[k]} 2>&1 &
                  # python3 train.py --algo mappo --env lasercar --exp_name eval --save_history True --use_render True --model_name ${model_name[i]} --model_dir ${model_dir[i]} --assignment ${assignment[j]} --num_agents ${num_agents[l]} --goal_num ${goal_num[m]} --scenario ${scenario[k]}
                  # echo "python3 train.py --algo mappo --env lasercar --exp_name eval --save_history True --use_render True --model_name ${model_name[i]} --model_dir ${model_dir[i]} --assignment ${assignment[j]} --num_agents ${num_agents[l]} --goal_num ${goal_num[m]} --scenario ${scenario[k]} "
                  while (( $(jobs -rp | wc -l) >= MAX_PROCS )); do
                      echo $(jobs -rp | wc -l)
                      sleep 1
                  done
              fi
          else
              for p in "${!metric[@]}"; do

                  if [ -f "evalres/trainAt_${model_name[i]}_evalAt_${scenario[k]}_${assignment[j]}_${metric[p]}_${num_agents[l]}_${goal_num[m]}.pkl" ]; then
                      # echo "trainAt_${model_name[i]}_evalAt_${scenario[k]}_${assignment[j]}_${metric[p]}_${num_agents[l]}_${goal_num[m]}.pkl"
                      # echo "File already exists, skipping..."
                      sleep 0.01
                  else
                      count=$((count+1))
                      echo $count $p
                      # nohup python3 train.py --algo mappo --env lasercar --exp_name eval --save_history True --cuda False --use_render True --metric ${metric[p]} --model_name ${model_name[i]} --model_dir ${model_dir[i]} --assignment ${assignment[j]} --num_agents ${num_agents[l]} --goal_num ${goal_num[m]} --scenario ${scenario[k]}  2>&1 &
                      python3 train.py --algo mappo --env lasercar --exp_name eval --save_history True --use_render True --metric ${metric[p]} --model_name ${model_name[i]} --model_dir ${model_dir[i]} --assignment ${assignment[j]} --num_agents ${num_agents[l]} --goal_num ${goal_num[m]} --scenario ${scenario[k]}
                      # echo "python3 train.py --algo mappo --env lasercar --exp_name eval --save_history True --use_render True --metric ${metric[p]} --model_name ${model_name[i]} --model_dir ${model_dir[i]} --assignment ${assignment[j]} --num_agents ${num_agents[l]} --goal_num ${goal_num[m]} --scenario ${scenario[k]}"
                      while (( $(jobs -rp | wc -l) >= MAX_PROCS )); do
                          echo $(jobs -rp | wc -l)
                          sleep 1
                      done
                  fi
              done
          fi
        done
      done
    done
  done
done