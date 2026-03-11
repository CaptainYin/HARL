import os
import pickle
import numpy as np
import csv

model_name = [
    "MAPPOHPN_Cube_4v4",
    "MAPPOHPN_Cube_5v5",
    "MAPPOHPN_Cube_10v10",
    "MAPPOHPN_unmaze_5v5",
    "MAPPOHPN_unmaze_10v10",
    "MAPPOHPN_unmaze_15v15",
    "MAPPOHPN_CA_5v5",
    "MAPPOHPN_CA_10v10",
    "MAPPOHPN_CA_15v15",
]
model_dir = [
    "/project/HARL-main/examples/results/lasercar/Cube/mappo/MAPPOHPN_Cube_4v4_randompos/seed-00001-2025-05-14-03-25-27/models",
    "/project/HARL-main/examples/results/lasercar/Cube/mappo/MAPPOHPN_Cube_5v5_randompos/seed-00001-2025-05-14-05-27-10/models",
    "/project/HARL-main/examples/results/lasercar/Cube/mappo/MAPPOHPN_Cube_10v10_randompos/seed-00001-2025-05-14-05-25-12/models",
    "/project/HARL-main/examples/results/lasercar/Unmaze/mappo/nnhpn_scale10_rt_anum5_cnum5/seed-00001-2025-05-12-23-49-00/models",
    "/project/HARL-main/examples/results/lasercar/Unmaze/mappo/nnhpn_scale10_rt_anum10_cnum10/seed-00001-2025-05-10-14-24-00/models",
    "/project/HARL-main/examples/results/lasercar/Unmaze/mappo/nnhpn_scale10_rt_anum15_cnum15/seed-00001-2025-05-10-14-27-04/models",
    "/project/HARL-main/examples/results/lasercar/CA/mappo/nnhpnCA_scale10_rt_anum5_cnum5/seed-00001-2025-05-12-10-12-39/models",
    "/project/HARL-main/examples/results/lasercar/CA/mappo/nnhpnCA_scale10_rt_anum10_cnum10/seed-00001-2025-05-12-10-13-17/models",
    "/project/HARL-main/examples/results/lasercar/CA/mappo/nnhpnCA_scale10_rt_anum15_cnum15/seed-00001-2025-05-12-10-14-16/models",
]
assignment = ["AMARL", "lapjv", "greedy"]
metric = ["euclidean", "dijstra"]
num_agents = [5, 10, 15,20, 25, 30, 35, 40, 45, 50]
goal_num = [5, 10, 15,20, 25, 30, 35, 40, 45, 50,100]
scenario = ["CA", "Cube", "Warehouse", "Unmaze", 'scenario2','scenario3','scenario6']

# prepare output CSV
out_csv = 'evalres/results_summary100.csv'
os.makedirs(os.path.dirname(out_csv), exist_ok=True)
with open(out_csv, 'w', newline='') as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow([
        'model_name', 'assignment', 'metric', 'num_agents',
        'goal_num', 'scenario', 'Max_Step', 'SR', 'Obs_CR', 'Car_CR', 'TimeoutR'
    ])
    # iterate over all combinations
    for mn in model_name:
        for assign in assignment:
            for sc in scenario:
                for na in num_agents:
                    for gn in goal_num:
                        if assign == 'AMARL':
                            fname = f"evalres/trainAt_{mn}_evalAt_{sc}_{assign}_dijstra_{na}_{gn}.pkl"
                            mets = ['dijstra']
                        else:
                            fname_list = []
                            mets = metric
                        # handle metrics
                        for met in mets:
                            if assign == 'AMARL':
                                pkl_path = fname
                            else:
                                pkl_path = f"evalres/trainAt_{mn}_evalAt_{sc}_{assign}_{met}_{na}_{gn}.pkl"
                            if not os.path.isfile(pkl_path):
                                continue
                            # load and compute means
                            with open(pkl_path, 'rb') as f:
                                Max_Step_arr, SR_arr, Obs_CR_arr, Car_CR_arr, TimeoutR_arr = pickle.load(f)
                            Max_Step = np.mean(Max_Step_arr)
                            SR       = np.mean(SR_arr)
                            Obs_CR   = np.mean(Obs_CR_arr)
                            Car_CR   = np.mean(Car_CR_arr)
                            TimeoutR = np.mean(TimeoutR_arr)
                            writer.writerow([
                                mn, assign, met, na, gn, sc,
                                Max_Step, SR, Obs_CR, Car_CR, TimeoutR
                            ])
print(f"Results written to {out_csv}")
# end of script
