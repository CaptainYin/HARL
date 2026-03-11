# import pyrvo2
import numpy as np
import os, sys, csv, itertools, multiprocessing as mp
import math, time
# from pyrvo2 import *
import matplotlib
matplotlib.use('Agg')  # Set non-interactive backend before importing pyplot
import matplotlib.pyplot as plt

from shapely.geometry import Polygon, Point, LineString
sys.path.append(os.path.dirname(__file__))
sys.path.append(os.path.join(os.getcwd(), "../../../../MinMax-MTSP-master/partition"))  
sys.path.append(os.path.join(os.getcwd(), "../../../../HARL-main/"))  
from env_core import EnvCore
from SimEnv import line_to_rectangle,get_edges, SimulationEnvironment_Scenario2, SimulationEnvironment_Scenario6, \
    SimulationEnvironment_CubeCollection, SimulationEnvironment_Warehouse, SimulationEnvironment_CA
import yaml
def load_yaml_as_dict(path):
    with open(path, 'r') as f:
        return yaml.safe_load(f) 

BASE_CFG=load_yaml_as_dict("/project/HARL-main/harl/configs/envs_cfgs/lasercar.yaml")
def evaluate(cfg, n_eval=100, seed0=42):
    """Run n_eval episodes with the given cfg, return average metrics."""
    ms=sr = oc = cc = to = 0.0
    env = EnvCore(cfg)
    # use different seeds for reproducibility
    for i in range(n_eval):
        env.seed(seed0 + i)
        _ = env.reset()
        # run until all done
        while True:
            obs, state, rew, dones, infos, _ = env.step()
            if np.array(dones).all():
                break
        m, _, s, o, c, t = env.render(mode='rgb_array',gif_name='unmaze_nhorca.gif')
        ms+=m; sr+= s; oc += o; cc += c; to += t
    return ms/n_eval, sr/n_eval, oc/n_eval, cc/n_eval, to/n_eval

def worker(params):
    """Worker for one grid combo."""
    nd, mn, th, tho = params
    cfg = BASE_CFG.copy()
    cfg['neighborDist']    = nd
    cfg['maxNeighbors']    = mn
    cfg['timeHorizon']     = th
    cfg['timeHorizonObst'] = tho
    
    ms_acc = sr_acc = oc_acc = cc_acc = to_acc = 0.0
    scenarios = {
        "Unmaze": (4, 4),
        "CA": (4, 4),
        "Cube": (8, 8),
        "Warehouse": (8, 8),
        "scenario1": (4, 4),
        "scenario3": (8, 8),
        "scenario6": (8, 8)
    }
    for scenario, (goal_num, num_agents) in scenarios.items():
        cfg['goal_num'] = goal_num
        cfg['num_agents'] = num_agents
        cfg['scenario'] = scenario
        print(f"[PID {os.getpid()}] Evaluating ND={nd}, MN={mn}, TH={th}, THO={tho}, scenario={scenario}")
        ms, sr, oc, cc, to = evaluate(cfg, n_eval=10)
        print(f"[PID {os.getpid()}] → ms={ms:.3f},sr={sr:.3f}, obs={oc:.3f}, car={cc:.3f}, to={to:.3f}")
        ms_acc += ms
        sr_acc += sr
        oc_acc += oc
        cc_acc += cc
        to_acc += to
    return (
        nd, mn, th, tho,
        ms_acc / len(scenarios),
        sr_acc / len(scenarios),
        oc_acc / len(scenarios),
        cc_acc / len(scenarios),
        to_acc / len(scenarios)
    )


    # ms,sr, oc, cc, to = evaluate(cfg, n_eval=10)
    # print(f"[PID {os.getpid()}] → ms={ms:.3f},sr={sr:.3f}, obs={oc:.3f}, car={cc:.3f}, to={to:.3f}")
    # return (nd, mn, th, tho, ms, sr, oc, cc, to)

def main():
    # search results under unmaze
    # neighborDist_range     = [0.8]
    # maxNeighbors_range     = [8]
    # timeHorizon_range      = [0.2]
    # timeHorizonObst_range  = [0.3]
    neighborDist_range     = [1.1] #7
    maxNeighbors_range     = [8] #4
    timeHorizon_range      = [0.3] #5
    timeHorizonObst_range  = [0.5] #6
    
    combos = list(itertools.product(
        neighborDist_range,
        maxNeighbors_range,
        timeHorizon_range,
        timeHorizonObst_range
    ))

    out_path = "nhorca_gridsearch_results1.csv"
    with mp.Pool(mp.cpu_count()) as pool, \
         open(out_path, "w", newline='') as csvfile:

        writer = csv.writer(csvfile)
        writer.writerow([
            "neighborDist", "maxNeighbors",
            "timeHorizon", "timeHorizonObst",
            "max_step","success_rate", "obs_collision_rate",
            "car_collision_rate", "timeout_rate"
        ])

        for result in pool.imap_unordered(worker, combos):
            writer.writerow(result)

    print(f"Grid search complete. Results saved to {out_path}")

if __name__ == "__main__":
    main()
