"""IPPO algorithm (Independent PPO).

This implementation mirrors MAPPO's actor-side PPO update but is intended for
independent learning of each agent's policy. Use with share_param=False to train
separate policies per agent. The critic setup is handled by the runner.
"""
from harl.algorithms.actors.mappo import MAPPO

class IPPO(MAPPO):
    pass

