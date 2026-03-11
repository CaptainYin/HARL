import gym
from gym import spaces
import numpy as np
# from env_core import EnvCore
# from envs.env_wrappers import DummyVecEnv,SubprocVecEnv,ShareDummyVecEnv,ShareSubprocVecEnv

class ContinuousActionEnv(object):
    """
    对于连续动作环境的封装
    Wrapper for continuous action environment.
    """

    def __init__(self,all_args):
        self.env = EnvCore(all_args)
        self.num_agent = self.env.agent_num
        self.n = self.env.agent_num

        self.signal_obs_dim = self.env.obs_dim
        self.signal_action_dim = self.env.action_dim
        self.share_observation_dim=self.env.share_observation_dim
        self.state_dim_peragent=self.env.state_dim_peragent
        if all_args.use_feature_prune:
            self.share_observation_dim=self.env.share_observation_dim-self.state_dim_peragent
        # if true, action is a number 0...N, otherwise action is a one-hot N-dimensional vector
        self.discrete_action_input = False
        self.movable = True
        # configure spaces
        self.action_space = []
        self.observation_space = []
        self.share_observation_space = []

        share_obs_dim = 0
        total_action_space = []
        for agent in range(self.num_agent):
            # physical action space
            u_action_space = spaces.Box(low=np.array([0, -self.env.default_omega],dtype=np.float32), high=np.array([self.env.default_speed, self.env.default_omega],dtype=np.float32), dtype=np.float32)
            if self.movable:
                total_action_space.append(u_action_space)

            # total action space
            self.action_space.append(total_action_space[0])

            # observation space
            # share_obs_dim += self.signal_obs_dim-3*(self.num_agent-1)# feature prune
            share_obs_dim += self.signal_obs_dim
            self.observation_space.append(spaces.Box(low=-np.inf, high=+np.inf, shape=(self.signal_obs_dim,), dtype=np.float32))  # [-inf,inf]

        self.share_observation_space = [
            # spaces.Box(low=-np.inf, high=+np.inf, shape=(self.signal_obs_dim+self.share_observation_dim,), dtype=np.float32)
            spaces.Box(low=-np.inf, high=+np.inf, shape=(self.share_observation_dim,), dtype=np.float32)
            for _ in range(self.num_agent)
        ]

    def step(self, actions):
        """
        输入actions维度假设：
        # actions shape = (5, 2, 5)
        # 5个线程的环境，里面有2个智能体，每个智能体的动作是一个one_hot的5维编码

        Input actions dimension assumption:
        # actions shape = (5, 2, 5)
        # 5 threads of environment, there are 2 agents inside, and each agent's action is a 5-dimensional one_hot encoding
        """
        results = self.env.step(actions)
        obs, share_obs,rews, dones, infos = results
        return np.stack(obs), np.stack(share_obs),np.stack(rews), np.stack(dones), infos
    # def step(self, actions):
    #     results = self.env.step(actions)
    #     obs,rews, dones, infos = results
    #     return np.stack(obs), np.stack(rews), np.stack(dones), infos
    def reset(self):
        obs,share_obs = self.env.reset()
        return np.stack(obs),np.stack(share_obs)
    # def reset(self):
    #     obs = self.env.reset()
    #     return np.stack(obs)
    def close(self):
        pass

    def render(self, trajectory, mode="rgb_array",gif_name=None,plot_sensor_line=True):
        self.env.render(trajectory=trajectory, mode=mode,gif_name=gif_name,plot_sensor_line=plot_sensor_line)

    def seed(self, seed):
        self.env.seed(seed)
        pass
# def make_train_env(all_args):
#     all_args.Car_resettype=all_args.Car_resettype_train
#     all_args.add_sensor_noise=all_args.trainwith_sensor_noise
#     def get_env_fn(rank):
#         def init_env():
#             if not isinstance(all_args.scenario_name,list):
#                 all_args.scenario=all_args.subtask_scenario_list[rank%len(all_args.subtask_scenario_list)] if all_args.scenario_name=='mixed' else all_args.scenario_name
#             else:
#                 all_args.scenario=all_args.scenario_name[rank]
#             env = ContinuousActionEnv(all_args)
#             env.seed(all_args.seed + rank)
#             return env
#         return init_env
#     if all_args.n_rollout_threads == 1:
#         return ShareDummyVecEnv([get_env_fn(0)])

#     else:
#         return ShareSubprocVecEnv([get_env_fn(i) for i in range(all_args.n_rollout_threads)],in_series=all_args.n_rollout_threads_in_series)

# def make_eval_env(all_args):
#     all_args.Car_resettype=all_args.Car_resettype_eval
#     all_args.add_sensor_noise=False
#     def get_env_fn(rank):
#         def init_env():
#             if all_args.eval_scenario_name=='mixed':
#                 all_args.scenario=all_args.subtask_scenario_list[rank%len(all_args.subtask_scenario_list)]
#             elif all_args.eval_scenario_name=='goal_mixed':
#                 all_args.scenario=all_args.goal_scenario_list[rank%len(all_args.goal_scenario_list)] 
#             else:
#                 all_args.scenario=all_args.eval_scenario_name
#             env = ContinuousActionEnv(all_args)
#             env.seed(all_args.seed + rank +1000)
#             return env
#         return init_env
#     if all_args.n_eval_rollout_threads == 1:
#         return ShareDummyVecEnv([get_env_fn(0)])
#     else:
#         return ShareSubprocVecEnv([get_env_fn(i) for i in range(all_args.n_eval_rollout_threads)],in_series=all_args.n_eval_rollout_threads_in_series)

# def make_TSCL_train_env_list(all_args):
#     trainenvlist=[]
#     for scenario in all_args.subtask_scenario_list:
#         all_args.scenario_name=scenario
#         trainenvlist.append(make_train_env(all_args))
#     return trainenvlist
# def make_TSCL_eval_env_list(all_args):
#     evalenvlist=[]
#     for scenario in all_args.subtask_scenario_list:
#         all_args.eval_scenario_name=scenario
#         evalenvlist.append(make_eval_env(all_args))
#     return evalenvlist
# def make_TSCL_goal_env_list(all_args):
#     all_args.eval_scenario_name='goal_mixed'
#     return make_eval_env(all_args)