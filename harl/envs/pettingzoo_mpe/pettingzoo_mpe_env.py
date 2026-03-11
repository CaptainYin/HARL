import copy
import importlib
import logging
import numpy as np
import supersuit as ss
from gymnasium import spaces
logging.basicConfig()
logging.getLogger().setLevel(logging.ERROR)


class PettingZooMPEEnv:
    def __init__(self, args):
        self.args = copy.deepcopy(args)
        self.scenario = args["scenario"]
        del self.args["scenario"]
        self.discrete = True
        if (
            "continuous_actions" in self.args
            and self.args["continuous_actions"] == True
        ):
            self.discrete = False
        if "max_cycles" in self.args:
            self.max_cycles = self.args["max_cycles"]
            self.args["max_cycles"] += 1
        else:
            self.max_cycles = 25
            self.args["max_cycles"] = 26
        self.cur_step = 0
        self.module = importlib.import_module("pettingzoo.mpe." + self.scenario)
        self.env = ss.pad_action_space_v0(
            ss.pad_observations_v0(self.module.parallel_env(**self.args))
        )
        self.env.reset()
        self.n_agents = self.env.num_agents
        self.agents = self.env.agents
        self.share_observation_space = self.repeat(self.env.state_space)
        self.observation_space = self.unwrap(self.env.observation_spaces)
        if self.scenario == "simple_reference_sparserew_v1":
            self.observation_space=[spaces.Box(low=-np.inf, high=np.inf, shape=(21,), dtype=np.float32) for _ in range(2)]
            self.share_observation_space=[spaces.Box(low=-np.inf, high=np.inf, shape=(42,), dtype=np.float32) for _ in range(2)]
        self.action_space = self.unwrap(self.env.action_spaces)
        self._seed = 0

    def step(self, actions):
        """
        return local_obs, global_state, rewards, dones, infos, available_actions
        """
        if self.discrete:
            obs, rew, term, trunc, info = self.env.step(self.wrap(actions.flatten()))
        else:
            obs, rew, term, trunc, info = self.env.step(self.wrap(actions))
        if self.scenario == "simple_reference_sparserew_v1":

            info['agent_0']['prev_her']=self.her0
            info['agent_1']['prev_her']=self.her1
            info['agent_0']['her']=obs['agent_0'][21:]
            info['agent_1']['her']=obs['agent_1'][21:]
            self.her0=obs['agent_0'][21:]
            self.her1=obs['agent_1'][21:]
            obs['agent_0']=obs['agent_0'][:21]
            obs['agent_1']=obs['agent_1'][:21]
            s_obs = self.repeat(np.concatenate([obs['agent_0'],obs['agent_1']]))
        else:
            s_obs = self.repeat(self.env.state())
        self.cur_step += 1
        if self.cur_step == self.max_cycles:
            trunc = {agent: True for agent in self.agents}
            for agent in self.agents:
                info[agent]["bad_transition"] = True
        dones = {agent: term[agent] or trunc[agent] for agent in self.agents}
        
        total_reward = sum([rew[agent] for agent in self.agents])
        rewards = [[total_reward]] * self.n_agents
        return (
            self.unwrap(obs),
            s_obs,
            rewards,
            self.unwrap(dones),
            self.unwrap(info),
            self.get_avail_actions(),
        )

    def reset(self):
        """Returns initial observations and states"""
        self._seed += 1
        # print("cur_step", self.cur_step)
        self.cur_step = 0

        if self.scenario == "simple_reference_sparserew_v1":
            obs = self.unwrap(self.env.reset(seed=self._seed))
            self.her0=obs[0][21:]
            self.her1=obs[1][21:]

            obs[0]=obs[0][:21]
            obs[1]=obs[1][:21]
     
            state=np.concatenate([obs[0],obs[1]])
            s_obs = self.repeat(state)
        else:
            obs = self.unwrap(self.env.reset(seed=self._seed))
            s_obs = self.repeat(self.env.state())

        return obs, s_obs, self.get_avail_actions()

    def get_avail_actions(self):
        if self.discrete:
            avail_actions = []
            for agent_id in range(self.n_agents):
                avail_agent = self.get_avail_agent_actions(agent_id)
                avail_actions.append(avail_agent)
            return avail_actions
        else:
            return None

    def get_avail_agent_actions(self, agent_id):
        """Returns the available actions for agent_id"""
        return [1] * self.action_space[agent_id].n

    def render(self):
        self.env.render()

    def close(self):
        self.env.close()

    def seed(self, seed):
        self._seed = seed

    def wrap(self, l):
        d = {}
        for i, agent in enumerate(self.agents):
            d[agent] = l[i]
        return d

    def unwrap(self, d):
        l = []
        for agent in self.agents:
            l.append(d[agent])
        return l

    def repeat(self, a):
        return [a for _ in range(self.n_agents)]
