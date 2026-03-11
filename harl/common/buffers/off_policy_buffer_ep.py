"""Off-policy buffer."""
import numpy as np
import torch
from harl.common.buffers.off_policy_buffer_base import OffPolicyBufferBase


class OffPolicyBufferEP(OffPolicyBufferBase):
    """Off-policy buffer that uses Environment-Provided (EP) state."""

    def __init__(self, args, share_obs_space, num_agents, obs_spaces, act_spaces):
        """Initialize off-policy buffer.
        Args:
            args: (dict) arguments
            share_obs_space: (gym.Space or list) share observation space
            num_agents: (int) number of agents
            obs_spaces: (gym.Space or list) observation spaces
            act_spaces: (gym.Space) action spaces
        """
        super(OffPolicyBufferEP, self).__init__(
            args, share_obs_space, num_agents, obs_spaces, act_spaces
        )

        # Buffer for share observations
        self.share_obs = np.zeros(
            (self.buffer_size, *self.share_obs_shape), dtype=np.float32
        )

        # Buffer for next share observations
        self.next_share_obs = np.zeros(
            (self.buffer_size, *self.share_obs_shape), dtype=np.float32
        )

        # Buffer for rewards received by agents at each timestep
        self.rewards = np.zeros((self.buffer_size, 1), dtype=np.float32)

        # Buffer for done and termination flags
        self.dones = np.full((self.buffer_size, 1), False)
        self.terms = np.full((self.buffer_size, 1), False)

    def sample(self):
        """Sample data for training.
        Returns:
            sp_share_obs: (batch_size, *dim)
            sp_obs: (n_agents, batch_size, *dim)
            sp_actions: (n_agents, batch_size, *dim)
            sp_available_actions: (n_agents, batch_size, *dim)
            sp_reward: (batch_size, 1)
            sp_done: (batch_size, 1)
            sp_valid_transitions: (n_agents, batch_size, 1)
            sp_term: (batch_size, 1)
            sp_next_share_obs: (batch_size, *dim)
            sp_next_obs: (n_agents, batch_size, *dim)
            sp_next_available_actions: (n_agents, batch_size, *dim)
            sp_gamma: (batch_size, 1)
        """
        self.update_end_flag()  # update the current end flag
        indice = torch.randperm(self.cur_size).numpy()[
            : self.batch_size
        ]  # sample indice, shape: (batch_size, )
        # get data at the beginning indice
        sp_share_obs = self.share_obs[indice]
        sp_obs = np.array(
            [self.obs[agent_id][indice] for agent_id in range(self.num_agents)]
        )
        sp_actions = np.array(
            [self.actions[agent_id][indice] for agent_id in range(self.num_agents)]
        )
        sp_valid_transitions = np.array(
            [
                self.valid_transitions[agent_id][indice]
                for agent_id in range(self.num_agents)
            ]
        )
        if self.act_spaces[0].__class__.__name__ == "Discrete":
            sp_available_actions = np.array(
                [
                    self.available_actions[agent_id][indice]
                    for agent_id in range(self.num_agents)
                ]
            )

        # compute the indices along n steps
        indices = [indice]
        
        for _ in range(self.n_step - 1):
            indices.append(self.next(indices[-1]))

        # get data at the last indice
        sp_done = self.dones[indices[-1]]
        sp_term = self.terms[indices[-1]]
        sp_next_share_obs = self.next_share_obs[indices[-1]]
        sp_next_obs = np.array(
            [
                self.next_obs[agent_id][indices[-1]]
                for agent_id in range(self.num_agents)
            ]
        )
        if self.act_spaces[0].__class__.__name__ == "Discrete":
            sp_next_available_actions = np.array(
                [
                    self.next_available_actions[agent_id][indices[-1]]
                    for agent_id in range(self.num_agents)
                ]
            )

        # compute accumulated rewards and the corresponding gamma
        gamma_buffer = np.ones(self.n_step + 1)
        for i in range(1, self.n_step + 1):
            gamma_buffer[i] = gamma_buffer[i - 1] * self.gamma
        sp_reward = np.zeros((self.batch_size, 1))
        gammas = np.full(self.batch_size, self.n_step)
        
        for n in range(self.n_step - 1, -1, -1):
            now = indices[n]
            gammas[self.end_flag[now] > 0] = n + 1
            sp_reward[self.end_flag[now] > 0] = 0.0
            sp_reward = self.rewards[now] + self.gamma * sp_reward
        sp_gamma = gamma_buffer[gammas].reshape(self.batch_size, 1)

        if self.act_spaces[0].__class__.__name__ == "Discrete":
            return (
                sp_share_obs,
                sp_obs,
                sp_actions,
                sp_available_actions,
                sp_reward,
                sp_done,
                sp_valid_transitions,
                sp_term,
                sp_next_share_obs,
                sp_next_obs,
                sp_next_available_actions,
                sp_gamma,
            )
        else:
            return (
                sp_share_obs,
                sp_obs,
                sp_actions,
                None,
                sp_reward,
                sp_done,
                sp_valid_transitions,
                sp_term,
                sp_next_share_obs,
                sp_next_obs,
                None,
                sp_gamma,
            )

    def next(self, indices):
        """Get next indices"""
        return (
            indices + (1 - self.end_flag[indices]) * self.n_rollout_threads
        ) % self.buffer_size

    def update_end_flag(self):
        """Update current end flag for computing n-step return.
        End flag is True at the steps which are the end of an episode or the latest but unfinished steps.
        """
        self.unfinished_index = (
            self.idx - np.arange(self.n_rollout_threads) - 1 + self.cur_size
        ) % self.cur_size
        self.end_flag = self.dones.copy().squeeze()  # (buffer_size, )
        self.end_flag[self.unfinished_index] = True



class OffPolicyBufferEPHer(OffPolicyBufferBase):
    """Off-policy Her buffer that uses Environment-Provided (EP) state with HER support."""

    def __init__(self, args, share_obs_space, num_agents, obs_spaces, act_spaces, her_args=None):
        """Initialize off-policy buffer with HER support.
        Args:
            args: (dict) arguments
            share_obs_space: (gym.Space or list) share observation space
            num_agents: (int) number of agents
            obs_spaces: (gym.Space or list) observation spaces
            act_spaces: (gym.Space) action spaces
            her_args: (dict) arguments for HER, e.g., {'n_sampled_goal': 4, 'goal_selection_strategy': 'future'}
        """
        super(OffPolicyBufferEPHer, self).__init__(
            args, share_obs_space, num_agents, obs_spaces, act_spaces
        )

        # Initialize existing buffers
        self.share_obs = np.zeros(
            (self.buffer_size, *self.share_obs_shape), dtype=np.float32
        )
        self.next_share_obs = np.zeros(
            (self.buffer_size, *self.share_obs_shape), dtype=np.float32
        )
        self.rewards = np.zeros((self.buffer_size, 1), dtype=np.float32)
        self.dones = np.full((self.buffer_size, 1), False)
        self.terms = np.full((self.buffer_size, 1), False)

        # Initialize HER-related parameters
        if her_args is None:
            her_args = {}
        self.herstate=np.zeros((self.buffer_size,num_agents,5), dtype=np.float32)
        self.agent_pos=np.zeros((self.buffer_size,num_agents,2), dtype=np.float32)
        self.agent_pos_prev=np.zeros((self.buffer_size,num_agents,2), dtype=np.float32)
        self.goala_pos=np.zeros((self.buffer_size,num_agents,2), dtype=np.float32)
        self.goalb_ind=np.zeros((self.buffer_size,num_agents,1), dtype=np.float32)
        self.landmark_pos_rel=np.zeros((self.buffer_size,6), dtype=np.float32)
        self.goalb_pos=np.zeros((self.buffer_size,num_agents,2), dtype=np.float32)

        self.n_sampled_goal = her_args.get("n_sampled_goal", 4)
        self.goal_selection_strategy = her_args.get("goal_selection_strategy", "future")  # 'future', 'final', 'episode'
        self.her_ratio = her_args.get("her_ratio", 0.8)  # Fraction of samples to apply HER

        # Initialize episode tracking arrays
        self.episode_start = np.zeros(self.buffer_size, dtype=np.int32)
        self.episode_length = np.zeros(self.buffer_size, dtype=np.int32)

        # Assume that goals are part of the observation. Define goal dimensions.
        # For example, if the last 2 dimensions of the observation are the goal.
        self.goal_dim = args.get("goal_dim", 2)  # Modify as per your environment
        self.achieved_goal_dim = args.get("achieved_goal_dim", 2)  # Dimensions for achieved goal

    def insert(self, data):
        """Insert data into buffer with episode tracking.
        Args:
            data: a tuple of (share_obs, obs, actions, available_actions, reward, done, valid_transitions, term, next_share_obs, next_obs, next_available_actions)
        """
        (share_obs, obs, actions, available_actions, reward, done, valid_transitions, term, next_share_obs, next_obs, next_available_actions,herdata,prev_herdata) = data
        agent_pos=herdata[:,:,0:2]
        agent_pos_prev=prev_herdata[:,:,0:2]
        goala_pos=herdata[:,:,2:4]
        goalb_ind=herdata[:,:,4]
        landmark_pos_rel=obs.transpose(1,0,2)[:,0,2:8]
        goalb_pos = landmark_pos_rel[np.arange(self.n_rollout_threads)[:, None], np.arange(self.num_agents), goalb_ind.astype(int) * 2:goalb_ind.astype(int) * 2 + 2] + agent_pos[:, 0, None]

        length = share_obs.shape[0]
        if self.idx + length <= self.buffer_size:  # no overflow
            s, e = self.idx, self.idx + length
            self.share_obs[s:e], self.rewards[s:e], self.dones[s:e], self.terms[s:e], self.next_share_obs[s:e] = share_obs.copy(), reward.copy(), done.copy(), term.copy(), next_share_obs.copy()
            self.herstate[s:e]=herdata.copy()
            self.agent_pos[s:e]=agent_pos.copy()
            self.agent_pos_prev[s:e]=agent_pos_prev.copy()
            self.goala_pos[s:e]=goala_pos.copy()
            self.goalb_ind[s:e]=goalb_ind.copy()
            self.landmark_pos_rel[s:e]=landmark_pos_rel.copy()
            self.goalb_pos[s:e]=goalb_pos.copy()
            for agent_id in range(self.num_agents):
                self.obs[agent_id][s:e], self.actions[agent_id][s:e], self.valid_transitions[agent_id][s:e] = obs[agent_id].copy(), actions[agent_id].copy(), valid_transitions[agent_id].copy()
                if self.act_spaces[agent_id].__class__.__name__ == "Discrete":
                    self.available_actions[agent_id][s:e], self.next_available_actions[agent_id][s:e] = available_actions[agent_id].copy(), next_available_actions[agent_id].copy()
                self.next_obs[agent_id][s:e] = next_obs[agent_id].copy()
        else:  # overflow
            len1 = self.buffer_size - self.idx  # length of first segment
            len2 = length - len1  # length of second segment
            # insert first segment
            s, e = self.idx, self.buffer_size
            self.share_obs[s:e], self.rewards[s:e], self.dones[s:e], self.terms[s:e], self.next_share_obs[s:e] = share_obs[0:len1].copy(), reward[0:len1].copy(), done[0:len1].copy(), term[0:len1].copy(), next_share_obs[0:len1].copy()
            self.herstate[s:e]=herdata[0:len1].copy()
            self.agent_pos[s:e]=agent_pos[0:len1].copy()
            self.agent_pos_prev[s:e]=agent_pos_prev[0:len1].copy()
            self.goala_pos[s:e]=goala_pos[0:len1].copy()
            self.goalb_ind[s:e]=goalb_ind[0:len1].copy()
            self.landmark_pos_rel[s:e]=landmark_pos_rel[0:len1].copy()
            self.goalb_pos[s:e]=goalb_pos[0:len1].copy()

            for agent_id in range(self.num_agents):
                self.obs[agent_id][s:e], self.actions[agent_id][s:e], self.valid_transitions[agent_id][s:e] = obs[agent_id][0:len1].copy(), actions[agent_id][0:len1].copy(), valid_transitions[agent_id][0:len1].copy()
                if self.act_spaces[agent_id].__class__.__name__ == "Discrete":
                    self.available_actions[agent_id][s:e], self.next_available_actions[agent_id][s:e] = available_actions[agent_id][0:len1].copy(), next_available_actions[agent_id][0:len1].copy()
                self.next_obs[agent_id][s:e] = next_obs[agent_id][0:len1].copy()

            # insert second segment
            s,e = 0, len2
            self.share_obs[s:e], self.rewards[s:e], self.dones[s:e], self.terms[s:e], self.next_share_obs[s:e]= share_obs[len1:length].copy(), reward[len1:length].copy(), done[len1:length].copy(), term[len1:length].copy(),next_share_obs[len1:length].copy()
            self.herstate[s:e]=herdata[len1:length].copy()
            self.agent_pos[s:e]=agent_pos[len1:length].copy()
            self.agent_pos_prev[s:e]=agent_pos_prev[len1:length].copy()
            self.goala_pos[s:e]=goala_pos[len1:length].copy()
            self.goalb_ind[s:e]=goalb_ind[len1:length].copy()
            self.landmark_pos_rel[s:e]=landmark_pos_rel[len1:length].copy()
            self.goalb_pos[s:e]=goalb_pos[len1:length].copy()
            for agent_id in range(self.num_agents):
                self.obs[agent_id][s:e], self.actions[agent_id][s:e], self.valid_transitions[agent_id][s:e] = obs[agent_id][len1:length].copy(), actions[agent_id][len1:length].copy(), valid_transitions[agent_id][len1:length].copy()
                if self.act_spaces[agent_id].__class__.__name__ == "Discrete":
                    self.available_actions[agent_id][s:e], self.next_available_actions[agent_id][s:e] = available_actions[agent_id][len1:length].copy(), next_available_actions[agent_id][len1:length].copy()
                self.next_obs[agent_id][s:e] = next_obs[agent_id][len1:length].copy()

        # Update episode tracking
        for i in range(length):
            current_idx = (self.idx + i) % self.buffer_size
            previous_idx = (current_idx - length) % self.buffer_size
            if self.cur_size==0 or self.dones[previous_idx]:#previous transition is episode ends or buffer is empty
                self.episode_start[current_idx] = current_idx
                self.episode_length[current_idx] = 1
            else:
                self.episode_start[current_idx] = self.episode_start[previous_idx]
                self.episode_length[current_idx] = self.episode_length[previous_idx] + 1
                s,e=self.episode_start[current_idx],self.episode_start[current_idx]+(self.episode_length[current_idx]-1)*self.n_rollout_threads
                assert e==current_idx,f"e:{e},current_idx:{current_idx}"
                print(f"current_idx:{current_idx},s:{s},e:{e}")
                # update episode length for all transitions in the episode
                for j in range(self.episode_length[current_idx]):
                    self.episode_length[(s+j*self.n_rollout_threads)%self.n_rollout_threads]=self.episode_length[current_idx]

        # Update index and current size
        self.idx = (self.idx + length) % self.buffer_size  # update index
        self.cur_size = min(self.cur_size + length, self.buffer_size)  

    def sample(self):
        """Sample data for training with HER.
        Returns:
            sp_share_obs: (batch_size, *dim)
            sp_obs: (n_agents, batch_size, *dim)
            sp_actions: (n_agents, batch_size, *dim)
            sp_available_actions: (n_agents, batch_size, *dim)
            sp_reward: (batch_size, 1)
            sp_done: (batch_size, 1)
            sp_valid_transitions: (n_agents, batch_size, 1)
            sp_term: (batch_size, 1)
            sp_next_share_obs: (batch_size, *dim)
            sp_next_obs: (n_agents, batch_size, *dim)
            sp_next_available_actions: (n_agents, batch_size, *dim)
            sp_gamma: (batch_size, 1)
        """
        self.update_end_flag()  # update the current end flag
        indice = torch.randint(0, self.cur_size, (self.batch_size,)).numpy()  # sample indices

        # Determine which samples will use HER
        her_indices = np.random.choice(self.batch_size, size=int(self.her_ratio * self.batch_size), replace=False)

        # Initialize arrays for HER
        idx = indice[her_indices]
        ep_start = self.episode_start[idx]
        ep_len = self.episode_length[idx]
        assert (ep_len > 0).all(), f"ep_len: exist ep_len <=0"

        if self.goal_selection_strategy == "final":
            sampled_idx = (ep_start + (ep_len - 1)*self.n_rollout_threads) % self.buffer_size
        elif self.goal_selection_strategy == "future":
            future_len=ep_len-(idx-ep_start)%self.buffer_size//self.n_rollout_threads
            sampled_idx = (np.random.randint(0, future_len)*self.n_rollout_threads+idx)% self.buffer_size
        elif self.goal_selection_strategy == "episode":
            sampled_idx = (ep_start + np.random.randint(0, ep_len)*self.n_rollout_threads) % self.buffer_size
        else:
            raise ValueError(f"Invalid goal_selection_strategy: {self.goal_selection_strategy}")


        sp_obs = np.array([self.obs[agent_id][indice] for agent_id in range(self.num_agents)])
        sp_next_obs = np.array([self.next_obs[agent_id][indice] for agent_id in range(self.num_agents)])
        # sp_share_obs = self.share_obs[indice].copy()
        # sp_next_share_obs = self.next_share_obs[indice].copy()
        sp_agent_pos = self.agent_pos[indice].copy()
        sp_agent_pos_prev = self.agent_pos_prev[indice].copy()
        sp_goalb = self.goalb_pos[indice].transpose(1,0,2).copy()
        sp_goala = self.goala_pos[indice].transpose(1,0,2).copy()
        sp_goalb_ind = self.goalb_ind[indice].copy()
        print(idx.shape,sampled_idx.shape)
        for i in range(len(idx)):
            for j in range(self.num_agents):
                sp_goalb[j,idx[i]]=self.goalb_pos[sampled_idx[i],j]
                sp_obs[j,idx[i],2+sp_goalb_ind[idx[i],j]*2:4+sp_goalb_ind[idx[i],j]*2]=self.goala_pos[sampled_idx[i],j]-sp_agent_pos_prev[idx[i],j]
                sp_obs[j,idx[i],2+sp_goalb_ind[idx[i],1-j]*2:4+sp_goalb_ind[idx[i],1-j]*2]=self.goala_pos[sampled_idx[i],1-j]-sp_agent_pos_prev[idx[i],j]
                sp_next_obs[j,idx[i],2+sp_goalb_ind[idx[i],j]*2:4+sp_goalb_ind[idx[i],j]*2]=self.goala_pos[sampled_idx[i],j]-sp_agent_pos[idx[i],j]
                sp_next_obs[j,idx[i],2+sp_goalb_ind[idx[i],1-j]*2:4+sp_goalb_ind[idx[i],1-j]*2]=self.goala_pos[sampled_idx[i],1-j]-sp_agent_pos[idx[i],j]
        sp_share_obs=np.repeat(sp_obs,self.num_agents,axis=-1)
        sp_next_share_obs=np.repeat(sp_next_obs,self.num_agents,axis=-1)
        print("sp_share_obs",sp_share_obs.shape,sp_next_share_obs.shape)
        # sp_obs[:,idx,2+sp_goalb_ind[:,:]*2:4+sp_goalb_ind[:,:]*2]= self.goala_pos[sampled_idx]-sp_agent_pos_prev[idx]
        # sp_next_obs[:,idx,2+sp_goalb_ind[:,:]*2:4+sp_goalb_ind[:,:]*2]= self.goala_pos[sampled_idx]-sp_agent_pos[idx]
        # Resample goals for HER samples
        # Replace goals in observations and next observations
        # resampled_goals = self.herstate[sampled_idx].transpose(1,0,2)
        # print(sp_obs.shape,sp_share_obs.shape)#(2, 1000, 21) (1000, 42)
        # print(resampled_goals.shape)#( 2,800, 5)agent.state.p_pos,agent.goal_a.state.p_pos,np.array([agent.goalb_ind])
        # her_agent_pos=resampled_goals[:,:,0:2]
        #[agent.state.p_vel] + entity_pos + [agent.goal_b.color] + comm, 0:2,2:8,8:11,11:21
        #entity.state.p_pos - agent.state.p_pos
        # for agent_id in range(self.num_agents):
        #     sp_obs[agent_id][her_indices, -self.goal_dim:]
        #     sp_next_obs[agent_id][her_indices, -self.goal_dim:] = resampled_goals[:, -self.goal_dim:]
        # sp_share_obs[her_indices, -self.goal_dim:] = resampled_goals[:, -self.goal_dim:]
        # sp_next_share_obs[her_indices, -self.goal_dim:] = resampled_goals[:, -self.goal_dim:]

        sp_reward = self.compute_reward(sp_goala,sp_goalb)
        sp_done = self.dones[indice].copy()
        sp_term = self.terms[indice].copy()

        # Gather other necessary data
        sp_actions = np.array([self.actions[agent_id][indice] for agent_id in range(self.num_agents)])
        sp_valid_transitions = np.array([self.valid_transitions[agent_id][indice] for agent_id in range(self.num_agents)])
        if self.act_spaces[0].__class__.__name__ == "Discrete":
            sp_available_actions = np.array([self.available_actions[agent_id][indice] for agent_id in range(self.num_agents)])
            sp_next_available_actions = np.array([self.next_available_actions[agent_id][indice] for agent_id in range(self.num_agents)])
        else:
            sp_available_actions,sp_next_available_actions = None,None

        # Compute the indices along n steps
        indices_steps = [indice]
        for _ in range(self.n_step - 1):
            indices_steps.append(self.next(indices_steps[-1]))

        # Get data at the last index
        last_idx = indices_steps[-1]
        sp_done_last = self.dones[last_idx]
        sp_term_last = self.terms[last_idx]
        sp_next_share_obs_last = self.next_share_obs[last_idx]
        sp_next_obs_last = np.array([self.next_obs[agent_id][last_idx] for agent_id in range(self.num_agents)])
        if self.act_spaces[0].__class__.__name__ == "Discrete":
            sp_next_available_actions_last = np.array([self.next_available_actions[agent_id][last_idx] for agent_id in range(self.num_agents)])
        else:
            sp_next_available_actions_last = None

        # Compute accumulated rewards and the corresponding gamma
        gamma_buffer = np.ones(self.n_step + 1)
        for i in range(1, self.n_step + 1):
            gamma_buffer[i] = gamma_buffer[i - 1] * self.gamma
        sp_acc_reward = np.zeros((self.batch_size, 1))
        gammas = np.full(self.batch_size, self.n_step)
        for n in range(self.n_step - 1, -1, -1):
            now = indices_steps[n]
            gammas[self.end_flag[now] > 0] = n + 1
            sp_acc_reward[self.end_flag[now] > 0] = 0.0
            sp_acc_reward = self.rewards[now] + self.gamma * sp_acc_reward
        sp_gamma = gamma_buffer[gammas].reshape(self.batch_size, 1)

        if self.act_spaces[0].__class__.__name__ == "Discrete":
            return (sp_share_obs, sp_obs, sp_actions, sp_available_actions, sp_reward, sp_done, sp_valid_transitions, sp_term, sp_next_share_obs, sp_next_obs, sp_next_available_actions, sp_gamma)
        else:
            return (sp_share_obs, sp_obs, sp_actions, None, sp_reward, sp_done, sp_valid_transitions, sp_term, sp_next_share_obs, sp_next_obs, None, sp_gamma)

    def compute_reward(self, goala_pos, goalb_pos):
        agent_reward= np.linalg.norm(goala_pos - goalb_pos, axis=-1)
        agent_reward = 1 if agent_reward > 0.1 else 0
        return -agent_reward

    def next(self, indices):
        """Get next indices"""
        return (
            indices + (1 - self.end_flag[indices]) * self.n_rollout_threads
        ) % self.buffer_size

    def update_end_flag(self):
        """Update current end flag for computing n-step return.
        End flag is True at the steps which are the end of an episode or the latest but unfinished steps.
        """
        self.unfinished_index = (
            self.idx - np.arange(self.n_rollout_threads) - 1 + self.cur_size
        ) % self.cur_size
        self.end_flag = self.dones.copy().squeeze()  # (buffer_size, )
        self.end_flag[self.unfinished_index] = True

    def get_mean_rewards(self):
        """Get mean rewards of the buffer"""
        return np.mean(self.rewards[: self.cur_size])
