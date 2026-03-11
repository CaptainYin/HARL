"""MAPPO algorithm."""
import numpy as np
import torch
import torch.nn as nn
from typing import Optional
from harl.utils.envs_tools import check, get_shape_from_obs_space, get_shape_from_act_space
from harl.utils.models_tools import get_grad_norm
from harl.algorithms.actors.on_policy_base import OnPolicyBase
from harl.algorithms.modules.curiosity import CuriosityModule


class MAPPO(OnPolicyBase):
    def __init__(self, args, obs_space, act_space, device=torch.device("cpu")):
        """Initialize MAPPO algorithm.
        Args:
            args: (dict) arguments.
            obs_space: (gym.spaces or list) observation space.
            act_space: (gym.spaces) action space.
            device: (torch.device) device to use for tensor operations.
        """
        super(MAPPO, self).__init__(args, obs_space, act_space, device)

        self.clip_param = args["clip_param"]
        self.ppo_epoch = args["ppo_epoch"]
        self.actor_num_mini_batch = args["actor_num_mini_batch"]
        self.entropy_coef = args["entropy_coef"]
        self.use_max_grad_norm = args["use_max_grad_norm"]
        self.max_grad_norm = args["max_grad_norm"]

        # Curiosity (forward dynamics) module optional settings
        self.use_curiosity = args.get("use_curiosity", False)
        if self.use_curiosity:
            obs_shape_tuple = get_shape_from_obs_space(self.obs_space)
            # flatten observation dimension
            if len(obs_shape_tuple) == 1:
                obs_dim = obs_shape_tuple[0]
            else:
                obs_dim = int(np.prod(obs_shape_tuple))

            # action handling (supports Discrete, Box, MultiDiscrete, MultiBinary, Tuple variant used in project)
            if act_space.__class__.__name__ == "Discrete":
                action_dim = act_space.n
                self._curiosity_discrete = True
                self._num_actions = act_space.n
            elif act_space.__class__.__name__ == "Tuple":
                # Custom Tuple space (observed earlier set to 3 via get_shape_from_act_space). Treat as continuous vector of that length.
                action_dim = get_shape_from_act_space(act_space)
                self._curiosity_discrete = False
                self._num_actions = None
            else:
                action_dim = get_shape_from_act_space(act_space)
                self._curiosity_discrete = False
                self._num_actions = None
            hidden_sizes = args.get("curiosity_hidden_sizes", [256, 256])
            self.curiosity_coef = args.get("curiosity_coef", 0.01)
            self.curiosity_loss_coef = args.get("curiosity_loss_coef", 1.0)
            self.curiosity_module = CuriosityModule(
                obs_dim=obs_dim,
                action_dim=action_dim,
                hidden_sizes=hidden_sizes,
                device=device,
            )
            self.curiosity_optimizer = torch.optim.Adam(
                self.curiosity_module.parameters(), lr=args.get("curiosity_lr", args.get("lr", 3e-4))
            )
        else:
            self.curiosity_module = None
            self.curiosity_optimizer = None

    def update(self, sample):
        """Update actor network.
        Args:
            sample: (Tuple) contains data batch with which to update networks.
        Returns:
            policy_loss: (torch.Tensor) actor(policy) loss value.
            dist_entropy: (torch.Tensor) action entropies.
            actor_grad_norm: (torch.Tensor) gradient norm from actor update.
            imp_weights: (torch.Tensor) importance sampling weights.
        """
        (
            obs_batch,
            rnn_states_batch,
            actions_batch,
            masks_batch,
            active_masks_batch,
            old_action_log_probs_batch,
            adv_targ,
            available_actions_batch,
        ) = sample

        old_action_log_probs_batch = check(old_action_log_probs_batch).to(**self.tpdv)
        adv_targ = check(adv_targ).to(**self.tpdv)

        active_masks_batch = check(active_masks_batch).to(**self.tpdv)

        # reshape to do in a single forward pass for all steps
        action_log_probs, dist_entropy, _ = self.evaluate_actions(
            obs_batch,
            rnn_states_batch,
            actions_batch,
            masks_batch,
            available_actions_batch,
            active_masks_batch,
        )
        # update actor
        imp_weights = getattr(torch, self.action_aggregation)(
            torch.exp(action_log_probs - old_action_log_probs_batch),
            dim=-1,
            keepdim=True,
        )

        surr1 = imp_weights * adv_targ
        surr2 = torch.clamp(imp_weights, 1.0 - self.clip_param, 1.0 + self.clip_param) * adv_targ

        if self.use_policy_active_masks:
            policy_action_loss = (
                -torch.sum(torch.min(surr1, surr2), dim=-1, keepdim=True) * active_masks_batch
            ).sum() / active_masks_batch.sum()
        else:
            policy_action_loss = -torch.sum(torch.min(surr1, surr2), dim=-1, keepdim=True).mean()

        policy_loss = policy_action_loss

        self.actor_optimizer.zero_grad()

        (policy_loss - dist_entropy * self.entropy_coef).backward()

        if self.use_max_grad_norm:
            actor_grad_norm = nn.utils.clip_grad_norm_(self.actor.parameters(), self.max_grad_norm)
        else:
            actor_grad_norm = get_grad_norm(self.actor.parameters())

        self.actor_optimizer.step()

        return policy_loss, dist_entropy, actor_grad_norm, imp_weights

    def train(self, actor_buffer, advantages, state_type):
        """Perform a training update for non-parameter-sharing MAPPO using minibatch GD.
        Args:
            actor_buffer: (OnPolicyActorBuffer) buffer containing training data related to actor.
            advantages: (np.ndarray) advantages.
            state_type: (str) type of state.
        Returns:
            train_info: (dict) contains information regarding training update (e.g. loss, grad norms, etc).
        """
        train_info = {
            "policy_loss": 0,
            "dist_entropy": 0,
            "actor_grad_norm": 0,
            "ratio": 0,
            "curiosity_loss": 0,
            "intrinsic_reward_mean": 0,
        }

        if np.all(actor_buffer.active_masks[:-1] == 0.0):
            return train_info

        if state_type == "EP":
            advantages_copy = advantages.copy()
            advantages_copy[actor_buffer.active_masks[:-1] == 0.0] = np.nan
            mean_advantages = np.nanmean(advantages_copy)
            std_advantages = np.nanstd(advantages_copy)
            advantages = (advantages - mean_advantages) / (std_advantages + 1e-5)

        # If curiosity enabled, compute intrinsic rewards from forward model.
        if self.use_curiosity:
            # Need obs_t, action_t, obs_{t+1}. Buffer stores obs[0:T] and obs[1:T+1]. Use flattened arrays.
            obs_t = actor_buffer.obs[:-1]  # (T, N, *obs)
            obs_tp1 = actor_buffer.obs[1:]
            actions = actor_buffer.actions  # (T, N, A) or (T,N) if discrete earlier
            T, N = actions.shape[0], actions.shape[1]
            obs_flat = obs_t.reshape(T * N, -1)
            next_obs_flat = obs_tp1.reshape(T * N, -1)
            act_flat = actions.reshape(T * N, -1)
            # Convert to tensors
            obs_tensor = torch.as_tensor(obs_flat, dtype=torch.float32, device=self.device)
            next_obs_tensor = torch.as_tensor(next_obs_flat, dtype=torch.float32, device=self.device)
            act_tensor = torch.as_tensor(act_flat, dtype=torch.float32, device=self.device)
            if self._curiosity_discrete:
                # In that case actions were stored as one-dim (T,N,1) maybe. Flatten accordingly
                act_tensor = act_tensor.view(-1)
            with torch.no_grad():
                # Forward pass for intrinsic reward (no gradient to avoid policy shaping the model during rollout update step).
                _, per_sample_mse_rw, _ = self.curiosity_module(
                    obs_tensor, act_tensor, next_obs_tensor, discrete=self._curiosity_discrete, num_actions=self._num_actions
                )
            intrinsic_rewards = per_sample_mse_rw.detach().cpu().numpy().reshape(T, N, 1)
            intrinsic_reward_mean = intrinsic_rewards.mean()
            train_info["intrinsic_reward_mean"] = intrinsic_reward_mean
            # Add scaled intrinsic rewards to (already normalized) advantages BEFORE batching
            advantages = advantages + self.curiosity_coef * intrinsic_rewards.squeeze(-1)

            # Optimize curiosity model separately using all samples (supervised MSE)
            pred_next, per_sample_mse_all, curiosity_loss = self.curiosity_module(
                obs_tensor, act_tensor, next_obs_tensor, discrete=self._curiosity_discrete, num_actions=self._num_actions
            )
            self.curiosity_optimizer.zero_grad()
            (curiosity_loss * self.curiosity_loss_coef).backward()
            self.curiosity_optimizer.step()
            train_info["curiosity_loss"] = curiosity_loss.item()

        for _ in range(self.ppo_epoch):
            if self.use_recurrent_policy:
                data_generator = actor_buffer.recurrent_generator_actor(
                    advantages, self.actor_num_mini_batch, self.data_chunk_length
                )
            elif self.use_naive_recurrent_policy:
                data_generator = actor_buffer.naive_recurrent_generator_actor(
                    advantages, self.actor_num_mini_batch
                )
            else:
                data_generator = actor_buffer.feed_forward_generator_actor(
                    advantages, self.actor_num_mini_batch
                )

            for sample in data_generator:
                policy_loss, dist_entropy, actor_grad_norm, imp_weights = self.update(sample)

                train_info["policy_loss"] += policy_loss.item()
                train_info["dist_entropy"] += dist_entropy.item()
                train_info["actor_grad_norm"] += actor_grad_norm
                train_info["ratio"] += imp_weights.mean()

        num_updates = self.ppo_epoch * self.actor_num_mini_batch

        for k in train_info.keys():
            train_info[k] /= num_updates

        return train_info

    def share_param_train(self, actor_buffer, advantages, num_agents, state_type):
        """Perform a training update for parameter-sharing MAPPO using minibatch GD.
        Args:
            actor_buffer: (list[OnPolicyActorBuffer]) buffer containing training data related to actor.
            advantages: (np.ndarray) advantages.
            num_agents: (int) number of agents.
            state_type: (str) type of state.
        Returns:
            train_info: (dict) contains information regarding training update (e.g. loss, grad norms, etc).
        """
        train_info = {"policy_loss": 0, "dist_entropy": 0, "actor_grad_norm": 0, "ratio": 0,
                      "curiosity_loss": 0, "intrinsic_reward_mean": 0}

        if state_type == "EP":
            advantages_ori_list = []
            advantages_copy_list = []
            for agent_id in range(num_agents):
                advantages_ori = advantages.copy()
                advantages_ori_list.append(advantages_ori)
                advantages_copy = advantages.copy()
                # print(advantages_copy.shape,actor_buffer[agent_id].active_masks[:-1].shape)
                advantages_copy[actor_buffer[agent_id].active_masks[:-1] == 0.0] = np.nan
                advantages_copy_list.append(advantages_copy)
            advantages_ori_tensor = np.array(advantages_ori_list)
            advantages_copy_tensor = np.array(advantages_copy_list)
            mean_advantages = np.nanmean(advantages_copy_tensor)
            std_advantages = np.nanstd(advantages_copy_tensor)
            normalized_advantages = (advantages_ori_tensor - mean_advantages) / (
                std_advantages + 1e-5
            )
            advantages_list = []
            for agent_id in range(num_agents):
                advantages_list.append(normalized_advantages[agent_id])
        elif state_type == "FP":
            active_masks_collector = [actor_buffer[i].active_masks for i in range(num_agents) ]
            active_masks_array = np.stack(active_masks_collector, axis=2)
            advantages_copy = advantages.copy()
            advantages_copy[active_masks_array[:-1] == 0.0] = np.nan
            mean_advantages = np.nanmean(advantages_copy)
            std_advantages = np.nanstd(advantages_copy)
            advantages = (advantages - mean_advantages) / (std_advantages + 1e-5)
            advantages_list = [np.squeeze(advantages[:, :, agent_id]) for agent_id in range(num_agents)]
        else:
            raise NotImplementedError

        # Curiosity for shared-parameter: treat all agents' transitions jointly
        if self.use_curiosity:
            # actor_buffer is list per agent
            obs_t_list = [buf.obs[:-1] for buf in actor_buffer]  # list of (T,N,obs)
            obs_tp1_list = [buf.obs[1:] for buf in actor_buffer]
            actions_list = [buf.actions for buf in actor_buffer]
            # concatenate along agent dimension
            obs_t = np.concatenate(obs_t_list, axis=1)  # (T, N*A, obs)
            obs_tp1 = np.concatenate(obs_tp1_list, axis=1)
            actions = np.concatenate(actions_list, axis=1)
            T, NA = actions.shape[0], actions.shape[1]
            obs_flat = obs_t.reshape(T * NA, -1)
            next_obs_flat = obs_tp1.reshape(T * NA, -1)
            act_flat = actions.reshape(T * NA, -1)
            obs_tensor = torch.as_tensor(obs_flat, dtype=torch.float32, device=self.device)
            next_obs_tensor = torch.as_tensor(next_obs_flat, dtype=torch.float32, device=self.device)
            act_tensor = torch.as_tensor(act_flat, dtype=torch.float32, device=self.device)
            if self._curiosity_discrete:
                act_tensor = act_tensor.view(-1)
            with torch.no_grad():
                _, per_sample_mse_rw, _ = self.curiosity_module( obs_tensor, act_tensor, next_obs_tensor, discrete=self._curiosity_discrete, num_actions=self._num_actions )
            intrinsic_rewards = per_sample_mse_rw.detach().cpu().numpy().reshape(T, NA)
            train_info["intrinsic_reward_mean"] = intrinsic_rewards.mean()
            # advantages shape (T,N,A); intrinsic_rewards already (T, N*A)
            intrinsic_reshaped = intrinsic_rewards  # (T, N*A)
            # split back per agent
            per_agent = np.split(intrinsic_reshaped, num_agents, axis=1)
            for agent_id in range(num_agents):
                # per_agent[agent_id] shape (T, N); advantages slice may be (T,N,1)
                # print(advantages.shape)
                # adv_slice = advantages[:, :, agent_id]
                adv_slice = advantages_list[agent_id]
                if adv_slice.ndim == 3 and adv_slice.shape[-1] == 1:
                    adv_slice = adv_slice.squeeze(-1)
                adv_slice = adv_slice + self.curiosity_coef * per_agent[agent_id]
                # # write back (expand if original had singleton)
                # if advantages[:, :, agent_id].ndim == 3 and advantages[:, :, agent_id].shape[-1] == 1:
                #     advantages[:, :, agent_id] = adv_slice[..., None]
                # else:
                #     advantages[:, :, agent_id] = adv_slice
                advantages_list[agent_id] = adv_slice
            # optimize curiosity model
            _, _, curiosity_loss = self.curiosity_module( obs_tensor, act_tensor, next_obs_tensor, discrete=self._curiosity_discrete, num_actions=self._num_actions )
            self.curiosity_optimizer.zero_grad()
            (curiosity_loss * self.curiosity_loss_coef).backward()
            self.curiosity_optimizer.step()
            train_info["curiosity_loss"] = curiosity_loss.item()
            
        for _ in range(self.ppo_epoch):
            data_generators = []
            for agent_id in range(num_agents):
                if self.use_recurrent_policy:
                    data_generator = actor_buffer[agent_id].recurrent_generator_actor(
                        advantages_list[agent_id],
                        self.actor_num_mini_batch,
                        self.data_chunk_length,
                    )
                elif self.use_naive_recurrent_policy:
                    data_generator = actor_buffer[agent_id].naive_recurrent_generator_actor(
                        advantages_list[agent_id], self.actor_num_mini_batch
                    )
                else:
                    data_generator = actor_buffer[agent_id].feed_forward_generator_actor(
                        advantages_list[agent_id], self.actor_num_mini_batch
                    )
                data_generators.append(data_generator)

            for _ in range(self.actor_num_mini_batch):
                batches = [[] for _ in range(8)]
                for generator in data_generators:
                    sample = next(generator)
                    for i in range(8):
                        batches[i].append(sample[i])
                for i in range(7):
                    batches[i] = np.concatenate(batches[i], axis=0)
                if batches[7][0] is None:
                    batches[7] = None
                else:
                    batches[7] = np.concatenate(batches[7], axis=0)
                policy_loss, dist_entropy, actor_grad_norm, imp_weights = self.update(
                    tuple(batches)
                )

                train_info["policy_loss"] += policy_loss.item()
                train_info["dist_entropy"] += dist_entropy.item()
                train_info["actor_grad_norm"] += actor_grad_norm
                train_info["ratio"] += imp_weights.mean()

        num_updates = self.ppo_epoch * self.actor_num_mini_batch

        for k in train_info.keys():
            train_info[k] /= num_updates

        return train_info
