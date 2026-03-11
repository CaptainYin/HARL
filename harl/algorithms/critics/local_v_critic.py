"""Local V Critic for IPPO (per-agent value function on local observations)."""
import torch
import torch.nn as nn
from harl.utils.models_tools import get_grad_norm, huber_loss, mse_loss, update_linear_schedule
from harl.utils.envs_tools import check
from harl.models.value_function_models.v_net import VNet


class LocalVCritic:
    def __init__(self, args, obs_space, device=torch.device("cpu")):
        self.args = args
        self.device = device
        self.tpdv = dict(dtype=torch.float32, device=device)

        self.clip_param = args["clip_param"]
        self.critic_epoch = args["critic_epoch"]
        self.critic_num_mini_batch = args["critic_num_mini_batch"]
        self.data_chunk_length = args["data_chunk_length"]
        self.value_loss_coef = args["value_loss_coef"]
        self.max_grad_norm = args["max_grad_norm"]
        self.huber_delta = args["huber_delta"]

        self.use_recurrent_policy = args["use_recurrent_policy"]
        self.use_naive_recurrent_policy = args["use_naive_recurrent_policy"]
        self.use_max_grad_norm = args["use_max_grad_norm"]
        self.use_clipped_value_loss = args["use_clipped_value_loss"]
        self.use_huber_loss = args["use_huber_loss"]
        self.use_policy_active_masks = args["use_policy_active_masks"]

        self.critic_lr = args["critic_lr"]
        self.opti_eps = args["opti_eps"]
        self.weight_decay = args["weight_decay"]

        self.obs_space = obs_space
        self.critic = VNet(args, self.obs_space, self.device)

        self.critic_optimizer = torch.optim.Adam(
            self.critic.parameters(), lr=self.critic_lr, eps=self.opti_eps, weight_decay=self.weight_decay
        )

    def lr_decay(self, episode, episodes):
        update_linear_schedule(self.critic_optimizer, episode, episodes, self.critic_lr)

    def get_values(self, obs, rnn_states_critic, masks):
        values, rnn_states_critic = self.critic(obs, rnn_states_critic, masks)
        return values, rnn_states_critic

    def cal_value_loss(self, values, value_preds_batch, return_batch, value_normalizer=None):
        value_pred_clipped = value_preds_batch + (values - value_preds_batch).clamp(
            -self.clip_param, self.clip_param
        )
        if value_normalizer is not None:
            value_normalizer.update(return_batch)
            error_clipped = value_normalizer.normalize(return_batch) - value_pred_clipped
            error_original = value_normalizer.normalize(return_batch) - values
        else:
            error_clipped = return_batch - value_pred_clipped
            error_original = return_batch - values

        if self.use_huber_loss:
            value_loss_clipped = huber_loss(error_clipped, self.huber_delta)
            value_loss_original = huber_loss(error_original, self.huber_delta)
        else:
            value_loss_clipped = mse_loss(error_clipped)
            value_loss_original = mse_loss(error_original)

        value_loss = torch.max(value_loss_original, value_loss_clipped) if self.use_clipped_value_loss else value_loss_original
        return value_loss.mean()

    def update(self, sample, value_normalizer=None):
        (
            obs_batch,
            rnn_states_critic_batch,
            value_preds_batch,
            return_batch,
            masks_batch,
        ) = sample

        value_preds_batch = check(value_preds_batch).to(**self.tpdv)
        return_batch = check(return_batch).to(**self.tpdv)

        values, _ = self.get_values(obs_batch, rnn_states_critic_batch, masks_batch)
        value_loss = self.cal_value_loss(values, value_preds_batch, return_batch, value_normalizer=value_normalizer)

        self.critic_optimizer.zero_grad()
        (value_loss * self.value_loss_coef).backward()
        if self.use_max_grad_norm:
            critic_grad_norm = nn.utils.clip_grad_norm_(self.critic.parameters(), self.max_grad_norm)
        else:
            critic_grad_norm = get_grad_norm(self.critic.parameters())
        self.critic_optimizer.step()
        return value_loss, critic_grad_norm, value_preds_batch

    def train(self, critic_buffer, value_normalizer=None):
        train_info = {"value_loss": 0, "critic_grad_norm": 0, "value_preds": 0}
        for _ in range(self.critic_epoch):
            if self.use_recurrent_policy:
                data_generator = critic_buffer.recurrent_generator_critic(self.critic_num_mini_batch, self.data_chunk_length)
            elif self.use_naive_recurrent_policy:
                data_generator = critic_buffer.naive_recurrent_generator_critic(self.critic_num_mini_batch)
            else:
                data_generator = critic_buffer.feed_forward_generator_critic(self.critic_num_mini_batch)
            for sample in data_generator:
                value_loss, critic_grad_norm, value_preds_batch = self.update(sample, value_normalizer=value_normalizer)
                train_info["value_loss"] += value_loss.item()
                train_info["critic_grad_norm"] += critic_grad_norm
                train_info["value_preds"] += value_preds_batch.mean().item()
        num_updates = self.critic_epoch * self.critic_num_mini_batch
        for k in train_info.keys():
            train_info[k] /= num_updates
        return train_info

    def prep_training(self):
        self.critic.train()

    def prep_rollout(self):
        self.critic.eval()
