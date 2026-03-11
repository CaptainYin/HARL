"""Runner for IPPO with per-agent local critics."""
import numpy as np
import torch
from harl.runners.on_policy_base_runner import OnPolicyBaseRunner
from harl.common.buffers.on_policy_critic_buffer_ep import OnPolicyCriticBufferEP
from harl.common.buffers.on_policy_critic_buffer_fp import OnPolicyCriticBufferFP
from harl.algorithms.critics.v_critic import VCritic
from harl.utils.trans_tools import _t2n


class OnPolicyIPPORunner(OnPolicyBaseRunner):
    """IPPO Runner that overrides critic-related pieces to be per-agent local."""


    def train(self):
        """IPPO training loop: compute per-agent advantages and update actors and local critics."""
        actor_train_infos = []

        # compute advantages
        if self.value_normalizer is not None:
            advantages = self.critic_buffer.returns[ :-1 ] - self.value_normalizer.denormalize(self.critic_buffer.value_preds[:-1])
        else:
            advantages = ( self.critic_buffer.returns[:-1] - self.critic_buffer.value_preds[:-1] )

        actor_train_info = self.actor[0].share_param_train( self.actor_buffer, advantages.copy(), self.num_agents, self.state_type )
        for _ in torch.randperm(self.num_agents):
            actor_train_infos.append(actor_train_info)

        critic_train_info = self.critic.train(self.critic_buffer, self.value_normalizer)

        return actor_train_infos, critic_train_info

