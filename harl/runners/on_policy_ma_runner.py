import numpy as np
import torch
from harl.runners.on_policy_base_runner import OnPolicyBaseRunner

class OnPolicyMARunner(OnPolicyBaseRunner):
    """Runner for on-policy MA algorithms."""
    def train(self):
        """Training procedure for MAPPO."""
        actor_train_infos = []
        # compute advantages
        if self.value_normalizer is not None:
            advantages = self.critic_buffer.returns[ :-1 ] - self.value_normalizer.denormalize(self.critic_buffer.value_preds[:-1])
        else:
            advantages = ( self.critic_buffer.returns[:-1] - self.critic_buffer.value_preds[:-1] )

        # update actors
        if self.share_param:
            # print(advantages.shape)
            actor_train_info = self.actor[0].share_param_train( self.actor_buffer, advantages.copy(), self.num_agents, self.state_type )
            for _ in torch.randperm(self.num_agents):
                actor_train_infos.append(actor_train_info)
        else:
            for agent_id in range(self.num_agents):
                if self.state_type == "EP":
                    actor_train_info = self.actor[agent_id].train( self.actor_buffer[agent_id], advantages.copy(), "EP" )
                elif self.state_type == "FP":
                    actor_train_info = self.actor[agent_id].train( self.actor_buffer[agent_id], advantages[:, :, agent_id].copy(), "FP", )
                actor_train_infos.append(actor_train_info)

        # update critic
        critic_train_info = self.critic.train(self.critic_buffer, self.value_normalizer)
        return actor_train_infos, critic_train_info
