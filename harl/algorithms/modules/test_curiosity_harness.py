import torch
from curiosity import CuriosityModule


def run_basic():
    obs_dim = 12
    action_dim = 3
    batch = 5
    module = CuriosityModule(obs_dim=obs_dim, action_dim=action_dim, hidden_sizes=(32,32))
    obs = torch.randn(batch, obs_dim)
    next_obs = torch.randn(batch, obs_dim)
    actions = torch.randn(batch, action_dim)
    pred_next, per_sample_mse, loss = module(obs, actions, next_obs, discrete=False)
    print('pred_next.shape', pred_next.shape)
    print('per_sample_mse.shape', per_sample_mse.shape)
    print('loss', loss.item())

    # discrete test
    num_actions = 7
    disc_module = CuriosityModule(obs_dim=obs_dim, action_dim=num_actions, hidden_sizes=(32,32))
    disc_actions = torch.randint(0, num_actions, (batch,))
    pred_next2, per_sample_mse2, loss2 = disc_module(obs, disc_actions, next_obs, discrete=True, num_actions=num_actions)
    print('discrete pred_next.shape', pred_next2.shape)
    print('discrete per_sample_mse.shape', per_sample_mse2.shape)
    print('discrete loss', loss2.item())

if __name__ == '__main__':
    run_basic()
