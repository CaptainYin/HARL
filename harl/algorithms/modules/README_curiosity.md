# Curiosity Module Usage

The intrinsic curiosity module implements a simple forward dynamics predictor f(s_t, a_t) -> \hat{s}_{t+1}. 
The intrinsic reward r_int is the per-sample mean squared prediction error between \hat{s}_{t+1} and s_{t+1}.

## Enabling in IPPO (and shared-parameter IPPO)
Add/modify the following entries in your args dictionary when constructing the `IPPO` object (or via your config file):

```
use_curiosity: true
curiosity_coef: 0.01              # scales intrinsic reward added to advantages
curiosity_loss_coef: 1.0          # scales curiosity model supervised loss
curiosity_hidden_sizes: [256,256] # MLP layers
curiosity_lr: 3e-4                # (optional) learning rate for curiosity optimizer (defaults to actor lr)
```

The algorithm will:
1. Flatten observations (assumes vector observations; extend the module for image encoders if needed).
2. One-hot encode discrete actions or use continuous actions directly.
3. Compute intrinsic rewards for all collected transitions before PPO minibatching.
4. Add scaled intrinsic rewards to (normalized) advantages: `advantages += curiosity_coef * mse`.
5. Optimize the forward model with MSE over the same batch of transitions.

## Notes & Extensions
- For high-dimensional observations (e.g., images), insert a CNN encoder before the forward model.
- To combine extrinsic and intrinsic rewards at the environment level instead, you could modify the runner to add r_int to env rewards prior to GAE; current integration adds to advantages directly.
- You can log `intrinsic_reward_mean` and `curiosity_loss` from `train_info` for monitoring.

## API
The core module lives in `harl/algorithms/modules/curiosity.py`:
```
CuriosityModule(obs_dim, action_dim, hidden_sizes=(256,256))
# Forward call:
pred_next, per_sample_mse, loss = module(obs, actions, next_obs, discrete=False, num_actions=None)
```
