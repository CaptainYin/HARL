from pettingzoo.classic import rps_v2
import time
# env = rps_v2.env(render_mode="human")
# env.reset(seed=42)

# for agent in env.agent_iter():
#     observation, reward, termination, truncation, info = env.last()

#     if termination or truncation:
#         action = None
#     else:
#         action = env.action_space(agent).sample() # this is where you would insert your policy
#     time.sleep(1)
#     env.step(action)
# env.close()


# from pettingzoo.mpe import simple_reference_sparserew_v1

# env = simple_reference_sparserew_v1.parallel_env(render_mode="human")
# observations, infos = env.reset()

# while env.agents:
#     # this is where you would insert your policy
#     actions = {agent: env.action_space(agent).sample() for agent in env.agents}
#     # print(env.action_space) 
#     time.sleep(1)

#     observations, rewards, terminations, truncations, infos = env.step(actions)
#     print(observations['agent_0'].shape)#21
# env.close()

time.sleep(5)