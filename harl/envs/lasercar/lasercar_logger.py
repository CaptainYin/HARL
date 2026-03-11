from gymnasium.utils import seeding
from harl.common.base_logger import BaseLogger
import numpy as np
class LasercarLogger(BaseLogger):
    def get_task_name(self):
        return f"{self.env_args['scenario']}"
    def eval_init(self):
        """Initialize the logger for evaluation."""
        self.total_num_steps = (
            self.episode
            * self.algo_args["train"]["episode_length"]
            * self.algo_args["train"]["n_rollout_threads"]
        )
        self.eval_episode_rewards = []
        self.one_episode_rewards = []
        self.succ_episode_steps = []
        self.succ_rate = []
        self.obs_collision_rate = []
        self.car_collision_rate = []
        self.timeout_rate = []
        for eval_i in range(self.algo_args["eval"]["n_eval_rollout_threads"]):
            self.one_episode_rewards.append([])
            self.eval_episode_rewards.append([])
    def eval_thread_done(self, tid):
        """Log evaluation information."""
        self.eval_episode_rewards[tid].append(np.sum(self.one_episode_rewards[tid], axis=0))
        self.one_episode_rewards[tid] = []
        succ_car=np.array([car_info["reached_goal"] for car_info in self.eval_infos[tid]])
        obs_collision=np.array([car_info["obs_collision"] for car_info in self.eval_infos[tid]])
        car_collision=np.array([car_info["car_collision"] for car_info in self.eval_infos[tid]])
        timeout=np.array([car_info["timeout"] for car_info in self.eval_infos[tid]])
        if succ_car.all():
            self.succ_episode_steps.append(max([car_info["step"] for car_info in self.eval_infos[tid]]))
        if self.env_args['reachgoalset']:
            self.succ_rate.append((self.eval_infos[tid][0]["goal_reach"]!=0).astype(np.float32).mean())
        else:
            self.succ_rate.append(np.mean(succ_car))
        self.obs_collision_rate.append(np.mean(obs_collision))
        self.car_collision_rate.append(np.mean(car_collision))
        self.timeout_rate.append(np.mean(timeout))

    def eval_log(self, eval_episode):
        """Log evaluation information."""
        self.eval_episode_rewards = np.concatenate(
            [rewards for rewards in self.eval_episode_rewards if rewards]
        )
        eval_env_infos = {
            "eval_average_episode_rewards": self.eval_episode_rewards,
            "eval_max_episode_rewards": [np.max(self.eval_episode_rewards)],
            "episode_succ_rate": [len(self.succ_episode_steps)/eval_episode],
            "car_succ_rate": [np.mean(self.succ_rate)],
            "obs_collision_rate": [np.mean(self.obs_collision_rate)],
            "car_collision_rate": [np.mean(self.car_collision_rate)],
            "car_timeout_rate": [np.mean(self.timeout_rate)],
        }
        self.log_env(eval_env_infos)
        eval_avg_rew = np.mean(self.eval_episode_rewards)
        print("Evaluation average episode reward is {}, episode Succ_rate {}, car_succ_rate {} obs_collision_rate {}, car_collision_rate {}, car_timeout_rate {}".format(eval_avg_rew,eval_env_infos["episode_succ_rate"][0],
                                                                                     eval_env_infos["car_succ_rate"][0],
                                                                                     eval_env_infos["obs_collision_rate"][0],
                                                                                     eval_env_infos["car_collision_rate"][0],
                                                                                     eval_env_infos["car_timeout_rate"][0]))
        self.log_file.write(
            ",".join(map(str, [self.total_num_steps, eval_avg_rew])) + "\n"
        )
        self.log_file.flush()