import numpy as np
import torch
from fish_env import FishEnv, PPOLoggingCallback
from stable_baselines3 import PPO
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.vec_env import SubprocVecEnv


if __name__ == "__main__":
    N = 10
    L = .4

    mu = np.asarray([0.0] +[1e-4] * (N - 2))
    Cf = np.asarray([.12] * (N - 1))
    Cd = np.asarray([3.] + [1.] * (N - 2))

    # parallel environments
    def make_env():
        return FishEnv(L=L, N=N, mu=mu, freq=2., Cf=Cf, Cd=Cd, 
                       k_lo=1e-3, k_hi=10.0, 
                       head_len_lo=0.3, head_len_hi=0.6,
                       amp_lo=np.deg2rad(15), amp_hi=np.deg2rad(25),
                       reward_weight=[0., 2., -1., 0.])

    num_envs = 8
    vec_env = SubprocVecEnv([lambda: make_env() for _ in range(num_envs)])
    # check_env(env)
 
    # RL Agent
    # device = torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")
    model = PPO("MlpPolicy", 
                vec_env, 
                verbose=1, 
                learning_rate=0.0003, 
                n_steps=256, 
                batch_size=64, 
                n_epochs=10,
                ent_coef=0.005,
                clip_range=0.2,
                tensorboard_log="./res/ppo_log_1/",
                device=torch.device("cpu"))

    ppo_callback = PPOLoggingCallback()
    model.learn(total_timesteps=150000, callback=ppo_callback)

    file_name = "ppo_fish_2"
    model.save("res/" + file_name)    
