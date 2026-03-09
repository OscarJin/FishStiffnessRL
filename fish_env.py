import numpy as np
import gymnasium as gym
from gymnasium import spaces
import fish_sim
from numpy.typing import NDArray
from stable_baselines3.common.callbacks import BaseCallback
from fish_viz import get_head_yaw, calc_efficiency, get_tail_amplitude


class FishEnv(gym.Env):
    def __init__(self, 
                 L, N,
                 mu, freq, 
                 Cf, Cd, 
                 k_lo, k_hi,
                 head_len_lo, head_len_hi,
                 amp_lo, amp_hi,
                 reward_weight,
                 phase=0., bias=0., Cl_fin=2 * np.pi, Cd_fin=np.asarray([1.2, 0.01]),
                 fluid_density=1000., fluid_vel=np.asarray([0., 0.],),
                 sim_freq=50, t_start=0., t_end=10., timeout=30.,
                 ):
        '''
        `L`: total length of fish
        `head_len_lo`: lower bound of head length (in BL)
        `head_len_hi`: upper bound of head length (in BL)
        `N`: number of links
        `link_length`: length of each link
        `k_lo`: lower bound of stiffness
        `k_hi`: upper bound of stiffness
        `mu`: damping of each joint

        Driving angle = amp * sin(2*pi*freq*t+phase)+bias

        `amp_lo`: lower bound of driving joint amplitude
        `amp_hi`: upper bound of driving joint amplitude
        `freq`: driving frequency
        `phase`: driving phase
        `bias`: driving bias
        `Cf`: friction coeff. of each link (except caudal fin)
        `Cd`: drag coeff. of each link (except caudal fin)
        `Cl_fin`: lift coeff. of caudal fin (CL = Cl_fin * |AoA|)
        `Cd_fin`: drag coeff. of caudal fin (CD = Cd_fin[0] + Cd_fin[1] * AoA^2)
        `fluid_density`: density of fluid (default water: 1000.)
        `fluid_vel`: velocity of fluid (default zero)
        `sim_freq`: simulation frequency
        `t_start`: simulation start time (default 0 sec)
        `t_end`: simulation end time (default 10 sec)

        `reward_weight`: reward weight
        '''
        super(FishEnv, self).__init__()

        # fluid environment
        self.fluid = fish_sim.Fluid(fluid_density, fluid_vel)

        # fish shape
        self.L = L
        self.geometry = fish_sim.Geometry(self.L)

        # Links
        self.N = N
        self.fin_len = .2 * self.L
        self.mass_total = 0.
        self.dist2tail = 0.
        
        # Joints
        self.mu = mu 
        self.freq = freq
        self.phase = phase
        self.bias = bias
        self.joints = fish_sim.Joint(self.N, np.zeros(self.N - 1), self.mu, 0., self.freq, self.phase)

        # Hydrodynamics
        self.hydrodyn = fish_sim.HydroDyn(self.N, Cf, Cd, Cl_fin, Cd_fin)
        
        # Observation space: head x, y, vx, vy, theta, omega
        # all normalized
        self.observation_space = spaces.Box(
            low=np.asarray([-50, -2., -np.inf, -np.inf, -1, -np.inf]), 
            high=np.asarray([0, 2., np.inf, np.inf, 1, np.inf]), 
            dtype=np.float32)

        # Action space (normalized to [-1, 1])
        # Actions: (N-2) joint stiffness + head length + amp
        self.k_lo = k_lo
        self.k_hi = k_hi
        self.k_log_lo = np.log10(k_lo)
        self.k_log_hi = np.log10(k_hi)
        self.head_len_lo = head_len_lo
        self.head_len_hi = head_len_hi
        self.amp_lo = amp_lo 
        self.amp_hi = amp_hi
        self.action_size = self.N
        self.action_space = spaces.Box(low=-1., high=1., shape=(self.action_size,), dtype=np.float32)

        # simulation setup
        self.sim_freq = sim_freq
        self.t_start = t_start
        self.t_end = t_end
        self.timeout = timeout

        # record reset data
        self.reset_stable_vel = 0.
        self.reset_efficiency = 0.
        self.reset_y = 0.
        self.reset_yaw = 0.
        self.reset_state = None
        self.reset_info = None

        self.reward_weight = reward_weight
        self.reward_min = -2.
    
    def action2sim_inp(self, action):
        # k_values = self.k_lo + (action[:self.N - 2] + 1) * (self.k_hi - self.k_lo) / 2.0
        log_k = self.k_log_lo + (action[:self.N - 2] + 1) * (self.k_log_hi - self.k_log_lo) / 2.0
        k_values = 10 ** log_k
        head_len = self.head_len_lo + (action[-2] + 1) * (self.head_len_hi - self.head_len_lo) / 2.0
        amp = self.amp_lo + (action[-1] + 1) * (self.amp_hi - self.amp_lo) / 2.0
        return (np.concatenate(([0.], np.clip(k_values, self.k_lo, self.k_hi))), 
                float(np.clip(head_len, self.head_len_lo, self.head_len_hi)) * self.L,
                float(np.clip(amp, self.amp_lo, self.amp_hi)))
    
    def inp2actions(self, k_values, head_len, amp):
        # k_values not including the driving joint
        action = np.zeros(self.action_size)
        log_k = np.log10(k_values)
        action[:self.N - 2] = 2 * (log_k - self.k_log_lo) / (self.k_log_hi - self.k_log_lo) - 1
        action[-2] = 2 * (head_len - self.head_len_lo) / (self.head_len_hi - self.head_len_lo) - 1
        action[-1] = 2 * (amp - self.amp_hi) / (self.amp_hi - self.amp_lo) - 1
        # action[-2] = 0
        # action[-1] = 0
        return np.clip(action, -1, 1)


    def run_sim(self, action):
        try:
            k_val, head_len, amp = self.action2sim_inp(action)
            # Links
            mid_len = (self.L - head_len - self.fin_len) / (self.N - 2)
            link_length = np.asarray([head_len] + [mid_len] * (self.N - 2) + [self.fin_len])
            links = fish_sim.Link(self.N, self.L, link_length, self.fluid, self.geometry)
            self.mass_total = links.m_total
            self.dist2tail = links.length[-1]-links.lc[-1]
            # Joints
            self.joints.k = k_val
            self.joints.amp = amp
            # Initial state
            q_init = np.zeros(4 + 2 * self.N)
            q_init[5: self.N + 4] = np.full(self.N - 1, amp * np.sin(self.phase) + self.bias)
            I1 = float(links.Izz[0])
            I2 = float(links.Izz[1])
            dTheta = self.joints.amp * (2 * np.pi * self.joints.freq) * np.cos(self.joints.phase)
            q_init[self.N + 4] = - I2 * dTheta / (I1 + I2)
            q_init[self.N + 5] = I1 * dTheta / (I1 + I2)
            q0 = fish_sim.get_q0(q_init, links)

            sim = fish_sim.FishSimulator(self.sim_freq, self.t_start, self.t_end, q0, links, self.joints, self.hydrodyn)
            return sim.run(self.timeout)
        except fish_sim.RuntimeError as e:
            return e 
        
    def extract_speed(self, q_traj: NDArray):
        vxs_g = q_traj[:, 2 * self.N]
        vys_g = q_traj[:, 3 * self.N]
        thetas = q_traj[:, 4 * self.N]
        # transform to head frame
        vxs = -vxs_g * np.cos(thetas) - vys_g * np.sin(thetas)
        vys = vxs_g * np.sin(thetas) - vys_g * np.cos(thetas)
        vel = np.sqrt(vxs_g ** 2 + vys_g ** 2)
        return vxs, vys, vel
    
    def _get_state(self, q_traj: NDArray):
        vxs, vys, _ = self.extract_speed(q_traj)
        x_normalized = q_traj[-1, 0] / self.L
        y_normalized = q_traj[-1, self.N] / self.L
        vx_normalized = np.clip(vxs[-1] / self.L, self.observation_space.low[2], self.observation_space.high[2])
        vy_normalized = np.clip(vys[-1] / self.L, self.observation_space.low[3], self.observation_space.high[3])
        theta_normalized = np.clip(q_traj[-1, 4 * self.N] / np.pi, self.observation_space.low[4], self.observation_space.high[4])
        omega_normalized = np.clip(q_traj[-1, 5 * self.N] / np.pi, self.observation_space.low[5], self.observation_space.high[5])
        state = np.asarray([x_normalized, y_normalized, vx_normalized, vy_normalized, theta_normalized, omega_normalized], dtype=np.float32)
        return state

    def step(self, action):
        try:
            # run simulation
            results = self.run_sim(action)
            if isinstance(results, fish_sim.RuntimeError):
                raise RuntimeError(str(results))

            # extract results
            t_traj = np.asarray(results.t_traj, dtype=np.float32)
            q_traj = np.asarray(results.q_traj, dtype=np.float32)
            # Ft_rec = np.asarray(results.Ft_rec, dtype=np.float32)
            M_rec = np.asarray(results.M_rec, dtype=np.float32)

            # extract head velocity
            vxs, vys, vel = self.extract_speed(q_traj)
            vel_stable_normalized = float(np.mean(vel[len(vel) // 2: ]) / self.L)

            # Reward: maximize vel
            state = self._get_state(q_traj)           
            speed_gain = vel_stable_normalized - self.reset_stable_vel
            thrust_eff = calc_efficiency(self.N, self.mass_total, self.freq, t_traj, q_traj, M_rec)
            eff_gain = thrust_eff - self.reset_efficiency
            # deviation_penalty = abs(state[1]) - self.reset_y
            vys_g = q_traj[:, 3 * self.N]
            deviation_penalty = abs(float(np.mean(vys_g[len(vys_g) // 2: ]) / self.L)) - self.reset_y
            yaws = get_head_yaw(vxs, vys)
            max_yaw = np.max(yaws[len(vel) // 2: ])
            min_yaw = np.min(yaws[len(vel) // 2: ])
            # yaw_penalty = float(max_yaw - min_yaw) / np.pi - self.reset_yaw

            # total reward
            in_bounds = (self.observation_space.low[:2] <= state[:2]).all() and \
                        (state[:2] <= self.observation_space.high[:2]).all() and \
                        (float(max_yaw - min_yaw) < np.pi)
            
            base_reward = self.reward_weight[0] * eff_gain + \
                        self.reward_weight[1] * speed_gain
                        # self.reward_weight[2] * deviation_penalty
                        # self.reward_weight[3] * yaw_penalty

            # Exponential amplitude penalty
            tail_amp = get_tail_amplitude(t_traj, q_traj, self.N, self.dist2tail) / self.L 
            lambda_amp = 4.0
            amp_penalty = np.exp(-lambda_amp * (tail_amp ** 2)) if base_reward > 0 else 1.  # only when postive reward

            reward = max(base_reward * amp_penalty + self.reward_weight[2] * deviation_penalty, self.reward_min) \
                if in_bounds and 0. < thrust_eff < 1. else self.reward_min

            done = True
            truncated = False
            info = {
                # results
                "t_traj": t_traj,
                "q_traj": q_traj,
                "vel_stable": vel_stable_normalized,
                "thrust_eff": thrust_eff,
                "tail_amp": tail_amp,
                # reward terms
                "reward_eff": self.reward_weight[0] * eff_gain,
                "reward_speed": self.reward_weight[1] * speed_gain,
                "reward_deviation": self.reward_weight[2] * deviation_penalty,
                # "reward_yaw": self.reward_weight[3] * yaw_penalty,
            }
        except RuntimeError as e:
            print(str(e))
            state = np.zeros(self.observation_space.shape, dtype=np.float32)
            reward = self.reward_min   # strong penalty
            done = False
            truncated = True
            info = {
                "t_traj": None,
                "q_traj": None,
                "vel_stable": 0.,
                "thrust_eff": 0.,
                "error_message": str(e)
            }

        return state, reward, done, truncated, info

    def reset(self, seed=None):
        if self.reset_state is None or self.reset_info["q_traj"] is None:
            head_len_rand = np.random.uniform(self.head_len_lo, self.head_len_hi)
            mid_len_rand = (1 - head_len_rand - self.fin_len / self.L) / (self.N - 2)
            reset_k_hi = 5. * self.freq * self.freq
            k_rand = np.random.uniform(.9 * reset_k_hi, reset_k_hi)
            k_values_rand = k_rand * np.asarray([np.pow(1 - (head_len_rand + mid_len_rand * (i + 1)), 4) for i in range(self.N - 2)])
            amp_rand = np.random.uniform(self.amp_lo, self.amp_hi)
            
            action_rand = self.inp2actions(k_values_rand, head_len_rand, amp_rand)

            try:
                # run simulation
                results = self.run_sim(action_rand)
                if isinstance(results, fish_sim.RuntimeError):
                    raise RuntimeError(str(results))
                
                # extract results
                t_traj = np.asarray(results.t_traj, dtype=np.float32)
                q_traj = np.asarray(results.q_traj, dtype=np.float32)
                # Ft_rec = np.asarray(results.Ft_rec, dtype=np.float32)
                M_rec = np.asarray(results.M_rec, dtype=np.float32)

                self.reset_state = self._get_state(q_traj)
                vxs, vys, vel = self.extract_speed(q_traj)
                
                reset_stable_vel = float(np.mean(vel[len(vel) // 2: ]) / self.L)
                thrust_eff = calc_efficiency(self.N, self.mass_total, self.freq, t_traj, q_traj, M_rec)
                self.reset_stable_vel = reset_stable_vel
                self.reset_efficiency = thrust_eff

                yaws = get_head_yaw(vxs, vys)
                max_yaw = np.max(yaws[len(vel) // 2: ])
                min_yaw = np.min(yaws[len(vel) // 2: ])
                self.reset_yaw = float(max_yaw - min_yaw) / np.pi
                vys_g = q_traj[:, 3 * self.N]
                self.reset_y = abs(float(np.mean(vys_g[len(vys_g) // 2: ]) / self.L))

                self.reset_info = {
                    "t_traj": t_traj,
                    "q_traj": q_traj,
                    "vel_stable": reset_stable_vel,
                    "thrust_eff": thrust_eff,
                }
                print(reset_stable_vel, thrust_eff*100, self.reset_y, self.reset_yaw)
            except RuntimeError as e:
                self.reset_state = np.zeros(self.observation_space.shape, dtype=np.float32)
                self.reset_info = {
                    "t_traj": None,
                    "q_traj": None,
                    "vel_stable": 0.,
                    "thrust_eff": 0.,
                    "error": str(e),
                }
        return self.reset_state, self.reset_info
                
        # if seed is not None:
        #     np.random.seed(seed)
        
        # # action_rand = np.random.normal(0, 0.5, size=(self.N - 2,))
        # action_rand = np.random.uniform(-1, 1, size=(self.action_size,))
        # action_rand[:self.N - 2] = np.sort(action_rand[:self.N - 2], kind='mergesort')[::-1]
        
        # try:
        #     # run simulation
        #     results = self.run_sim(action_rand)
        #     if isinstance(results, fish_sim.RuntimeError):
        #         raise RuntimeError(str(results))
            
        #     # extract results
        #     t_traj = np.asarray(results.t_traj, dtype=np.float32)
        #     q_traj = np.asarray(results.q_traj, dtype=np.float32)
        #     Ft_rec = np.asarray(results.Ft_rec, dtype=np.float32)
        #     M_rec = np.asarray(results.M_rec, dtype=np.float32)

        #     # extract head velocity
        #     vxs, vys, vel = self.extract_speed(q_traj)
        #     state = self._get_state(q_traj)
        #     yaws = get_head_yaw(vxs, vys)
        #     max_yaw = np.max(yaws[len(vel) // 2: ])
        #     min_yaw = np.min(yaws[len(vel) // 2: ])
        #     self.reset_yaw = float(max_yaw - min_yaw)
        #     self.reset_stable_vel = float(np.mean(vel[len(vel) // 2: ]) / self.L)
        #     # self.reset_y = abs(state[1])
        #     vys_g = q_traj[:, 3 * self.N]
        #     self.reset_y = abs(float(np.mean(vys_g[len(vys_g) // 2: ]) / self.L))
        #     thrust_eff = calc_efficiency(self.N, t_traj, q_traj, Ft_rec, M_rec)
        #     self.reset_efficiency = thrust_eff if 0. < thrust_eff < 1. else 0.
        #     info = {
        #         "t_traj": t_traj,
        #         "q_traj": q_traj,
        #         "vel_stable": self.reset_stable_vel,
        #         "thrust_eff": thrust_eff,
        #     }
        # except RuntimeError as e:
        #     print(str(e))
        #     state = np.zeros(self.observation_space.shape, dtype=np.float32)
        #     self.reset_stable_vel = 0.
        #     # self.reset_y = self.observation_space.high[1]
        #     self.reset_y = 1.
        #     self.reset_yaw = np.pi / 2
        #     self.reset_efficiency = 0.
        #     info = {
        #         "t_traj": None,
        #         "q_traj": None,
        #         "vel_stable": 0.,
        #         "thrust_eff": 0.,
        #         "error": str(e),
        #     }

        # return state, info

    def render(self, mode='human'):
        pass


class PPOLoggingCallback(BaseCallback):
    def __init__(self, verbose = 1):
        super().__init__(verbose)
        self.episode_vel = []       # episode velocity buffer
        self.vel_history = []       # average velocity history
        self.episode_eff = []       # episode efficiency buffer
        self.eff_history = []       # average efficiency history
        self.episode_rewards = []    # episode reward buffer
        self.reward_keys = ['speed', 'eff', 'deviation', 'yaw']
        self.episode_reward_terms = {k: [] for k in self.reward_keys}
        self.rewards_history = []    # average reward history
        self.losses = {"policy_loss": [], "value_loss": [], "entropy_loss": []}
        self.n_updates = 0
    
    def _on_step(self):
        """log training metrics at each step"""
        # Store losses from logger
        if "train/policy_gradient_loss" in self.model.logger.name_to_value:
            self.losses["policy_loss"].append(self.model.logger.name_to_value.get("train/policy_gradient_loss", 0))
            self.losses["value_loss"].append(self.model.logger.name_to_value.get("train/value_loss", 0))
            self.losses["entropy_loss"].append(self.model.logger.name_to_value.get("train/entropy_loss", 0))
        
        # Store velocity for averaging
        if "infos" in self.locals:
            for _info in self.locals["infos"]:
                self.episode_vel.append(_info["vel_stable"])
                self.episode_eff.append(_info["thrust_eff"])
                for k in self.reward_keys:
                    reward_key = f"reward_{k}"
                    if reward_key in _info:
                        self.episode_reward_terms[k].append(_info[reward_key])
        
        if "rewards" in self.locals:
            self.episode_rewards.extend(self.locals["rewards"])
        
        return True
    
    def _on_rollout_end(self):
        self.n_updates += 1
        avg_vel = np.mean(self.episode_vel) if self.episode_vel else 0.
        self.vel_history.append(avg_vel)
        avg_eff = np.mean(self.episode_eff) if self.episode_eff else 0.
        self.eff_history.append(avg_eff)
        avg_reward = np.mean(self.episode_rewards) if self.episode_rewards else 0.
        avg_reward_components = {
            k: np.mean(self.episode_reward_terms[k]) if self.episode_reward_terms[k] else 0.0
            for k in self.reward_keys
        }
        self.rewards_history.append(avg_reward)
        cur_policy_loss = self.losses["policy_loss"][-1] if self.losses["policy_loss"] else 0.
        cur_value_loss = self.losses["value_loss"][-1] if self.losses["value_loss"] else 0.
        cur_entropy_loss = self.losses["entropy_loss"][-1] if self.losses["entropy_loss"] else 0.

        self.logger.record("train/avg_vel", avg_vel)
        self.logger.record("train/avg_eff", avg_eff)
        self.logger.record("train/avg_reward", avg_reward)
        # Print logs
        if self.model.verbose != 2:
            print("\n------------------------------------")
            print(f"n_updates: {self.n_updates}")
            print(f"avg_vel: {avg_vel:.4f}")
            print(f"avg_eff: {avg_eff:.4f}")
            if self.episode_rewards:
                print(f"avg_reward: {avg_reward:.4f}")
                for k in self.reward_keys:
                    print(f"reward_{k}: {avg_reward_components[k]:.4f}")
            if self.losses["policy_loss"]:
                print(f"policy_gradient_loss: {cur_policy_loss:.4f}")
                print(f"value_loss: {cur_value_loss:.4f}")
                print(f"entropy_loss: {cur_entropy_loss:.4f}")
            print("------------------------------------\n")

        self.episode_vel.clear()
        self.episode_rewards.clear()
        for k in self.reward_keys:
            self.episode_reward_terms[k].clear()

