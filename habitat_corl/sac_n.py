# Inspired by:
# 1. paper for SAC-N: https://arxiv.org/abs/2110.01548
# 2. implementation: https://github.com/snu-mllab/EDAC
import argparse
# The only difference from the original implementation:
# default pytorch weight initialization,
# without custom rlkit init & uniform init for last layers.
from typing import Any, Dict, List, Optional, Tuple, Union
from copy import deepcopy
from dataclasses import asdict, dataclass
import math
import os
import random
import uuid

import gym
import numpy as np
import torch
from gym import ObservationWrapper, ActionWrapper
from torch.distributions import Normal
import torch.nn as nn
from tqdm import trange
import wandb
from tqdm import tqdm

import habitat
from habitat.utils.visualizations.utils import images_to_video, \
    observations_to_image
from habitat_baselines.config.default import get_config
from habitat_corl.common.utils import restructure_results, train_eval_split
from habitat_corl.replay_buffer import ReplayBuffer, get_input_dims
from habitat_corl.shortest_path_dataset import sample_transitions, \
    register_position_sensor, \
    calc_mean_std, get_stored_episodes, batch_generator

# general utils
TensorBatch = List[torch.Tensor]

class HabitatWrapper(ObservationWrapper):
    def reset(self, **kwargs):
        """Resets the environment, returning a modified observation using :meth:`self.observation`."""
        obs = self.env.reset(**kwargs)
        return self.observation(obs)

    def step(self, action):
        """Returns a modified observation using :meth:`self.observation` after calling :meth:`env.step`."""
        observation = self.env.step(action)
        return self.observation(observation)

class ContinuousActionWrapper(ActionWrapper):
    def __init__(self, env):
        super().__init__(env)
        self.action_space = gym.spaces.Box(
            low=0.0, high=1.0, shape=(1,), dtype=np.float32
        )
    def _quat_to_xy_heading(self, quat):
        from habitat.utils.geometry_utils import quaternion_rotate_vector
        from habitat.tasks.utils import cartesian_to_polar
        direction_vector = np.array([0, 0, -1])

        heading_vector = quaternion_rotate_vector(quat, direction_vector)

        phi = cartesian_to_polar(-heading_vector[2], heading_vector[0])[1]
        return np.array([phi], dtype=np.float32)

    def _get_heading(self):
        env = self.env
        while hasattr(env, "env"):
            env = env.env
        agent_state = env.sim.get_agent_state()
        rotation_world_agent = agent_state.rotation
        return self._quat_to_xy_heading(rotation_world_agent.inverse())

    def action(self, action):
        action = action[0]
        # clip action to [0, 1]
        action = np.clip(action, 0, 1)
        target_angle = action * (2 * np.pi) - np.pi

        current_heading = self._get_heading()

        # tolerance = 7.5 degrees, in radians
        tolerance = 7.5 * np.pi / 180

        # if the current heading is within tolerance of the target angle,
        # move forward = 1
        if np.abs(current_heading - target_angle) < tolerance:
            return 1

        # if the current heading is greater than the target angle,
        # turn left = 3
        if current_heading > target_angle:
            return 3

        # if the current heading is less than the target angle,
        # turn right = 2
        if current_heading < target_angle:
            return 2

    def step(self, action):
        obs = self.env.step(self.action(action))
        return obs



def wrap_env(
    env: gym.Env,
    state_mean: Union[np.ndarray, float] = 0.0,
    state_std: Union[np.ndarray, float] = 1.0,
    reward_scale: float = 1.0,
    used_inputs=None,
) -> gym.Env:
    if used_inputs is None:
        used_inputs = ["postion", "heading", "pointgoal"]

    def state_to_vector(state):
        return np.concatenate([state[key] for key in used_inputs], axis=-1)

    def normalize_state(state):
        return (state - state_mean) / state_std

    def transform_state(state):
        raw_state = deepcopy(state)
        state = state_to_vector(state)
        return normalize_state(state), raw_state

    def scale_reward(reward):
        return reward_scale * reward

    env = HabitatWrapper(env)
    env.observation = transform_state
    env = ContinuousActionWrapper(env)
    return env


def soft_update(target: nn.Module, source: nn.Module, tau: float):
    for target_param, source_param in zip(target.parameters(),
                                          source.parameters()):
        target_param.data.copy_(
            (1 - tau) * target_param.data + tau * source_param.data)


def wandb_init(config) -> None:
    wandb.init(
        config=config,
        project=config.PROJECT,
        group=config.GROUP,
        name=config.NAME,
        id=str(uuid.uuid4()),
        mode="disabled"
    )
    wandb.run.save()


def set_seed(
    seed: int, env: Optional[gym.Env] = None, deterministic_torch: bool = False
):
    if env is not None:
        env.seed(seed)
        env.action_space.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    if deterministic_torch:
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True


# SAC Actor & Critic implementation
class VectorizedLinear(nn.Module):
    def __init__(self, in_features: int, out_features: int,
                 ensemble_size: int):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.ensemble_size = ensemble_size

        self.weight = nn.Parameter(
            torch.empty(ensemble_size, in_features, out_features))
        self.bias = nn.Parameter(torch.empty(ensemble_size, 1, out_features))

        self.reset_parameters()

    def reset_parameters(self):
        # default pytorch init for nn.Linear module
        for layer in range(self.ensemble_size):
            nn.init.kaiming_uniform_(self.weight[layer], a=math.sqrt(5))

        fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight[0])
        bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
        nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # input: [ensemble_size, batch_size, input_size]
        # weight: [ensemble_size, input_size, out_size]
        # out: [ensemble_size, batch_size, out_size]
        return x @ self.weight + self.bias


class Actor(nn.Module):
    def __init__(
        self, state_dim: int, action_dim: int, hidden_dim: int,
        max_action: float = 1.0
    ):
        super().__init__()
        self.trunk = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )
        # with separate layers works better than with Linear(hidden_dim, 2 * action_dim)
        self.mu = nn.Linear(hidden_dim, action_dim)
        self.log_sigma = nn.Linear(hidden_dim, action_dim)

        # init as in the EDAC paper
        for layer in self.trunk[::2]:
            torch.nn.init.constant_(layer.bias, 0.1)

        torch.nn.init.uniform_(self.mu.weight, -1e-3, 1e-3)
        torch.nn.init.uniform_(self.mu.bias, -1e-3, 1e-3)
        torch.nn.init.uniform_(self.log_sigma.weight, -1e-3, 1e-3)
        torch.nn.init.uniform_(self.log_sigma.bias, -1e-3, 1e-3)

        self.action_dim = action_dim
        self.max_action = max_action

    def forward(
        self,
        state: torch.Tensor,
        deterministic: bool = False,
        need_log_prob: bool = False,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        hidden = self.trunk(state)
        mu, log_sigma = self.mu(hidden), self.log_sigma(hidden)

        # clipping params from EDAC paper, not as in SAC paper (-20, 2)
        log_sigma = torch.clamp(log_sigma, -5, 2)
        policy_dist = Normal(mu, torch.exp(log_sigma))

        if deterministic:
            action = mu
        else:
            action = policy_dist.rsample()

        tanh_action, log_prob = torch.tanh(action), None
        if need_log_prob:
            # change of variables formula (SAC paper, appendix C, eq 21)
            log_prob = policy_dist.log_prob(action).sum(axis=-1)
            log_prob = log_prob - torch.log(1 - tanh_action.pow(2) + 1e-6).sum(
                axis=-1)

        return tanh_action * self.max_action, log_prob

    @torch.no_grad()
    def act(self, state: np.ndarray, device: str) -> np.ndarray:
        deterministic = not self.training
        state = torch.tensor(state, device=device, dtype=torch.float32)
        action = self(state, deterministic=deterministic)[0].cpu().numpy()
        return action


class VectorizedCritic(nn.Module):
    def __init__(
        self, state_dim: int, action_dim: int, hidden_dim: int,
        num_critics: int
    ):
        super().__init__()
        self.critic = nn.Sequential(
            VectorizedLinear(state_dim + action_dim, hidden_dim, num_critics),
            nn.ReLU(),
            VectorizedLinear(hidden_dim, hidden_dim, num_critics),
            nn.ReLU(),
            VectorizedLinear(hidden_dim, hidden_dim, num_critics),
            nn.ReLU(),
            VectorizedLinear(hidden_dim, 1, num_critics),
        )
        # init as in the EDAC paper
        for layer in self.critic[::2]:
            torch.nn.init.constant_(layer.bias, 0.1)

        torch.nn.init.uniform_(self.critic[-1].weight, -3e-3, 3e-3)
        torch.nn.init.uniform_(self.critic[-1].bias, -3e-3, 3e-3)

        self.num_critics = num_critics

    def forward(self, state: torch.Tensor,
                action: torch.Tensor) -> torch.Tensor:
        # [batch_size, state_dim + action_dim]
        state_action = torch.cat([state, action], dim=-1)
        # [num_critics, batch_size, state_dim + action_dim]
        state_action = state_action.unsqueeze(0).repeat_interleave(
            self.num_critics, dim=0
        )
        # [num_critics, batch_size]
        q_values = self.critic(state_action).squeeze(-1)
        return q_values


class SACN:
    def __init__(
        self,
        actor: Actor,
        actor_optimizer: torch.optim.Optimizer,
        critic: VectorizedCritic,
        critic_optimizer: torch.optim.Optimizer,
        gamma: float = 0.99,
        tau: float = 0.005,
        alpha_learning_rate: float = 1e-4,
        device: str = "cpu",  # noqa
    ):
        self.device = device

        self.actor = actor
        self.critic = critic
        with torch.no_grad():
            self.target_critic = deepcopy(self.critic)

        self.actor_optimizer = actor_optimizer
        self.critic_optimizer = critic_optimizer

        self.tau = tau
        self.gamma = gamma

        # adaptive alpha setup
        self.target_entropy = -float(self.actor.action_dim)
        self.log_alpha = torch.tensor(
            [0.0], dtype=torch.float32, device=self.device, requires_grad=True
        )
        self.alpha_optimizer = torch.optim.Adam([self.log_alpha],
                                                lr=alpha_learning_rate)
        self.alpha = self.log_alpha.exp().detach()

    def _alpha_loss(self, state: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            action, action_log_prob = self.actor(state, need_log_prob=True)

        loss = (-self.log_alpha * (
                action_log_prob + self.target_entropy)).mean()

        return loss

    def _actor_loss(self, state: torch.Tensor) -> Tuple[
        torch.Tensor, float, float]:
        action, action_log_prob = self.actor(state, need_log_prob=True)
        q_value_dist = self.critic(state, action)
        assert q_value_dist.shape[0] == self.critic.num_critics
        q_value_min = q_value_dist.min(0).values
        # needed for logging
        q_value_std = q_value_dist.std(0).mean().item()
        batch_entropy = -action_log_prob.mean().item()

        assert action_log_prob.shape == q_value_min.shape
        loss = (self.alpha * action_log_prob - q_value_min).mean()

        return loss, batch_entropy, q_value_std

    def _critic_loss(
        self,
        state: torch.Tensor,
        action: torch.Tensor,
        reward: torch.Tensor,
        next_state: torch.Tensor,
        done: torch.Tensor,
    ) -> torch.Tensor:
        with torch.no_grad():
            next_action, next_action_log_prob = self.actor(
                next_state, need_log_prob=True
            )
            q_next = self.target_critic(next_state, next_action).min(0).values
            q_next = q_next - self.alpha * next_action_log_prob
            assert q_next.unsqueeze(-1).shape == done.shape == reward.shape
            q_target = reward + self.gamma * (1 - done) * q_next.unsqueeze(-1)

        q_values = self.critic(state, action)
        # [ensemble_size, batch_size] - [1, batch_size]
        loss = ((q_values - q_target.view(1, -1)) ** 2).mean(dim=1).sum(dim=0)

        return loss

    def update(self, batch: ReplayBuffer, used_inputs: List[str]) -> Dict[
        str, float]:
        state, action, reward, next_state, done = batch.to_tensor(self.device,
                                                                  used_inputs,
                                                                  continuous_actions=True)
        # Usually updates are done in the following order: critic -> actor -> alpha
        # But we found that EDAC paper uses reverse (which gives better results)

        # Alpha update
        alpha_loss = self._alpha_loss(state)
        self.alpha_optimizer.zero_grad()
        alpha_loss.backward()
        self.alpha_optimizer.step()

        self.alpha = self.log_alpha.exp().detach()

        # Actor update
        actor_loss, actor_batch_entropy, q_policy_std = self._actor_loss(state)
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        # Critic update
        critic_loss = self._critic_loss(state, action, reward, next_state,
                                        done)
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        #  Target networks soft update
        with torch.no_grad():
            soft_update(self.target_critic, self.critic, tau=self.tau)
            # for logging, Q-ensemble std estimate with the random actions:
            # a ~ U[-max_action, max_action]
            max_action = self.actor.max_action
            random_actions = -max_action + 2 * max_action * torch.rand_like(
                action)

            q_random_std = self.critic(state, random_actions).std(
                0).mean().item()

        update_info = {
            "alpha_loss": alpha_loss.item(),
            "critic_loss": critic_loss.item(),
            "actor_loss": actor_loss.item(),
            "batch_entropy": actor_batch_entropy,
            "alpha": self.alpha.item(),
            "q_policy_std": q_policy_std,
            "q_random_std": q_random_std,
        }
        return update_info

    def state_dict(self) -> Dict[str, Any]:
        state = {
            "actor": self.actor.state_dict(),
            "critic": self.critic.state_dict(),
            "target_critic": self.target_critic.state_dict(),
            "log_alpha": self.log_alpha.item(),
            "actor_optim": self.actor_optimizer.state_dict(),
            "critic_optim": self.critic_optimizer.state_dict(),
            "alpha_optim": self.alpha_optimizer.state_dict(),
        }
        return state

    def load_state_dict(self, state_dict: Dict[str, Any]):
        self.actor.load_state_dict(state_dict["actor"])
        self.critic.load_state_dict(state_dict["critic"])
        self.target_critic.load_state_dict(state_dict["target_critic"])
        self.actor_optimizer.load_state_dict(state_dict["actor_optim"])
        self.critic_optimizer.load_state_dict(state_dict["critic_optim"])
        self.alpha_optimizer.load_state_dict(state_dict["alpha_optim"])
        self.log_alpha.data[0] = state_dict["log_alpha"]
        self.alpha = self.log_alpha.exp().detach()


@torch.no_grad()
def eval_actor(
    env,
    actor,
    device,
    episodes,
    seed,
    max_traj_len=1000,
    used_inputs=["pointgoal_with_gps_compass"],
    video=False,
    video_dir="demos",
    video_prefix="demo",
    ignore_stop=False,
    succes_distance=0.2,
):
    def make_videos(observations_list, output_prefix, ep_id):
        prefix = output_prefix + "_{}".format(ep_id)
        # make dir if it does not exist
        os.makedirs(video_dir, exist_ok=True)
        # check for directories in output_prefix
        if "/" in output_prefix:
            dirs = [video_dir] + output_prefix.split("/")[0:-1]
            dirs = "/".join(dirs)
            os.makedirs(dirs, exist_ok=True)
        images_to_video(observations_list, output_dir=video_dir,
                        video_name=prefix)

    # run the agent for n_episodes
    # env = habitat.Env(config=env._config)
    env.episodes = episodes
    env.episode_iterator = iter(episodes)
    env.seed(seed)  # needed?
    results = []
    for i in tqdm(range(len(episodes)), desc="eval", leave=False):
        video_frames = []
        observations, raw = env.reset()
        for step in range(max_traj_len):
            action = actor.act(observations, device)
            # print(action)
            observations, raw = env.step(action)
            info = env.get_metrics()
            if video:
                frame = observations_to_image(raw, info)
                video_frames.append(frame)
            # stop if close to goal
            position = env.sim.get_agent_state().position
            goal = env.current_episode.goals[0].position
            distance = np.linalg.norm(np.array(position) - np.array(goal))
            # print(f"\t{distance}")
            if env.episode_over:
                # print(f"Episode {i} finished after {step} steps")
                break
            if ignore_stop and distance < succes_distance:
                info["success"] = True
                break

        info = env.get_metrics()
        results.append(info)
        if video:
            make_videos(video_frames, video_prefix, i)
    return restructure_results(results)


def return_reward_range(dataset, max_episode_steps):
    returns, lengths = [], []
    ep_ret, ep_len = 0.0, 0
    for r, d in zip(dataset["rewards"], dataset["terminals"]):
        ep_ret += float(r)
        ep_len += 1
        if d or ep_len == max_episode_steps:
            returns.append(ep_ret)
            lengths.append(ep_len)
            ep_ret, ep_len = 0.0, 0
    lengths.append(ep_len)  # but still keep track of number of steps
    assert sum(lengths) == len(dataset["rewards"])
    return min(returns), max(returns)


def modify_reward(dataset, env_name, max_episode_steps=1000):
    if any(s in env_name for s in ("halfcheetah", "hopper", "walker2d")):
        min_ret, max_ret = return_reward_range(dataset, max_episode_steps)
        dataset["rewards"] /= max_ret - min_ret
        dataset["rewards"] *= max_episode_steps
    elif "antmaze" in env_name:
        dataset["rewards"] -= 1.0


def train(config):
    set_seed(config.SEED,
             deterministic_torch=config.RL.SAC_N.deterministic_torch)
    wandb_init(config)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    mean_std = calc_mean_std(config.TASK_CONFIG, used_inputs=config.MODEL.used_inputs)

    # data, evaluation, env setup
    with wrap_env(
        habitat.Env(config=config.TASK_CONFIG),
        state_mean=mean_std["used"][0],
        state_std=mean_std["used"][1],
        used_inputs=config.MODEL.used_inputs,
    ) as env:
        state_dim = get_input_dims(config)
        action_dim = 1
        # action_dim = env.action_space.n

        train_episodes, eval_episodes = train_eval_split(
            env=env,
            config=config,
            n_eval_episodes=config.RL.SAC_N.eval_episodes
        )


        # Actor & Critic setup
        actor = Actor(state_dim, action_dim, config.RL.SAC_N.hidden_dim,
                      config.RL.SAC_N.max_action)
        actor.to(device)
        actor_optimizer = torch.optim.Adam(actor.parameters(),
                                           lr=config.RL.SAC_N.actor_learning_rate)
        critic = VectorizedCritic(
            state_dim, action_dim, config.RL.SAC_N.hidden_dim,
            config.RL.SAC_N.num_critics
        )
        critic.to(device)
        critic_optimizer = torch.optim.Adam(
            critic.parameters(), lr=config.RL.SAC_N.critic_learning_rate
        )

        trainer = SACN(
            actor=actor,
            actor_optimizer=actor_optimizer,
            critic=critic,
            critic_optimizer=critic_optimizer,
            gamma=config.RL.SAC_N.gamma,
            tau=config.RL.SAC_N.tau,
            alpha_learning_rate=config.RL.SAC_N.alpha_learning_rate,
            device=device,
        )
        # saving config to the checkpoint
        if hasattr(config,
                   "CHECKPOINT_FOLDER") and config.CHECKPOINT_FOLDER is not None:
            print(f"Checkpoints path: {config.CHECKPOINT_FOLDER}")
            os.makedirs(config.CHECKPOINT_FOLDER, exist_ok=True)
            with open(os.path.join(config.CHECKPOINT_FOLDER, "config.yaml"),
                      "w") as f:
                f.write(config.dump())

        total_updates = 0.0
        evaluations = []
        batch_gen = batch_generator(
            config.TASK_CONFIG,
            n_transitions=config.RL.SAC_N.batch_size,
            groups=train_episodes,
            # use_full_dataset=False,
            use_full_dataset=config.RL.SAC_N.load_full_dataset,
            # n_batches=config.RL.SAC_N.eval_every * config.RL.SAC_N.num_updates_on_epoch,
            n_batches=config.RL.SAC_N.num_updates_on_epoch,
            datasets=[f"state_{x}" for x in config.MODEL.used_inputs] + \
                     [f"next_state_{x}" for x in config.MODEL.used_inputs] + \
                     ["action", "reward", "done"],
            continuous=True,
        )
        for epoch in trange(config.RL.SAC_N.num_epochs, desc="Training"):
            # training
            for _ in trange(config.RL.SAC_N.num_updates_on_epoch, desc="Epoch",
                            leave=False):
                batch = next(batch_gen)
                batch.normalize_states(mean_std)
                batch.to_tensor(device=device)
                update_info = trainer.update(batch,
                                             used_inputs=config.MODEL.used_inputs)

                if total_updates % config.RL.SAC_N.log_every == 0:
                    wandb.log({"epoch": epoch, **update_info})

                total_updates += 1

            # evaluation
            t = total_updates
            if epoch % config.RL.SAC_N.eval_every == 0 or epoch == config.RL.SAC_N.num_epochs - 1:
                print(f"Time steps: {t + 1}")
                eval_scores = eval_actor(
                    env,
                    actor,
                    device=device,
                    episodes=eval_episodes,
                    seed=config.SEED,
                    used_inputs=config.MODEL.used_inputs,
                    video=True,
                    # video=epoch == config.RL.SAC_N.num_epochs - 1,
                    video_dir=config.VIDEO_DIR,
                    video_prefix="sac_n/sac_n",
                    succes_distance=config.TASK_CONFIG.TASK.SUCCESS_DISTANCE,
                    ignore_stop=config.RL.SAC_N.ignore_stop,
                )
                evaluations.append(eval_scores)
                print("---------------------------------------")
                print(
                    f"Evaluation over {config.RL.SAC_N.eval_episodes} episodes: "
                )
                for key in eval_scores:
                    print(f"{key}: {np.mean(eval_scores[key])}")
                print("---------------------------------------")
                torch.save(
                    trainer.state_dict(),
                    os.path.join(config.CHECKPOINT_FOLDER,
                                 f"checkpoint_{total_updates}.pt"),
                )
                for key in eval_scores:
                    wandb.log(
                        {
                            f"eval/{key}_mean": np.mean(eval_scores[key]),
                            f"eval/{key}_std": np.std(eval_scores[key])
                        },
                        step=total_updates,
                    )

    wandb.finish()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str,
                        default="habitat_corl/configs/sacn_pointnav.yaml")
    args = parser.parse_args()

    config = get_config(args.config)
    register_position_sensor(config.TASK_CONFIG)
    train(config)


if __name__ == "__main__":
    main()
