# source: https://github.com/sfujim/TD3_BC
# https://arxiv.org/pdf/2106.06860.pdf
import argparse
from typing import Any, Dict, List, Optional, Tuple, Union
import copy
from dataclasses import asdict, dataclass
import os
from pathlib import Path
import random
import uuid

import gym
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import wandb
from tqdm import trange

import habitat
from habitat_baselines.config.default import get_config
from habitat_corl.common.utils import restructure_results, train_eval_split, \
    eval_actor, wandb_init, set_seed, get_goal
from habitat_corl.common.wrappers import wrap_env
from habitat_corl.replay_buffer import get_input_dims, ReplayBuffer
from habitat_corl.shortest_path_dataset import register_new_sensors, \
    calc_mean_std, batch_generator

TensorBatch = List[torch.Tensor]


def soft_update(target: nn.Module, source: nn.Module, tau: float):
    for target_param, source_param in zip(target.parameters(), source.parameters()):
        target_param.data.copy_((1 - tau) * target_param.data + tau * source_param.data)



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


class Actor(nn.Module):
    def __init__(self, state_dim: int, action_dim: int, max_action: float):
        super(Actor, self).__init__()

        self.net = nn.Sequential(
            nn.Linear(state_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, action_dim),
            nn.Tanh(),
        )

        self.max_action = max_action

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        return self.max_action * self.net(state)

    @torch.no_grad()
    def act(self, state: np.ndarray, device: str = "cpu") -> np.ndarray:
        state = torch.tensor(state.reshape(1, -1), device=device, dtype=torch.float32)
        return self(state).cpu().data.numpy().flatten()


class Critic(nn.Module):
    def __init__(self, state_dim: int, action_dim: int):
        super(Critic, self).__init__()

        self.net = nn.Sequential(
            nn.Linear(state_dim + action_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 1),
        )

    def forward(self, state: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        sa = torch.cat([state, action], 1)
        return self.net(sa)


class TD3_BC:  # noqa
    def __init__(
        self,
        max_action: float,
        actor: nn.Module,
        actor_optimizer: torch.optim.Optimizer,
        critic_1: nn.Module,
        critic_1_optimizer: torch.optim.Optimizer,
        critic_2: nn.Module,
        critic_2_optimizer: torch.optim.Optimizer,
        discount: float = 0.99,
        tau: float = 0.005,
        policy_noise: float = 0.2,
        noise_clip: float = 0.5,
        policy_freq: int = 2,
        alpha: float = 2.5,
        device: str = "cpu",
    ):
        self.actor = actor
        self.actor_target = copy.deepcopy(actor)
        self.actor_optimizer = actor_optimizer
        self.critic_1 = critic_1
        self.critic_1_target = copy.deepcopy(critic_1)
        self.critic_1_optimizer = critic_1_optimizer
        self.critic_2 = critic_2
        self.critic_2_target = copy.deepcopy(critic_2)
        self.critic_2_optimizer = critic_2_optimizer

        self.max_action = max_action
        self.discount = discount
        self.tau = tau
        self.policy_noise = policy_noise
        self.noise_clip = noise_clip
        self.policy_freq = policy_freq
        self.alpha = alpha

        self.total_it = 0
        self.device = device

    def train(self, batch: ReplayBuffer, used_inputs: List[str]) -> Dict[str, float]:
        log_dict = {}
        self.total_it += 1

        state, action, reward, next_state, done = batch.to_tensor(
            device=self.device,
            state_keys=used_inputs,
            continuous_actions=True,
        )
        not_done = 1 - done

        with torch.no_grad():
            # Select action according to actor and add clipped noise
            noise = (torch.randn_like(action) * self.policy_noise).clamp(
                -self.noise_clip, self.noise_clip
            )

            next_action = (self.actor_target(next_state) + noise).clamp(
                -self.max_action, self.max_action
            )

            # Compute the target Q value
            target_q1 = self.critic_1_target(next_state, next_action)
            target_q2 = self.critic_2_target(next_state, next_action)
            target_q = torch.min(target_q1, target_q2)
            target_q = reward + not_done * self.discount * target_q

        # Get current Q estimates
        current_q1 = self.critic_1(state, action)
        current_q2 = self.critic_2(state, action)

        # Compute critic loss
        critic_loss = F.mse_loss(current_q1, target_q) + F.mse_loss(current_q2, target_q)
        log_dict["critic_loss"] = critic_loss.item()
        # Optimize the critic
        self.critic_1_optimizer.zero_grad()
        self.critic_2_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_1_optimizer.step()
        self.critic_2_optimizer.step()

        # Delayed actor updates
        if self.total_it % self.policy_freq == 0:

            # Compute actor loss
            pi = self.actor(state)
            q = self.critic_1(state, pi)
            lmbda = self.alpha / q.abs().mean().detach()

            actor_loss = -lmbda * q.mean() + F.mse_loss(pi, action)
            log_dict["actor_loss"] = actor_loss.item()
            # Optimize the actor
            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            self.actor_optimizer.step()

            # Update the frozen target models
            soft_update(self.critic_1_target, self.critic_1, self.tau)
            soft_update(self.critic_2_target, self.critic_2, self.tau)
            soft_update(self.actor_target, self.actor, self.tau)

        return log_dict

    def state_dict(self) -> Dict[str, Any]:
        return {
            "critic_1": self.critic_1.state_dict(),
            "critic_1_optimizer": self.critic_1_optimizer.state_dict(),
            "critic_2": self.critic_2.state_dict(),
            "critic_2_optimizer": self.critic_2_optimizer.state_dict(),
            "actor": self.actor.state_dict(),
            "actor_optimizer": self.actor_optimizer.state_dict(),
            "total_it": self.total_it,
        }

    def load_state_dict(self, state_dict: Dict[str, Any]):
        self.critic_1.load_state_dict(state_dict["critic_1"])
        self.critic_1_optimizer.load_state_dict(state_dict["critic_1_optimizer"])
        self.critic_1_target = copy.deepcopy(self.critic_1)

        self.critic_2.load_state_dict(state_dict["critic_2"])
        self.critic_2_optimizer.load_state_dict(state_dict["critic_2_optimizer"])
        self.critic_2_target = copy.deepcopy(self.critic_2)

        self.actor.load_state_dict(state_dict["actor"])
        self.actor_optimizer.load_state_dict(state_dict["actor_optimizer"])
        self.actor_target = copy.deepcopy(self.actor)

        self.total_it = state_dict["total_it"]


def train(config):
    algo_config = config.RL.TD3_BC
    task_config = config.TASK_CONFIG
    set_seed(config.SEED)
    wandb_init(config)

    mean_std = calc_mean_std(config.TASK_CONFIG,
                             used_inputs=config.MODEL.used_inputs)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    with wrap_env(
        habitat.Env(config=task_config),
        state_mean=mean_std["used"][0],
        state_std=mean_std["used"][1],
        used_inputs=config.MODEL.used_inputs,
        continuous=True,
        ignore_stop=True,
        turn_angle=task_config.SIMULATOR.TURN_ANGLE,
    ) as env:

        state_dim = get_input_dims(config)
        action_dim = env.action_space.shape[0]

        # if config.normalize_reward:
        #     modify_reward(dataset, config.env)

        train_episodes, eval_episodes = train_eval_split(
            env=env,
            config=config,
            n_eval_episodes=algo_config.eval_episodes,
            single_goal=algo_config.single_goal,
        )

        batch_gen = batch_generator(
            task_config,
            n_transitions=algo_config.batch_size,
            groups=train_episodes,
            use_full_dataset=algo_config.load_full_dataset,
            datasets=[f"state_{x}" for x in config.MODEL.used_inputs] + \
                     [f"next_state_{x}" for x in config.MODEL.used_inputs] + \
                     ["action", "reward", "done"],
            continuous=True,
            single_goal=get_goal(algo_config, eval_episodes),
            normalization_data=mean_std
        )

        max_action = float(env.action_space.high[0])

        if hasattr(config,
                   "CHECKPOINT_FOLDER") and config.CHECKPOINT_FOLDER is not None:
            print(f"Checkpoints path: {config.CHECKPOINT_FOLDER}")
            os.makedirs(config.CHECKPOINT_FOLDER, exist_ok=True)
            with open(os.path.join(config.CHECKPOINT_FOLDER, "config.yaml"),
                      "w") as f:
                f.write(config.dump())

        # Set seeds
        seed = config.SEED
        set_seed(seed, env)

        actor = Actor(state_dim, action_dim, max_action).to(device)
        actor_optimizer = torch.optim.Adam(actor.parameters(), lr=algo_config.learning_rate)

        critic_1 = Critic(state_dim, action_dim).to(device)
        critic_1_optimizer = torch.optim.Adam(critic_1.parameters(), lr=algo_config.learning_rate)
        critic_2 = Critic(state_dim, action_dim).to(device)
        critic_2_optimizer = torch.optim.Adam(critic_2.parameters(), lr=algo_config.learning_rate)

        wandb.watch(actor)
        wandb.watch(critic_1)
        wandb.watch(critic_2)

        kwargs = {
            "max_action": max_action,
            "actor": actor,
            "actor_optimizer": actor_optimizer,
            "critic_1": critic_1,
            "critic_1_optimizer": critic_1_optimizer,
            "critic_2": critic_2,
            "critic_2_optimizer": critic_2_optimizer,
            "discount": algo_config.discount,
            "tau": algo_config.tau,
            "device": device,
            # TD3
            "policy_noise": algo_config.policy_noise * max_action,
            "noise_clip": algo_config.noise_clip * max_action,
            "policy_freq": algo_config.policy_freq,
            # TD3 + BC
            "alpha": algo_config.alpha,
        }

        print("---------------------------------------")
        print(f"Training TD3 + BC, Scene: {task_config.DATASET.CONTENT_SCENES}"
              f", Seed: {seed}, ignore_stop: {algo_config.ignore_stop}, "
              f"single_goal: {algo_config.single_goal}")
        print("---------------------------------------")

        # Initialize actor
        trainer = TD3_BC(**kwargs)

        if algo_config.load_model != "":
            policy_file = Path(algo_config.load_model)
            trainer.load_state_dict(torch.load(policy_file))
            actor = trainer.actor

        evaluations = []
        for t in trange(int(algo_config.max_timesteps), desc="Training"):
            batch = next(batch_gen)
            log_dict = trainer.train(batch, used_inputs=config.MODEL.used_inputs)
            wandb.log(log_dict, step=trainer.total_it)
            # Evaluate episode
            if (t + 1) % algo_config.eval_freq == 0:
                print(f"Time steps: {t + 1}")
                eval_scores = eval_actor(
                    env,
                    actor,
                    device=device,
                    episodes=eval_episodes,
                    seed=config.SEED,
                    used_inputs=config.MODEL.used_inputs,
                    # video=True,
                    video=t == algo_config.max_timesteps - 1,
                    video_dir=config.VIDEO_DIR,
                    video_prefix="td3_bc/td3_bc",
                    success_distance=task_config.TASK.SUCCESS_DISTANCE,
                    ignore_stop=algo_config.ignore_stop,
                )
                evaluations.append(eval_scores)
                print("---------------------------------------")
                print(
                    f"Evaluation over {algo_config.eval_episodes} episodes: "
                )
                for key in eval_scores:
                    print(f"{key}: {np.mean(eval_scores[key])}")
                print("---------------------------------------")
                torch.save(
                    trainer.state_dict(),
                    os.path.join(config.CHECKPOINT_FOLDER,
                                 f"checkpoint_{trainer.total_it}.pt"),
                )
                for key in eval_scores:
                    wandb.log(
                        {
                            f"eval/{key}_mean": np.mean(eval_scores[key]),
                            f"eval/{key}_std": np.std(eval_scores[key])
                        },
                        step=trainer.total_it,
                    )

        wandb.finish()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str,
                        default="habitat_corl/configs/td3_bc_pointnav.yaml")
    args = parser.parse_args()

    config = get_config(args.config)
    register_new_sensors(config.TASK_CONFIG)
    train(config)


if __name__ == "__main__":
    main()
