# source: https://github.com/gwthomas/IQL-PyTorch
# https://arxiv.org/pdf/2110.06169.pdf
import argparse
from typing import Any, Callable, Dict, List, Optional, Tuple, Union
import copy
from dataclasses import asdict, dataclass
import os
from pathlib import Path
import random
import uuid

import gym
import numpy as np
import torch
from torch.distributions import MultivariateNormal
import torch.nn as nn
import torch.nn.functional as F
from torch.optim.lr_scheduler import CosineAnnealingLR
import wandb
from tqdm import trange

import habitat
from habitat_baselines.config.default import get_config
from habitat_corl.common.wrappers import wrap_env
from habitat_corl.common.utils import set_seed, wandb_init, train_eval_split, \
    get_goal, eval_actor
from habitat_corl.replay_buffer import ReplayBuffer, get_input_dims
from habitat_corl.shortest_path_dataset import register_new_sensors, \
    calc_mean_std, batch_generator

TensorBatch = List[torch.Tensor]


EXP_ADV_MAX = 100.0
LOG_STD_MIN = -5.0
LOG_STD_MAX = 2.0

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


def asymmetric_l2_loss(u: torch.Tensor, tau: float) -> torch.Tensor:
    return torch.mean(torch.abs(tau - (u < 0).float()) * u**2)


class Squeeze(nn.Module):
    def __init__(self, dim=-1):
        super().__init__()
        self.dim = dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x.squeeze(dim=self.dim)


class MLP(nn.Module):
    def __init__(
        self,
        dims,
        activation_fn: Callable[[], nn.Module] = nn.ReLU,
        output_activation_fn: Callable[[], nn.Module] = None,
        squeeze_output: bool = False,
    ):
        super().__init__()
        n_dims = len(dims)
        if n_dims < 2:
            raise ValueError("MLP requires at least two dims (input and output)")

        layers = []
        for i in range(n_dims - 2):
            layers.append(nn.Linear(dims[i], dims[i + 1]))
            layers.append(activation_fn())
        layers.append(nn.Linear(dims[-2], dims[-1]))
        if output_activation_fn is not None:
            layers.append(output_activation_fn())
        if squeeze_output:
            if dims[-1] != 1:
                raise ValueError("Last dim must be 1 when squeezing")
            layers.append(Squeeze(-1))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class GaussianPolicy(nn.Module):
    def __init__(
        self,
        state_dim: int,
        act_dim: int,
        max_action: float,
        hidden_dim: int = 256,
        n_hidden: int = 2,
    ):
        super().__init__()
        self.net = MLP([state_dim, *([hidden_dim] * n_hidden), act_dim])
        self.log_std = nn.Parameter(torch.zeros(act_dim, dtype=torch.float32))
        self.max_action = max_action

    def forward(self, obs: torch.Tensor) -> MultivariateNormal:
        mean = self.net(obs)
        std = torch.exp(self.log_std.clamp(LOG_STD_MIN, LOG_STD_MAX))
        scale_tril = torch.diag(std)
        return MultivariateNormal(mean, scale_tril=scale_tril)

    @torch.no_grad()
    def act(self, state: np.ndarray, device: str = "cpu"):
        state = torch.tensor(state.reshape(1, -1), device=device, dtype=torch.float32)
        dist = self(state)
        action = dist.mean if not self.training else dist.sample()
        action = torch.clamp(self.max_action * action, -self.max_action, self.max_action)
        return action.cpu().data.numpy().flatten()


class DeterministicPolicy(nn.Module):
    def __init__(
        self,
        state_dim: int,
        act_dim: int,
        max_action: float,
        hidden_dim: int = 256,
        n_hidden: int = 2,
    ):
        super().__init__()
        self.net = MLP(
            [state_dim, *([hidden_dim] * n_hidden), act_dim],
            output_activation_fn=nn.Tanh,
        )
        self.max_action = max_action

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        return self.net(obs)

    @torch.no_grad()
    def act(self, state: np.ndarray, device: str = "cpu"):
        state = torch.tensor(state.reshape(1, -1), device=device, dtype=torch.float32)
        return (
            torch.clamp(self(state) * self.max_action, -self.max_action, self.max_action)
            .cpu()
            .data.numpy()
            .flatten()
        )


class TwinQ(nn.Module):
    def __init__(
        self, state_dim: int, action_dim: int, hidden_dim: int = 256, n_hidden: int = 2
    ):
        super().__init__()
        dims = [state_dim + action_dim, *([hidden_dim] * n_hidden), 1]
        self.q1 = MLP(dims, squeeze_output=True)
        self.q2 = MLP(dims, squeeze_output=True)

    def both(
        self, state: torch.Tensor, action: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        sa = torch.cat([state, action], 1)
        return self.q1(sa), self.q2(sa)

    def forward(self, state: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        return torch.min(*self.both(state, action))


class ValueFunction(nn.Module):
    def __init__(self, state_dim: int, hidden_dim: int = 256, n_hidden: int = 2):
        super().__init__()
        dims = [state_dim, *([hidden_dim] * n_hidden), 1]
        self.v = MLP(dims, squeeze_output=True)

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        return self.v(state)


class ImplicitQLearning:
    def __init__(
        self,
        max_action: float,
        actor: nn.Module,
        actor_optimizer: torch.optim.Optimizer,
        q_network: nn.Module,
        q_optimizer: torch.optim.Optimizer,
        v_network: nn.Module,
        v_optimizer: torch.optim.Optimizer,
        iql_tau: float = 0.7,
        beta: float = 3.0,
        max_steps: int = 1000000,
        discount: float = 0.99,
        tau: float = 0.005,
        device: str = "cpu",
    ):
        self.max_action = max_action
        self.qf = q_network
        self.q_target = copy.deepcopy(self.qf).requires_grad_(False).to(device)
        self.vf = v_network
        self.actor = actor
        self.v_optimizer = v_optimizer
        self.q_optimizer = q_optimizer
        self.actor_optimizer = actor_optimizer
        self.actor_lr_schedule = CosineAnnealingLR(self.actor_optimizer, max_steps)
        self.iql_tau = iql_tau
        self.beta = beta
        self.discount = discount
        self.tau = tau

        self.total_it = 0
        self.device = device

    def _update_v(self, observations, actions, log_dict) -> torch.Tensor:
        # Update value function
        with torch.no_grad():
            target_q = self.q_target(observations, actions)

        v = self.vf(observations)
        adv = target_q - v
        v_loss = asymmetric_l2_loss(adv, self.iql_tau)
        log_dict["value_loss"] = v_loss.item()
        self.v_optimizer.zero_grad(set_to_none=True)
        v_loss.backward()
        self.v_optimizer.step()
        return adv

    def _update_q(
        self,
        next_v,
        observations,
        actions,
        rewards,
        terminals,
        log_dict,
    ):
        targets = rewards + (1.0 - terminals.float()) * self.discount * next_v.detach()
        qs = self.qf.both(observations, actions)
        q_loss = sum(F.mse_loss(q, targets) for q in qs) / len(qs)
        log_dict["q_loss"] = q_loss.item()
        self.q_optimizer.zero_grad(set_to_none=True)
        q_loss.backward()
        self.q_optimizer.step()

        # Update target Q network
        soft_update(self.q_target, self.qf, self.tau)

    def _update_policy(self, adv, observations, actions, log_dict):
        exp_adv = torch.exp(self.beta * adv.detach()).clamp(max=EXP_ADV_MAX)
        policy_out = self.actor(observations)
        if isinstance(policy_out, torch.distributions.Distribution):
            bc_losses = -policy_out.log_prob(actions)
        elif torch.is_tensor(policy_out):
            if policy_out.shape != actions.shape:
                raise RuntimeError("Actions shape missmatch")
            bc_losses = torch.sum((policy_out - actions) ** 2, dim=1)
        else:
            raise NotImplementedError
        policy_loss = torch.mean(exp_adv * bc_losses)
        log_dict["actor_loss"] = policy_loss.item()
        self.actor_optimizer.zero_grad(set_to_none=True)
        policy_loss.backward()
        self.actor_optimizer.step()
        self.actor_lr_schedule.step()

    def train(self, batch: ReplayBuffer, used_inputs: List[str]) -> Dict[str, float]:
        self.total_it += 1
        (
            observations,
            actions,
            rewards,
            next_observations,
            dones,
        ) = batch.to_tensor(
            device=self.device,
            state_keys=used_inputs,
            continuous_actions=True,
        )
        log_dict = {}

        with torch.no_grad():
            next_v = self.vf(next_observations)
        # Update value function
        adv = self._update_v(observations, actions, log_dict)
        rewards = rewards.squeeze(dim=-1)
        dones = dones.squeeze(dim=-1)
        # Update Q function
        self._update_q(next_v, observations, actions, rewards, dones, log_dict)
        # Update actor
        self._update_policy(adv, observations, actions, log_dict)

        return log_dict

    def state_dict(self) -> Dict[str, Any]:
        return {
            "qf": self.qf.state_dict(),
            "q_optimizer": self.q_optimizer.state_dict(),
            "vf": self.vf.state_dict(),
            "v_optimizer": self.v_optimizer.state_dict(),
            "actor": self.actor.state_dict(),
            "actor_optimizer": self.actor_optimizer.state_dict(),
            "actor_lr_schedule": self.actor_lr_schedule.state_dict(),
            "total_it": self.total_it,
        }

    def load_state_dict(self, state_dict: Dict[str, Any]):
        self.qf.load_state_dict(state_dict["qf"])
        self.q_optimizer.load_state_dict(state_dict["q_optimizer"])
        self.q_target = copy.deepcopy(self.qf)

        self.vf.load_state_dict(state_dict["vf"])
        self.v_optimizer.load_state_dict(state_dict["v_optimizer"])

        self.actor.load_state_dict(state_dict["actor"])
        self.actor_optimizer.load_state_dict(state_dict["actor_optimizer"])
        self.actor_lr_schedule.load_state_dict(state_dict["actor_lr_schedule"])

        self.total_it = state_dict["total_it"]


def train(config):
    algo_config = config.RL.IQL
    task_config = config.TASK_CONFIG
    set_seed(config.SEED)
    wandb_init(config)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    mean_std = calc_mean_std(config.TASK_CONFIG,
                             used_inputs=config.MODEL.used_inputs)

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
            single_goal=get_goal(algo_config, eval_episodes)
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

        q_network = TwinQ(state_dim, action_dim).to(device)
        v_network = ValueFunction(state_dim).to(device)
        actor = (
            DeterministicPolicy(state_dim, action_dim, max_action)
            if algo_config.iql_deterministic
            else GaussianPolicy(state_dim, action_dim, max_action)
        ).to(device)
        v_optimizer = torch.optim.Adam(v_network.parameters(), lr=3e-5)
        q_optimizer = torch.optim.Adam(q_network.parameters(), lr=3e-5)
        actor_optimizer = torch.optim.Adam(actor.parameters(), lr=3e-5)

        kwargs = {
            "max_action": max_action,
            "actor": actor,
            "actor_optimizer": actor_optimizer,
            "q_network": q_network,
            "q_optimizer": q_optimizer,
            "v_network": v_network,
            "v_optimizer": v_optimizer,
            "discount": algo_config.discount,
            "tau": algo_config.tau,
            "device": device,
            # IQL
            "beta": algo_config.beta,
            "iql_tau": algo_config.iql_tau,
            "max_steps": algo_config.max_timesteps,
        }

        print("---------------------------------------")
        print(f"Training IQL, Scene: {task_config.DATASET.CONTENT_SCENES}"
              f", Seed: {seed}, ignore_stop: {algo_config.ignore_stop}, "
              f"single_goal: {algo_config.single_goal}")
        print("---------------------------------------")

        # Initialize actor
        trainer = ImplicitQLearning(**kwargs)

        wandb.watch(trainer.actor, log="all")
        wandb.watch(trainer.qf, log="all")
        wandb.watch(trainer.vf, log="all")

        if algo_config.load_model != "":
            policy_file = Path(algo_config.load_model)
            trainer.load_state_dict(torch.load(policy_file))
            actor = trainer.actor

        evaluations = []
        for t in trange(int(algo_config.max_timesteps)):
            batch = next(batch_gen)
            batch.normalize_states(mean_std)
            batch.to_tensor(device=device)
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
                    video_prefix="iql/iql",
                    succes_distance=task_config.TASK.SUCCESS_DISTANCE,
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
