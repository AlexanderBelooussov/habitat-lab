# source: https://github.com/young-geng/CQL/tree/934b0e8354ca431d6c083c4e3a29df88d4b0a24d
# STRONG UNDER-PERFORMANCE ON PART OF ANTMAZE TASKS. BUT IN IQL PAPER IT WORKS SOMEHOW
# https://arxiv.org/pdf/2006.04779.pdf
import argparse
from typing import Any, Dict, List, Tuple
from copy import deepcopy
import os
from pathlib import Path

import numpy as np
import torch
import tqdm
from torch.distributions import Normal, TanhTransform, TransformedDistribution
import torch.nn as nn
import torch.nn.functional as F
import wandb

import habitat
from habitat_baselines.config.default import get_config
from habitat_corl.common.utils import train_eval_split, \
    eval_actor, wandb_init, set_seed, get_goal, remove_unreachable
from habitat_corl.common.wrappers import wrap_env
from habitat_corl.common.replay_buffer import get_input_dims, ReplayBuffer
from habitat_corl.common.shortest_path_dataset import register_new_sensors, \
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


def extend_and_repeat(tensor: torch.Tensor, dim: int, repeat: int) -> torch.Tensor:
    return tensor.unsqueeze(dim).repeat_interleave(repeat, dim=dim)


def init_module_weights(module: torch.nn.Module, orthogonal_init: bool = False):
    if isinstance(module, nn.Linear):
        if orthogonal_init:
            nn.init.orthogonal_(module.weight, gain=np.sqrt(2))
            nn.init.constant_(module.bias, 0.0)
        else:
            nn.init.xavier_uniform_(module.weight, gain=1e-2)


class ReparameterizedTanhGaussian(nn.Module):
    def __init__(
        self, log_std_min: float = -20.0, log_std_max: float = 2.0, no_tanh: bool = False
    ):
        super().__init__()
        self.log_std_min = log_std_min
        self.log_std_max = log_std_max
        self.no_tanh = no_tanh

    def log_prob(
        self, mean: torch.Tensor, log_std: torch.Tensor, sample: torch.Tensor
    ) -> torch.Tensor:
        log_std = torch.clamp(log_std, self.log_std_min, self.log_std_max)
        std = torch.exp(log_std)
        if self.no_tanh:
            action_distribution = Normal(mean, std)
        else:
            action_distribution = TransformedDistribution(
                Normal(mean, std), TanhTransform(cache_size=1)
            )
        return torch.sum(action_distribution.log_prob(sample), dim=-1)

    def forward(
        self, mean: torch.Tensor, log_std: torch.Tensor, deterministic: bool = False
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        log_std = torch.clamp(log_std, self.log_std_min, self.log_std_max)
        std = torch.exp(log_std)

        if self.no_tanh:
            action_distribution = Normal(mean, std)
        else:
            action_distribution = TransformedDistribution(
                Normal(mean, std), TanhTransform(cache_size=1)
            )

        if deterministic:
            action_sample = torch.tanh(mean)
        else:
            action_sample = action_distribution.rsample()

        log_prob = torch.sum(action_distribution.log_prob(action_sample), dim=-1)

        return action_sample, log_prob


class TanhGaussianPolicy(nn.Module):
    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        max_action: float,
        log_std_multiplier: float = 1.0,
        log_std_offset: float = -1.0,
        orthogonal_init: bool = False,
        no_tanh: bool = False,
    ):
        super().__init__()
        self.observation_dim = state_dim
        self.action_dim = action_dim
        self.max_action = max_action
        self.orthogonal_init = orthogonal_init
        self.no_tanh = no_tanh

        self.base_network = nn.Sequential(
            nn.Linear(state_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 2 * action_dim),
        )

        if orthogonal_init:
            self.base_network.apply(lambda m: init_module_weights(m, True))
        else:
            init_module_weights(self.base_network[-1], False)

        self.log_std_multiplier = Scalar(log_std_multiplier)
        self.log_std_offset = Scalar(log_std_offset)
        self.tanh_gaussian = ReparameterizedTanhGaussian(no_tanh=no_tanh)

    def log_prob(
        self, observations: torch.Tensor, actions: torch.Tensor
    ) -> torch.Tensor:
        if actions.ndim == 3:
            observations = extend_and_repeat(observations, 1, actions.shape[1])
        base_network_output = self.base_network(observations)
        mean, log_std = torch.split(base_network_output, self.action_dim, dim=-1)
        log_std = self.log_std_multiplier() * log_std + self.log_std_offset()
        return self.tanh_gaussian.log_prob(mean, log_std, actions)

    def forward(
        self,
        observations: torch.Tensor,
        deterministic: bool = False,
        repeat: bool = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if repeat is not None:
            observations = extend_and_repeat(observations, 1, repeat)
        base_network_output = self.base_network(observations)
        # print(f"base_network_output: {base_network_output}")
        mean, log_std = torch.split(base_network_output, self.action_dim, dim=-1)
        # check for nans
        if torch.isnan(mean).any():
            torch.set_printoptions(profile="full")
            # print("mean is nan")
            # print(f"mean: {mean}")
            print(f"base_network_output: {base_network_output}")
            print(f"action_dim: {self.action_dim}")
            print(f"observations: {observations}")
        if torch.isnan(log_std).any():
            print("log_std is nan")
            print(f"log_std: {log_std}")
        # print(f"log_std 1: {log_std}")
        log_std = self.log_std_multiplier() * log_std + self.log_std_offset()
        # check for nans
        if torch.isnan(log_std).any():
            print("log_std 2 is nan")
            print(f"log_std: {log_std}")
        actions, log_probs = self.tanh_gaussian(mean, log_std, deterministic)
        return self.max_action * actions, log_probs

    @torch.no_grad()
    def act(self, state: np.ndarray, device: str = "cpu"):
        state = torch.tensor(state.reshape(1, -1), device=device, dtype=torch.float32)
        with torch.no_grad():
            actions, _ = self(state, not self.training)
        return actions.cpu().data.numpy().flatten()


class FullyConnectedQFunction(nn.Module):
    def __init__(
        self,
        observation_dim: int,
        action_dim: int,
        orthogonal_init: bool = False,
    ):
        super().__init__()
        self.observation_dim = observation_dim
        self.action_dim = action_dim
        self.orthogonal_init = orthogonal_init

        self.network = nn.Sequential(
            nn.Linear(observation_dim + action_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 1),
        )
        if orthogonal_init:
            self.network.apply(lambda m: init_module_weights(m, True))
        else:
            init_module_weights(self.network[-1], False)

    def forward(self, observations: torch.Tensor, actions: torch.Tensor) -> torch.Tensor:
        multiple_actions = False
        batch_size = observations.shape[0]
        if actions.ndim == 3 and observations.ndim == 2:
            multiple_actions = True
            observations = extend_and_repeat(observations, 1, actions.shape[1]).reshape(
                -1, observations.shape[-1]
            )
            actions = actions.reshape(-1, actions.shape[-1])
        input_tensor = torch.cat([observations, actions], dim=-1)
        q_values = torch.squeeze(self.network(input_tensor), dim=-1)
        if multiple_actions:
            q_values = q_values.reshape(batch_size, -1)
        return q_values


class Scalar(nn.Module):
    def __init__(self, init_value: float):
        super().__init__()
        self.constant = nn.Parameter(torch.tensor(init_value, dtype=torch.float32))

    def forward(self) -> nn.Parameter:
        return self.constant


class ContinuousCQL:
    def __init__(
        self,
        critic_1,
        critic_1_optimizer,
        critic_2,
        critic_2_optimizer,
        actor,
        actor_optimizer,
        target_entropy: float,
        discount: float = 0.99,
        alpha_multiplier: float = 1.0,
        use_automatic_entropy_tuning: bool = True,
        backup_entropy: bool = False,
        policy_lr: bool = 3e-4,
        qf_lr: bool = 3e-4,
        soft_target_update_rate: float = 5e-3,
        bc_steps=100000,
        target_update_period: int = 1,
        cql_n_actions: int = 10,
        cql_importance_sample: bool = True,
        cql_lagrange: bool = False,
        cql_target_action_gap: float = -1.0,
        cql_temp: float = 1.0,
        cql_min_q_weight: float = 5.0,
        cql_max_target_backup: bool = False,
        cql_clip_diff_min: float = -np.inf,
        cql_clip_diff_max: float = np.inf,
        device: str = "cpu",
    ):
        super().__init__()

        self.discount = discount
        self.target_entropy = target_entropy
        self.alpha_multiplier = alpha_multiplier
        self.use_automatic_entropy_tuning = use_automatic_entropy_tuning
        self.backup_entropy = backup_entropy
        self.policy_lr = policy_lr
        self.qf_lr = qf_lr
        self.soft_target_update_rate = soft_target_update_rate
        self.bc_steps = bc_steps
        self.target_update_period = target_update_period
        self.cql_n_actions = cql_n_actions
        self.cql_importance_sample = cql_importance_sample
        self.cql_lagrange = cql_lagrange
        self.cql_target_action_gap = cql_target_action_gap
        self.cql_temp = cql_temp
        self.cql_min_q_weight = cql_min_q_weight
        self.cql_max_target_backup = cql_max_target_backup
        self.cql_clip_diff_min = cql_clip_diff_min
        self.cql_clip_diff_max = cql_clip_diff_max
        self._device = device

        self.total_it = 0

        self.critic_1 = critic_1
        self.critic_2 = critic_2

        self.target_critic_1 = deepcopy(self.critic_1).to(device)
        self.target_critic_2 = deepcopy(self.critic_2).to(device)

        self.actor = actor

        self.actor_optimizer = actor_optimizer
        self.critic_1_optimizer = critic_1_optimizer
        self.critic_2_optimizer = critic_2_optimizer

        if self.use_automatic_entropy_tuning:
            self.log_alpha = Scalar(0.0)
            self.alpha_optimizer = torch.optim.Adam(
                self.log_alpha.parameters(),
                lr=self.policy_lr,
            )
        else:
            self.log_alpha = None

        self.log_alpha_prime = Scalar(1.0)
        self.alpha_prime_optimizer = torch.optim.Adam(
            self.log_alpha_prime.parameters(),
            lr=self.qf_lr,
        )

        self.total_it = 0

    def update_target_network(self, soft_target_update_rate: float):
        soft_update(self.target_critic_1, self.critic_1, soft_target_update_rate)
        soft_update(self.target_critic_2, self.critic_2, soft_target_update_rate)

    def _alpha_and_alpha_loss(self, observations: torch.Tensor, log_pi: torch.Tensor):
        if self.use_automatic_entropy_tuning:
            alpha_loss = -(
                self.log_alpha() * (log_pi + self.target_entropy).detach()
            ).mean()
            alpha = self.log_alpha().exp() * self.alpha_multiplier
        else:
            alpha_loss = observations.new_tensor(0.0)
            alpha = observations.new_tensor(self.alpha_multiplier)
        return alpha, alpha_loss

    def _policy_loss(
        self,
        observations: torch.Tensor,
        actions: torch.Tensor,
        new_actions: torch.Tensor,
        alpha: torch.Tensor,
        log_pi: torch.Tensor,
    ) -> torch.Tensor:
        if self.total_it <= self.bc_steps:
            log_probs = self.actor.log_prob(observations, actions)
            policy_loss = (alpha * log_pi - log_probs).mean()
        else:
            q_new_actions = torch.min(
                self.critic_1(observations, new_actions),
                self.critic_2(observations, new_actions),
            )
            policy_loss = (alpha * log_pi - q_new_actions).mean()
        return policy_loss

    def _q_loss(
        self, observations, actions, next_observations, rewards, dones, alpha, log_dict
    ):
        q1_predicted = self.critic_1(observations, actions)
        q2_predicted = self.critic_2(observations, actions)

        if self.cql_max_target_backup:
            new_next_actions, next_log_pi = self.actor(
                next_observations, repeat=self.cql_n_actions
            )
            target_q_values, max_target_indices = torch.max(
                torch.min(
                    self.target_critic_1(next_observations, new_next_actions),
                    self.target_critic_2(next_observations, new_next_actions),
                ),
                dim=-1,
            )
            next_log_pi = torch.gather(
                next_log_pi, -1, max_target_indices.unsqueeze(-1)
            ).squeeze(-1)
        else:
            new_next_actions, next_log_pi = self.actor(next_observations)
            target_q_values = torch.min(
                self.target_critic_1(next_observations, new_next_actions),
                self.target_critic_2(next_observations, new_next_actions),
            )

        if self.backup_entropy:
            target_q_values = target_q_values - alpha * next_log_pi

        target_q_values = target_q_values.unsqueeze(-1)
        td_target = rewards + (1.0 - dones) * self.discount * target_q_values
        td_target = td_target.squeeze(-1)
        qf1_loss = F.mse_loss(q1_predicted, td_target.detach())
        qf2_loss = F.mse_loss(q2_predicted, td_target.detach())

        # CQL
        batch_size = actions.shape[0]
        action_dim = actions.shape[-1]
        cql_random_actions = actions.new_empty(
            (batch_size, self.cql_n_actions, action_dim), requires_grad=False
        ).uniform_(-1, 1)
        cql_current_actions, cql_current_log_pis = self.actor(
            observations, repeat=self.cql_n_actions
        )
        cql_next_actions, cql_next_log_pis = self.actor(
            next_observations, repeat=self.cql_n_actions
        )
        cql_current_actions, cql_current_log_pis = (
            cql_current_actions.detach(),
            cql_current_log_pis.detach(),
        )
        cql_next_actions, cql_next_log_pis = (
            cql_next_actions.detach(),
            cql_next_log_pis.detach(),
        )

        cql_q1_rand = self.critic_1(observations, cql_random_actions)
        cql_q2_rand = self.critic_2(observations, cql_random_actions)
        cql_q1_current_actions = self.critic_1(observations, cql_current_actions)
        cql_q2_current_actions = self.critic_2(observations, cql_current_actions)
        cql_q1_next_actions = self.critic_1(observations, cql_next_actions)
        cql_q2_next_actions = self.critic_2(observations, cql_next_actions)

        cql_cat_q1 = torch.cat(
            [
                cql_q1_rand,
                torch.unsqueeze(q1_predicted, 1),
                cql_q1_next_actions,
                cql_q1_current_actions,
            ],
            dim=1,
        )
        cql_cat_q2 = torch.cat(
            [
                cql_q2_rand,
                torch.unsqueeze(q2_predicted, 1),
                cql_q2_next_actions,
                cql_q2_current_actions,
            ],
            dim=1,
        )
        cql_std_q1 = torch.std(cql_cat_q1, dim=1)
        cql_std_q2 = torch.std(cql_cat_q2, dim=1)

        if self.cql_importance_sample:
            random_density = np.log(0.5**action_dim)
            cql_cat_q1 = torch.cat(
                [
                    cql_q1_rand - random_density,
                    cql_q1_next_actions - cql_next_log_pis.detach(),
                    cql_q1_current_actions - cql_current_log_pis.detach(),
                ],
                dim=1,
            )
            cql_cat_q2 = torch.cat(
                [
                    cql_q2_rand - random_density,
                    cql_q2_next_actions - cql_next_log_pis.detach(),
                    cql_q2_current_actions - cql_current_log_pis.detach(),
                ],
                dim=1,
            )

        cql_qf1_ood = torch.logsumexp(cql_cat_q1 / self.cql_temp, dim=1) * self.cql_temp
        cql_qf2_ood = torch.logsumexp(cql_cat_q2 / self.cql_temp, dim=1) * self.cql_temp

        """Subtract the log likelihood of data"""
        cql_qf1_diff = torch.clamp(
            cql_qf1_ood - q1_predicted,
            self.cql_clip_diff_min,
            self.cql_clip_diff_max,
        ).mean()
        cql_qf2_diff = torch.clamp(
            cql_qf2_ood - q2_predicted,
            self.cql_clip_diff_min,
            self.cql_clip_diff_max,
        ).mean()

        if self.cql_lagrange:
            alpha_prime = torch.clamp(
                torch.exp(self.log_alpha_prime()), min=0.0, max=1000000.0
            )
            cql_min_qf1_loss = (
                alpha_prime  # noqa
                * self.cql_min_q_weight  # noqa
                * (cql_qf1_diff - self.cql_target_action_gap)  # noqa
            )
            cql_min_qf2_loss = (
                alpha_prime  # noqa
                * self.cql_min_q_weight  # noqa
                * (cql_qf2_diff - self.cql_target_action_gap)  # noqa
            )

            self.alpha_prime_optimizer.zero_grad()
            alpha_prime_loss = (-cql_min_qf1_loss - cql_min_qf2_loss) * 0.5
            alpha_prime_loss.backward(retain_graph=True)
            self.alpha_prime_optimizer.step()
        else:
            cql_min_qf1_loss = cql_qf1_diff * self.cql_min_q_weight
            cql_min_qf2_loss = cql_qf2_diff * self.cql_min_q_weight
            alpha_prime_loss = observations.new_tensor(0.0)
            alpha_prime = observations.new_tensor(0.0)

        qf_loss = qf1_loss + qf2_loss + cql_min_qf1_loss + cql_min_qf2_loss

        log_dict.update(
            dict(
                qf1_loss=qf1_loss.item(),
                qf2_loss=qf2_loss.item(),
                alpha=alpha.item(),
                average_qf1=q1_predicted.mean().item(),
                average_qf2=q2_predicted.mean().item(),
                average_target_q=target_q_values.mean().item(),
            )
        )

        log_dict.update(
            dict(
                cql_std_q1=cql_std_q1.mean().item(),
                cql_std_q2=cql_std_q2.mean().item(),
                cql_q1_rand=cql_q1_rand.mean().item(),
                cql_q2_rand=cql_q2_rand.mean().item(),
                cql_min_qf1_loss=cql_min_qf1_loss.mean().item(),
                cql_min_qf2_loss=cql_min_qf2_loss.mean().item(),
                cql_qf1_diff=cql_qf1_diff.mean().item(),
                cql_qf2_diff=cql_qf2_diff.mean().item(),
                cql_q1_current_actions=cql_q1_current_actions.mean().item(),
                cql_q2_current_actions=cql_q2_current_actions.mean().item(),
                cql_q1_next_actions=cql_q1_next_actions.mean().item(),
                cql_q2_next_actions=cql_q2_next_actions.mean().item(),
                alpha_prime_loss=alpha_prime_loss.item(),
                alpha_prime=alpha_prime.item(),
            )
        )

        return qf_loss, alpha_prime, alpha_prime_loss

    def train(self, batch: ReplayBuffer, used_inputs: List[str]) -> Dict[str, float]:
        observations, actions, rewards, next_observations, dones = batch.to_tensor(
            device=self._device,
            state_keys=used_inputs,
            continuous_actions=True,
        )
        self.total_it += 1
        # print(f"observations: {observations}")
        new_actions, log_pi = self.actor(observations)

        alpha, alpha_loss = self._alpha_and_alpha_loss(observations, log_pi)

        """ Policy loss """
        policy_loss = self._policy_loss(
            observations, actions, new_actions, alpha, log_pi
        )

        log_dict = dict(
            log_pi=log_pi.mean().item(),
            policy_loss=policy_loss.item(),
            alpha_loss=alpha_loss.item(),
            alpha=alpha.item(),
        )

        """ Q function loss """
        qf_loss, alpha_prime, alpha_prime_loss = self._q_loss(
            observations, actions, next_observations, rewards, dones, alpha, log_dict
        )

        if self.use_automatic_entropy_tuning:
            self.alpha_optimizer.zero_grad()
            alpha_loss.backward()
            self.alpha_optimizer.step()

        self.actor_optimizer.zero_grad()
        policy_loss.backward()
        self.actor_optimizer.step()

        self.critic_1_optimizer.zero_grad()
        self.critic_2_optimizer.zero_grad()
        qf_loss.backward(retain_graph=True)
        self.critic_1_optimizer.step()
        self.critic_2_optimizer.step()

        if self.total_it % self.target_update_period == 0:
            self.update_target_network(self.soft_target_update_rate)

        return log_dict

    def state_dict(self) -> Dict[str, Any]:
        return {
            "actor": self.actor.state_dict(),
            "critic1": self.critic_1.state_dict(),
            "critic2": self.critic_2.state_dict(),
            "critic1_target": self.target_critic_1.state_dict(),
            "critic2_target": self.target_critic_2.state_dict(),
            "critic_1_optimizer": self.critic_1_optimizer.state_dict(),
            "critic_2_optimizer": self.critic_2_optimizer.state_dict(),
            "actor_optim": self.actor_optimizer.state_dict(),
            "sac_log_alpha": self.log_alpha,
            "sac_log_alpha_optim": self.alpha_optimizer.state_dict(),
            "cql_log_alpha": self.log_alpha_prime,
            "cql_log_alpha_optim": self.alpha_prime_optimizer.state_dict(),
            "total_it": self.total_it,
        }

    def load_state_dict(self, state_dict: Dict[str, Any]):
        self.actor.load_state_dict(state_dict=state_dict["actor"])
        self.critic_1.load_state_dict(state_dict=state_dict["critic1"])
        self.critic_2.load_state_dict(state_dict=state_dict["critic2"])

        self.target_critic_1.load_state_dict(state_dict=state_dict["critic1_target"])
        self.target_critic_2.load_state_dict(state_dict=state_dict["critic2_target"])

        self.critic_1_optimizer.load_state_dict(
            state_dict=state_dict["critic_1_optimizer"]
        )
        self.critic_2_optimizer.load_state_dict(
            state_dict=state_dict["critic_2_optimizer"]
        )
        self.actor_optimizer.load_state_dict(state_dict=state_dict["actor_optim"])

        self.log_alpha = state_dict["sac_log_alpha"]
        self.alpha_optimizer.load_state_dict(
            state_dict=state_dict["sac_log_alpha_optim"]
        )

        self.log_alpha_prime = state_dict["cql_log_alpha"]
        self.alpha_prime_optimizer.load_state_dict(
            state_dict=state_dict["cql_log_alpha_optim"]
        )
        self.total_it = state_dict["total_it"]


def init_trainer(algo_config, state_dim, action_dim, device, max_action, action_space_shape, **kwargs):
    critic_1 = FullyConnectedQFunction(state_dim, action_dim,
                                       algo_config.orthogonal_init).to(
        device
    )
    critic_2 = FullyConnectedQFunction(state_dim, action_dim,
                                       algo_config.orthogonal_init).to(
        device
    )
    critic_1_optimizer = torch.optim.Adam(list(critic_1.parameters()),
                                          algo_config.qf_lr)
    critic_2_optimizer = torch.optim.Adam(list(critic_2.parameters()),
                                          algo_config.qf_lr)

    actor = TanhGaussianPolicy(
        state_dim, action_dim, max_action,
        orthogonal_init=algo_config.orthogonal_init
    ).to(device)
    actor_optimizer = torch.optim.Adam(actor.parameters(),
                                       algo_config.policy_lr)

    kwargs = {
        "critic_1": critic_1,
        "critic_2": critic_2,
        "critic_1_optimizer": critic_1_optimizer,
        "critic_2_optimizer": critic_2_optimizer,
        "actor": actor,
        "actor_optimizer": actor_optimizer,
        "discount": algo_config.discount,
        "soft_target_update_rate": algo_config.soft_target_update_rate,
        "device": device,
        # CQL
        "target_entropy": -np.prod(action_space_shape).item(),
        "alpha_multiplier": algo_config.alpha_multiplier,
        "use_automatic_entropy_tuning": algo_config.use_automatic_entropy_tuning,
        "backup_entropy": algo_config.backup_entropy,
        "policy_lr": algo_config.policy_lr,
        "qf_lr": algo_config.qf_lr,
        "bc_steps": algo_config.bc_steps,
        "target_update_period": algo_config.target_update_period,
        "cql_n_actions": algo_config.cql_n_actions,
        "cql_importance_sample": algo_config.cql_importance_sample,
        "cql_lagrange": algo_config.cql_lagrange,
        "cql_target_action_gap": algo_config.cql_target_action_gap,
        "cql_temp": algo_config.cql_temp,
        "cql_min_q_weight": algo_config.cql_min_q_weight,
        "cql_max_target_backup": algo_config.cql_max_target_backup,
        "cql_clip_diff_min": algo_config.cql_clip_diff_min,
        "cql_clip_diff_max": algo_config.cql_clip_diff_max,
    }

    # Initialize actor
    trainer = ContinuousCQL(**kwargs)
    return trainer

def train(config):
    algo_config = config.RL.CQL
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
        model_config=config.MODEL,
        continuous=algo_config.continuous,
        ignore_stop=algo_config.ignore_stop,
        turn_angle=task_config.SIMULATOR.TURN_ANGLE,
    ) as env:
        state_dim = get_input_dims(config)
        action_dim = env.action_space.shape[0]
        action_space_shape = env.action_space.shape

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

        print("---------------------------------------")
        print(f"Training CQL, Scene: {task_config.DATASET.CONTENT_SCENES}"
              f", Seed: {seed}, ignore_stop: {algo_config.ignore_stop}, "
              f"single_goal: {algo_config.single_goal}")
        print("---------------------------------------")

        trainer = init_trainer(algo_config, state_dim, action_dim, device,
                               max_action, action_space_shape)

        if algo_config.load_model != "":
            policy_file = Path(algo_config.load_model)
            trainer.load_state_dict(torch.load(policy_file))
        actor = trainer.actor

        evaluations = []
        for t in tqdm.tqdm(range(int(algo_config.max_timesteps))):
            batch = next(batch_gen)
            log_dict = trainer.train(batch,
                                     used_inputs=config.MODEL.used_inputs)
            try:
                wandb.log(log_dict, step=trainer.total_it)
            except:
                pass
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
                    video_prefix="cql/cql",
                    success_distance=task_config.TASK.SUCCESS_DISTANCE,
                    ignore_stop=algo_config.ignore_stop,
                )
                eval_scores = remove_unreachable(eval_scores)
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
                    try:
                        wandb.log(
                            {
                                f"eval/{key}_mean": np.mean(eval_scores[key]),
                                f"eval/{key}_std": np.std(eval_scores[key])
                            },
                            step=trainer.total_it,
                        )
                    except:
                        pass

        wandb.finish()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str,
                        default="habitat_corl/configs/cql_pointnav.yaml")
    args = parser.parse_args()

    config = get_config(args.config)
    register_new_sensors(config.TASK_CONFIG)
    train(config)


if __name__ == "__main__":
    main()
