# Inspired by:
# 1. paper for SAC-N: https://arxiv.org/abs/2110.01548
# 2. implementation: https://github.com/snu-mllab/EDAC
# 3. SAC-Discrete: Soft Actor-Critic for Discrete Action Settings
# 4. Implementation: https://github.com/p-christ/Deep-Reinforcement-Learning-Algorithms-with-PyTorch/blob/master/agents/actor_critic_agents/SAC_Discrete.py
import argparse
import faulthandler
import math
import os
from copy import deepcopy
# The only difference from the original implementation:
# default pytorch weight initialization,
# without custom rlkit init & uniform init for last layers.
from typing import Any, Dict, List, Tuple

import numpy as np
import torch
import torch.nn as nn
import wandb
from tqdm import trange

import habitat
from habitat_baselines.config.default import get_config
from habitat_corl.common.utils import train_eval_split, \
    set_seed, wandb_init, eval_actor, get_goal, remove_unreachable
from habitat_corl.common.wrappers import wrap_env
from habitat_corl.common.replay_buffer import ReplayBuffer, get_input_dims
from habitat_corl.common.shortest_path_dataset import register_new_sensors, \
    calc_mean_std, batch_generator

# general utils
TensorBatch = List[torch.Tensor]


def soft_update(target: nn.Module, source: nn.Module, tau: float):
    for target_param, source_param in zip(target.parameters(),
                                          source.parameters()):
        target_param.data.copy_(
            (1 - tau) * target_param.data + tau * source_param.data)


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
            nn.Linear(hidden_dim, action_dim),  # changed
            nn.Softmax()  # changed
        )

        # init as in the EDAC paper
        for layer in self.trunk[::2]:
            torch.nn.init.constant_(layer.bias, 0.1)

        self.action_dim = action_dim
        self.max_action = max_action

    def forward(
        self,
        state: torch.Tensor,
        deterministic: bool = False,
        need_log_prob: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:  # changed
        prob = self.trunk(state)
        if deterministic:
            action = prob.argmax(dim=-1)
        else:
            action = torch.distributions.Categorical(prob).sample()

        z = prob == 0.0
        z = z.float() * 1e-8

        log_prob = torch.log(prob + z)

        return action, prob, log_prob

    @torch.no_grad()
    def act(self, state: np.ndarray, device: str) -> np.ndarray:  # changed
        deterministic = not self.training
        state = torch.tensor(state, device=device, dtype=torch.float32)
        action, _, _ = self(state, deterministic=deterministic)
        action = action.cpu().numpy()

        return action


class VectorizedCritic(nn.Module):
    def __init__(
        self, state_dim: int, action_dim: int, hidden_dim: int,
        num_critics: int
    ):
        super().__init__()
        self.critic = nn.Sequential(
            VectorizedLinear(state_dim, hidden_dim, num_critics),  # changed
            nn.ReLU(),
            VectorizedLinear(hidden_dim, hidden_dim, num_critics),
            nn.ReLU(),
            VectorizedLinear(hidden_dim, hidden_dim, num_critics),
            nn.ReLU(),
            VectorizedLinear(hidden_dim, action_dim, num_critics),  # changed
        )
        # init as in the EDAC paper
        for layer in self.critic[::2]:
            torch.nn.init.constant_(layer.bias, 0.1)

        torch.nn.init.uniform_(self.critic[-1].weight, -3e-3, 3e-3)
        torch.nn.init.uniform_(self.critic[-1].bias, -3e-3, 3e-3)

        self.num_critics = num_critics

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        # [num_critics, batch_size, state_dim + action_dim]
        state = state.unsqueeze(0).repeat_interleave(
            self.num_critics, dim=0
        )
        # [num_critics, batch_size]
        q_values = self.critic(state).squeeze(-1)
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
        # self.target_entropy = -float(self.actor.action_dim)
        self.target_entropy = -np.log((1/self.actor.action_dim)) * 0.98
        self.log_alpha = torch.tensor(
            [0.0], dtype=torch.float32, device=self.device, requires_grad=True
        )
        self.alpha_optimizer = torch.optim.Adam([self.log_alpha],
                                                lr=alpha_learning_rate)
        self.alpha = self.log_alpha.exp().detach()

    def _alpha_loss(self, state: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            action, action_prob, action_log_prob = self.actor(
                state,
                need_log_prob=True
            )

        term2 = (-self.log_alpha * (
            action_log_prob + self.target_entropy))
        loss1 = (action_prob * term2).sum(dim=-1).mean()  # changed
        loss2 = term2.mean()

        return loss2

    def _actor_loss(self, state: torch.Tensor) -> Tuple[
        torch.Tensor, float, float]:
        action, action_prob, action_log_prob = self.actor(state,
                                                          need_log_prob=True)
        q_value_dist = self.critic(state)
        assert q_value_dist.shape[0] == self.critic.num_critics
        q_value_min = q_value_dist.min(0).values
        # needed for logging
        q_value_std = q_value_dist.std(0).mean().item()
        batch_entropy = -action_log_prob.mean().item()

        assert action_log_prob.shape == q_value_min.shape
        inside = self.alpha * action_log_prob - q_value_min  # changed
        loss = (action_prob * inside).sum(dim=-1).mean()  # changed

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
            next_action, next_action_prob, next_action_log_prob = self.actor(
                next_state, need_log_prob=True
            )
            q_next = self.target_critic(next_state)
            min_qf_next_target = \
                next_action_prob * (
                    torch.min(q_next, dim=0)[0] -
                    self.alpha * next_action_log_prob
                )  # changed
            q_next = min_qf_next_target.sum(dim=1).unsqueeze(-1)
            assert q_next.shape == done.shape == reward.shape
            q_target = reward + self.gamma * (1 - done) * q_next

        q_values = self.critic(state)
        # choose q_values for actions
        reshaped = action.view(1, -1, 1).repeat_interleave(
            self.critic.num_critics, dim=0
        )  # changed
        q_values = q_values.gather(2, reshaped.long()).squeeze(-1)  # changed
        # [ensemble_size, batch_size] - [1, batch_size]
        loss = ((q_values - q_target.view(1, -1)) ** 2).mean(dim=1).sum(dim=0)  # unchanged

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

        update_info = {
            "alpha_loss": alpha_loss.item(),
            "critic_loss": critic_loss.item(),
            "actor_loss": actor_loss.item(),
            "batch_entropy": actor_batch_entropy,
            "alpha": self.alpha.item(),
            "q_policy_std": q_policy_std,
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


def init_trainer(algo_config, state_dim, action_dim, device, **kwargs):
    # Actor & Critic setup
    actor = Actor(state_dim, action_dim, algo_config.hidden_dim,
                  algo_config.max_action)
    actor.to(device)
    actor_optimizer = torch.optim.Adam(actor.parameters(),
                                       lr=algo_config.actor_learning_rate)
    critic = VectorizedCritic(
        state_dim, action_dim, algo_config.hidden_dim,
        algo_config.num_critics
    )
    critic.to(device)
    critic_optimizer = torch.optim.Adam(
        critic.parameters(), lr=algo_config.critic_learning_rate
    )

    trainer = SACN(
        actor=actor,
        actor_optimizer=actor_optimizer,
        critic=critic,
        critic_optimizer=critic_optimizer,
        gamma=algo_config.gamma,
        tau=algo_config.tau,
        alpha_learning_rate=algo_config.alpha_learning_rate,
        device=device,
    )
    return trainer

def train(config):
    algo_config = config.RL.SAC_N
    set_seed(config.SEED,
             deterministic_torch=algo_config.deterministic_torch)
    wandb_init(config)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    mean_std = calc_mean_std(config.TASK_CONFIG,
                             used_inputs=config.MODEL.used_inputs)

    # data, evaluation, env setup
    with wrap_env(
        habitat.Env(config=config.TASK_CONFIG),
        state_mean=mean_std["used"][0],
        state_std=mean_std["used"][1],
        model_config=config.MODEL,
        continuous=False,
        ignore_stop=algo_config.ignore_stop,
        turn_angle=config.TASK_CONFIG.SIMULATOR.TURN_ANGLE,
    ) as env:
        state_dim = get_input_dims(config)
        action_dim = env.action_space.n

        train_episodes, eval_episodes = train_eval_split(
            env=env,
            config=config,
            n_eval_episodes=algo_config.eval_episodes,
            single_goal=algo_config.single_goal,
        )

        trainer = init_trainer(algo_config, state_dim, action_dim, device)
        actor = trainer.actor

        wandb.watch(trainer.actor, log="all")
        wandb.watch(trainer.critic, log="all")

        # saving config to the checkpoint
        if hasattr(config,
                   "CHECKPOINT_FOLDER") and config.CHECKPOINT_FOLDER is not None:
            print(f"Checkpoints path: {config.CHECKPOINT_FOLDER}")
            os.makedirs(config.CHECKPOINT_FOLDER, exist_ok=True)
            with open(os.path.join(config.CHECKPOINT_FOLDER, "config.yaml"),
                      "w") as f:
                f.write(config.dump())

        total_updates = 0
        evaluations = []
        batch_gen = batch_generator(
            config.TASK_CONFIG,
            n_transitions=algo_config.batch_size,
            groups=train_episodes,
            # use_full_dataset=False,
            use_full_dataset=algo_config.load_full_dataset,
            # n_batches=algo_config.eval_every * algo_config.num_updates_on_epoch,
            n_batches=algo_config.num_updates_on_epoch,
            datasets=[f"state_{x}" for x in config.MODEL.used_inputs] + \
                     [f"next_state_{x}" for x in config.MODEL.used_inputs] + \
                     ["action", "reward", "done"],
            ignore_stop=algo_config.ignore_stop,
            continuous=False,
            single_goal=get_goal(algo_config, eval_episodes)
        )
        for epoch in trange(algo_config.num_epochs, desc="Training"):
            # training
            for _ in trange(algo_config.num_updates_on_epoch, desc="Epoch",
                            leave=False):
                batch = next(batch_gen)
                batch.normalize_states(mean_std)
                batch.to_tensor(device=device)
                update_info = trainer.update(batch,
                                             used_inputs=config.MODEL.used_inputs)

                if total_updates % algo_config.log_every == 0:
                    wandb.log(
                        {"epoch": epoch, **update_info},
                        step=total_updates
                    )

                total_updates += 1

            # evaluation
            t = total_updates
            if (epoch + 1) % algo_config.eval_every == 0 or epoch == algo_config.num_epochs - 1:
                print(f"Time steps: {t + 1}")
                eval_scores = eval_actor(
                    env,
                    actor,
                    device=device,
                    episodes=eval_episodes,
                    seed=config.SEED,
                    used_inputs=config.MODEL.used_inputs,
                    # video=True,
                    video=epoch == algo_config.num_epochs - 1,
                    video_dir=config.VIDEO_DIR,
                    video_prefix="sac_n_d/sac_n_d",
                    success_distance=config.TASK_CONFIG.TASK.SUCCESS_DISTANCE,
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
    register_new_sensors(config.TASK_CONFIG)
    train(config)


if __name__ == "__main__":
    faulthandler.enable()
    main()
