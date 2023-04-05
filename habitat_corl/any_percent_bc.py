import argparse
import os
import random
import uuid
from copy import deepcopy
from pathlib import Path
from typing import Any, Dict, List, Optional

import gym
import h5py
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import vaex
import wandb
from tqdm import tqdm

import habitat
from habitat.utils.visualizations.utils import images_to_video, \
    observations_to_image
from habitat_baselines.config.default import get_config
from habitat_corl.common.wrappers import wrap_env
from habitat_corl.replay_buffer import ReplayBuffer, get_input_dims
from habitat_corl.shortest_path_dataset import sample_transitions, \
    register_new_sensors, \
    calc_mean_std, get_stored_episodes, batch_generator
from habitat_corl.common.utils import set_seed, wandb_init, eval_actor, \
    get_goal, train_eval_split

TensorBatch = List[torch.Tensor]


def soft_update(target: nn.Module, source: nn.Module, tau: float):
    for target_param, source_param in zip(target.parameters(),
                                          source.parameters()):
        target_param.data.copy_(
            (1 - tau) * target_param.data + tau * source_param.data)


def keep_best_trajectories(
    config,
    groups: List[str],
    frac: float,
    discount: float,
    max_episode_steps: int = 1000,
):
    if frac == 1.0:
        return groups
    returns = []

    for idx, group in enumerate(groups):
        df = vaex.open(config.DATASET.SP_DATASET_PATH, group=group)
        rewards = deepcopy(df["reward"].values)
        dones = deepcopy(df["done"].values)
        cur_return = 0
        reward_scale = 1.0
        for i, (reward, done) in enumerate(zip(rewards, dones)):
            cur_return += reward_scale * reward
            reward_scale *= discount
            if done or i == max_episode_steps:
                returns.append(cur_return)
                break
        df.close()

    sort_ord = np.argsort(returns, axis=0)[::-1].reshape(-1)
    top_trajs = sort_ord[: int(frac * len(sort_ord))]
    return [groups[i] for i in top_trajs]


class Actor(nn.Module):
    def __init__(self, config, env):
        super(Actor, self).__init__()

        self.env = env
        self.config = config

        self.device = torch.device(
            "cuda:0" if torch.cuda.is_available() else "cpu")

        self.linear_input_size = get_input_dims(config)

        action_dim = env.action_space.n
        self.net = nn.Sequential(
            nn.Linear(self.linear_input_size, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, action_dim),
            nn.Tanh(),
        ).to(self.device)

    def forward(self, state) -> torch.Tensor:
        if isinstance(state, np.ndarray):
            state = torch.tensor(state, dtype=torch.float32, device=self.device)
        return self.net(state)

    @torch.no_grad()
    def act(self, state, device: str = "cpu") -> int:
        if isinstance(state, np.ndarray):
            state = torch.tensor(state, dtype=torch.float32, device=device)

        # make sure net is on the same device as state
        if device != self.device:
            self.net.to(device)

        action = self.forward(state)
        action = torch.argmax(action).item()

        # move net back to original device
        if device != self.device:
            self.net.to(self.device)

        return action


class BC:  # noqa
    def __init__(
        self,
        actor: nn.Module,
        actor_optimizer: torch.optim.Optimizer,
        discount: float = 0.99,
        device: str = "cpu",
    ):
        self.actor = actor
        self.actor_optimizer = actor_optimizer
        self.discount = discount

        self.total_it = 0
        self.device = device

    def train(self, batch: ReplayBuffer, used_inputs) -> Dict[str, float]:
        log_dict = {}
        self.total_it += 1

        state, action, _, _, _ = batch.to_tensor(state_keys=used_inputs)

        # Compute actor loss
        pi = self.actor(state)
        # actor_loss = F.mse_loss(pi, action)
        actor_loss = F.cross_entropy(pi, action.long())
        log_dict["actor_loss"] = actor_loss.item()
        # Optimize the actor
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        return log_dict

    def state_dict(self) -> Dict[str, Any]:
        return {
            "actor": self.actor.state_dict(),
            "actor_optimizer": self.actor_optimizer.state_dict(),
            "total_it": self.total_it,
        }

    def load_state_dict(self, state_dict: Dict[str, Any]):
        self.actor.load_state_dict(state_dict["actor"])
        self.actor_optimizer.load_state_dict(state_dict["actor_optimizer"])
        self.total_it = state_dict["total_it"]


def train(config):
    if "position" in config.MODEL.used_inputs:
        config.defrost()
        config.TASK_CONFIG = register_new_sensors(config.TASK_CONFIG)
        config.freeze()

    set_seed(config.SEED)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    mean_std = calc_mean_std(config.TASK_CONFIG,
                             used_inputs=config.MODEL.used_inputs)

    with wrap_env(
        habitat.Env(config=config.TASK_CONFIG),
        state_mean=mean_std["used"][0],
        state_std=mean_std["used"][1],
        used_inputs=config.MODEL.used_inputs,
        continuous=False,
        ignore_stop=config.RL.BC.ignore_stop,
    ) as env:
        # dataset = d4rl.qlearning_dataset(env)
        train_episodes, eval_episodes = train_eval_split(
            env=env,
            config=config,
            n_eval_episodes=config.RL.BC.eval_episodes,
            single_goal=config.RL.BC.single_goal,
        )
        train_episodes = keep_best_trajectories(
            config=config.TASK_CONFIG,
            groups=train_episodes,
            discount=config.RL.BC.DISCOUNT,
            frac=config.RL.BC.FRAC,
            max_episode_steps=config.RL.BC.MAX_TRAJ_LEN,
        )

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

        actor = Actor(config, env)
        actor_optimizer = torch.optim.Adam(actor.parameters(), lr=3e-4)

        kwargs = {
            "actor": actor,
            "actor_optimizer": actor_optimizer,
            "discount": config.RL.BC.DISCOUNT,
            "device": device,
        }

        print("---------------------------------------")
        print(
            f"Training BC, task: {config.BASE_TASK_CONFIG_PATH}, Seed: {seed}")
        print("---------------------------------------")

        # Initialize policy
        trainer = BC(**kwargs)

        if config.RL.BC.LOAD_MODEL != "":
            policy_file = Path(config.load_model)
            trainer.load_state_dict(torch.load(policy_file))
            actor = trainer.actor

        wandb_init(config)

        evaluations = []
        batch_gen = batch_generator(
            config.TASK_CONFIG,
            n_transitions=config.RL.BC.batch_size,
            groups=train_episodes,
            use_full_dataset=config.RL.BC.load_full_dataset,
            datasets=[f"state_{x}" for x in config.MODEL.used_inputs] + \
                     ["action"],
            continuous=False,
            ignore_stop=config.RL.BC.ignore_stop,
            single_goal=get_goal(config.RL.BC, eval_episodes)
        )
        for t in tqdm(range(int(config.NUM_UPDATES)), desc="Training"):
            batch = next(batch_gen)
            batch.normalize_states(mean_std)
            log_dict = trainer.train(batch,
                                     used_inputs=config.MODEL.used_inputs)
            wandb.log(log_dict, step=trainer.total_it)
            # Evaluate episode
            if (t + 1) % config.RL.BC.EVAL_FREQ == 0:
                print(f"Time steps: {t + 1}")
                eval_scores = eval_actor(
                    env,
                    actor,
                    device=device,
                    episodes=eval_episodes,
                    seed=config.SEED,
                    used_inputs=config.MODEL.used_inputs,
                    video=t == config.NUM_UPDATES - 1,
                    video_dir=config.VIDEO_DIR,
                    video_prefix="bc/bc",
                    ignore_stop=config.RL.BC.ignore_stop,
                    succes_distance=config.TASK_CONFIG.TASK.SUCCESS_DISTANCE,
                )
                evaluations.append(eval_scores)
                print("---------------------------------------")
                print(
                    f"Evaluation over {config.RL.BC.eval_episodes} episodes: "
                )
                for key in eval_scores:
                    print(f"{key}: {np.mean(eval_scores[key])}")
                print("---------------------------------------")
                torch.save(
                    trainer.state_dict(),
                    os.path.join(config.CHECKPOINT_FOLDER,
                                 f"checkpoint_{t}.pt"),
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
                        default="habitat_corl/configs/bc_pointnav.yaml")
    args = parser.parse_args()
    config = get_config(args.config)
    register_new_sensors(config.TASK_CONFIG)
    train(config)


if __name__ == "__main__":
    main()
