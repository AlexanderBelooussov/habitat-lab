# source: https://github.com/sfujim/TD3_BC
# https://arxiv.org/pdf/2106.06860.pdf
import argparse
from typing import List

import numpy as np
import torch
import torch.nn as nn
import wandb

import habitat
from habitat_baselines.config.default import get_config
from habitat_corl.common.utils import train_eval_split, \
    eval_actor, set_seed
from habitat_corl.common.wrappers import wrap_env
from habitat_corl.common.shortest_path_dataset import register_new_sensors

TensorBatch = List[torch.Tensor]


class Actor(nn.Module):
    def __init__(self, action_dim: int):
        super(Actor, self).__init__()
        self.n_actions = action_dim

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

    @torch.no_grad()
    def act(self, state: np.ndarray, device: str = "cpu") -> np.ndarray:
        action = np.random.choice(self.n_actions)
        return action


def train(config):
    algo_config = config.RL.RANDOM
    task_config = config.TASK_CONFIG
    set_seed(config.SEED)

    with wrap_env(
        habitat.Env(config=task_config),
        model_config=config.MODEL,
        continuous=algo_config.continuous,
        ignore_stop=algo_config.ignore_stop,
        turn_angle=task_config.SIMULATOR.TURN_ANGLE,
    ) as env:
        action_dim = env.action_space.n

        train_episodes, eval_episodes = train_eval_split(
            env=env,
            config=config,
            n_eval_episodes=algo_config.eval_episodes,
            single_goal=algo_config.single_goal,
        )

        # Set seeds
        seed = config.SEED
        set_seed(seed, env)

        actor = Actor(action_dim)

        print("---------------------------------------")
        print(f"Random Agent, Scene: {task_config.DATASET.CONTENT_SCENES}"
              f", Seed: {seed}, ignore_stop: {algo_config.ignore_stop}, "
              f"single_goal: {algo_config.single_goal}")
        print("---------------------------------------")

        evaluations = []
        eval_scores = eval_actor(
            env,
            actor,
            device="cpu",
            episodes=eval_episodes,
            seed=config.SEED,
            used_inputs=config.MODEL.used_inputs,
            video=True,
            video_dir=config.VIDEO_DIR,
            video_prefix="random/random",
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

        wandb.finish()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str,
                        default="habitat_corl/configs/random_pointnav.yaml")
    args = parser.parse_args()

    config = get_config(args.config)
    register_new_sensors(config.TASK_CONFIG)
    train(config)


if __name__ == "__main__":
    main()
