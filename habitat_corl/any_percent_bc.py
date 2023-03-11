import argparse
from typing import Any, Dict, List, Optional, Tuple, Union

from tqdm import tqdm
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

import habitat
from habitat.utils.visualizations.utils import images_to_video, \
    observations_to_image
from habitat_baselines.common.baseline_registry import baseline_registry

from habitat_corl.replay_buffer import ReplayBuffer
from habitat_baselines.config.default import get_config
from habitat_corl.shortest_path_dataset import generate_shortest_path_dataset, \
    sample_transitions, register_position_sensor, dataset_episodes, \
    calc_mean_std
from habitat_baselines.il.common.encoders.resnet_encoders import \
    ResnetRGBEncoder, VlnResnetDepthEncoder, ResnetSemSeqEncoder

from scipy.spatial.transform import Rotation as R

TensorBatch = List[torch.Tensor]


def soft_update(target: nn.Module, source: nn.Module, tau: float):
    for target_param, source_param in zip(target.parameters(),
                                          source.parameters()):
        target_param.data.copy_(
            (1 - tau) * target_param.data + tau * source_param.data)


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
    normalization_stats=None,
):
    if normalization_stats is None:
        normalization_stats = {}

    def make_videos(observations_list, output_prefix, ep_id):
        prefix = output_prefix + "_{}".format(ep_id)
        images_to_video(observations_list, output_dir=video_dir,
                        video_name=prefix)

    # run the agent for n_episodes
    env.episodes = episodes
    env.episode_iterator = iter(episodes)
    results = []
    for i in tqdm(range(len(episodes)), desc="eval", leave=False):
        video_frames = []
        env.seed(seed)  # needed?
        observations = env.reset()
        for step in range(max_traj_len):
            state = []
            for k in used_inputs:
                obs = observations[k]
                if k in normalization_stats:
                    obs = (obs - normalization_stats[k][0]) / \
                          (normalization_stats[k][1] + 1e-8)
                state.extend(obs)
            state = torch.tensor(state, dtype=torch.float).to(device)
            action = actor.act(state)
            action_name = env.task.get_action_name(action)
            observations = env.step(action)
            info = env.get_metrics()
            if video:
                frame = observations_to_image(observations, info)
                video_frames.append(frame)
            if action_name == "STOP" or env.episode_over:
                print(f"Episode {i} finished after {step} steps")
                break

        info = env.get_metrics()
        results.append(info)
        if video:
            make_videos(video_frames, video_prefix, i)

    # reformulate results
    final = {}
    for k in results[0].keys():
        if k == "top_down_map":
            continue
        final[k] = [r[k] for r in results]
    return final


def keep_best_trajectories(
    dataset: ReplayBuffer,
    frac: float,
    discount: float,
    max_episode_steps: int = 1000,
):
    # TODO: make this work with hdf5
    ids_by_trajectories = []
    returns = []
    cur_ids = []
    cur_return = 0
    reward_scale = 1.0
    for i, (reward, done) in enumerate(zip(dataset.rewards, dataset.dones)):
        cur_return += reward_scale * reward
        cur_ids.append(i)
        reward_scale *= discount
        if done or len(cur_ids) == max_episode_steps:
            ids_by_trajectories.append(list(cur_ids))
            returns.append(cur_return)
            cur_ids = []
            cur_return = 0
            reward_scale = 1.0

    sort_ord = np.argsort(returns, axis=0)[::-1].reshape(-1)
    top_trajs = sort_ord[: int(frac * len(sort_ord))]

    order = []
    for i in top_trajs:
        order += ids_by_trajectories[i]
    order = np.array(order)

    dataset.dones = np.take(dataset.dones, order, axis=0)
    dataset.rewards = np.take(dataset.rewards, order, axis=0)
    dataset.actions = np.take(dataset.actions, order, axis=0)
    for key in dataset.states:
        dataset.states[key] = np.take(dataset.states[key], order, axis=0)
        dataset.next_states[key] = np.take(dataset.next_states[key], order,
                                           axis=0)


class Actor(nn.Module):
    def __init__(self, config, env):
        super(Actor, self).__init__()

        self.env = env
        self.config = config

        self.device = torch.device(
            "cuda:0" if torch.cuda.is_available() else "cpu")

        # model_config = self.config.MODEL
        # model_config.defrost()
        # model_config.TORCH_GPU_ID = 0
        # model_config.freeze()
        #
        # observation_space = self.env.observation_space
        # policy = baseline_registry.get_policy(self.config.RL.POLICY.name)
        # self.policy = policy.from_config(
        #     self.config, observation_space, self.env.action_space
        # )
        # self.policy.to(self.device)

        self.linear_input_size = 0
        if "pointgoal_with_gps_compass" in config.MODEL.used_inputs:
            self.linear_input_size += config.TASK_CONFIG.TASK.POINTGOAL_WITH_GPS_COMPASS_SENSOR.DIMENSIONALITY
        if "proximity" in config.MODEL.used_inputs:
            self.linear_input_size += 1
        if "agent_map_coord" in config.MODEL.used_inputs:
            self.linear_input_size += 2
        if "agent_angle" in config.MODEL.used_inputs:
            self.linear_input_size += 1
        if "position" in config.MODEL.used_inputs:
            self.linear_input_size += 3
        if "heading" in config.MODEL.used_inputs:
            self.linear_input_size += 1
        if "pointgoal" in config.MODEL.used_inputs:
            self.linear_input_size += config.TASK_CONFIG.TASK.POINTGOAL_SENSOR.DIMENSIONALITY

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
        state.to(self.device)
        return self.net(state)

    @torch.no_grad()
    def act(self, state, device: str = "cpu") -> int:
        action = self.forward(state)
        action = torch.argmax(action).item()
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

    def train(self, batch: TensorBatch, used_inputs) -> Dict[str, float]:
        log_dict = {}
        self.total_it += 1

        states = batch.states
        actions = batch.actions

        # transform state to tensor
        transitions = []
        for i in range(len(actions)):
            transition_obs = []
            for k in used_inputs:
                transition_obs.extend(states[k][i])
            transitions.append(
                torch.tensor(transition_obs, device=self.device))
        states = torch.stack(transitions)

        # Compute actor loss
        pi = self.actor(states)
        # actor_loss = F.mse_loss(pi, action)
        actor_loss = F.cross_entropy(pi, actions.long())
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


def train(config_path):
    config = get_config(config_path)
    if "position" in config.MODEL.used_inputs:
        config.defrost()
        config.TASK_CONFIG = register_position_sensor(config.TASK_CONFIG)
        config.freeze()

    set_seed(config.SEED)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    with habitat.Env(config=config.TASK_CONFIG) as env:
        # dataset = d4rl.qlearning_dataset(env)
        all_episodes = env.episodes
        # eval_episodes = np.random.choice(all_episodes,
        #                                  config.RL.BC.EVAL_EPISODES,
        #                                  replace=False)
        # train_episodes = np.setdiff1d(all_episodes, eval_episodes)
        # env.episodes = train_episodes
        ep_ids = dataset_episodes(config.TASK_CONFIG)[-1]
        ep_ids = [ep_ids]
        eval_episodes = [ep for ep in all_episodes if
                         str(ep.episode_id) in ep_ids]

        mean_std = calc_mean_std(config.TASK_CONFIG)
        # keep_best_trajectories(dataset, config.RL.BC.FRAC,
        #                        config.RL.BC.DISCOUNT)

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
        for t in tqdm(range(int(config.NUM_UPDATES)), desc="Training"):
            batch = sample_transitions(
                config.TASK_CONFIG,
                config.RL.BC.BATCH_SIZE,
                groups=[f"17DRP5sb8fy/{ep.episode_id}" for ep in eval_episodes]
            )
            batch.normalize_states(mean_std)
            batch.to_tensor(device=device)
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
                    video_prefix="bc",
                    normalization_stats=mean_std,
                )
                evaluations.append(eval_scores)
                print("---------------------------------------")
                print(
                    f"Evaluation over {config.RL.BC.EVAL_EPISODES} episodes: "
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
    train(args.config)


if __name__ == "__main__":
    main()
