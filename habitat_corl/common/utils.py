import os
import random
import uuid
from typing import List, Dict, Any, Optional

import gym
import numpy as np
import torch
import wandb
from tqdm import tqdm

from habitat.utils.visualizations.utils import images_to_video, \
    observations_to_image
from habitat_corl.shortest_path_dataset import get_stored_episodes


def restructure_results(info_list: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Restructure the result from the evaluation into a dictionary.

    Args:
        info_list: a list of dictionaries containing the evaluation results.

    Returns:
        a dictionary containing the evaluation results.
    """
    # reformulate results
    final = {}
    for k in info_list[0].keys():
        if k == "top_down_map":
            continue
        final[k] = [r[k] for r in info_list]
    return final


def train_eval_split(env, config, n_eval_episodes: int = 10, single_goal=False):
    all_episodes = env.episodes
    eval_episodes = np.random.choice(all_episodes,
                                     n_eval_episodes,
                                     replace=False)
    train_episodes = np.setdiff1d(all_episodes, eval_episodes)
    train_episodes = [
        f"{ep.scene_id.split('/')[-1].split('.')[0]}/{ep.episode_id}" for ep in
        train_episodes]

    # check that data exists for episodes
    available_episodes = get_stored_episodes(config.TASK_CONFIG)
    actual_train_episodes = []
    for ep in train_episodes:
        scene_id, episode_id = ep.split("/")
        if episode_id in available_episodes[scene_id]:
            actual_train_episodes.append(ep)
    train_episodes = actual_train_episodes

    # if we want to evaluate on a single goal
    # we set the goal to be the same for all eval episodes
    if single_goal:
        goal = eval_episodes[0].goals[0]
        for ep in eval_episodes:
            ep.goals = [goal]

    return train_episodes, eval_episodes

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
    env_ptr = env
    while hasattr(env_ptr, "env"):
        env_ptr = env_ptr.env
    env_ptr.episodes = episodes
    env_ptr.episode_iterator = iter(episodes)
    env_ptr.seed(seed)  # needed?
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

def get_goal(algorithn_config, eval_episodes):
    if algorithn_config.single_goal:
        goal = eval_episodes[0].goals[0].position
    else:
        goal = None
    return goal
