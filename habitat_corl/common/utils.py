from typing import List, Dict, Any

import numpy as np

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


def train_eval_split(env, config, n_eval_episodes: int = 10):
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
    return train_episodes, eval_episodes
