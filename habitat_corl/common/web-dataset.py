import argparse
import gc
import os

import numpy as np

import habitat
from habitat_baselines.config.default import get_config
from tqdm import trange

from habitat_corl.common.replay_buffer import ReplayBuffer
from habitat_corl.common.shortest_path_dataset import register_new_sensors, \
    get_stored_scenes, get_stored_episodes, dataset_to_dhf5

scene_dict = {
        "medium": "17DRP5sb8fy",
        "small": "Pm6F8kyY3z2",
        # "large": "XcA2TqTSSAj",
        "large": "ac26ZMwG7aT",
        "long_hallway": "Vt2qJdWjCF2",
        "xl": "uNb9QFRL6hY",
    }

def get_dataset(config, overwrite=False):
    # if overwrite, delete the file
    if overwrite:
        file_path = config.DATASET.WEB_DATASET_PATH
        if os.path.exists(file_path):
            os.remove(file_path)

    action_dict = {
        "STOP": 0,
        "MOVE_FORWARD": 1,
        "TURN_LEFT": 2,
        "TURN_RIGHT": 3,
    }

    dataset = ReplayBuffer()

    stored_eps = {}
    for scene in get_stored_scenes(config):
        stored_eps[scene] = get_stored_episodes(config, scene)

    total_transitions = 0

    with habitat.Env(config=config) as env:
        for episode in trange(len(env.episodes)):
            observations = env.reset()

            # check if episode is already stored
            scene = env.current_episode.scene_id.split("/")[-1].split(".")[0]
            episode_id = env.current_episode.episode_id
            if not overwrite:
                stored_episodes = stored_eps.get(scene, [])
                if episode_id in stored_episodes:
                    continue

            actions = env.current_episode.reference_replay[1:]
            n_actions = 0
            for action in actions:
                if "LOOK" in action.action:
                    continue
                total_transitions += 1
                n_actions += 1
                dataset.append_observations(observations, exclude=["rgb", "depth", "semantic"])
                observations = env.step(action=action.action)
                info = env.get_metrics()
                dataset.append_action(action_dict[action.action])
                dataset.append_reward(info["success"])
                dataset.append_next_observations(observations, exclude=["rgb", "depth", "semantic"])

                if action.action == "STOP":
                    dataset.append_done(True)
                    if info["success"]:
                        goal = observations["position"]
                    else:
                        # find the closest goal
                        goals = np.array([goal.position for goal in env.current_episode.goals])
                        goal = goals[np.argmin(np.linalg.norm(goals - observations["position"], axis=1))]

                    goal = np.array([goal] * n_actions)
                    dataset.extend_states(goal, key="goal_position")
                    dataset.extend_next_states(goal, key="goal_position")

                    scene = np.array([scene] * n_actions)
                    episode_id = np.array([episode_id] * n_actions)

                    dataset.extend_scenes(scene)
                    dataset.extend_episodes(episode_id)

                    break
                else:
                    dataset.append_done(False)
            if (episode + 1) % 100 == 0:
                dataset_to_dhf5(dataset, config)
                dataset = ReplayBuffer()
                gc.collect()
        dataset_to_dhf5(dataset, config)
    print(f"Num transitions: {total_transitions}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--scene", type=str, default="small")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    scene = args.scene
    overwrite = args.overwrite
    config_path = "habitat_corl/configs/web.yaml"
    config = get_config(config_path)
    register_new_sensors(config.TASK_CONFIG, ["heading_vec", "position"])

    config.defrost()
    config.TASK_CONFIG.DATASET.WEB_DATASET_PATH = f"data/web_datasets/web_dataset_{scene}.hdf5"
    config.TASK_CONFIG.DATASET.CONTENT_SCENES = [scene_dict[scene]]
    config.freeze()

    get_dataset(config.TASK_CONFIG, overwrite=overwrite)

if __name__ == "__main__":
    main()
