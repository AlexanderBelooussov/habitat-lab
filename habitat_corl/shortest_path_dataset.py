import argparse
from typing import Any

import habitat
from habitat import registry
from habitat.tasks.nav.nav import EpisodicGPSSensor
from habitat.utils.geometry_utils import quaternion_from_coeff, \
    quaternion_rotate_vector
from habitat_baselines.config.default import get_config
from habitat.tasks.nav.shortest_path_follower import ShortestPathFollower
from habitat_corl.replay_buffer import ReplayBuffer
from tqdm import tqdm
import numpy as np
from gym import spaces
import h5py
import os


@registry.register_sensor(name="PositionSensor")
class PositionSensor(EpisodicGPSSensor):
    r"""
    The agents current location in the global coordinate frame.
    Args:
        sim: reference to the simulator for calculating task observations.
        config: Contains the DIMENSIONALITY field for the number of dimensions to express the agents position
    Attributes:
        _dimensionality: number of dimensions used to specify the agents position
    """
    cls_uuid: str = "position"

    def _get_observation_space(self, *args: Any, **kwargs: Any):
        return spaces.Box(
            low=np.finfo(np.float32).min,
            high=np.finfo(np.float32).max,
            shape=(3,),
            dtype=np.float32,
        )

    def get_observation(
        self, observations, episode, *args: Any, **kwargs: Any
    ):
        agent_state = self._sim.get_agent_state()
        return agent_state.position


def register_position_sensor(config):
    config.defrost()
    config.TASK.AGENT_POSITION_SENSOR = habitat.config.Config()
    config.TASK.AGENT_POSITION_SENSOR.TYPE = "PositionSensor"
    config.TASK.SENSORS.append("AGENT_POSITION_SENSOR")
    config.freeze()
    return config


def dataset_to_dhf5(dataset: ReplayBuffer, config):
    dataset.episode_ids = np.array(dataset.episode_ids)
    dataset.scenes = np.array(dataset.scenes)
    dataset.actions = np.array(dataset.actions)
    dataset.rewards = np.array(dataset.rewards)
    dataset.dones = np.array(dataset.dones)

    file_path = config.DATASET.SP_DATASET_PATH
    # make dirs
    all_scenes = set([scene for scene in dataset.scenes])
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    with h5py.File(file_path, "a") as hf:
        for scene in tqdm(all_scenes, desc="Saving scenes", leave=False):
            idxs = np.where(dataset.scenes == scene)[0]
            episode_ids = set(dataset.episode_ids[idxs])
            for episode_id in tqdm(episode_ids, desc="Saving episodes", leave=False):
                if f"{scene}/{episode_id}" in hf:
                    # delete old dataset
                    del hf[f"{scene}/{episode_id}"]
                grp = hf.create_group(f"{scene}/{episode_id}")
                ep_idxs = np.where(dataset.episode_ids == episode_id)[0]
                for key in dataset.states.keys():
                    dataset.states[key] = np.array(dataset.states[key])
                    grp.create_dataset(f"states/{key}", data=dataset.states[key][ep_idxs], compression="gzip")
                    dataset.next_states[key] = np.array(dataset.next_states[key])
                    grp.create_dataset(f"next_states/{key}", data=dataset.next_states[key][ep_idxs], compression="gzip")

                grp.create_dataset("actions", data=dataset.actions[ep_idxs], compression="gzip")
                grp.create_dataset("rewards", data=dataset.rewards[ep_idxs], compression="gzip")
                grp.create_dataset("dones", data=dataset.dones[ep_idxs], compression="gzip")




def generate_shortest_path_dataset(config, train_episodes=None, max_traj_len=1000):
    if isinstance(config, str):
        config = get_config(config)

    dataset = ReplayBuffer()

    if hasattr(config, "TASK_CONFIG"):
        config = config.TASK_CONFIG

    config = register_position_sensor(config)

    with habitat.Env(config) as env:
        if train_episodes is not None:
            env.episodes = train_episodes

        shortest_path_follower = ShortestPathFollower(env.sim, config.TASK.SUCCESS_DISTANCE)

        if config.DATASET.EPISODES == -1:
            n_episodes = len(env.episodes)
        else:
            n_episodes = min(config.DATASET.EPISODES, len(env.episodes))
        # for episode in tqdm(range(len(env.episodes)), desc="Generating shortest path dataset"):
        for episode in tqdm(range(n_episodes), desc="Generating shortest path dataset"):
            obs = env.reset()
            episode_id = env.current_episode.episode_id
            scene = env.current_episode.scene_id.split("/")[-1].split(".")[0]
            # print("Episode: ", episode)
            for step in range(max_traj_len):
                dataset.append_observations(obs)

                # print("Step: ", step)
                action = shortest_path_follower.get_next_action(env.current_episode.goals[0].position)
                # select position with value 1
                action = action.argmax()
                action_name = env.task.get_action_name(action)
                # print("Action: ", action)
                obs = env.step(action)
                # print("Observation: ", obs)
                # print("Distance to goal: ", env.current_episode.info["geodesic_distance"])
                info = env.get_metrics()
                dataset.append_scene(scene)
                dataset.append_episode_id(episode_id)


                # reward calculation
                if action_name != "STOP":
                    reward = 0.0
                else:
                    reward = info["success"]

                # add to dataset
                dataset.append_next_observations(obs)
                dataset.append_action(action)
                dataset.append_reward(reward)
                dataset.append_done(action_name == "STOP")

                # check if action is stop
                if action_name == "STOP":
                    break
            if (episode+1) % 100 == 0:
                dataset_to_dhf5(dataset, config)
                dataset = ReplayBuffer()
    dataset_to_dhf5(dataset, config)
    return dataset


def sample_transitions(
    config,
    n_transitions=256,
    groups=None,
    datasets=None
):
    if datasets is None:
        datasets = [
            "states/position",
            "states/heading",
            "states/pointgoal",
            "actions"
        ]


    transitions = ReplayBuffer()
    with h5py.File(config.DATASET.SP_DATASET_PATH, "r") as hf:
        if groups is None:
            groups = []
            for scene in hf:
                for episode in hf[scene]:
                    groups.append(f"{scene}/{episode}")

        for i in tqdm(range(n_transitions), desc="Sampling transitions", leave=False, disable=True):
            grp = hf[np.random.choice(groups)]
            step = np.random.randint(len(grp["actions"]))

            # now add to transitions
            for dataset in datasets:
                if "next_states" in dataset:
                    transitions.append_next_observations(grp[dataset][step], dataset.split("/")[1])
                elif "states" in dataset:
                    transitions.append_observations(grp[dataset][step], dataset.split("/")[1])
                elif "actions" in dataset:
                    transitions.append_action(grp[dataset][step])
                elif "rewards" in dataset:
                    transitions.append_reward(grp[dataset][step])

    return transitions


def dataset_episodes(config):
    with h5py.File(config.DATASET.SP_DATASET_PATH, "r") as hf:
        return [f"{episode}" for scene in hf for episode in hf[scene]]


def calc_mean_std(config, groups=None):
    with h5py.File(config.DATASET.SP_DATASET_PATH, "r") as hf:
        if groups is None:
            groups = []
            for scene in hf:
                for episode in hf[scene]:
                    groups.append(f"{scene}/{episode}")
        # get all types of state datasets
        state_datasets = [x for x in hf[groups[0]]["states"] if x not in ["rgb", "depth"]]
        stats = {}
        for ds in state_datasets:
            all_data = []
            for grp in groups:
                all_data.append(hf[grp]["states"][ds][:])
            all_data = np.concatenate(all_data, axis=0)
            stats[ds] = (np.mean(all_data, axis=0), np.std(all_data, axis=0))
    return stats

def main():
    scene_dict = {
        "medium": "17DRP5sb8fy",
        "small": "Pm6F8kyY3z2",
        "large": "XcA2TqTSSAj",
        "long_hallway": "Vt2qJdWjCF2",
        "xl": "uNb9QFRL6hY",
    }
    parser = argparse.ArgumentParser()
    parser.add_argument("--scene", type=str, default="medium")

    args = parser.parse_args()

    config = habitat.get_config("configs/tasks/pointnav_mp3d_medium.yaml")

    scene = args.scene

    if scene == "all":
        scenes = ["medium", "small", "large", "long_hallway", "xl"]
    elif scene in scene_dict:
        scenes = [scene]
    else:
        raise ValueError("Invalid scene")
    for scene in scenes:
        print("Generating dataset for scene: ", scene)
        config.defrost()
        if scene in scene_dict:
            config.DATASET.CONTENT_SCENES = [scene_dict[scene]]
        if scene == "long_hallway":
            config.DATASET.DATA_PATH = "data/datasets/pointnav/mp3d/v1/test/test.json.gz"
        config.DATASET.EPISODES = -1
        config.freeze()
        generate_shortest_path_dataset(config)

if __name__ == "__main__":
    # config = habitat.get_config("configs/tasks/pointnav_mp3d_medium.yaml")
    # sample_transitions(config, 256)
    main()

