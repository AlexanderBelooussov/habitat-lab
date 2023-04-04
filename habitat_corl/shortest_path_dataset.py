import argparse
import copy
import gc
import sys
from typing import Any

import vaex

import habitat
from habitat import registry
from habitat.tasks.nav.nav import EpisodicGPSSensor
from habitat.tasks.utils import cartesian_to_polar
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

non_image_state_datasets = [
    "state_position",
    "state_heading",
    "state_gps",
    "state_compass",
    "state_pointgoal",
    "state_pointgoal_with_gps_compass",
    "state_goal_position",
]


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


@registry.register_sensor(name="GoalPositionSensor")
class GoalPositionSensor(EpisodicGPSSensor):
    r"""
    The agents current location in the global coordinate frame.
    Args:
        sim: reference to the simulator for calculating task observations.
        config: Contains the DIMENSIONALITY field for the number of dimensions to express the agents position
    Attributes:
        _dimensionality: number of dimensions used to specify the agents position
    """
    cls_uuid: str = "goal_position"

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
        goal_position = episode.goals[0].position
        return goal_position


def register_position_sensor(config):
    config.defrost()
    config.TASK.AGENT_POSITION_SENSOR = habitat.config.Config()
    config.TASK.AGENT_POSITION_SENSOR.TYPE = "PositionSensor"
    config.TASK.SENSORS.append("AGENT_POSITION_SENSOR")

    config.TASK.AGENT_GOAL_POSITION_SENSOR = habitat.config.Config()
    config.TASK.AGENT_GOAL_POSITION_SENSOR.TYPE = "GoalPositionSensor"
    config.TASK.SENSORS.append("AGENT_GOAL_POSITION_SENSOR")
    config.freeze()
    return config


def dataset_to_dhf5(dataset: ReplayBuffer, config):
    dataset.episode_ids = np.array(dataset.episode_ids, dtype=np.uint32)
    dataset.scenes = np.array(dataset.scenes)
    dataset.actions = np.array(dataset.actions, dtype=np.uint8)
    dataset.rewards = np.array(dataset.rewards, dtype=np.bool)
    dataset.dones = np.array(dataset.dones, dtype=np.bool)

    file_path = config.DATASET.SP_DATASET_PATH

    # add "state" prefix to state keys
    # reduce depth size by converting float32 to uint8
    dataset.states = {f"state_{k}": v for k, v in dataset.states.items()}
    dataset.next_states = {f"next_state_{k}": v for k, v in
                           dataset.next_states.items()}
    if 'state_depth' in dataset.states:
        dataset.states['state_depth'] = (
                np.array(dataset.states['state_depth']) * 255).astype(np.uint8)
    if 'next_state_depth' in dataset.next_states:
        dataset.next_states['next_state_depth'] = (np.array(
            dataset.next_states['next_state_depth']) * 255).astype(np.uint8)

    df = vaex.from_arrays(
        # episode_id=dataset.episode_ids,
        # scene=dataset.scenes,
        action=dataset.actions,
        reward=dataset.rewards,
        done=dataset.dones,
        **dataset.states,
        **dataset.next_states,
    )

    # export each scene/episode to a group
    # get all scenes
    all_scenes = set([scene for scene in dataset.scenes])
    for scene in tqdm(all_scenes):
        # get all episodes
        idxs = np.where(dataset.scenes == scene)[0]
        episode_ids = set(dataset.episode_ids[idxs])
        for episode_id in episode_ids:
            idxs = np.where((dataset.episode_ids == episode_id) & (
                    dataset.scenes == scene))[0]
            df_ep = df[min(idxs):max(idxs) + 1]
            df_ep.export_hdf5(file_path, progress=False, mode="a",
                              group=f"{scene}/{episode_id}")


def get_stored_scenes(config):
    file_path = config.DATASET.SP_DATASET_PATH
    try:
        with h5py.File(file_path, "r") as hf:
            scenes = list(hf.keys())
        return scenes
    except:
        return []


def get_stored_episodes(config, scene=None):
    file_path = config.DATASET.SP_DATASET_PATH
    with h5py.File(file_path, "r") as hf:
        if scene is None:
            episodes = {}
            for scene in hf:
                episodes[scene] = list(hf[scene].keys())
        else:
            episodes = list(hf[scene].keys())
    return episodes


def get_stored_groups(config):
    file_path = config.DATASET.SP_DATASET_PATH
    with h5py.File(file_path, "r") as hf:
        groups = []
        for scene in hf:
            for episode in hf[scene]:
                groups.append(f"{scene}/{episode}")
    return groups


def generate_shortest_path_dataset(config, train_episodes=None,
                                   max_traj_len=1000, overwrite=False):
    # if overwrite, delete the file
    if overwrite:
        file_path = config.DATASET.SP_DATASET_PATH
        if os.path.exists(file_path):
            os.remove(file_path)

    if isinstance(config, str):
        config = get_config(config)

    dataset = ReplayBuffer()

    if hasattr(config, "TASK_CONFIG"):
        config = config.TASK_CONFIG

    config = register_position_sensor(config)

    stored_eps = {}
    for scene in get_stored_scenes(config):
        stored_eps[scene] = get_stored_episodes(config, scene)

    with habitat.Env(config) as env:
        if train_episodes is not None:
            env.episodes = train_episodes

        shortest_path_follower = ShortestPathFollower(env.sim,
                                                      config.TASK.SUCCESS_DISTANCE)

        if config.DATASET.EPISODES == -1:
            n_episodes = len(env.episodes)
        else:
            n_episodes = min(config.DATASET.EPISODES, len(env.episodes))
        # for episode in tqdm(range(len(env.episodes)), desc="Generating shortest path dataset"):
        for episode in tqdm(range(n_episodes),
                            desc="Generating shortest path dataset"):
            obs = env.reset()
            episode_id = env.current_episode.episode_id
            scene = env.current_episode.scene_id.split("/")[-1].split(".")[0]

            # check if episode is already stored
            if not overwrite:
                stored_episodes = stored_eps.get(scene, [])
                if episode_id in stored_episodes:
                    continue

            # print("Episode: ", episode)
            for step in range(max_traj_len):
                dataset.append_observations(obs)

                # print("Step: ", step)
                action = shortest_path_follower.get_next_action(
                    env.current_episode.goals[0].position)
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
            if (episode + 1) % 100 == 0:
                dataset_to_dhf5(dataset, config)
                dataset = ReplayBuffer()
                gc.collect()
    dataset_to_dhf5(dataset, config)
    return dataset


def load_full_dataset(config, groups=None, datasets=None, continuous=False,
                      ignore_stop=False, single_goal=None):
    if datasets is None:
        datasets = [
            "states/position",
            "states/heading",
            "states/pointgoal",
            "action"
        ]

    datasets = [dataset.replace(f"s/", "_") for dataset in datasets]
    rpb = ReplayBuffer()
    groups = get_stored_groups(config) if groups is None else groups

    for group in tqdm(groups, desc="Loading dataset"):
        df = vaex.open(f"{config.DATASET.SP_DATASET_PATH}", group=group)
        n_steps = len(df['action'].values)
        if ignore_stop:
            n_steps -= 1
        for dataset in datasets:
            if "next_state" in dataset:
                rpb.extend_next_states(
                    copy.deepcopy(df[dataset].values[:n_steps]),
                    dataset.split("state_")[1])
            elif "state" in dataset:
                rpb.extend_states(
                    copy.deepcopy(df[dataset].values[:n_steps]),
                    dataset.split("state_")[1])
            elif "action" in dataset:
                ds = "action"
                actions = copy.deepcopy(df[ds].values[:n_steps])
                if not continuous and ignore_stop:
                    actions = np.where(actions == 0, 0, actions - 1)
                rpb.extend_actions(actions)
            elif "reward" in dataset:
                ds = "reward"
                if not ignore_stop and single_goal is None:
                    rpb.extend_rewards(copy.deepcopy(df[ds].values[:n_steps]))
                else:
                    # np array of current positions
                    positions = copy.deepcopy(df["next_state_position"].values[:-1])
                    # np array of goal positions
                    if single_goal is None:
                        goals = copy.deepcopy(df["state_goal_position"].values[:-1])
                    else:
                        goals = np.array([single_goal] * n_steps)
                    # np array of distances between current position and goal
                    distances = np.linalg.norm(positions - goals, axis=1)
                    # np array of rewards
                    rewards = np.where(
                        distances < config.TASK.SUCCESS_DISTANCE, 1, 0)
                    rpb.extend_rewards(rewards)

            elif "done" in dataset:
                ds = "done"
                if not ignore_stop:
                    rpb.extend_dones(copy.deepcopy(df[ds].values[:n_steps]))
                else:
                    dones = copy.deepcopy(df[ds].values[:-2])
                    dones = np.append(dones, True)
                    rpb.extend_dones(dones)

        df.close()
    # print size of dataset in memory (MBs)
    print(f"Dataset size: {sys.getsizeof(rpb) / 1024 / 1024:.3f} MBs\n"
          f"Number of transitions: {rpb.num_steps}\n"
          f"Number of episodes: {len(groups)}")
    if continuous:
        rpb.to_continuous_actions()
    rpb.to_numpy()
    return rpb


def prepare_batches(config, n_batches=5000, n_transitions=256, groups=None,
                    datasets=None, continuous=False, ignore_stop=False):
    """
    prepare multiple batches so that they can be used in an efficient way
    """
    if datasets is None:
        datasets = [
            "state_position",
            "state_heading",
            "state_pointgoal",
            "action"
        ]
    batches = [ReplayBuffer() for _ in range(n_batches)]
    groups = get_stored_groups(config) if groups is None else groups

    # sample a group

    # put all transitions into the batches
    # switch to next batch for each transition
    # until all batches are full
    batch_idx = 0
    transitions_added = 0
    # prepare tqdm
    pbar = tqdm(total=n_batches * n_transitions,
                desc="Preparing batches")
    while transitions_added < n_transitions * n_batches:
        replay = ReplayBuffer()
        replay.from_hdf5_group(config.DATASET.SP_DATASET_PATH,
                               np.random.choice(groups))
        if continuous:
            replay.to_continuous_actions()
        # now add to transitions
        for step in range(replay.num_steps):
            if step == replay.num_steps - 1 and ignore_stop:
                continue
            for dataset in datasets:
                if "next_state" in dataset:
                    kind = dataset.split("state_")[1]
                    batches[batch_idx].append_next_observations(
                        replay.next_states[kind][step],
                        kind
                    )
                elif "state" in dataset:
                    kind = dataset.split("state_")[1]
                    batches[batch_idx].append_observations(
                        replay.states[kind][step],
                        kind
                    )
                elif "action" in dataset:
                    action = replay.actions[step]
                    # shift actions to fill gap left by stop
                    if not continuous and ignore_stop:
                        action = action - 1
                    batches[batch_idx].append_action(action)
                elif "reward" in dataset:
                    if ignore_stop and step + 1 == replay.num_steps - 1:
                        batches[batch_idx].append_reward(
                            replay.rewards[step + 1])
                    else:
                        batches[batch_idx].append_reward(replay.rewards[step])
                elif "done" in dataset:
                    if ignore_stop and step + 1 == replay.num_steps - 1:
                        batches[batch_idx].append_done(replay.dones[step + 1])
                    else:
                        batches[batch_idx].append_done(replay.dones[step])
            batch_idx = (batch_idx + 1) % n_batches
            transitions_added += 1
            pbar.update(1)
            if transitions_added >= n_transitions * n_batches:
                break
    return batches


def batch_generator(
    config,
    n_transitions=256,
    groups=None,
    datasets=None,
    use_full_dataset=False,
    n_batches=5000,
    continuous=False,
    ignore_stop=False,
    single_goal=None
):
    if continuous:
        ignore_stop = True
    if not use_full_dataset:
        while True:
            batches = prepare_batches(
                config,
                n_transitions=n_transitions,
                groups=groups,
                datasets=datasets,
                n_batches=n_batches,
                continuous=continuous,
                ignore_stop=ignore_stop
            )
            while len(batches) > 0:
                batch = batches.pop(0)
                yield batch
    else:
        dataset = load_full_dataset(
            config,
            groups=groups,
            datasets=datasets,
            continuous=continuous,
            ignore_stop=ignore_stop,
            single_goal=single_goal
        )
        while True:
            yield dataset.sample(n_transitions)


def sample_transitions(
    config,
    n_transitions=256,
    groups=None,
    datasets=None,
    use_full_dataset=True,
):
    gen = batch_generator(
        config,
        n_transitions=n_transitions,
        groups=groups,
        datasets=datasets,
        use_full_dataset=use_full_dataset
    )
    batch = next(gen)
    return batch


def calc_mean_std(config, groups=None, used_inputs=None):
    means = []
    stds = []
    with h5py.File(config.DATASET.SP_DATASET_PATH, "r") as hf:
        groups = get_stored_groups(config) if groups is None else groups
        # get all types of state datasets
        state_datasets = [x for x in hf[groups[0]]['columns'] if
                          x in non_image_state_datasets]
        assert len(state_datasets) > 0, "No state datasets found"
        stats = {}
        for ds in state_datasets:
            count = 0
            mean = None
            std = None
            all_data = []
            for grp in groups:
                count += 1
                data = hf[grp]["columns"][ds]['data']
                all_data.append(data[...])
                if count % 100 == 0:
                    # early stopping if mean and std do not change anymore
                    ad = np.concatenate(all_data, axis=0)
                    new_mean = np.mean(ad, axis=0)
                    new_std = np.std(ad, axis=0)
                    if mean is None:
                        mean = new_mean
                        std = new_std
                    else:
                        mean_dif = np.linalg.norm(mean - new_mean)
                        std_dif = np.linalg.norm(std - new_std)
                        if mean_dif < 1e-3 and std_dif < 1e-3:
                            break
                        else:
                            mean = new_mean
                            std = new_std
            all_data = np.concatenate(all_data, axis=0)
            stats[ds] = (np.mean(all_data, axis=0), np.std(all_data, axis=0))



    if used_inputs is not None:
        used_mean = []
        used_std = []
        for ds in used_inputs:
            state_name = f"state_{ds}"
            used_mean.append(stats[state_name][0])
            used_std.append(stats[state_name][1])
        used_mean = np.concatenate(used_mean, axis=0)
        used_std = np.concatenate(used_std, axis=0)
        stats["used"] = (used_mean, used_std)
    return stats


def longest_cont_action_sequence(config, groups=None):
    with h5py.File(config.DATASET.SP_DATASET_PATH, "r") as hf:
        groups = get_stored_groups(config) if groups is None else groups
        max_len = 0
        for grp in groups:
            actions = hf[grp]["actions"][...]
            current_value = actions[0]
            current_len = 1
            for a in actions[1:]:
                if a == current_value:
                    current_len += 1
                else:
                    max_len = max(max_len, current_len)
                    current_len = 1
                    current_value = a

    return max_len


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
    parser.add_argument("--overwrite", action="store_true")

    args = parser.parse_args()

    config = habitat.get_config(
        "configs/tasks/pointnav_mp3d_sp_dataset_generation.yaml")

    scene = args.scene
    overwrite = args.overwrite

    if scene == "all":
        scenes = ["medium", "small", "large", "long_hallway", "xl"]
    elif scene in scene_dict:
        scenes = [scene]
    else:
        raise ValueError("Invalid scene")
    original_path = None
    for scene in scenes:
        print("Generating dataset for scene: ", scene)
        sys.stdout.flush()
        config.defrost()
        if scene in scene_dict:
            config.DATASET.CONTENT_SCENES = [scene_dict[scene]]
        if scene == "long_hallway":
            config.DATASET.DATA_PATH = "data/datasets/pointnav/mp3d/v1/test/test.json.gz"
        config.DATASET.EPISODES = -1
        if original_path is None:
            original_path = config.DATASET.SP_DATASET_PATH
        path = original_path
        path = path.split(".")[0]
        path += f"_{scene}_no_depth.hdf5"
        config.DATASET.SP_DATASET_PATH = path
        config.freeze()
        generate_shortest_path_dataset(config, overwrite=overwrite)


if __name__ == "__main__":
    # config = habitat.get_config("configs/tasks/pointnav_mp3d_medium.yaml")
    # sample_transitions(config, 256)
    main()
