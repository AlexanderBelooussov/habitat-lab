import argparse
import copy
import gc
import pickle
import sys
from typing import Any, Union

import vaex

import habitat
from habitat import registry, Config
from habitat.tasks.nav.nav import EpisodicGPSSensor, HeadingSensor
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
    "state_heading_vec",
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


@registry.register_sensor(name="HeadingVecSensor")
class HeadingVecSensor(HeadingSensor):
    cls_uuid: str = "heading_vec"

    def get_observation(
        self, observations, episode, *args: Any, **kwargs: Any
    ):
        heading = \
            super().get_observation(observations, episode, *args, **kwargs)[0]
        vec = [np.cos(heading), np.sin(heading)]
        return vec


def register_new_sensors(config, sensors=None):
    if sensors is None:
        sensors = ["position", "goal_position", "heading_vec"]

    config.defrost()
    if "position" in sensors:
        config.TASK.AGENT_POSITION_SENSOR = habitat.config.Config()
        config.TASK.AGENT_POSITION_SENSOR.TYPE = "PositionSensor"
        config.TASK.SENSORS.append("AGENT_POSITION_SENSOR")
    if "goal_position" in sensors:
        config.TASK.AGENT_GOAL_POSITION_SENSOR = habitat.config.Config()
        config.TASK.AGENT_GOAL_POSITION_SENSOR.TYPE = "GoalPositionSensor"
        config.TASK.SENSORS.append("AGENT_GOAL_POSITION_SENSOR")
    if "heading_vec" in sensors:
        config.TASK.AGENT_HEADING_VEC_SENSOR = habitat.config.Config()
        config.TASK.AGENT_HEADING_VEC_SENSOR.TYPE = "HeadingVecSensor"
        config.TASK.SENSORS.append("AGENT_HEADING_VEC_SENSOR")

    config.freeze()
    return config


def dataset_to_dhf5(dataset: ReplayBuffer, config):
    try:
        dataset.episode_ids = np.array(dataset.episode_ids, dtype=np.uint32)
    except ValueError:
        dataset.episode_ids = np.array(dataset.episode_ids)
    dataset.scenes = np.array(dataset.scenes)
    dataset.actions = np.array(dataset.actions, dtype=np.uint8)
    dataset.rewards = np.array(dataset.rewards, dtype=np.bool)
    dataset.dones = np.array(dataset.dones, dtype=np.bool)

    if hasattr(config.DATASET, 'SP_DATASET_PATH'):
        file_path = config.DATASET.SP_DATASET_PATH
    elif hasattr(config.DATASET, 'WEB_DATASET_PATH'):
        file_path = config.DATASET.WEB_DATASET_PATH
    else:
        raise ValueError("No dataset path specified in config")

    os.makedirs(os.path.dirname(file_path), exist_ok=True)

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
    if isinstance(config, str):
        path = config
    elif hasattr(config.DATASET, "SP_DATASET_PATH"):
        path = config.DATASET.SP_DATASET_PATH
    elif hasattr(config.DATASET, "WEB_DATASET_PATH"):
        path = config.DATASET.WEB_DATASET_PATH

    try:
        with h5py.File(path, "r") as hf:
            scenes = list(hf.keys())
        return scenes
    except:
        return []


def get_stored_episodes(config, scene=None):
    if isinstance(config, str):
        path = config
    elif hasattr(config.DATASET, "SP_DATASET_PATH"):
        path = config.DATASET.SP_DATASET_PATH
    elif hasattr(config.DATASET, "WEB_DATASET_PATH"):
        path = config.DATASET.WEB_DATASET_PATH

    with h5py.File(path, "r") as hf:
        if scene is None:
            episodes = {}
            for scene in hf:
                episodes[scene] = list(hf[scene].keys())
        else:
            episodes = list(hf[scene].keys())
    return episodes


def get_stored_groups(config):
    if isinstance(config, str):
        path = config
    elif hasattr(config.DATASET, "SP_DATASET_PATH"):
        path = config.DATASET.SP_DATASET_PATH
    elif hasattr(config.DATASET, "WEB_DATASET_PATH"):
        path = config.DATASET.WEB_DATASET_PATH

    with h5py.File(path, "r") as hf:
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

    config = register_new_sensors(config)

    stored_eps = {}
    for scene in get_stored_scenes(config):
        stored_eps[scene] = get_stored_episodes(config, scene)

    with habitat.Env(config) as env:
        if train_episodes is not None:
            env.episodes = train_episodes
            env.episode_iterator = iter(env.episodes)

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
                dataset.append_done(action_name == "STOP" or env.episode_over)

                # check if action is stop
                if action_name == "STOP" or env.episode_over:
                    break
            if (episode + 1) % 100 == 0:
                dataset_to_dhf5(dataset, config)
                dataset = ReplayBuffer()
                gc.collect()
    dataset_to_dhf5(dataset, config)
    return dataset

def get_pickle_path(config, ignore_stop, single_goal, continuous):
    path = f"data/dataset"
    scene = config.DATASET.CONTENT_SCENES[0]
    path += f"_scene_{scene}"
    if hasattr(config.DATASET, "SP_DATASET_PATH"):
        path += f"_sp"
        if "debug" in config.DATASET.SP_DATASET_PATH:
            path += f"_debug"
    if hasattr(config.DATASET, "WEB_DATASET_PATH"):
        path += f"_web"
    if continuous:
        path += f"_continuous"
    if ignore_stop:
        path += f"_ignore_stop"
    if single_goal is not None:
        path += f"_single_goal"

    path += ".pkl"
    return path
def save_as_pickle(
    obj,
    config,
    groups,
    datasets,
    ignore_stop,
    single_goal,
    continuous
):
    path = get_pickle_path(config, ignore_stop, single_goal, continuous)

    d = {
        "groups": groups,
        "datasets": datasets,
        "ignore_stop": ignore_stop,
        "single_goal": single_goal,
        "continuous": continuous,
        "obj": obj
    }


    with open(path, "wb") as f:
        pickle.dump(d, f, protocol=pickle.HIGHEST_PROTOCOL)


def load_from_pickle(file_path):
    with open(file_path, "rb") as f:
        return pickle.load(f)


def check_pickles(
    config,
    groups,
    datasets,
    ignore_stop,
    single_goal,
    continuous
):
    path = get_pickle_path(config, ignore_stop, single_goal, continuous)
    if os.path.exists(path):
        d = load_from_pickle(path)
        if d["groups"] == groups \
            and d["datasets"] == datasets \
            and d["ignore_stop"] == ignore_stop \
            and d["single_goal"] == single_goal \
            and d["continuous"] == continuous:
            return d["obj"]
    return None



def load_full_dataset(config, groups=None, datasets=None, continuous=False,
                      ignore_stop=False, single_goal=None,
                      frac=1.0, discount=0.99, max_episode_steps=1000,
                      normalization_data=None):
    if datasets is None:
        datasets = [
            "states/position",
            "states/heading",
            "states/pointgoal",
            "action"
        ]
    # pkl = check_pickles(config, groups, datasets, ignore_stop, single_goal, continuous)

    paths = []
    if hasattr(config.DATASET, "SP_DATASET_PATH"):
        paths.append(config.DATASET.SP_DATASET_PATH)
    if hasattr(config.DATASET, "WEB_DATASET_PATH"):
        paths.append(config.DATASET.WEB_DATASET_PATH)

    datasets = [dataset.replace(f"s/", "_") for dataset in datasets]
    rpb = ReplayBuffer()

    for file_path in paths:
        path_groups = get_stored_groups(file_path)
        if groups is not None:
            intersect = list(set(groups) & set(path_groups))
            if len(intersect) > 0:
                path_groups = intersect
        for group in tqdm(path_groups[:10], desc="Loading dataset"):
            rpb.from_hdf5_group(
                file_path=file_path,
                group=group,
                datasets=datasets,
                ignore_stop=ignore_stop,
                single_goal=single_goal,
                continuous=continuous,
            )
    # print size of dataset in memory (MBs)
    print(f"Dataset size: {sys.getsizeof(rpb) / 1024 / 1024:.3f} MBs\n"
          f"Number of transitions: {rpb.num_steps}\n"
          f"Number of episodes: {rpb.num_episodes}\n")
    if continuous:
        rpb.to_continuous_actions()

    rpb.to_numpy()

    if normalization_data is not None:
        rpb.normalize_states(normalization_data)

    if frac < 1.0:
        # keep the best trajectories (for BC-x%)
        trajectories = list(set(rpb.episode_ids))
        returns = []
        for traj in trajectories:
            idxs = np.where(rpb.episode_ids == traj)[0]
            rewards = rpb.rewards[idxs]
            dones = rpb.dones[idxs]
            cur_return = 0
            reward_scale = 1.0
            for i, (reward, done) in enumerate(zip(rewards, dones)):
                cur_return += reward_scale * reward
                reward_scale *= discount
                if done or i == max_episode_steps:
                    returns.append(cur_return)
                    break

        sort_ord = np.argsort(returns, axis=0)[::-1].reshape(-1)
        top_trajs = sort_ord[: int(frac * len(sort_ord))]

        trajectories = [trajectories[i] for i in top_trajs]

        # filter out the worst trajectories
        idxs = np.where(np.isin(rpb.episode_ids, trajectories))[0]
        rpb.filter(idxs)

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
    single_goal=None,
    frac=1.0,
    discount=0.99,
    max_episode_steps=1000,
    normalization_data=None
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
            single_goal=single_goal,
            frac=frac,
            discount=discount,
            max_episode_steps=max_episode_steps,
            normalization_data=normalization_data
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

    if "state_heading_vec" in stats:
        stats["state_heading_vec"] = (np.zeros(2), np.ones(2))
    if "state_position" in stats and "state_goal_position" in stats:
        stats["state_goal_position"] = stats["state_position"]

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
        # "large": "XcA2TqTSSAj",
        "large": "ac26ZMwG7aT",
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
        scenes = ["medium", "small", "large", "xl", "long_hallway"]
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


def augment_dataset():
    paths = [
        "data/sp_datasets/datasets_medium_no_depth.hdf5",
        "data/sp_datasets/fulldatasets_medium_no_depth.hdf5",
        "data/sp_datasets/datasets_small_no_depth.hdf5",
        "data/sp_datasets/datasets_large_no_depth.hdf5",
        "data/sp_datasets/datasets_long_hallway_no_depth.hdf5",
        "data/sp_datasets/datasets_xl_no_depth.hdf5",
    ]

    for path in paths:
        groups = []
        with h5py.File(path, "r") as hf:
            for scene in hf:
                for episode in hf[scene]:
                    groups.append(f"{scene}/{episode}")
        for group in tqdm(groups, desc="Augmenting dataset"):
            # add a dataset "heading_vec", containing the heading vector of the agent
            # as a 2D vector
            df = vaex.open(path, group=group)
            if "state_heading_vec" in df.get_column_names():
                print(f"Skipping {group}")
            heading = df["state_heading"].values
            heading_vec = np.stack([np.cos(heading), np.sin(heading)], axis=1)
            # flatten last dimension
            heading_vec = heading_vec.reshape(heading_vec.shape[:-1])
            df.add_column("state_heading_vec", heading_vec)

            # same for next_state
            heading = df["next_state_heading"].values
            heading_vec = np.stack([np.cos(heading), np.sin(heading)], axis=1)
            # flatten last dimension
            heading_vec = heading_vec.reshape(heading_vec.shape[:-1])
            df.add_column("next_state_heading_vec", heading_vec)

            df.export_hdf5(path, group=group, mode="a")
            df.close()


if __name__ == "__main__":
    # config = habitat.get_config("configs/tasks/pointnav_mp3d_medium.yaml")
    # sample_transitions(config, 256)
    main()
    #
    # augment_dataset()
