import argparse
import copy
import sys
import time
from typing import Dict, Union

from tqdm import tqdm, trange

import habitat
import torch
import numpy as np
import vaex

from habitat_baselines.il.common.encoders.resnet_encoders import \
    ResnetRGBEncoder, VlnResnetDepthEncoder

config = habitat.get_config("configs/tasks/objectnav_mp3d_il_xl.yaml")

eps = 1e-8


class ReplayBuffer:
    def __init__(self):
        self.dones = []
        self.states = {}
        self.actions = []
        self.rewards = []
        self.next_states = {}
        self.scenes = []
        self.episode_ids = []

    @property
    def is_numpy(self):
        return isinstance(self.dones, np.ndarray)

    def from_hdf5_group(self, file_path, group, ignore_stop=False,
                        continuous=False, single_goal=None, datasets=None):
        df = vaex.open(file_path, group=group)

        if datasets is None:
            datasets = [col for col in df.get_column_names()]

        n_steps = len(df['action'].values)
        if n_steps < 2:
            return
        if ignore_stop:
            n_steps -= 1
        for dataset in datasets:
            if "next_state" in dataset:
                self.extend_next_states(
                    copy.deepcopy(df[dataset].values[:n_steps]),
                    dataset.split("state_")[1]
                )
            elif "state" in dataset:
                self.extend_states(
                    copy.deepcopy(df[dataset].values[:n_steps]),
                    dataset.split("state_")[1]
                )
            elif "action" in dataset:
                ds = "action"
                actions = copy.deepcopy(df[ds].values[:n_steps])
                if not continuous and ignore_stop:
                    actions = np.where(actions == 0, 0, actions - 1)
                self.extend_actions(actions)
            elif "reward" in dataset:
                ds = "reward"
                if not ignore_stop and single_goal is None:
                    self.extend_rewards(copy.deepcopy(df[ds].values[:n_steps]))
                else:
                    # np array of current positions
                    positions = copy.deepcopy(
                        df["next_state_position"].values[:n_steps])
                    # np array of goal positions
                    if single_goal is None:
                        goals = copy.deepcopy(
                            df["state_goal_position"].values[:n_steps])
                    else:
                        goals = np.array([single_goal] * n_steps)
                    # np array of distances between current position and goal
                    distances = np.linalg.norm(positions - goals, axis=1)
                    # np array of rewards
                    rewards = np.where(
                        distances < config.TASK.SUCCESS_DISTANCE, 1, 0)
                    self.extend_rewards(rewards)

            elif "done" in dataset:
                ds = "done"
                if not ignore_stop:
                    self.extend_dones(copy.deepcopy(df[ds].values[:n_steps]))
                else:
                    dones = copy.deepcopy(df[ds].values[:-2])
                    dones = np.append(dones, True)
                    self.extend_dones(dones)
        scene = np.array([group.split("/")[0]] * n_steps)
        self.extend_scenes(scene)
        episode = np.array([group.split("/")[1]] * n_steps)
        self.extend_episodes(episode)
        df.close()

        # df = vaex.open(file_path, group=group)
        # states = [col for col in df.get_column_names() if col.startswith("state_")]
        # next_states = [col for col in df.get_column_names() if col.startswith("next_state_")]
        # for state in states:
        #     name = state.split("state_")[1]
        #     self.states[name] = copy.deepcopy(df[state].values)
        # for next_state in next_states:
        #     name = next_state.split("next_state_")[1]
        #     self.next_states[name] = copy.deepcopy(df[next_state].values)
        #
        # scene = group.split("/")[0]
        # ep = group.split("/")[1]
        # self.dones = copy.deepcopy(df["done"].values)
        # self.actions = copy.deepcopy(df["action"].values)
        # self.rewards = copy.deepcopy(df["reward"].values)
        #
        # if ignore_stop:
        #     self.dones = copy.deepcopy(self.dones[:-1])
        #     self.dones[-1] = True
        #     last_reward = self.rewards[-1]
        #     self.rewards = copy.deepcopy(self.rewards[:-1])
        #     self.rewards[-1] += last_reward
        #     self.actions = copy.deepcopy(self.actions[:-1])
        #     self.actions -= 1
        #     for key in self.states:
        #         self.states[key] = copy.deepcopy(self.states[key][:-1])
        #     for key in self.next_states:
        #         self.next_states[key] = copy.deepcopy(self.next_states[key][:-1])
        #
        # df.close()
        #
        # self.scenes = [scene] * len(self.dones)
        # self.scenes = np.array(self.scenes)
        # self.episode_ids = [ep] * len(self.dones)
        # self.episode_ids = np.array(self.episode_ids)

    def to_numpy(self):
        for key in self.states:
            self.states[key] = np.array(self.states[key])
        for key in self.next_states:
            self.next_states[key] = np.array(self.next_states[key])
        self.dones = np.array(self.dones)
        self.actions = np.array(self.actions)
        self.rewards = np.array(self.rewards)
        self.scenes = np.array(self.scenes)
        self.episode_ids = np.array(self.episode_ids)


    def append_observations(self, observations, key=None, exclude=None):
        if key is None:
            if exclude is None:
                exclude = []
            for key in observations:
                if key in exclude:
                    continue
                if "state_" in key:
                    key = key.split("state_")[1]
                if key not in self.states:
                    self.states[key] = []
                self.states[key].append(observations[key])
        else:
            if "state_" in key:
                key = key.split("state_")[1]
            if key not in self.states:
                self.states[key] = np.array([])
            self.states[key] = np.array.append(self.states[key], observations)

    def append_next_observations(self, observations, key=None, exclude=None):
        if key is None:
            if exclude is None:
                exclude = []
            for key in observations:
                if key in exclude:
                    continue
                if "state_" in key:
                    key = key.split("state_")[1]
                if key not in self.next_states:
                    self.next_states[key] = []
                self.next_states[key].append(observations[key])
        else:
            if "state_" in key:
                key = key.split("state_")[1]
            if key not in self.next_states:
                self.next_states[key] = []
            self.next_states[key].append(observations)

    def append_action(self, action):
        self.actions.append(action)

    def append_reward(self, reward):
        self.rewards.append(reward)

    def extend_rewards(self, rewards: np.ndarray):
        self.rewards.extend(rewards)

    def extend_actions(self, actions: np.ndarray):
        self.actions.extend(actions)

    def extend_dones(self, dones: np.ndarray):
        self.dones.extend(dones)

    def extend_episodes(self, episodes: np.ndarray):
        self.episode_ids.extend(episodes)

    def extend_scenes(self, scenes: np.ndarray):
        self.scenes.extend(scenes)

    def extend_next_states(self, next_states: Union[Dict[str, np.ndarray], np.ndarray], key=None):
        if key is None:
            for key in next_states:
                if key not in self.next_states:
                    self.next_states[key] = []
                self.next_states[key].extend(next_states[key])
        else:
            if key not in self.next_states:
                self.next_states[key] = []
            self.next_states[key].extend(next_states)

    def extend_states(self, states: Union[Dict[str, np.ndarray], np.ndarray], key=None):
        if key is None:
            for key in states:
                if key not in self.states:
                    self.states[key] = []
                self.states[key].extend(states[key])
        else:
            if key not in self.states:
                self.states[key] = []
            self.states[key].extend(states)

    def append_done(self, done):
        self.dones.append(done)

    def append_scene(self, scene):
        self.scenes.append(scene)

    def append_episode_id(self, episode_id):
        self.episode_ids.append(episode_id)
    def to_tensor(self, device=None, state_keys=None, continuous_actions=False, n_actions=4):
        if device is None:
            device = torch.device(
                "cuda:0" if torch.cuda.is_available() else "cpu")
        if state_keys is None:
            state_keys = list(self.states.keys())
        state = []
        next_state = []
        for key in state_keys:
            state.append(torch.tensor(
                self.states[key],
                dtype=torch.float
            ).to(device))
            if key in self.next_states:
                next_state.append(torch.tensor(
                    self.next_states[key],
                    dtype=torch.float
                ).to(device))
        # action = torch.tensor(self.actions, dtype=torch.float).unsqueeze(-1).to(device)
        action = torch.tensor(self.actions, dtype=torch.float).to(device)
        # if continuous_actions:
        #     action = torch.nn.functional.one_hot(action, n_actions).float().to(device)
        reward = torch.tensor(self.rewards, dtype=torch.float).unsqueeze(1).to(device)
        done = torch.tensor(self.dones, dtype=torch.float).unsqueeze(1).to(device)
        state = torch.cat(state, dim=1).to(device)
        if len (next_state) > 0:
            next_state = torch.cat(next_state, dim=1).to(device)
        else:
            next_state = torch.zeros_like(state)

        return state, action, reward, next_state, done

    def sample(self, batch_size):
        assert self.is_numpy

        idx = np.random.randint(0, len(self.actions), batch_size)
        states = {}
        next_states = {}
        for key in self.states.keys():
            states[key] = self.states[key][idx]
            if key in self.next_states:
                next_states[key] = self.next_states[key][idx]

        actions = self.actions[idx]
        rewards = self.rewards[idx] if len(self.rewards) > 0 else np.array([])
        dones = self.dones[idx] if len(self.dones) > 0 else np.array([])

        sample = ReplayBuffer()
        sample.states = states
        sample.next_states = next_states
        sample.actions = actions
        sample.rewards = rewards
        sample.dones = dones

        return sample

    def normalize_states(self, mean_std: Dict) -> Dict:
        normalized = 0 if "depth" not in self.states else 1  # depth is already normalized
        goal = len(self.states.keys())
        for key in list(mean_std.keys()):
            state_key = key.replace("state_", "")
            if state_key in self.states:
                self.states[state_key] = (self.states[state_key] - mean_std[key][0]) / (
                    mean_std[key][1] + 1e-8)
                normalized += 1
            if state_key in self.next_states:
                self.next_states[state_key] = (self.next_states[state_key] - mean_std[key][0]) / (
                    mean_std[key][1] + 1e-8)
        assert normalized == goal

    def _normalize_rgb(self):
        # rgb_states = self.states['rgb'] / 255.0
        # self.states['rgb'] = rgb_states
        pass  # done in the encoder

    def _normalize_depth(self):
        # mean = np.mean(self.states['depth'])
        # std = np.std(self.states['depth'])
        # self.states['depth'] = (self.states['depth'] - mean) / (std + eps)
        pass  # done in the encoder

    def _normalize_semantic(self):
        pass  # done in the encoder

    def _nomralize_default(self, key):
        states = self.states[key]
        mean = np.mean(states)
        std = np.std(states)
        self.states[key] = (states - mean) / (std + eps)

    def _normalize_compass(self):
        states = self.states['compass']
        min = -np.pi
        max = np.pi
        states = (states - min) / (max - min)
        self.states['compass'] = states

    def _normalize_polar(self, key):
        # polar coordinates are in the range of [-pi, pi]
        # except the first dimension which is in the range of [0, +inf]
        states = np.array(self.states[key])

        # check number of dimensions for polar coordinates
        num_dims = states.shape[-1]

        for i in range(num_dims):
            # if i == 0:  # distance r is in the range of [0, +inf]
            #     min = 0.0
            #     max = np.amax(states[..., i])
            #     states[..., i] = (states[..., i] - min) / (max - min + eps)
            # else:  # angles theta and phi are in the range of [?, ?]
            #     # TODO: fix so that the ranges of rotation are accounted for
            #     min = np.amin(states[..., i])
            #     max = np.amax(states[..., i])
            #     states[..., i] = (states[..., i] - min) / (max - min + eps)
            mean = np.mean(states[..., i])
            std = np.std(states[..., i])
            states[..., i] = (states[..., i] - mean) / (std + eps)

        self.states[key] = states

    @property
    def num_steps(self):
        return len(self.actions)

    @property
    def num_episodes(self):
        return len(set(zip(self.scenes, self.episode_ids)))

    def __sizeof__(self):
        size = 0
        for key in self.states.keys():
            size += sys.getsizeof(self.states[key])
            if key in self.next_states:
                size += sys.getsizeof(self.next_states[key])
        size += sys.getsizeof(self.actions)
        size += sys.getsizeof(self.rewards)
        size += sys.getsizeof(self.dones)
        return size


    def to_continuous_actions(self, forward=0.25, turn=30, add_noise=0.25):
        def add_noise_to_vec(vec):
            rad_angle = np.arctan2(vec[1], vec[0])
            return normalize_angle([rad_angle])

        def normalize_angle(angle):
            a = angle[0]
            # add some noise to the angle
            # keep in range [-pi, pi]
            # and make sure it does not turn more than turn/2
            turn_rad = np.deg2rad(turn)
            noise = np.random.normal(0, turn_rad * add_noise)
            noise = np.clip(noise, -turn_rad / 2, turn_rad / 2)
            a += noise
            if a > np.pi:
                a -= 2 * np.pi
            elif a < -np.pi:
                a += 2 * np.pi
            return [np.cos(a), np.sin(a)]

        actions = np.zeros((len(self.actions), 2))
        for i, action in tqdm(enumerate(self.actions)):
            if action == 1 or action == 0:
                if "heading_vec" in self.states:
                    actions[i] = self.states['heading_vec'][i]
                else:
                    actions[i] = normalize_angle(self.states['heading'][i])
            else:
                # look for next different action
                for j in range(i + 1, len(self.actions)):
                    if self.actions[j] != action or j == len(self.actions) - 1:
                        if "heading_vec" in self.states:
                            heading_vec = self.states['heading_vec'][j]
                            actions[i] = add_noise_to_vec(heading_vec)
                        else:
                            actions[i] = normalize_angle(self.states['heading'][j])
                        break

        self.actions = actions


    def filter(self, indices):
        """Filter by keeping only the indices specified."""
        self.states = {key: value[indices] for key, value in self.states.items()}
        self.next_states = {key: value[indices] for key, value in self.next_states.items()}
        self.actions = self.actions[indices]
        self.rewards = self.rewards[indices]
        if len(self.dones) > 0:
            self.dones = self.dones[indices]
        if len(self.episode_ids) > 0:
            self.episode_ids = self.episode_ids[indices]
        if len(self.scenes) > 0:
            self.scenes = self.scenes[indices]

def generate_dataset(
    cfg, num_episodes=None
):
    possible_actions = cfg.TASK.POSSIBLE_ACTIONS

    dataset = ReplayBuffer()

    if hasattr(cfg, "TASK_CONFIG"):
        cfg = cfg.TASK_CONFIG
    with habitat.Env(cfg) as env:
        total_success = 0
        spl = 0

        num_episodes = min(num_episodes, len(env.episodes))

        print(
            "Replaying {}/{} episodes".format(num_episodes, len(env.episodes)))
        for ep_id in range(num_episodes):
            step_index = 1
            total_reward = 0.0

            observations = env.reset()
            info = env.get_metrics()

            episode = env.current_episode

            for data in env.current_episode.reference_replay[step_index:]:
                dataset.append_observations(observations)

                action = possible_actions.index(data.action)
                action_name = env.task.get_action_name(action)

                observations = env.step(action=action)
                info = env.get_metrics()

                dataset.append_action(action)
                if action_name == "STOP":
                    dataset.append_reward(info["success"])
                else:
                    dataset.append_reward(0)
                dataset.append_next_observations(observations)
                dataset.append_done(action_name == "STOP")

                if action_name == "STOP":
                    break
            print("Total reward for trajectory: {}".format(total_reward))

            if len(episode.reference_replay) <= 500:
                total_success += info["success"]
                spl += info["spl"]

        print("SPL: {}, {}, {}".format(spl / num_episodes, spl, num_episodes))
        print("Success: {}, {}, {}".format(total_success / num_episodes,
                                           total_success, num_episodes))

        dataset.to_tensor()
        # print(replay_buffer["states"].shape)
        # print(replay_buffer["states"][0])
        dataset.states['rgb'] = process_rgb(env, dataset)
        dataset.states['depth'] = process_depth(env, dataset)
        return dataset


def get_input_dims(config):
    linear_input_size = 0
    if "pointgoal_with_gps_compass" in config.MODEL.used_inputs:
        linear_input_size += config.TASK_CONFIG.TASK.POINTGOAL_WITH_GPS_COMPASS_SENSOR.DIMENSIONALITY
    if "proximity" in config.MODEL.used_inputs:
        linear_input_size += 1
    if "agent_map_coord" in config.MODEL.used_inputs:
        linear_input_size += 2
    if "agent_angle" in config.MODEL.used_inputs:
        linear_input_size += 1
    if "position" in config.MODEL.used_inputs:
        linear_input_size += 3
    if "heading" in config.MODEL.used_inputs:
        linear_input_size += 1
    if "pointgoal" in config.MODEL.used_inputs:
        linear_input_size += config.TASK_CONFIG.TASK.POINTGOAL_SENSOR.DIMENSIONALITY
    if "goal_position" in config.MODEL.used_inputs:
        linear_input_size += 3
    if "heading_vec" in config.MODEL.used_inputs:
        linear_input_size += 2
    if "depth" in config.MODEL.used_inputs:
        linear_input_size += config.MODEL.DEPTH_ENCODER.output_size

    return linear_input_size


def process_rgb(env, dataset):
    cfg = habitat.get_config("habitat_corl/configs/bc_objectnav.yaml")
    encoder = ResnetRGBEncoder(observation_space=env.observation_space)
    observations = dataset.states
    output = encoder(observations)
    print(output)
    print(output.shape)
    return output


def process_depth(env, dataset):
    cfg = habitat.get_config("habitat_corl/configs/bc_objectnav.yaml")
    encoder = VlnResnetDepthEncoder(observation_space=env.observation_space)
    observations = dataset.states
    output = encoder(observations)
    print(output)
    print(output.shape)
    return output


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--path", type=str, default="replays/demo_1.json.gz"
    )
    parser.add_argument(
        "--num-episodes", type=int, default=1
    )
    args = parser.parse_args()
    cfg = config
    cfg.defrost()
    cfg.DATASET.DATA_PATH = args.path

    # Set to a high limit to allow replaying episodes with
    # number of steps greater than ObjectNav episode step
    # limit.
    cfg.ENVIRONMENT.MAX_EPISODE_STEPS = 5000

    # Set to a high limit to allow loading episodes with
    # number of steps greater than ObjectNav episode step
    # limit in the replay buffer.
    cfg.DATASET.MAX_REPLAY_STEPS = 5000
    cfg.freeze()

    generate_dataset(
        cfg,
        num_episodes=args.num_episodes
    )


if __name__ == "__main__":
    main()
