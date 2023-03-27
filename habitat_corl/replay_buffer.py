import argparse
import sys
from typing import Dict, Union

import habitat
import torch
import numpy as np
import vaex

from habitat.utils.visualizations.utils import observations_to_image, \
    images_to_video, append_text_to_image
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

    def from_hdf5_group(self, file_path, group):
        df = vaex.open(file_path, group=group)
        states = [col for col in df.get_column_names() if col.startswith("state_")]
        next_states = [col for col in df.get_column_names() if col.startswith("next_state_")]
        for state in states:
            name = state.split("state_")[1]
            self.states[name] = df[state].values
        for next_state in next_states:
            name = next_state.split("next_state_")[1]
            self.next_states[name] = df[next_state].values

        scene = group.split("/")[0]
        ep = group.split("/")[1]
        self.dones = df["done"].values
        self.actions = df["action"].values
        self.rewards = df["reward"].values
        self.scenes = [scene] * len(self.dones)
        self.scenes = np.array(self.scenes)
        self.episode_ids = [ep] * len(self.dones)
        self.episode_ids = np.array(self.episode_ids)


    def append_observations(self, observations, key=None):
        if key is None:
            for key in observations:
                if "state_" in key:
                    key = key.split("state_")[1]
                if key not in self.states:
                    self.states[key] = []
                self.states[key].append(observations[key])
        else:
            if "state_" in key:
                key = key.split("state_")[1]
            if key not in self.states:
                self.states[key] = []
            self.states[key].append(observations)

    def append_next_observations(self, observations, key=None):
        if key is None:
            for key in observations:
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
        action = torch.tensor(self.actions, dtype=torch.float).unsqueeze(-1).to(device)
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
        idx = np.random.randint(0, len(self.actions), batch_size)
        states = {}
        next_states = {}
        for key in self.states.keys():
            states[key] = np.array(self.states[key])[idx]
            if key in self.next_states:
                next_states[key] = np.array(self.next_states[key])[idx]

        actions = np.array(self.actions)[idx]
        rewards = np.array(self.rewards)[idx] if len(self.rewards) > 0 else np.array([])
        dones = np.array(self.dones)[idx] if len(self.dones) > 0 else np.array([])

        sample = ReplayBuffer()
        sample.states = states
        sample.next_states = next_states
        sample.actions = actions
        sample.rewards = rewards
        sample.dones = dones

        return sample

    def normalize_states(self, mean_std: Dict) -> Dict:
        for key in list(mean_std.keys()):
            if key in self.states:
                self.states[key] = (self.states[key] - mean_std[key][0]) / (
                        mean_std[key][1] + 1e-8)
            if key in self.next_states:
                self.next_states[key] = (self.next_states[key] - mean_std[key][0]) / (
                        mean_std[key][1] + 1e-8)

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


    def to_continuous_actions(self, forward=0.25, turn=10):
        # actions = np.zeros((len(self.actions), max(self.actions) + 1))
        # for i, action in enumerate(self.actions):
        #     # get amount of repeats of current action
        #     repeats = 1
        #     for j in range(i + 1, len(self.actions)):
        #         if action == self.actions[j]:
        #             repeats += 1
        #         else:
        #             break
        #     actions[i, action] = repeats
        #
        # actions *= [1, forward, turn * np.pi / 180, turn * np.pi / 180]
        # self.actions = actions
        def normalize_angle(angle):
            return (angle + np.pi) / (2 * np.pi)
        actions = np.zeros(len(self.actions))
        for i in reversed(range(len(self.actions))):
            action = self.actions[i]
            if action == 1 or action == 0:
                actions[i] = normalize_angle(self.states['heading'][i])
            else:
                # look for next different action
                for j in range(i + 1, len(self.actions)):
                    if self.actions[j] != action or j == len(self.actions) - 1:
                        actions[i] = normalize_angle(self.states['heading'][j])
                        break

        self.actions = actions

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
