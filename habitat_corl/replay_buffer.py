import argparse
from typing import Dict

import habitat
import torch
import numpy as np

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

    def append_observations(self, observations, key=None):
        if key is None:
            for key in observations:
                if key not in self.states:
                    self.states[key] = []
                self.states[key].append(observations[key])
        else:
            if key not in self.states:
                self.states[key] = []
            self.states[key].append(observations)

    def append_next_observations(self, observations, key=None):
        if key is None:
            for key in observations:
                if key not in self.next_states:
                    self.next_states[key] = []
                self.next_states[key].append(observations[key])
        else:
            if key not in self.next_states:
                self.next_states[key] = []
            self.next_states[key].append(observations)

    def append_action(self, action):
        self.actions.append(action)

    def append_reward(self, reward):
        self.rewards.append(reward)

    def append_done(self, done):
        self.dones.append(done)

    def append_scene(self, scene):
        self.scenes.append(scene)

    def append_episode_id(self, episode_id):
        self.episode_ids.append(episode_id)
    def to_tensor(self, device=None):
        if device is None:
            device = torch.device(
                "cuda:0" if torch.cuda.is_available() else "cpu")
        for key in self.states.keys():
            self.states[key] = torch.tensor(
                self.states[key],
                dtype=torch.float
            ).to(device)
            if key in self.next_states:
                self.next_states[key] = torch.tensor(
                    self.next_states[key],
                    dtype=torch.float
                ).to(device)
        self.actions = torch.tensor(self.actions, dtype=torch.float).to(device)
        self.rewards = torch.tensor(self.rewards, dtype=torch.float).to(device)
        self.dones = torch.tensor(self.dones, dtype=torch.float).to(device)
        return self

    def sample(self, batch_size):
        idx = np.random.randint(0, len(self.actions), batch_size)
        states = {}
        next_states = {}
        for key in self.states.keys():
            states[key] = self.states[key][idx]
            next_states[key] = self.next_states[key][idx]
        actions = self.actions[idx]
        rewards = self.rewards[idx]
        dones = self.dones[idx]

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
