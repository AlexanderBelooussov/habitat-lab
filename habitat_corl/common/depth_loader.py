import time

import torch
from habitat.sims import make_sim
from habitat.datasets import make_dataset
from habitat.tasks.registration import make_task
import quaternion
from habitat_baselines.il.common.encoders.resnet_encoders import \
    VlnResnetDepthEncoder
import numpy as np
from habitat_sim.agent import AgentState
from tqdm import trange


class DepthLoader():
    def __init__(self, model_config, task_config, observation_space):
        self.depth_encoder = VlnResnetDepthEncoder(
            observation_space,
            output_size=model_config.DEPTH_ENCODER.output_size,
            checkpoint=model_config.DEPTH_ENCODER.ddppo_checkpoint,
            backbone=model_config.DEPTH_ENCODER.backbone,
            trainable=model_config.DEPTH_ENCODER.trainable,
        )
        self.sim = make_sim(id_sim=task_config.SIMULATOR.TYPE,
                            config=task_config.SIMULATOR)

        self.dataset = make_dataset(
            id_dataset=task_config.DATASET.TYPE,
            config=task_config.DATASET,
        )
        self.task = make_task(
            task_config.TASK.TYPE,
            config=task_config.TASK,
            sim=self.sim,
            dataset=self.dataset,
        )
        self.current_episode = self.dataset.episodes[0]
        self.sim.reset()

    def get_depth_from_state(self, current_state):
        position = current_state.position
        rotation = current_state.rotation
        self.sim.agents[0].set_state(current_state)
        observations = self.sim.get_sensor_observations()
        observations["depth"] = np.expand_dims(observations["depth"], 0)
        observations["depth"] = torch.from_numpy(
            np.expand_dims(observations["depth"], -1)
        ).float()
        depth_encoding = self.depth_encoder(observations)[0].detach().numpy()
        return depth_encoding

    def get_depth_from_postion_rotation(self, position, rotation):
        agent_state = AgentState(position, rotation)
        return self.get_depth_from_state(agent_state)
        return depth_encoding


    def add_depth_to_dataset(self, dataset, next_state=False):
        depth_data = []
        start = time.time()
        for i in trange(dataset.num_steps, desc="Adding depth to dataset"):
            position = dataset.states['position'][i]
            heading_vec = dataset.states['heading_vec'][i]
            rotation = np.arctan2(heading_vec[1], heading_vec[0])
            axis_angle = rotation * np.array([0., 1., 0.])
            rotation = quaternion.from_rotation_vector(axis_angle)
            depth_encoding = self.get_depth_from_postion_rotation(position, rotation)
            depth_data.append(depth_encoding)
            if i % 10_000 == 0:
                print(f"{time.time()-start}: {i} / {dataset.num_steps}")
        dataset.states['depth'] = np.array(depth_data)

        if next_state:
            # add next state depth
            dataset.next_states['depth'] = np.empty_like(dataset.states['depth'])
            for i in trange(dataset.num_steps, desc="Adding next state depth to dataset"):
                if i == dataset.num_steps - 1:
                    dataset.next_states['depth'][i] = dataset.states['depth'][i]
                    continue
                current_episode = dataset.episode_ids[i]
                next_episode = dataset.episode_ids[i+1]
                if current_episode == next_episode:
                    dataset.next_states['depth'][i] = dataset.states['depth'][i+1]
                else:
                    dataset.next_states['depth'][i] = dataset.states['depth'][i]
        return dataset

