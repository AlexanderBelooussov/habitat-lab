# inspiration:
# 1. https://github.com/kzl/decision-transformer/blob/master/gym/decision_transformer/models/decision_transformer.py  # noqa
# 2. https://github.com/karpathy/minGPT
import argparse
from typing import Any, DefaultDict, Dict, List, Optional, Tuple, Union
from collections import defaultdict
from dataclasses import asdict, dataclass
import os
import random
import uuid

import gym  # noqa
import numpy as np
import torch
import torch.nn as nn
from gym import ObservationWrapper
from torch.nn import functional as F  # noqa
from torch.utils.data import DataLoader, IterableDataset
from tqdm.auto import tqdm, trange  # noqa
import wandb

import habitat
from habitat.utils.visualizations.utils import images_to_video, \
    observations_to_image
from habitat_corl.common.utils import restructure_results, train_eval_split, \
    set_seed, wandb_init
from habitat_corl.common.wrappers import wrap_env
from habitat_corl.replay_buffer import get_input_dims, ReplayBuffer
from habitat_corl.shortest_path_dataset import calc_mean_std, \
    get_stored_groups, register_position_sensor
from habitat_baselines.config.default import get_config


def strin_to_tuple(string, dtype):
    # https://www.geeksforgeeks.org/python-convert-tuple-string-to-integer-tuple/ # noqa
    return tuple(dtype(num) for num in
          string.replace('(', '').replace(')', '').replace('...',
                                                          '').split(
              ', '))


# some utils functionalities specific for Decision Transformer
def pad_along_axis(
    arr: np.ndarray, pad_to: int, axis: int = 0, fill_value: float = 0.0
) -> np.ndarray:
    pad_size = pad_to - arr.shape[axis]
    if pad_size <= 0:
        return arr

    npad = [(0, 0)] * arr.ndim
    npad[axis] = (0, pad_size)
    return np.pad(arr, pad_width=npad, mode="constant",
                  constant_values=fill_value)


def discounted_cumsum(x: np.ndarray, gamma: float) -> np.ndarray:
    cumsum = np.zeros_like(x)
    cumsum[-1] = x[-1]
    for t in reversed(range(x.shape[0] - 1)):
        cumsum[t] = x[t] + gamma * cumsum[t + 1]
    return cumsum


def load_trajectories(
    config, gamma: float = 1.0, groups=None, used_inputs=None, action_dim=4
) -> Tuple[List[DefaultDict[str, np.ndarray]], Dict[str, Any]]:
    if used_inputs is None:
        used_inputs = ["postion", "heading", "pointgoal"]
    groups = get_stored_groups(config.TASK_CONFIG) if groups is None else groups
    dataset = [ReplayBuffer() for _ in groups]
    for i, group in enumerate(groups):
        dataset[i].from_hdf5_group(
            config.TASK_CONFIG.DATASET.SP_DATASET_PATH,
            group,
            ignore_stop=config.RL.DT.ignore_stop
        )
    traj, traj_len = [], []

    data_, episode_step = defaultdict(list), 0
    for i, buffer in tqdm(enumerate(dataset), desc="Processing trajectories"):
        for episode_step in range(buffer.num_steps):
            state = [buffer.states[key][episode_step] for key in used_inputs]
            state = np.concatenate(state, axis=-1)
            data_["observations"].append(state)
            # one-hot encode actions
            data_["actions"].append(
                np.eye(action_dim)[buffer.actions[episode_step]]
            )

            data_["rewards"].append(buffer.rewards[episode_step])

        episode_data = {k: np.array(v, dtype=np.float32) for k, v in
                        data_.items()}
        # return-to-go if gamma=1.0, just discounted returns else
        episode_data["returns"] = discounted_cumsum(
            episode_data["rewards"], gamma=gamma
        )
        traj.append(episode_data)
        traj_len.append(episode_step)
        # reset trajectory buffer
        data_, episode_step = defaultdict(list), 0

    # needed for normalization, weighted sampling, other stats can be added also
    # concat all states to get mean and std
    all_obs = np.concatenate([d["observations"] for d in traj], axis=0)
    info = {
        "obs_mean": all_obs.mean(0, keepdims=True),
        "obs_std": all_obs.std(0, keepdims=True) + 1e-6,
        "traj_lens": np.array(traj_len),
    }
    return traj, info


class SequenceDataset(IterableDataset):
    def __init__(self, config, seq_len: int = 10,
                 reward_scale: float = 1.0, action_dim=4, groups=None):
        self.dataset, info = load_trajectories(
            config, gamma=1.0,
            used_inputs=config.MODEL.used_inputs,
            action_dim=action_dim,
            groups=groups
        )
        self.reward_scale = reward_scale
        self.seq_len = seq_len

        self.state_mean = info["obs_mean"]
        self.state_std = info["obs_std"]
        # https://github.com/kzl/decision-transformer/blob/e2d82e68f330c00f763507b3b01d774740bee53f/gym/experiment.py#L116 # noqa
        self.sample_prob = info["traj_lens"] / info["traj_lens"].sum()

    def __prepare_sample(self, traj_idx, start_idx):
        traj = self.dataset[traj_idx]
        # https://github.com/kzl/decision-transformer/blob/e2d82e68f330c00f763507b3b01d774740bee53f/gym/experiment.py#L128 # noqa
        states = traj["observations"][start_idx: start_idx + self.seq_len]
        actions = traj["actions"][start_idx: start_idx + self.seq_len]
        returns = traj["returns"][start_idx: start_idx + self.seq_len]
        time_steps = np.arange(start_idx, start_idx + self.seq_len)

        states = (states - self.state_mean) / self.state_std
        returns = returns * self.reward_scale
        # pad up to seq_len if needed
        mask = np.hstack(
            [np.ones(states.shape[0]),
             np.zeros(self.seq_len - states.shape[0])]
        )
        if states.shape[0] < self.seq_len:
            states = pad_along_axis(states, pad_to=self.seq_len)
            actions = pad_along_axis(actions, pad_to=self.seq_len)
            returns = pad_along_axis(returns, pad_to=self.seq_len)

        return states, actions, returns, time_steps, mask

    def __iter__(self):
        while True:
            traj_idx = np.random.choice(len(self.dataset), p=self.sample_prob)
            start_idx = random.randint(0,
                                       self.dataset[traj_idx]["rewards"].shape[
                                           0] - 1)
            yield self.__prepare_sample(traj_idx, start_idx)


# Decision Transformer implementation
class TransformerBlock(nn.Module):
    def __init__(
        self,
        seq_len: int,
        embedding_dim: int,
        num_heads: int,
        attention_dropout: float,
        residual_dropout: float,
    ):
        super().__init__()
        self.norm1 = nn.LayerNorm(embedding_dim)
        self.norm2 = nn.LayerNorm(embedding_dim)
        self.drop = nn.Dropout(residual_dropout)

        self.attention = nn.MultiheadAttention(
            embedding_dim, num_heads, attention_dropout
        )
        self.mlp = nn.Sequential(
            nn.Linear(embedding_dim, 4 * embedding_dim),
            nn.GELU(),
            nn.Linear(4 * embedding_dim, embedding_dim),
            nn.Dropout(residual_dropout),
        )
        # True value indicates that the corresponding position is not allowed to attend
        self.register_buffer(
            "causal_mask", ~torch.tril(torch.ones(seq_len, seq_len)).to(bool)
        )
        self.seq_len = seq_len

    # [batch_size, seq_len, emb_dim] -> [batch_size, seq_len, emb_dim]
    def forward(
        self, x: torch.Tensor, padding_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        causal_mask = self.causal_mask[: x.shape[1], : x.shape[1]]

        norm_x = self.norm1(x)
        attention_out = self.attention(
            query=norm_x,
            key=norm_x,
            value=norm_x,
            attn_mask=causal_mask,
            key_padding_mask=padding_mask,
            need_weights=False,
        )[0]
        # by default pytorch attention does not use dropout
        # after final attention weights projection, while minGPT does:
        # https://github.com/karpathy/minGPT/blob/7218bcfa527c65f164de791099de715b81a95106/mingpt/model.py#L70 # noqa
        x = x + self.drop(attention_out)
        x = x + self.mlp(self.norm2(x))
        return x


class DecisionTransformer(nn.Module):
    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        seq_len: int = 10,
        episode_len: int = 1000,
        embedding_dim: int = 128,
        num_layers: int = 4,
        num_heads: int = 8,
        attention_dropout: float = 0.0,
        residual_dropout: float = 0.0,
        embedding_dropout: float = 0.0,
        max_action: float = 1.0,
    ):
        super().__init__()
        self.emb_drop = nn.Dropout(embedding_dropout)
        self.emb_norm = nn.LayerNorm(embedding_dim)

        self.out_norm = nn.LayerNorm(embedding_dim)
        # additional seq_len embeddings for padding timesteps
        self.timestep_emb = nn.Embedding(episode_len + seq_len, embedding_dim)
        self.state_emb = nn.Linear(state_dim, embedding_dim)
        self.action_emb = nn.Linear(action_dim, embedding_dim)
        self.return_emb = nn.Linear(1, embedding_dim)

        self.blocks = nn.ModuleList(
            [
                TransformerBlock(
                    seq_len=3 * seq_len,
                    embedding_dim=embedding_dim,
                    num_heads=num_heads,
                    attention_dropout=attention_dropout,
                    residual_dropout=residual_dropout,
                )
                for _ in range(num_layers)
            ]
        )
        self.action_head = nn.Sequential(nn.Linear(embedding_dim, action_dim),
                                         nn.Tanh())
        self.seq_len = seq_len
        self.embedding_dim = embedding_dim
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.episode_len = episode_len
        self.max_action = max_action

        self.apply(self._init_weights)

    @staticmethod
    def _init_weights(module: nn.Module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if isinstance(module, nn.Linear) and module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.LayerNorm):
            torch.nn.init.zeros_(module.bias)
            torch.nn.init.ones_(module.weight)

    def forward(
        self,
        states: torch.Tensor,  # [batch_size, seq_len, state_dim]
        actions: torch.Tensor,  # [batch_size, seq_len, action_dim]
        returns_to_go: torch.Tensor,  # [batch_size, seq_len]
        time_steps: torch.Tensor,  # [batch_size, seq_len]
        padding_mask: Optional[torch.Tensor] = None,  # [batch_size, seq_len]
    ) -> torch.FloatTensor:
        batch_size, seq_len = states.shape[0], states.shape[1]
        # [batch_size, seq_len, emb_dim]
        time_emb = self.timestep_emb(time_steps)
        state_emb = self.state_emb(states) + time_emb
        act_emb = self.action_emb(actions) + time_emb
        returns_emb = self.return_emb(returns_to_go.unsqueeze(-1)) + time_emb

        # [batch_size, seq_len * 3, emb_dim], (r_0, s_0, a_0, r_1, s_1, a_1, ...)
        sequence = (
            torch.stack([returns_emb, state_emb, act_emb], dim=1)
            .permute(0, 2, 1, 3)
            .reshape(batch_size, 3 * seq_len, self.embedding_dim)
        )
        if padding_mask is not None:
            # [batch_size, seq_len * 3], stack mask identically to fit the sequence
            padding_mask = (
                torch.stack([padding_mask, padding_mask, padding_mask], dim=1)
                .permute(0, 2, 1)
                .reshape(batch_size, 3 * seq_len)
            )
        # LayerNorm and Dropout (!!!) as in original implementation,
        # while minGPT & huggingface uses only embedding dropout
        out = self.emb_norm(sequence)
        out = self.emb_drop(out)
        out = out.permute(1, 0, 2)

        for block in self.blocks:
            out = block(out, padding_mask=padding_mask)
        out = out.permute(1, 0, 2)
        out = self.out_norm(out)
        # [batch_size, seq_len, action_dim]
        # predict actions only from state embeddings
        out = self.action_head(out[:, 1::3]) * self.max_action
        return out


# Training and evaluation logic
@torch.no_grad()
def eval_rollout(
    model: DecisionTransformer,
    env: gym.Env,
    target_return: float,
    device: str = "cpu",
    video=False,
    video_dir: str = "demos",
    video_prefix: str = "dt",
    eval_iteration=0,
    ignore_stop=False,
    success_distance=0.2,
) -> Tuple[float, float]:
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

    video_frames = []
    states = torch.zeros(
        1, model.episode_len + 1, model.state_dim, dtype=torch.float,
        device=device
    )
    actions = torch.zeros(
        1, model.episode_len, model.action_dim, dtype=torch.float,
        device=device
    )
    returns = torch.zeros(1, model.episode_len + 1, dtype=torch.float,
                          device=device)
    time_steps = torch.arange(model.episode_len, dtype=torch.long,
                              device=device)
    time_steps = time_steps.view(1, -1)
    state, raw = env.reset()
    states[:] = torch.as_tensor(state, device=device)
    returns[:, 0] = torch.as_tensor(target_return, device=device)

    # cannot step higher than model episode len, as timestep embeddings will crash
    episode_return, episode_len = 0.0, 0.0
    for step in range(model.episode_len):
        # first select history up to step, then select last seq_len states,
        # step + 1 as : operator is not inclusive, last action is dummy with zeros
        # (as model will predict last, actual last values are not important)
        predicted_actions = model(  # fix this noqa!!!
            states[:, : step + 1][:, -model.seq_len:],  # noqa
            actions[:, : step + 1][:, -model.seq_len:],  # noqa
            returns[:, : step + 1][:, -model.seq_len:],  # noqa
            time_steps[:, : step + 1][:, -model.seq_len:],  # noqa
        )
        predicted_action = predicted_actions[0, -1].cpu().numpy()
        predicted_action = np.argmax(predicted_action)
        next_state, raw = env.step(predicted_action)
        info = env.get_metrics()
        if video:
            frame = observations_to_image(raw, info)
            video_frames.append(frame)
        reward = info["success"]
        # at step t, we predict a_t, get s_{t + 1}, r_{t + 1}
        actions[:, step] = torch.as_tensor(predicted_action)
        states[:, step + 1] = torch.as_tensor(next_state)
        returns[:, step + 1] = torch.as_tensor(returns[:, step] - reward)

        episode_return += reward
        episode_len += 1

        if env.episode_over :
            break
        position = env.sim.get_agent_state().position
        goal = env.current_episode.goals[0].position
        distance = np.linalg.norm(np.array(position) - np.array(goal))
        if ignore_stop and distance < success_distance:
            info["success"] = True
            break

    if video:
        make_videos(video_frames, video_prefix, eval_iteration)
    return info
    # return episode_return, episode_len


def train(config):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dt_config = config.RL.DT
    set_seed(dt_config.train_seed,
             deterministic_torch=dt_config.deterministic_torch)
    # init wandb session for logging
    wandb_init(config)

    env = habitat.Env(config.TASK_CONFIG)

    train_episodes, eval_episodes = train_eval_split(
        env=env,
        config=config,
        n_eval_episodes=dt_config.eval_episodes,
    )
    env.episodes = eval_episodes
    def ep_iter(eval_episodes):
        while True:
            for ep in eval_episodes:
                yield ep
    env.episode_iterator = ep_iter(eval_episodes)

    # data & dataloader setup
    dataset = SequenceDataset(
        config, seq_len=dt_config.seq_len,
        reward_scale=dt_config.reward_scale,
        action_dim=4-int(dt_config.ignore_stop),
        groups=train_episodes
    )
    trainloader = DataLoader(
        dataset,
        batch_size=dt_config.batch_size,
        pin_memory=True,
        num_workers=dt_config.num_workers,
    )
    # evaluation environment with state & reward preprocessing (as in dataset above)
    eval_env = wrap_env(
        env,
        used_inputs=config.MODEL.used_inputs,
        state_mean=dataset.state_mean,
        state_std=dataset.state_std,
        reward_scale=dt_config.reward_scale,
        ignore_stop=dt_config.ignore_stop,
        continuous=False
    )
    # model & optimizer & scheduler setup
    state_dim = get_input_dims(config)
    action_dim = eval_env.action_space.n
    model = DecisionTransformer(
        state_dim=state_dim,
        action_dim=action_dim,
        embedding_dim=dt_config.embedding_dim,
        seq_len=dt_config.seq_len,
        episode_len=dt_config.episode_len,
        num_layers=dt_config.num_layers,
        num_heads=dt_config.num_heads,
        attention_dropout=dt_config.attention_dropout,
        residual_dropout=dt_config.residual_dropout,
        embedding_dropout=dt_config.embedding_dropout,
        max_action=dt_config.max_action,
    ).to(device)
    betas = dt_config.betas
    betas = strin_to_tuple(betas, float)

    optim = torch.optim.AdamW(
        model.parameters(),
        lr=dt_config.learning_rate,
        weight_decay=dt_config.weight_decay,
        betas=betas,
    )
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optim,
        lambda steps: min((steps + 1) / dt_config.warmup_steps, 1),
    )
    # save config to the checkpoint
    if config.CHECKPOINT_FOLDER is not None:
        print(f"Checkpoints path: {config.CHECKPOINT_FOLDER}")
        os.makedirs(config.CHECKPOINT_FOLDER, exist_ok=True)
        with open(os.path.join(config.CHECKPOINT_FOLDER, "config.yaml"),
                  "w") as f:
            f.write(config.dump())

    print(f"Total parameters: {sum(p.numel() for p in model.parameters())}")
    trainloader_iter = iter(trainloader)
    for step in trange(dt_config.update_steps, desc="Training"):
        batch = next(trainloader_iter)
        states, actions, returns, time_steps, mask = [b.to(device) for b in
                                                      batch]
        # True value indicates that the corresponding key value will be ignored
        padding_mask = ~mask.to(torch.bool)
        predicted_actions = model(
            states=states,
            actions=actions,
            returns_to_go=returns,
            time_steps=time_steps,
            padding_mask=padding_mask,
        )
        loss = F.mse_loss(predicted_actions, actions.detach(),
                          reduction="none")
        # cross entropy loss instead of mse because of discrete actions
        # loss = F.cross_entropy(predicted_actions, actions.detach(),
        #                        reduction="none")
        # [batch_size, seq_len, action_dim] * [batch_size, seq_len, 1]
        loss = (loss * mask.unsqueeze(-1)).mean()

        optim.zero_grad()
        loss.backward()
        if dt_config.clip_grad is not None:
            torch.nn.utils.clip_grad_norm_(model.parameters(),
                                           dt_config.clip_grad)
        optim.step()
        scheduler.step()

        wandb.log(
            {
                "train_loss": loss.item(),
                "learning_rate": scheduler.get_last_lr()[0],
            },
            step=step,
        )

        # validation in the env for the actual online performance
        if step % dt_config.eval_every == 0 or step == dt_config.update_steps - 1:
            model.eval()
            target_returns = strin_to_tuple(dt_config.target_returns, float)
            for target_return in target_returns:
                eval_env.seed(dt_config.eval_seed)
                eval_scores = []
                for i in trange(dt_config.eval_episodes, desc="Evaluation",
                                leave=False):
                    result = eval_rollout(
                        model=model,
                        env=eval_env,
                        target_return=target_return * dt_config.reward_scale,
                        device=device,
                        video=True,
                        video_dir=config.VIDEO_DIR,
                        video_prefix=f"dt/{int(target_return)}/dt",
                        eval_iteration=i,
                        success_distance=config.TASK_CONFIG.TASK.SUCCESS_DISTANCE,
                        ignore_stop=dt_config.ignore_stop,
                    )
                    eval_scores.append(result)
                eval_scores = restructure_results(eval_scores)
                for key in eval_scores:
                    wandb.log(
                        {
                            f"eval/{key}_mean": np.mean(eval_scores[key]),
                            f"eval/{key}_std": np.std(eval_scores[key])
                        },
                        step=step,
                    )
            model.train()

    if config.CHECKPOINT_FOLDER is not None:
        checkpoint = {
            "model_state": model.state_dict(),
            "state_mean": dataset.state_mean,
            "state_std": dataset.state_std,
        }
        torch.save(checkpoint,
                   os.path.join(config.CHECKPOINT_FOLDER, "dt_checkpoint.pt"))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str,
                        default="habitat_corl/configs/dt_pointnav.yaml")
    args = parser.parse_args()

    config = get_config(args.config)
    register_position_sensor(config.TASK_CONFIG)
    train(config)


if __name__ == "__main__":
    main()
