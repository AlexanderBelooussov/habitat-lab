from habitat import Config
from habitat_baselines.config.default import get_config
from habitat_baselines.common.environments import get_env_class
from habitat_baselines.common.obs_transformers import \
    apply_obs_transforms_obs_space, get_active_obs_transforms
from habitat_baselines.il.env_based.common.rollout_storage import \
    RolloutStorage
from habitat_baselines.utils.env_utils import construct_envs


def main():
    config_path = "habitat-corl/configs/bc_objectnav.yaml"
    config = get_config(config_path, [])
    envs = construct_envs(
        config, get_env_class(config.ENV_NAME)
    )

    bc_config = config.ORL.BC

    observation_space = envs.observation_spaces[0]
    obs_transforms = get_active_obs_transforms(config)
    observation_space = apply_obs_transforms_obs_space(
        observation_space, obs_transforms
    )
    obs_space = observation_space

    rollouts = RolloutStorage(
        bc_config.num_steps,
        envs.num_envs,
        obs_space,
        envs.action_spaces[0],
        config.MODEL.STATE_ENCODER.hidden_size,
        config.MODEL.STATE_ENCODER.num_recurrent_layers,
    )
    print("generated dataset succesfully!")

    observation = envs.reset()
    print(observation)
    for x in observation:
        print(x.shape)

if __name__ == "__main__":
    main()
