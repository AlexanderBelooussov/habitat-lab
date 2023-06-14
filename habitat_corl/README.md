# Training
Working directory: habitat-lab/
```bash
python habitat_corl/scipts/train.py
```

### Options
* `--algorithm`: Algorithm to use, default: `sacn`, options: `sacn, dt, bc, td3_bc, sac_n, td3bc, bc_10, bc10, iql, edac, sacnd, sacn_d, td3bcd, td3bc_d, random, cql, cql_d, cqld, lb_sac, lbsac`
* `--task`: Task to use, default: `singlegoal`, options: `pointnav, objectnav, singlegoal, pointnavdepth`
* `--scene`: Scene to use, default: `medium`, options: `medium, large, small, xl, debug`
* `--ignore_stop`: Ignore stop action in the environment (Boolean), default: `False`
* `--n_eval_episodes`: Number of episodes to evaluate on (Integer), default: `100`
* `--seed`: Random seed to use (Integer), default: `1`
* `--web_dataset`: Use web dataset (Boolean), default: `False`
* `--web_dataset_only`: Use web dataset, without shortest path dataset (Boolean), default: `False`
* `--comment`: Comment to add to the run name (String), default: `""`
* `--group`: Additional change to group name (String), default: `""`
* `--n_layers`: Number of layers for the network, -1 for default (Integer), default: `-1`
* `--noise`: Noise to add to the actions (Float), default: `0.25`
* `--n_updates`: Number of updates to run , -1 for default(Integer), default: `-1`
* `--tau`: Tau for soft updates (Float), default: `-1`
* `--blind`: Use blind agent (in PointNavDepth) (Boolean), default: `False`


# Evaluating checkpoints
An example of evaluating training checkpoints can be seen in habitat_corl/scripts/eval_checkpoints.py
