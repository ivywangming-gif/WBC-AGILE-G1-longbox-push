# SPDX-License-Identifier: Apache-2.0
"""G1 longbox push task registration."""

import gymnasium as gym

from . import agents

gym.register(
    id="G1-LongBoxPush-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.g1_longbox_push_env_cfg:G1LongBoxPushEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:G1PickPlacePPORunnerCfg",
    },
)
