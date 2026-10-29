# Copyright (c) 2026, BipedalSF Project.
# SPDX-License-Identifier: BSD-3-Clause

"""SF-TRON2A biped velocity tracking task registration."""

import gymnasium as gym

from . import agents

gym.register(
    id="Bipedalsf-Locomotion-SF-Tron2A-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.env_cfg:SFTron2AEnvCfg",
        "default_agent": "rsl_rl",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:PPORunnerCfg",
    },
)
