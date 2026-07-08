# SPDX-License-Identifier: Apache-2.0
"""Reset/event helpers for G1 longbox push."""

import torch

from isaaclab.managers import SceneEntityCfg


def reset_robot_joints_to_default(
    env,
    env_ids: torch.Tensor,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
):
    """Reset robot joints to articulation default joint position/velocity."""
    robot = env.scene[asset_cfg.name]

    joint_pos = robot.data.default_joint_pos[env_ids].clone()
    joint_vel = robot.data.default_joint_vel[env_ids].clone()

    robot.write_joint_state_to_sim(joint_pos, joint_vel, env_ids=env_ids)
    robot.set_joint_position_target(joint_pos, env_ids=env_ids)
