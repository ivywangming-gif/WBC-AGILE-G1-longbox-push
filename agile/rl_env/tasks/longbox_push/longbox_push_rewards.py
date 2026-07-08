# SPDX-License-Identifier: Apache-2.0
"""Reward terms for G1 longbox push."""

import torch
from isaaclab.managers import SceneEntityCfg


def _finite(x: torch.Tensor, nan: float = 0.0, posinf: float = 0.0, neginf: float = 0.0) -> torch.Tensor:
    return torch.nan_to_num(x, nan=nan, posinf=posinf, neginf=neginf)


def _yaw_from_quat_wxyz(q: torch.Tensor) -> torch.Tensor:
    w, x, y, z = q[:, 0], q[:, 1], q[:, 2], q[:, 3]
    siny_cosp = 2.0 * (w * z + x * y)
    cosy_cosp = 1.0 - 2.0 * (y * y + z * z)
    return torch.atan2(siny_cosp, cosy_cosp)


def box_forward_progress(
    env,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("object"),
    start_x: float = 1.05,
    max_dx: float = 1.2,
) -> torch.Tensor:
    box = env.scene[asset_cfg.name]
    dx = box.data.root_pos_w[:, 0] - start_x
    return _finite(torch.clamp(dx, min=-0.2, max=max_dx), nan=-0.2)


def box_yaw_l2(
    env,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("object"),
) -> torch.Tensor:
    box = env.scene[asset_cfg.name]
    yaw = _yaw_from_quat_wxyz(box.data.root_quat_w)
    return _finite(yaw * yaw, nan=0.0, posinf=10.0, neginf=10.0)


def robot_height_exp(
    env,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    target_z: float = 0.80,
    std: float = 0.18,
) -> torch.Tensor:
    robot = env.scene[asset_cfg.name]
    z_err = robot.data.root_pos_w[:, 2] - target_z
    rew = torch.exp(-(z_err * z_err) / (std * std))
    return _finite(rew, nan=0.0)


def base_near_rear_face_exp(
    env,
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    box_cfg: SceneEntityCfg = SceneEntityCfg("object"),
    box_half_length_x: float = 0.8,
    target_distance: float = 0.45,
    std: float = 0.30,
) -> torch.Tensor:
    robot = env.scene[robot_cfg.name]
    box = env.scene[box_cfg.name]
    rear_face_x = box.data.root_pos_w[:, 0] - box_half_length_x
    base_to_rear = rear_face_x - robot.data.root_pos_w[:, 0]
    err = base_to_rear - target_distance
    rew = torch.exp(-(err * err) / (std * std))
    return _finite(rew, nan=0.0)
