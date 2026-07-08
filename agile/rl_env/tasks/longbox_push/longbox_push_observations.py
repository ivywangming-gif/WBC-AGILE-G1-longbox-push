# SPDX-License-Identifier: Apache-2.0
"""Observation terms for G1 longbox push."""

import torch
from isaaclab.managers import SceneEntityCfg


def _finite(x: torch.Tensor) -> torch.Tensor:
    return torch.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)


def _yaw_from_quat_wxyz(q: torch.Tensor) -> torch.Tensor:
    w, x, y, z = q[:, 0], q[:, 1], q[:, 2], q[:, 3]
    siny_cosp = 2.0 * (w * z + x * y)
    cosy_cosp = 1.0 - 2.0 * (y * y + z * z)
    return torch.atan2(siny_cosp, cosy_cosp)


def box_to_base_pos_w(
    env,
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    box_cfg: SceneEntityCfg = SceneEntityCfg("object"),
) -> torch.Tensor:
    robot = env.scene[robot_cfg.name]
    box = env.scene[box_cfg.name]
    return _finite(box.data.root_pos_w - robot.data.root_pos_w)


def base_to_rear_face_x(
    env,
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    box_cfg: SceneEntityCfg = SceneEntityCfg("object"),
    box_half_length_x: float = 0.8,
) -> torch.Tensor:
    robot = env.scene[robot_cfg.name]
    box = env.scene[box_cfg.name]
    rear_face_x = box.data.root_pos_w[:, 0] - box_half_length_x
    dist = rear_face_x - robot.data.root_pos_w[:, 0]
    return _finite(dist.unsqueeze(-1))


def box_yaw(
    env,
    box_cfg: SceneEntityCfg = SceneEntityCfg("object"),
) -> torch.Tensor:
    box = env.scene[box_cfg.name]
    yaw = _yaw_from_quat_wxyz(box.data.root_quat_w)
    return _finite(yaw.unsqueeze(-1))
