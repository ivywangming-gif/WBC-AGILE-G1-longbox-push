# SPDX-License-Identifier: Apache-2.0
"""Headless audit for G1-LongBoxPush-v0.

This script does not train. It only:
- creates the registered task,
- resets it,
- steps with zero actions,
- prints box/base/root metrics.
"""

import argparse
import math

from isaaclab.app import AppLauncher


parser = argparse.ArgumentParser(description="Audit G1 longbox push reset/step metrics.")
parser.add_argument("--task", type=str, default="G1-LongBoxPush-v0")
parser.add_argument("--num_envs", type=int, default=1)
parser.add_argument("--num_steps", type=int, default=200)
parser.add_argument("--print_every", type=int, default=50)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import gymnasium as gym
import torch

import agile.rl_env.tasks  # noqa: F401
from isaaclab_tasks.utils import load_cfg_from_registry


def quat_yaw_wxyz(q: torch.Tensor) -> torch.Tensor:
    """Return yaw from quaternion in wxyz convention."""
    w, x, y, z = q[:, 0], q[:, 1], q[:, 2], q[:, 3]
    siny_cosp = 2.0 * (w * z + x * y)
    cosy_cosp = 1.0 - 2.0 * (y * y + z * z)
    return torch.atan2(siny_cosp, cosy_cosp)


def get_asset(scene, name: str):
    try:
        return scene[name]
    except Exception:
        return getattr(scene, name)


def main():
    env_cfg = load_cfg_from_registry(args_cli.task, "env_cfg_entry_point")
    env_cfg.scene.num_envs = args_cli.num_envs

    env = gym.make(args_cli.task, cfg=env_cfg, render_mode=None)
    unwrapped = env.unwrapped

    env.reset()

    box = get_asset(unwrapped.scene, "object")
    robot = get_asset(unwrapped.scene, "robot")

    box_x0 = box.data.root_pos_w[:, 0].clone()
    base_x0 = robot.data.root_pos_w[:, 0].clone()

    action_dim = unwrapped.action_manager.total_action_dim
    actions = torch.zeros((unwrapped.num_envs, action_dim), device=unwrapped.device)

    print("=== audit_start ===")
    print(f"task={args_cli.task}")
    print(f"num_envs={unwrapped.num_envs}")
    print(f"action_dim={action_dim}")
    print(f"box_pos0={box.data.root_pos_w[0].detach().cpu().tolist()}")
    print(f"base_pos0={robot.data.root_pos_w[0].detach().cpu().tolist()}")
    print(f"robot_z0={robot.data.root_pos_w[0, 2].item():.4f}")
    print(f"box_yaw0={quat_yaw_wxyz(box.data.root_quat_w)[0].item():.4f}")

    for i in range(args_cli.num_steps):
        env.step(actions)

        if (i + 1) % args_cli.print_every == 0 or i == args_cli.num_steps - 1:
            box_pos = box.data.root_pos_w
            base_pos = robot.data.root_pos_w
            box_yaw = quat_yaw_wxyz(box.data.root_quat_w)

            print(
                "step="
                f"{i + 1} "
                f"box_x={box_pos[0, 0].item():.4f} "
                f"box_dx={(box_pos[0, 0] - box_x0[0]).item():.4f} "
                f"base_x={base_pos[0, 0].item():.4f} "
                f"base_dx={(base_pos[0, 0] - base_x0[0]).item():.4f} "
                f"robot_z={base_pos[0, 2].item():.4f} "
                f"box_yaw={box_yaw[0].item():.4f}"
            )

    print("=== audit_done ===")
    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
