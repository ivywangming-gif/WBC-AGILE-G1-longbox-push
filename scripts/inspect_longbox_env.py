# SPDX-License-Identifier: Apache-2.0
"""Inspect action terms and reset state for G1-LongBoxPush-v0."""

import argparse
from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--task", type=str, default="G1-LongBoxPush-v0")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import gymnasium as gym
import agile.rl_env.tasks  # noqa: F401
from isaaclab_tasks.utils import load_cfg_from_registry

env_cfg = load_cfg_from_registry(args_cli.task, "env_cfg_entry_point")
env_cfg.scene.num_envs = 1

env = gym.make(args_cli.task, cfg=env_cfg, render_mode=None)
u = env.unwrapped
env.reset()

print("=== action manager ===")
print("total_action_dim", u.action_manager.total_action_dim)
for name, term in u.action_manager._terms.items():
    print(f"\nACTION_TERM {name}")
    print("  class:", term.__class__)
    cfg = getattr(term, "cfg", None)
    print("  cfg:", cfg)
    for attr in ["joint_names", "scale", "offset", "use_default_offset", "asset_name"]:
        if cfg is not None and hasattr(cfg, attr):
            print(f"  cfg.{attr} =", getattr(cfg, attr))
    for attr in ["_joint_names", "_scale", "_offset", "_raw_actions", "_processed_actions"]:
        if hasattr(term, attr):
            value = getattr(term, attr)
            try:
                print(f"  {attr} shape/value:", value.shape if hasattr(value, "shape") else value)
            except Exception:
                print(f"  {attr}: <unprintable>")

robot = u.scene["robot"]
box = u.scene["object"]

print("\n=== reset state ===")
print("robot root pos:", robot.data.root_pos_w[0].detach().cpu().tolist())
print("box root pos:", box.data.root_pos_w[0].detach().cpu().tolist())
print("default joint pos first 10:", robot.data.default_joint_pos[0, :10].detach().cpu().tolist())
print("joint pos first 10:", robot.data.joint_pos[0, :10].detach().cpu().tolist())
print("joint names first 20:", robot.joint_names[:20])

env.close()
simulation_app.close()
