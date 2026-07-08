# SPDX-License-Identifier: Apache-2.0
"""Minimal Unitree G1 longbox push task skeleton.

Phase 1 goal:
- Register and reset a G1 + longbox scene.
- Spawn a 1.6 x 0.8 x 0.8 m rigid cuboid.
- Run headless smoke tests before adding custom rewards/audit.
"""

import isaaclab.sim as sim_utils
from isaaclab.sim.schemas.schemas_cfg import RigidBodyPropertiesCfg
from isaaclab.utils import configclass

from agile.rl_env.tasks.pick_place.g1.g1_pick_place_tracking_env_cfg import G1PickPlaceTrackingEnvCfg


@configclass
class G1LongBoxPushEnvCfg(G1PickPlaceTrackingEnvCfg):
    """Minimal longbox push config derived from official G1 pick-place."""

    def __post_init__(self) -> None:
        super().__post_init__()

        # Keep smoke tests tiny on WSL.
        self.scene.num_envs = 1
        self.scene.env_spacing = 4.0
        self.episode_length_s = 8.0

        # Remove table/fixture from pick-place. Longbox sits on the ground.
        self.scene.fixture_structure = None

        # Robot behind the longbox rear face, facing +x.
        # Box center x=1.05, length x=1.6 -> rear face x=0.25.
        self.scene.robot.init_state.pos = [-0.65, 0.0, 0.8]
        self.scene.robot.init_state.rot = [1.0, 0.0, 0.0, 0.0]

        # Replace pick-place USD object with procedural longbox.
        self.scene.object.prim_path = "{ENV_REGEX_NS}/Object"
        self.scene.object.spawn = sim_utils.CuboidCfg(
            size=(1.6, 0.8, 0.8),
            visual_material=sim_utils.PreviewSurfaceCfg(
                diffuse_color=(0.15, 0.25, 0.85),
                metallic=0.0,
            ),
            collision_props=sim_utils.CollisionPropertiesCfg(
                collision_enabled=True,
            ),
            rigid_props=RigidBodyPropertiesCfg(
                solver_position_iteration_count=16,
                solver_velocity_iteration_count=1,
                max_angular_velocity=1000.0,
                max_linear_velocity=1000.0,
                max_depenetration_velocity=5.0,
                disable_gravity=False,
            ),
            mass_props=sim_utils.MassPropertiesCfg(mass=35.0),
            physics_material=sim_utils.RigidBodyMaterialCfg(
                static_friction=0.9,
                dynamic_friction=0.8,
                restitution=0.0,
            ),
        )
        self.scene.object.init_state.pos = [1.05, 0.0, 0.4]
        self.scene.object.init_state.rot = [1.0, 0.0, 0.0, 0.0]

        # Make reset deterministic for the first smoke test.
        self.events.reset_object.params["pose_range"] = {
            "x": (0.0, 0.0),
            "y": (0.0, 0.0),
            "yaw": (0.0, 0.0),
        }

        # Pick-place object z bounds assumed table height; longbox center is z=0.4.
        self.terminations.object_out_of_bound.params["in_bound_range"] = {"z": [0.2, 1.2]}

        # Disable lift-specific reward/curriculum for push skeleton.
        # Pick-place has a curriculum term that references reward_name="lifting_object";
        # if we remove the reward, we must remove the matching curriculum too.
        if hasattr(self.rewards, "lifting_object"):
            self.rewards.lifting_object = None
        if hasattr(self.curriculum, "lifting_object_curriculum"):
            self.curriculum.lifting_object_curriculum = None

        # Keep the first skeleton simple: no curriculum weight scheduling yet.
        if hasattr(self.curriculum, "increase_object_pos_tracking"):
            self.curriculum.increase_object_pos_tracking = None
