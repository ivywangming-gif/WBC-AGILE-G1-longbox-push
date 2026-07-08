# SPDX-License-Identifier: Apache-2.0
"""Minimal Unitree G1 longbox push task.

Phase goals:
- Spawn Unitree G1 with hands.
- Spawn a 1.6 x 0.8 x 0.8 m longbox.
- Start G1 behind the box rear face.
- Replace pick-place trajectory rewards with longbox push reward skeleton.
"""

import isaaclab.sim as sim_utils
from isaaclab.managers import EventTermCfg as EventTerm, RewardTermCfg as RewTerm, SceneEntityCfg
from isaaclab.sim.schemas.schemas_cfg import RigidBodyPropertiesCfg
from isaaclab.utils import configclass

from agile.rl_env.tasks.longbox_push import longbox_push_events as lb_events
from agile.rl_env.tasks.longbox_push import longbox_push_rewards as lb_rewards
from agile.rl_env.tasks.pick_place.g1.g1_pick_place_tracking_env_cfg import G1PickPlaceTrackingEnvCfg


@configclass
class G1LongBoxPushEnvCfg(G1PickPlaceTrackingEnvCfg):
    """Longbox push config derived from official G1 pick-place."""

    def __post_init__(self) -> None:
        super().__post_init__()

        # Keep smoke tests tiny on WSL.
        self.scene.num_envs = 1
        self.scene.env_spacing = 4.0
        self.episode_length_s = 8.0

        # Remove pick-place table/fixture. Longbox sits on the ground.
        self.scene.fixture_structure = None

        # Robot behind longbox rear face, facing +x.
        # Box center x=1.05, length x=1.6 -> rear face x=0.25.
        self.scene.robot.init_state.pos = [-0.25, 0.0, 0.8]
        self.scene.robot.init_state.rot = [1.0, 0.0, 0.0, 0.0]
        self.scene.robot.init_state.joint_pos = {
            ".*_hip_pitch_joint": -0.10,
            ".*_knee_joint": 0.30,
            ".*_ankle_pitch_joint": -0.20,
            "left_shoulder_pitch_joint": 0.19,
            "left_shoulder_roll_joint": 0.4638,
            "left_shoulder_yaw_joint": -0.2448,
            "left_elbow_joint": 0.9777,
            "left_wrist_roll_joint": -0.0926,
            "left_wrist_pitch_joint": -0.0179,
            "left_wrist_yaw_joint": -0.0225,
            "left_hand_thumb_1_joint": 1.0,
            "left_hand_thumb_2_joint": 0.3,
        }

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

        # Disable pick-place trajectory reset.
        if hasattr(self.events, "reset_robot"):
            self.events.reset_robot = None

        # Explicitly reset joints to standing defaults.
        self.events.reset_robot_joints_to_default = EventTerm(
            func=lb_events.reset_robot_joints_to_default,
            mode="reset",
            params={"asset_cfg": SceneEntityCfg("robot")},
        )

        # Deterministic object reset for the first prototype.
        self.events.reset_object.params["pose_range"] = {
            "x": (0.0, 0.0),
            "y": (0.0, 0.0),
            "yaw": (0.0, 0.0),
        }

        # Longbox center z is 0.4, not table height.
        self.terminations.object_out_of_bound.params["in_bound_range"] = {"z": [0.2, 1.2]}

        # Remove pick-place trajectory rewards.
        for name in [
            "static_at_goal",
            "motion_global_anchor_pos",
            "motion_global_anchor_ori",
            "upper_body_joint_pos",
            "object_pos_tracking",
            "hand_object_tracking",
            "lifting_object",
            "nominal_posture_at_end",
        ]:
            if hasattr(self.rewards, name):
                setattr(self.rewards, name, None)

        # Remove pick-place curricula that reference removed rewards.
        if hasattr(self.curriculum, "lifting_object_curriculum"):
            self.curriculum.lifting_object_curriculum = None
        if hasattr(self.curriculum, "increase_object_pos_tracking"):
            self.curriculum.increase_object_pos_tracking = None

        # Minimal longbox push rewards.
        self.rewards.box_forward_progress = RewTerm(
            func=lb_rewards.box_forward_progress,
            weight=8.0,
            params={
                "asset_cfg": SceneEntityCfg("object"),
                "start_x": 1.05,
                "max_dx": 1.2,
            },
        )
        self.rewards.robot_height = RewTerm(
            func=lb_rewards.robot_height_exp,
            weight=1.5,
            params={
                "asset_cfg": SceneEntityCfg("robot"),
                "target_z": 0.80,
                "std": 0.18,
            },
        )
        self.rewards.base_near_rear_face = RewTerm(
            func=lb_rewards.base_near_rear_face_exp,
            weight=1.0,
            params={
                "robot_cfg": SceneEntityCfg("robot"),
                "box_cfg": SceneEntityCfg("object"),
                "box_half_length_x": 0.8,
                "target_distance": 0.45,
                "std": 0.30,
            },
        )
        self.rewards.box_yaw_l2 = RewTerm(
            func=lb_rewards.box_yaw_l2,
            weight=-1.0,
            params={
                "asset_cfg": SceneEntityCfg("object"),
            },
        )
