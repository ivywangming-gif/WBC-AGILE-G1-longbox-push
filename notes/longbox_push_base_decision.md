# G1 Longbox Push base-task decision

Decision:
- Primary base: `G1-PickPlace-Tracking-v0`
- Secondary reference: `Debug-G1-Object-v0`
- Standing/locomotion reference: `Velocity-Height-G1-v0`

Why primary base is pick-place:
- It uses G1 with hands: `G1_W_HANDS_AGILE_CFG`.
- It already has a rigid object in scene.
- It already contains hand-object tracking reward structure.
- It already contains object reset / object bounds / object-related observations.
- It has G1 pick-place task registration and PPO config.

Why not use Debug-G1-Object as main:
- It is useful for object spawn/debug.
- But it uses `G1_29DOF_DELAYED_DC_MOTOR`, not the hands config.
- It fixes the robot root link in the G1 debug config.
- It is not a locomotion/manipulation training task.

Longbox phase-1 target:
- Spawn longbox: 1.6 x 0.8 x 0.8 m
- box_start_x ≈ 1.05
- target_x ≈ 2.05
- G1 starts behind rear face
- First success: reset + headless step + audit logs
- Then train for box_dx >= 0.5 m
