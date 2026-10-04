# G1 variants generated from Unitree's real descriptions

Built by `tools/build_g1_variants.py` (re-run to regenerate). Each `scene_*_fixed_base.xml`
includes its model, sets `timestep=0.0004` (light finger links) and welds the pelvis to the world.

| model | source (third_party/unitree_ros/robots) | body motors | hand motors |
|---|---|---|---|
| `g1_29dof_dex1.xml` | `g1_description/g1_29dof_mode_15_with_dex1_1.urdf` | 29 | 4 (2 prismatic fingers per hand) |
| `g1_29dof_inspire_ftp.xml` | `g1_description/g1_29dof_rev_1_0_with_inspire_hand_FTP.urdf` | 29 | 12 (6 per hand; the rest are mimic joints) |
| `g1_29dof_brainco.xml` | `g1_with_brainco_hand/g1_29dof_mode_15_brainco_hand.urdf` | 29 | 12 (6 per hand; the rest are mimic joints) |

Actuator order: body in `LowCmd_` order (`left_hip_pitch` ... `right_wrist_yaw`), then left hand, then
right hand. Sensors: jointpos for all actuators, then jointvel, then jointactuatorfrc, then IMU sensors
(same layout as `g1_29dof_hand14.xml`).

Body dynamics (damping/armature/frictionloss) come from `g1_29dof.xml`. Hand joint dynamics
(damping 0.01, armature 0.001) are our choice, not Unitree's. Mimic joints (URDF `<mimic>`) are
`<equality><joint>` constraints. Tip joints with lower == upper limit were made fixed. Contact pairs
that overlap at the zero pose (thumb vs hand base) are excluded.
