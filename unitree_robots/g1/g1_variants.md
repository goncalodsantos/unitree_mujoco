# G1 variants generated from Unitree's real descriptions

Built by `tools/build_g1_variants.py` (re-run to regenerate). Each `scene_*_fixed_base.xml`
includes its model, sets `timestep=0.0004` (light finger links) and welds the pelvis to the world;
the same scene without `_fixed_base` leaves the pelvis free.

**All variants share the same body: `g1_29dof.xml`** (geometry, masses, joint classes, actuators, IMU).
Only the hands change: each hand sub-tree is cut from Unitree's real URDF and attached to the
`wrist_yaw_link` at the pose that URDF gives. The `g1_29dof.xml` rubber hand is removed and the wrist
inertia is replaced by the URDF's wrist-without-rubber-hand one (0.085 kg instead of 0.255 kg), so the
rubber hand's mass is not counted on top of the new hand.

| model | hand source (third_party/unitree_ros/robots) | body motors | hand motors |
|---|---|---|---|
| `g1_29dof_dex1.xml` | `g1_description/g1_29dof_mode_15_with_dex1_1.urdf` | 29 | 4 (2 prismatic fingers per hand) |
| `g1_29dof_inspire_ftp.xml` | `g1_description/g1_29dof_rev_1_0_with_inspire_hand_FTP.urdf` | 29 | 12 (6 per hand; the rest are mimic joints) |
| `g1_29dof_brainco.xml` | `g1_with_brainco_hand/g1_29dof_mode_15_brainco_hand.urdf` | 29 | 12 (6 per hand; the rest are mimic joints) |

Actuator order: body in `LowCmd_` order (`left_hip_pitch` ... `right_wrist_yaw`), then left hand, then
right hand. Sensors: jointpos for all actuators, then jointvel, then jointactuatorfrc, then IMU sensors
(same layout as `g1_29dof_hand14.xml`).

Hand joint dynamics (damping 0.01, armature 0.001) are our choice, not Unitree's. Mimic joints (URDF
`<mimic>`) are `<equality><joint>` constraints. Tip joints with lower == upper limit were made fixed.
Contact pairs that overlap at the zero pose (thumb vs hand base) are excluded.
