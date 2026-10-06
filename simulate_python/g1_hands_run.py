"""
Run the G1 sim with a dexterous hand/gripper attached, fixed pelvis.

- Body (29 motors) talks rt/lowcmd / rt/lowstate exactly like the plain
unitree_mujoco.py entry point. 
- The Hands talk their own DDS protocol, see
g1_hands_bridge.py (Dex3 and why it can't share LowCmd_) for details.

    conda activate g1
    cd third_party/unitree_mujoco/simulate_python
    python g1_hands_run.py                      # Dex3, 43 actuators (default)
    python g1_hands_run.py --hand dex1          # Dex1-1 gripper
    python g1_hands_run.py --hand inspire_ftp   # Inspire FTP
    python g1_hands_run.py --hand inspire_dfx   # Inspire DFX (Unitree's DFQ model, as in xr_teleoperate)
    python g1_hands_run.py --hand brainco       # BrainCo
    python g1_hands_run.py --hand dex1 --free-base   # pelvis not welded
"""

import argparse
import time
import threading
from threading import Thread

import mujoco
import mujoco.viewer

from unitree_sdk2py.core.channel import ChannelFactoryInitialize, ChannelPublisher, ChannelSubscriber
from unitree_sdk2py.idl.default import unitree_hg_msg_dds__LowState_
from unitree_sdk2py.idl.unitree_hg.msg.dds_ import LowCmd_, LowState_
from unitree_sdk2py.utils.thread import RecurrentThread

from g1_hands_bridge import BrainCoBridge, Dex1Bridge, HandBridge, InspireDFXBridge, InspireFTPBridge
from unitree_sdk2py_bridge import ElasticBand

DOMAIN_ID = 1
INTERFACE = "lo"
NUM_BODY_MOTOR = 29  # LowCmd_/LowState_ slots used by the G1 body

ROBOTS_DIR = "../unitree_robots/g1/"
HANDS = {
    "dex3": "scene_29dof_hand14_fixed_base.xml",
    "dex1": "scene_29dof_dex1_fixed_base.xml",
    "inspire_ftp": "scene_29dof_inspire_ftp_fixed_base.xml",
    "inspire_dfx": "scene_29dof_inspire_dfx_fixed_base.xml",
    "brainco": "scene_29dof_brainco_fixed_base.xml",
}
# Actuator names per hardware motor index, per side (the order each protocol uses).
# Inspire DFX: one topic for both hands, right hand first.
DFX_ORDER = ["pinky_proximal", "ring_proximal", "middle_proximal", "index_proximal", "thumb_proximal_pitch", "thumb_proximal_yaw"]
INSPIRE_ORDER = ["little_1", "ring_1", "middle_1", "index_1", "thumb_2", "thumb_1"]  # pinky, ring, middle, index, thumb-bend, thumb-rotation
BRAINCO_ORDER = ["thumb_proximal", "thumb_metacarpal", "index_proximal", "middle_proximal", "ring_proximal", "pinky_proximal"]  # Thumb, Thumb_aux, Index, Middle, Ring, Pinky

parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
parser.add_argument("--hand", choices=list(HANDS), default="dex3")
parser.add_argument("--domain-id", type=int, default=DOMAIN_ID, help="DDS domain (change it to run a second sim next to another one)")
parser.add_argument("--free-base", action="store_true", help="do not weld the pelvis to the world (the robot has no balance controller and will fall; use the elastic band, key 9)")
args = parser.parse_args()

locker = threading.Lock()

scene = HANDS[args.hand].replace("_fixed_base", "") if args.free_base else HANDS[args.hand]
mj_model = mujoco.MjModel.from_xml_path(ROBOTS_DIR + scene)
mj_data = mujoco.MjData(mj_model)

# DDS motor index -> actuator index for the body (identity: the models list the 29 body motors first).
BODY_MAP = {i: i for i in range(NUM_BODY_MOTOR)}
N_BODY_ACT = len(BODY_MAP)


def actuator_id(name: str) -> int:
    idx = mujoco.mj_name2id(mj_model, mujoco.mjtObj.mjOBJ_ACTUATOR, name)
    assert idx >= 0, f"actuator '{name}' not in {HANDS[args.hand]}"
    return idx


elastic_band = ElasticBand()
band_attached_link = mj_model.body("torso_link").id
viewer = mujoco.viewer.launch_passive(mj_model, mj_data, key_callback=elastic_band.MujuocoKeyCallback)


class BodyBridge29:
    """Same LowCmd_/LowState_ PD logic as UnitreeSdk2Bridge, but only for the body motors -
    deliberately NOT using mj_model.nu (which also counts the hand motors and would index
    past LowCmd_.motor_cmd's fixed 35 slots)."""

    def __init__(self, mj_model, mj_data):
        self.mj_model = mj_model
        self.mj_data = mj_data
        self.low_state = unitree_hg_msg_dds__LowState_()
        self.low_state_puber = ChannelPublisher("rt/lowstate", LowState_)
        self.low_state_puber.Init()
        self.low_cmd_suber = ChannelSubscriber("rt/lowcmd", LowCmd_)
        self.low_cmd_suber.Init(self.LowCmdHandler, 10)
        # See g1_hands_bridge.py for why this doesn't use mj_model.opt.timestep
        # (0.001s / 1000Hz) directly - too many high-rate RecurrentThreads
        # starve the CPU enough that .Start() itself stalls.
        self.state_thread = RecurrentThread(
            interval=0.01, target=self.PublishLowState, name="g1_hands_lowstate"
        )
        self.state_thread.Start()

    def LowCmdHandler(self, msg: LowCmd_):
        for dds_i, act in BODY_MAP.items():
            q = self.mj_data.sensordata[act]
            dq = self.mj_data.sensordata[act + mj_model.nu]
            self.mj_data.ctrl[act] = (
                msg.motor_cmd[dds_i].tau
                + msg.motor_cmd[dds_i].kp * (msg.motor_cmd[dds_i].q - q)
                + msg.motor_cmd[dds_i].kd * (msg.motor_cmd[dds_i].dq - dq)
            )

    def PublishLowState(self):
        for dds_i, act in BODY_MAP.items():
            self.low_state.motor_state[dds_i].q = self.mj_data.sensordata[act]
            self.low_state.motor_state[dds_i].dq = self.mj_data.sensordata[act + mj_model.nu]
            self.low_state.motor_state[dds_i].tau_est = self.mj_data.sensordata[act + 2 * mj_model.nu]
        self.low_state_puber.Write(self.low_state)


def make_hand_bridges():
    if args.hand == "dex3":
        assert mj_model.nu == N_BODY_ACT + 14, f"expected {N_BODY_ACT} body + 14 hand actuators, got {mj_model.nu}"
        return [HandBridge(mj_model, mj_data, "left", N_BODY_ACT), HandBridge(mj_model, mj_data, "right", N_BODY_ACT + 7)]
    if args.hand == "inspire_dfx":
        acts = [actuator_id(f"{side}_{n}_joint") for side in ("R", "L") for n in DFX_ORDER]
        return [InspireDFXBridge(mj_model, mj_data, acts)]
    bridges = []
    for side in ("left", "right"):
        if args.hand == "dex1":
            acts = [actuator_id(f"{side}_dex1_finger_joint_{k}") for k in (1, 2)]
            bridges.append(Dex1Bridge(mj_model, mj_data, side, acts))
        elif args.hand == "inspire_ftp":
            acts = [actuator_id(f"{side}_{n}_joint") for n in INSPIRE_ORDER]
            bridges.append(InspireFTPBridge(mj_model, mj_data, side, acts))
        elif args.hand == "brainco":
            acts = [actuator_id(f"{side}_{n}_joint") for n in BRAINCO_ORDER]
            bridges.append(BrainCoBridge(mj_model, mj_data, side, acts))
    return bridges


def simulation_thread():
    ChannelFactoryInitialize(args.domain_id, INTERFACE)
    body_bridge = BodyBridge29(mj_model, mj_data)
    hand_bridges = make_hand_bridges()

    while viewer.is_running():
        step_start = time.perf_counter()

        locker.acquire()
        if elastic_band.enable:
            mj_data.xfrc_applied[band_attached_link, :3] = elastic_band.Advance(
                mj_data.qpos[:3], mj_data.qvel[:3]
            )
        for hand in hand_bridges:
            hand.apply_ctrl()
        mujoco.mj_step(mj_model, mj_data)
        locker.release()

        elapsed = time.perf_counter() - step_start
        remaining = mj_model.opt.timestep - elapsed
        if remaining > 0:
            time.sleep(remaining)


def viewer_thread():
    while viewer.is_running():
        locker.acquire()
        viewer.sync()
        locker.release()
        time.sleep(0.02)


if __name__ == "__main__":
    Thread(target=viewer_thread).start()
    Thread(target=simulation_thread).start()
