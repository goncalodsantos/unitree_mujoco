"""
Run the G1 sim with Dex3 hands (g1_29dof_hand14, 43 actuators).

- Body (29 motors) still talks rt/lowcmd/rt/lowstate exactly like the plain
unitree_mujoco.py entry point. 
- Hands (7+7 motors) talk rt/dex3/{left,right} /cmd and /state instead - see g1_hands_bridge.py 
for why they can't share the body's LowCmd_/LowState_ channel (35-slot fixed array, would overflow
at 43 motors).

    conda activate g1
    cd third_party/unitree_mujoco/simulate_python
    python g1_hands_run.py
"""

import time
import threading
from threading import Thread

import mujoco
import mujoco.viewer

from unitree_sdk2py.core.channel import ChannelFactoryInitialize, ChannelPublisher, ChannelSubscriber
from unitree_sdk2py.idl.default import unitree_hg_msg_dds__LowState_
from unitree_sdk2py.idl.unitree_hg.msg.dds_ import LowCmd_, LowState_
from unitree_sdk2py.utils.thread import RecurrentThread

from g1_hands_bridge import HandBridge
from unitree_sdk2py_bridge import ElasticBand

MODEL_PATH = "../unitree_robots/g1/scene_29dof_hand14_fixed_base.xml"
DOMAIN_ID = 1
INTERFACE = "lo"
NUM_BODY_MOTOR = 29
LEFT_HAND_OFFSET = 29
RIGHT_HAND_OFFSET = 36

locker = threading.Lock()

mj_model = mujoco.MjModel.from_xml_path(MODEL_PATH)
mj_data = mujoco.MjData(mj_model)
assert mj_model.nu == 43, f"expected 43 actuators (29 body + 14 hand), got {mj_model.nu}"

elastic_band = ElasticBand()
band_attached_link = mj_model.body("torso_link").id
viewer = mujoco.viewer.launch_passive(mj_model, mj_data, key_callback=elastic_band.MujuocoKeyCallback)


class BodyBridge29:
    """Same LowCmd_/LowState_ PD logic as UnitreeSdk2Bridge, but hardcoded
    to the first 29 actuators only - deliberately NOT using mj_model.nu
    (43), which would index past LowCmd_.motor_cmd's fixed 35 slots."""

    def __init__(self, mj_model, mj_data):
        self.mj_model = mj_model
        self.mj_data = mj_data
        self.low_state = unitree_hg_msg_dds__LowState_()
        self.low_state_puber = ChannelPublisher("rt/lowstate", LowState_)
        self.low_state_puber.Init()
        self.low_cmd_suber = ChannelSubscriber("rt/lowcmd", LowCmd_)
        self.low_cmd_suber.Init(self.LowCmdHandler, 10)
        # See g1_hands_bridge.py for why this doesn't use mj_model.opt.timestep
        # (0.0004s / 2500Hz) directly - too many high-rate RecurrentThreads
        # starve the CPU enough that .Start() itself stalls.
        self.state_thread = RecurrentThread(
            interval=0.01, target=self.PublishLowState, name="g1_hands_lowstate"
        )
        self.state_thread.Start()

    def LowCmdHandler(self, msg: LowCmd_):
        for i in range(NUM_BODY_MOTOR):
            q = self.mj_data.sensordata[i]
            dq = self.mj_data.sensordata[i + mj_model.nu]
            self.mj_data.ctrl[i] = (
                msg.motor_cmd[i].tau
                + msg.motor_cmd[i].kp * (msg.motor_cmd[i].q - q)
                + msg.motor_cmd[i].kd * (msg.motor_cmd[i].dq - dq)
            )

    def PublishLowState(self):
        for i in range(NUM_BODY_MOTOR):
            self.low_state.motor_state[i].q = self.mj_data.sensordata[i]
            self.low_state.motor_state[i].dq = self.mj_data.sensordata[i + mj_model.nu]
            self.low_state.motor_state[i].tau_est = self.mj_data.sensordata[i + 2 * mj_model.nu]
        self.low_state_puber.Write(self.low_state)


def simulation_thread():
    ChannelFactoryInitialize(DOMAIN_ID, INTERFACE)
    body_bridge = BodyBridge29(mj_model, mj_data)
    left_hand = HandBridge(mj_model, mj_data, "left", LEFT_HAND_OFFSET)
    right_hand = HandBridge(mj_model, mj_data, "right", RIGHT_HAND_OFFSET)

    while viewer.is_running():
        step_start = time.perf_counter()

        locker.acquire()
        if elastic_band.enable:
            mj_data.xfrc_applied[band_attached_link, :3] = elastic_band.Advance(
                mj_data.qpos[:3], mj_data.qvel[:3]
            )
        left_hand.apply_ctrl()
        right_hand.apply_ctrl()
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
