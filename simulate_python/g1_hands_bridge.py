"""
DDS <-> MuJoCo bridge for the G1's Dex3 hands, meant to run alongside (not
instead of) the official UnitreeSdk2Bridge when using a body+hand model
(g1_29dof_hand14.xml, 43 actuators: 29 body + 7 left hand + 7 right hand).

- Why a separate bridge instead of extending UnitreeSdk2Bridge: the real Dex3
hand does NOT go over rt/lowcmd/rt/lowstate - those use unitree_hg's LowCmd_/
LowState_, whose motor_cmd/motor_state are a FIXED-size array of 35 slots.
29 body + 14 hand = 43 would overflow that array. 
- The real hand instead uses its own topics (rt/dex3/left/cmd, rt/dex3/right/cmd, .../state) with HandCmd_/
HandState_, which use variable-length sequences (7 motors each) instead of a
fixed 35-slot array. 
- So this bridge is deliberately independent of the body bridge, and the body bridge 
(unchanged, from unitree_sdk2py_bridge.py) must be limited to the first 29 actuators when used with this model. 
See g1_hands_run.py, which does that instead of using UnitreeSdk2Bridge directly.
"""

from unitree_sdk2py.core.channel import ChannelPublisher, ChannelSubscriber
from unitree_sdk2py.idl.default import (
    unitree_hg_msg_dds__HandCmd_,
    unitree_hg_msg_dds__HandState_,
    unitree_hg_msg_dds__PressSensorState_,
)
from unitree_sdk2py.idl.unitree_hg.msg.dds_ import HandCmd_, HandState_, MotorState_
from unitree_sdk2py.utils.thread import RecurrentThread

NUM_HAND_MOTOR = 7
# Pressure sensors per hand on the real G1 (per the Humanoid Everyday dataset docs).
# The MuJoCo model has no touch sensors, so these are published as zeros - an
# honest "no reading" - just so code indexing press_sensor_state works in sim too.
NUM_PRESS_SENSOR = 9

# Dex3 gains are much smaller than the arm's, matching Unitree's own
# example/g1/dex3/g1_dex3_example.cpp (kp~0.5-1.5, kd~0.1). Empirically
# (see g1_29dof_hand14 physics tests) these tiny/low-inertia finger joints
# also need a finer simulation timestep (see scene_29dof_hand14_fixed_base.xml)
# to stay stable even at these gains - the real robot's embedded finger
# control loop runs much faster than our DDS command rate.
DEFAULT_HAND_KP = 10.0
DEFAULT_HAND_KD = 0.5


class HandBridge:
    """One instance per hand. `actuator_offset` is the mj_data.ctrl/sensordata
    index where this hand's 7 motors start (29 for left, 36 for right, given
    the 29-body + 7-left + 7-right actuator ordering in g1_29dof_hand14.xml).
    """

    def __init__(self, mj_model, mj_data, side: str, actuator_offset: int):
        assert side in ("left", "right")
        self.mj_model = mj_model
        self.mj_data = mj_data
        self.side = side
        self.offset = actuator_offset
        self.total_motor = mj_model.nu  # for the sensordata block stride

        self.last_cmd = None

        self.state_puber = ChannelPublisher(f"rt/dex3/{side}/state", HandState_)
        self.state_puber.Init()
        self.cmd_suber = ChannelSubscriber(f"rt/dex3/{side}/cmd", HandCmd_)
        self.cmd_suber.Init(self._on_cmd, 10)
        # Publish state at a fixed, modest rate (100Hz) independent of the
        # physics timestep - the hand model needs a very fine timestep
        # (0.0004s = 2500Hz) for numerical stability, and firing a
        # RecurrentThread at that rate for state publishing (x2 hands, on
        # top of the physics loop itself already running there) starves
        # the CPU badly enough that RecurrentThread.Start() effectively
        # never returns. PD control (apply_ctrl) still runs every physics
        # step, called directly from the sim loop - only telemetry is slower.
        self.state_thread = RecurrentThread(
            interval=0.01, target=self._publish_state, name=f"dex3_{side}_state"
        )
        self.state_thread.Start()

    def _on_cmd(self, msg: HandCmd_) -> None:
        self.last_cmd = msg

    def apply_ctrl(self) -> None:
        """Call once per physics step (from the sim loop), same pattern as
        UnitreeSdk2Bridge.LowCmdHandler but for this hand's 7 actuators."""
        if self.last_cmd is None:
            return
        for i, motor_cmd in enumerate(self.last_cmd.motor_cmd):
            if i >= NUM_HAND_MOTOR:
                break
            idx = self.offset + i
            q = self.mj_data.sensordata[idx]
            dq = self.mj_data.sensordata[idx + self.total_motor]
            self.mj_data.ctrl[idx] = (
                motor_cmd.tau + motor_cmd.kp * (motor_cmd.q - q) + motor_cmd.kd * (motor_cmd.dq - dq)
            )

    def _publish_state(self) -> None:
        state = unitree_hg_msg_dds__HandState_()
        motor_states = []
        for i in range(NUM_HAND_MOTOR):
            idx = self.offset + i
            ms = MotorState_(
                mode=1,
                q=self.mj_data.sensordata[idx],
                dq=self.mj_data.sensordata[idx + self.total_motor],
                ddq=0.0,
                tau_est=self.mj_data.sensordata[idx + 2 * self.total_motor],
                temperature=[0, 0],
                vol=0.0,
                sensor=[0, 0],
                motorstate=0,
                reserve=[0, 0, 0, 0],
            )
            motor_states.append(ms)
        state.motor_state = motor_states
        state.press_sensor_state = [unitree_hg_msg_dds__PressSensorState_() for _ in range(NUM_PRESS_SENSOR)]
        self.state_puber.Write(state)
