"""
DDS <-> MuJoCo bridge for the G1's hands, meant to run alongside (not instead of) the
official UnitreeSdk2Bridge when using a body+hand model (g1_29dof_dex3.xml, 43 actuators:
29 body + 7 left hand + 7 right hand).

Why a separate bridge: the real Dex3 hand does NOT go over rt/lowcmd/rt/lowstate. Those use
unitree_hg LowCmd_/LowState_, whose motor arrays have a fixed 35 slots, and 29 body + 14 hand
= 43 would overflow them. The Dex3 uses its own topics (rt/dex3/{left,right}/cmd|state) with
HandCmd_/HandState_. The body bridge must be limited to the first 29 actuators (see g1_hands_run.py).
"""

import sys
from pathlib import Path

from unitree_sdk2py.core.channel import ChannelPublisher, ChannelSubscriber
from unitree_sdk2py.idl.default import (
    unitree_hg_msg_dds__HandState_,
    unitree_hg_msg_dds__PressSensorState_,
)
from unitree_sdk2py.idl.unitree_hg.msg.dds_ import HandCmd_, HandState_, MotorState_
from unitree_sdk2py.utils.thread import RecurrentThread

NUM_HAND_MOTOR = 7
# The MuJoCo model has no touch sensors: the pressure sensors are published as zeros.
NUM_PRESS_SENSOR = 9

DEFAULT_HAND_KP = 8.0
DEFAULT_HAND_KD = 0.2


class HandBridge:
    """One instance per hand. `actuator_offset` is the mj_data.ctrl/sensordata index where
    this hand's 7 motors start (29 for left, 36 for right)."""

    def __init__(self, mj_model, mj_data, side: str, actuator_offset: int):
        assert side in ("left", "right")
        self.mj_model = mj_model
        self.mj_data = mj_data
        self.side = side
        self.offset = actuator_offset
        self.total_motor = mj_model.nu  # stride of the sensordata blocks

        self.last_cmd = None

        self.state_puber = ChannelPublisher(f"rt/dex3/{side}/state", HandState_)
        self.state_puber.Init()
        self.cmd_suber = ChannelSubscriber(f"rt/dex3/{side}/cmd", HandCmd_)
        self.cmd_suber.Init(self._on_cmd, 10)
        # State is published at 100 Hz, independent of the 1 kHz physics step: a faster
        # RecurrentThread starves the CPU. PD control still runs every physics step.
        self.state_thread = RecurrentThread(
            interval=0.01, target=self._publish_state, name=f"dex3_{side}_state"
        )
        self.state_thread.Start()

    def _on_cmd(self, msg: HandCmd_) -> None:
        self.last_cmd = msg

    def apply_ctrl(self) -> None:
        """Call once per physics step from the sim loop."""
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


# ---------------------------------------------------------------------------
# Other hands. These use higher-level protocols, so each bridge converts the received
# command to a joint target and runs a joint-space PD. The PD gains are OUR choices for
# the simulated joints, not vendor values.
# ---------------------------------------------------------------------------
import numpy as np
from unitree_sdk2py.idl.default import unitree_go_msg_dds__MotorState_
from unitree_sdk2py.idl.unitree_go.msg.dds_ import MotorCmds_, MotorStates_

# Inspire FTP message types, taken from the inspire_hand_ws submodule. Its inspire_sdkpy package
# __init__ pulls in PyQt/pymodbus, so only the generated inspire_dds package is put on the path.
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "inspire_hand_ws/inspire_hand_sdk/inspire_sdkpy"))
from inspire_dds import inspire_hand_ctrl, inspire_hand_state  # noqa: E402


class _JointPdHand:
    """A group of simulated actuators held by a joint-space PD toward a target vector."""

    kp = 8.0
    kd = 0.2

    def __init__(self, mj_model, mj_data, actuator_indices, publish_name):
        self.m, self.d = mj_model, mj_data
        self.act = list(actuator_indices)
        self.total = mj_model.nu
        self.target = None  # set by the protocol handler
        # joint limits, used to map normalised commands to angles
        self.lo = np.array([mj_model.jnt_range[mj_model.actuator_trnid[a, 0], 0] for a in self.act])
        self.hi = np.array([mj_model.jnt_range[mj_model.actuator_trnid[a, 0], 1] for a in self.act])
        self.state_thread = RecurrentThread(interval=0.01, target=self._publish_state, name=publish_name)

    def start(self):
        self.state_thread.Start()

    def q(self):
        return np.array([self.d.sensordata[a] for a in self.act])

    def dq(self):
        return np.array([self.d.sensordata[a + self.total] for a in self.act])

    def apply_ctrl(self):
        if self.target is None:
            return
        q, dq = self.q(), self.dq()
        tau = self.kp * (self.target - q) - self.kd * dq
        for k, a in enumerate(self.act):
            self.d.ctrl[a] = tau[k]

    def _publish_state(self):  # implemented by the protocol subclass
        raise NotImplementedError


class Dex1Bridge(_JointPdHand):
    """Dex1-1 gripper: rt/dex1/{left,right}/cmd|state, unitree_go MotorCmds_/MotorStates_,
    ONE motor per gripper (cmds[0].q). q is the motor angle in rad, 0 = closed, ~5.4 rad = fully
    open (about 9 cm of jaw opening). The sim model has two prismatic fingers (finger_joint_1/2),
    mapped symmetrically from the joint's lower limit (closed) to its upper limit (open).

    Like Unitree's dex1_1_service, the command's kp/kd/dq/tau are honoured:
        tau_motor = tau + kp*(q_des - q) + kd*(dq_des - dq)        [motor side, rad]
    The motor torque becomes a force on each slide through r = finger travel / motor travel
    (m/rad): F = tau_motor / (2 r), i.e. a spring kp / (2 r^2) and a damper kd / (2 r^2) per finger.
    The real gains are far too stiff for 0.09 kg fingers integrated explicitly at 1 ms, so the
    stiffness is capped to a natural frequency of OMEGA_DT/dt, the damping is raised to critical
    damping (limited to c*dt/m <= 0.5), and the force is limited to the actuator range.
    A command with kp = kd = 0 falls back to the class kp/kd (N/m and N*s/m on the slides)."""

    kp = 300.0
    kd = 10.0
    Q_OPEN = 5.4
    OMEGA_DT = 0.3  # largest natural frequency of the simulated spring, as a fraction of 1/dt

    def __init__(self, mj_model, mj_data, side, actuator_indices):
        super().__init__(mj_model, mj_data, actuator_indices, f"dex1_{side}_state")
        assert len(self.act) == 2
        self.lo_closed = float(self.lo.max())  # joint value with the jaws closed (m)
        self.hi_open = float(self.hi.min())    # joint value with the jaws fully open (m)
        self.stroke = self.hi_open - self.lo_closed  # travel of each finger
        self.cmd = None
        self.r = self.stroke / self.Q_OPEN
        joint = mj_model.actuator_trnid[self.act[0], 0]
        self.mass = float(mj_model.body_mass[mj_model.jnt_bodyid[joint]] + mj_model.dof_armature[mj_model.jnt_dofadr[joint]])
        dt = mj_model.opt.timestep
        self.k_max = self.mass * (self.OMEGA_DT / dt) ** 2  # N/m the explicit integration keeps stable
        self.c_max = 0.5 * self.mass / dt                    # N*s/m (c*dt/m <= 0.5)
        self.f_max = float(mj_model.actuator_ctrlrange[self.act[0], 1])
        self.state_puber = ChannelPublisher(f"rt/dex1/{side}/state", MotorStates_)
        self.state_puber.Init()
        self.cmd_suber = ChannelSubscriber(f"rt/dex1/{side}/cmd", MotorCmds_)
        self.cmd_suber.Init(self._on_cmd, 10)
        self.start()

    def _on_cmd(self, msg):
        if msg.cmds:
            self.cmd = msg.cmds[0]
            frac = float(np.clip(self.cmd.q / self.Q_OPEN, 0.0, 1.0))
            self.target = np.full(2, self.lo_closed + frac * self.stroke)  # used by the fallback PD

    def apply_ctrl(self):
        c = self.cmd
        if c is None:
            return
        if c.kp == 0.0 and c.kd == 0.0:
            return super().apply_ctrl()
        x_des = self.lo_closed + float(np.clip(c.q, 0.0, self.Q_OPEN)) * self.r
        k = min(c.kp / (2 * self.r ** 2), self.k_max)
        damp = min(max(c.kd / (2 * self.r ** 2), 2.0 * np.sqrt(k * self.mass)), self.c_max)
        # One PD per finger: the real gripper couples the jaws with a rail, here each finger is
        # held to the same target (controlling only their mean leaves the difference unsprung).
        for a, x, dx in zip(self.act, self.q(), self.dq()):
            force = c.tau / (2 * self.r) + k * (x_des - float(x)) + damp * (c.dq * self.r - float(dx))
            self.d.ctrl[a] = float(np.clip(force, -self.f_max, self.f_max))

    def _publish_state(self):
        frac = float(np.clip((self.q().mean() - self.lo_closed) / self.stroke, 0.0, 1.0))
        ms = unitree_go_msg_dds__MotorState_()
        ms.mode = 1
        ms.q = frac * self.Q_OPEN
        ms.dq = float(self.dq().mean() / self.stroke * self.Q_OPEN)
        self.state_puber.Write(MotorStates_([ms]))


class BrainCoBridge(_JointPdHand):
    """BrainCo hand: rt/brainco/{left,right}/cmd|state, unitree_go MotorCmds_/MotorStates_, 6 motors in
    the order [Thumb, Thumb_aux, Index, Middle, Ring, Pinky], q normalised to [0,1] (0 = open,
    1 = closed). The speed (dq) is ignored. `actuator_indices` must follow that order."""

    kp = 8.0
    kd = 0.2

    def __init__(self, mj_model, mj_data, side, actuator_indices):
        super().__init__(mj_model, mj_data, actuator_indices, f"brainco_{side}_state")
        assert len(self.act) == 6
        self.state_puber = ChannelPublisher(f"rt/brainco/{side}/state", MotorStates_)
        self.state_puber.Init()
        self.cmd_suber = ChannelSubscriber(f"rt/brainco/{side}/cmd", MotorCmds_)
        self.cmd_suber.Init(self._on_cmd, 10)
        self.start()

    def _on_cmd(self, msg):
        if len(msg.cmds) >= 6:
            q = np.clip([c.q for c in msg.cmds[:6]], 0.0, 1.0)
            self.target = self.lo + q * (self.hi - self.lo)

    def _publish_state(self):
        frac = np.clip((self.q() - self.lo) / (self.hi - self.lo), 0.0, 1.0)
        states = []
        for k in range(6):
            ms = unitree_go_msg_dds__MotorState_()
            ms.mode = 1
            ms.q = float(frac[k])
            ms.dq = float(self.dq()[k] / (self.hi[k] - self.lo[k]))
            states.append(ms)
        self.state_puber.Write(MotorStates_(states))


class InspireFTPBridge(_JointPdHand):
    """Inspire FTP hand: rt/inspire_hand/ctrl/{l,r} and rt/inspire_hand/state/{l,r}, 6 motors in the
    order [pinky, ring, middle, index, thumb-bend, thumb-rotation]. Only angle control is simulated
    (mode bit 0): angle_set is 0..1000 with 1000 = open, 0 = closed, -1 = no change. Force/speed/
    position modes and the touch topic are ignored. `actuator_indices` must follow that order."""

    kp = 8.0
    kd = 0.2

    def __init__(self, mj_model, mj_data, side, actuator_indices):
        super().__init__(mj_model, mj_data, actuator_indices, f"inspire_{side}_state")
        assert len(self.act) == 6
        self.openness = np.ones(6)
        tag = "l" if side == "left" else "r"
        self.state_puber = ChannelPublisher(f"rt/inspire_hand/state/{tag}", inspire_hand_state)
        self.state_puber.Init()
        self.cmd_suber = ChannelSubscriber(f"rt/inspire_hand/ctrl/{tag}", inspire_hand_ctrl)
        self.cmd_suber.Init(self._on_cmd, 10)
        self.start()

    def _on_cmd(self, msg):
        if not (msg.mode & 0b0001) or len(msg.angle_set) < 6:
            return
        for k, v in enumerate(msg.angle_set[:6]):
            if v >= 0:  # -1 means "leave this finger as it is"
                self.openness[k] = np.clip(v / 1000.0, 0.0, 1.0)
        self.target = self.lo + (1.0 - self.openness) * (self.hi - self.lo)

    def _publish_state(self):
        openness = 1.0 - np.clip((self.q() - self.lo) / (self.hi - self.lo), 0.0, 1.0)
        angle = [int(round(1000 * o)) for o in openness]
        self.state_puber.Write(inspire_hand_state(
            pos_act=angle, angle_act=angle, force_act=[0] * 6, current=[0] * 6,
            err=[0] * 6, status=[0] * 6, temperature=[0] * 6))


class InspireDFXBridge(_JointPdHand):
    """Inspire DFX hand (both hands on ONE topic): rt/inspire/cmd and rt/inspire/state, unitree_go
    MotorCmds_/MotorStates_, 12 motors: right hand 0-5, left hand 6-11, each in the order
    [pinky, ring, middle, index, thumb-bend, thumb-rotation]. Only q is used, normalised to [0,1]
    with 0 = closed and 1 = open. `actuator_indices` must be the 12 actuators in that order.
    Follows the G1 service (dfx_inspire_service/inspire_g1.cpp, default namespace "inspire")."""

    kp = 8.0
    kd = 0.2

    def __init__(self, mj_model, mj_data, actuator_indices):
        super().__init__(mj_model, mj_data, actuator_indices, "inspire_dfx_state")
        assert len(self.act) == 12
        self.state_puber = ChannelPublisher("rt/inspire/state", MotorStates_)
        self.state_puber.Init()
        self.cmd_suber = ChannelSubscriber("rt/inspire/cmd", MotorCmds_)
        self.cmd_suber.Init(self._on_cmd, 10)
        self.start()

    def _on_cmd(self, msg):
        if len(msg.cmds) >= 12:
            openness = np.clip([c.q for c in msg.cmds[:12]], 0.0, 1.0)
            self.target = self.lo + (1.0 - openness) * (self.hi - self.lo)

    def _publish_state(self):
        openness = 1.0 - np.clip((self.q() - self.lo) / (self.hi - self.lo), 0.0, 1.0)
        states = []
        for k in range(12):
            ms = unitree_go_msg_dds__MotorState_()
            ms.mode = 1
            ms.q = float(openness[k])
            states.append(ms)
        self.state_puber.Write(MotorStates_(states))
