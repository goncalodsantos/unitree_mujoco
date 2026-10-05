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


# ---------------------------------------------------------------------------
# Other hands. Unlike Dex3 (unitree_hg HandCmd_/HandState_, direct q/kp/kd per motor),
# these use higher-level protocols, so each bridge converts the received command to a
# joint target and runs a joint-space PD on the simulated finger joints. The PD gains
# below are OUR choices for the simulated joints, not Unitree/vendor values.
# Protocol sources: unitreerobotics/dex1_1_service, brainco_hand_service and
# xr_teleoperate (Dex1, BrainCo), inspire_hand_ws IDL (Inspire FTP).
# ---------------------------------------------------------------------------
import numpy as np
from unitree_sdk2py.idl.default import unitree_go_msg_dds__MotorState_
from unitree_sdk2py.idl.unitree_go.msg.dds_ import MotorCmds_, MotorStates_

from inspire_idl import inspire_hand_ctrl, inspire_hand_state


class _JointPdHand:
    """A group of simulated actuators held by a joint-space PD toward a target vector."""

    kp = 10.0
    kd = 0.5

    def __init__(self, mj_model, mj_data, actuator_indices, publish_name):
        self.m, self.d = mj_model, mj_data
        self.act = list(actuator_indices)
        self.total = mj_model.nu
        self.target = None  # joint-space target, set by the protocol handler
        # joint limits of the actuated joints, used to map normalised commands to angles
        self.lo = np.array([mj_model.jnt_range[mj_model.actuator_trnid[a, 0], 0] for a in self.act])
        self.hi = np.array([mj_model.jnt_range[mj_model.actuator_trnid[a, 0], 1] for a in self.act])
        # Same 100 Hz state thread as HandBridge (see the note there about CPU starvation).
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
    ONE motor per gripper (cmds[0].q). q is the motor angle in rad, 0 = closed (calibrated
    closed), up to ~5.4 rad = fully open (xr_teleoperate's mapped range). The sim model has two
    prismatic fingers (finger_joint_1/2); both move 0 .. +upper limit symmetrically.

    Like Unitree's dex1_1_service (main.cpp), the command's kp/kd/dq/tau are honoured:
        tau_motor = tau + kp*(q_des - q) + kd*(dq_des - dq)        [motor side, rad]
    The motor torque is turned into a force on each slide through the transmission
    r = finger travel / motor travel (m/rad): one motor moves both fingers, so F = tau_motor / (2 r).
    That makes the real gains stiff for these light fingers, so kd is capped to what the physics
    timestep can integrate stably (c*dt/m <= 0.5) and the force to the actuator range. A command with
    kp = kd = 0 falls back to our own PD gains kp/kd below (N/m and N*s/m on the slides)."""

    kp = 300.0
    kd = 10.0
    Q_OPEN = 5.4

    def __init__(self, mj_model, mj_data, side, actuator_indices):
        super().__init__(mj_model, mj_data, actuator_indices, f"dex1_{side}_state")
        assert len(self.act) == 2
        self.hi_open = float(self.hi.min())  # shared opening per finger (m)
        self.cmd = None
        self.r = self.hi_open / self.Q_OPEN  # finger travel per rad of motor
        # largest damping force coefficient (N*s/m) the explicit integration keeps stable
        joint = mj_model.actuator_trnid[self.act[0], 0]
        mass = mj_model.body_mass[mj_model.jnt_bodyid[joint]] + mj_model.dof_armature[mj_model.jnt_dofadr[joint]]
        self.kd_max = 0.5 * mass / mj_model.opt.timestep * 2 * self.r ** 2  # as a motor-side kd
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
            self.target = np.full(2, frac * self.hi_open)  # used by the fallback PD

    def apply_ctrl(self):
        c = self.cmd
        if c is None:
            return
        if c.kp == 0.0 and c.kd == 0.0:
            return super().apply_ctrl()
        q_motor = float(self.q().mean()) / self.r
        dq_motor = float(self.dq().mean()) / self.r
        q_des = float(np.clip(c.q, 0.0, self.Q_OPEN))
        tau_motor = c.tau + c.kp * (q_des - q_motor) + min(c.kd, self.kd_max) * (c.dq - dq_motor)
        force = float(np.clip(tau_motor / (2 * self.r), -self.f_max, self.f_max))
        for a in self.act:
            self.d.ctrl[a] = force

    def _publish_state(self):
        frac = float(np.clip(self.q().mean() / self.hi_open, 0.0, 1.0))
        ms = unitree_go_msg_dds__MotorState_()
        ms.mode = 1
        ms.q = frac * self.Q_OPEN
        ms.dq = float(self.dq().mean() / self.hi_open * self.Q_OPEN)
        self.state_puber.Write(MotorStates_([ms]))


class BrainCoBridge(_JointPdHand):
    """BrainCo hand: rt/brainco/{left,right}/cmd|state, unitree_go MotorCmds_/MotorStates_, 6 motors in
    the order [Thumb, Thumb_aux, Index, Middle, Ring, Pinky] with q normalised to [0,1]
    (0 = open, 1 = closed) and dq (speed) normalised to [0,1] (ignored here).
    `actuator_indices` must follow that order."""

    kp = 2.0
    kd = 0.1

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
    """Inspire FTP hand: rt/inspire_hand/ctrl/{l,r} (inspire::inspire_hand_ctrl) and
    rt/inspire_hand/state/{l,r} (inspire::inspire_hand_state), 6 motors in the order
    [pinky, ring, middle, index, thumb-bend, thumb-rotation]. Only angle control is simulated
    (mode bit 0): angle_set is 0..1000 with 1000 = open, 0 = closed; -1 = no change.
    Force/speed/position modes and the touch topic are ignored.
    `actuator_indices` must follow that order."""

    kp = 8.0
    kd = 0.4

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
    """Inspire DFX hand (both hands on ONE topic): rt/inspire/cmd and rt/inspire/state,
    unitree_go MotorCmds_/MotorStates_, 12 motors: right hand 0-5, left hand 6-11, each in the
    order [pinky, ring, middle, index, thumb-bend, thumb-rotation]. Only q is used, normalised
    to [0,1] with 0 = closed and 1 = open (per Unitree's H1HandController example).
    `actuator_indices` must be the 12 actuators in that order (right hand first)."""

    kp = 8.0
    kd = 0.4

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
