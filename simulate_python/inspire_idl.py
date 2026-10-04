"""
DDS types for the Inspire FTP hand, as published by Inspire's own SDK (not part of unitree_sdk2py).

Transcribed from Inspire's inspire_hand_sdk IDL (hand_idl/), module `inspire`:

    struct inspire_hand_ctrl  { sequence<int16,6> pos_set, angle_set, force_set, speed_set; int8 mode; };
    struct inspire_hand_state { sequence<int16,6> pos_act, angle_act, force_act, current;
                                sequence<uint8,6> err, status, temperature; };

Topics: 
    rt/inspire_hand/ctrl/{l,r}
    rt/inspire_hand/state/{l,r}. 
The tactile topic (rt/inspire_hand/touch/*) is not simulated.

NOT verified against the real inspire_sdkpy package (not installable here): if a real
client does not match these types, compare the IDL typename/extensibility first.
"""

from dataclasses import dataclass

import cyclonedds.idl as idl
import cyclonedds.idl.annotations as annotate
import cyclonedds.idl.types as types


@dataclass
@annotate.final
@annotate.autoid("sequential")
class inspire_hand_ctrl(idl.IdlStruct, typename="inspire.inspire_hand_ctrl"):
    pos_set: types.sequence[types.int16, 6]
    angle_set: types.sequence[types.int16, 6]
    force_set: types.sequence[types.int16, 6]
    speed_set: types.sequence[types.int16, 6]
    mode: types.int8


@dataclass
@annotate.final
@annotate.autoid("sequential")
class inspire_hand_state(idl.IdlStruct, typename="inspire.inspire_hand_state"):
    pos_act: types.sequence[types.int16, 6]
    angle_act: types.sequence[types.int16, 6]
    force_act: types.sequence[types.int16, 6]
    current: types.sequence[types.int16, 6]
    err: types.sequence[types.uint8, 6]
    status: types.sequence[types.uint8, 6]
    temperature: types.sequence[types.uint8, 6]
