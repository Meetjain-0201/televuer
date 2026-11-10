"""Exercise actual receiver and wrapper without servers or controller constructors."""
import asyncio
from multiprocessing import Array,Value
from types import SimpleNamespace
import numpy as np
from televuer.televuer import TeleVuer
from televuer.tv_wrapper import TeleVuerWrapper,T_ROBOT_OPENXR,T_OPENXR_ROBOT
from televuer.controller_diagnostics import ControllerDiagnostics


def receiver():
    tv=TeleVuer.__new__(TeleVuer)
    tv.controller_diagnostics=ControllerDiagnostics()
    tv.head_pose_shared=Array('d',16)
    tv.head_received_monotonic=Value('d',0)
    tv.motion_data_ready_shared=Value('b',False)
    for side in ('left','right'):
        setattr(tv,side+'_arm_pose_shared',Array('d',16))
        for name in ('trigger','triggerValue','squeeze','squeezeValue','thumbstick','aButton','bButton'):
            setattr(tv,side+'_ctrl_'+name+'_shared',Value('d',0))
        setattr(tv,side+'_ctrl_thumbstickValue_shared',Array('d',2))
    return tv


def test_world_pose_and_event_receipt_are_exposed_without_head_subtraction():
    tv=receiver();head=np.eye(4);head[0,3]=7
    arm=np.eye(4);arm[2,3]=3
    asyncio.run(tv.on_cam_move(SimpleNamespace(value={'camera':{'matrix':head.T.flatten()}}),None))
    asyncio.run(tv.on_controller_move(SimpleNamespace(value={'left':arm.T.flatten(),'right':arm.T.flatten(),
                   'leftState':{},'rightState':{}}),None))
    wrapper=TeleVuerWrapper.__new__(TeleVuerWrapper)
    wrapper.tvuer=tv;wrapper.use_hand_tracking=False;wrapper.arm_reference_mode='head_yaw'
    data=wrapper.get_tele_data()
    np.testing.assert_allclose(data.left_controller_world_pose,T_ROBOT_OPENXR@arm@T_OPENXR_ROBOT)
    assert data.controller_poses_valid and data.head_source_age_seconds<.5
    assert data.controller_diagnostics['sequence']==1
    asyncio.run(tv.on_cam_move(SimpleNamespace(value={}),None))
    assert wrapper.get_tele_data().head_source_age_seconds==float('inf')
