"""Controller diagnostics are event receipts, never heartbeat or fabricated inputs."""
import importlib.util
from pathlib import Path
import pytest
spec=importlib.util.spec_from_file_location('diagnostics',Path(__file__).parents[1]/'src/televuer/controller_diagnostics.py')
module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)


def test_missing_and_repeated_samples_keep_original_event_age():
    state=module.ControllerDiagnostics()
    assert state.snapshot(now=12) is None
    state.accept({'leftState':{'triggerValue':0.2},'rightState':{'squeezeValue':0.8}},monotonic=10,unix_ms=10000)
    first=state.snapshot(now=11);second=state.snapshot(now=15)
    assert first['sequence']==second['sequence']==1
    assert second['sourceAgeSeconds']==5 and second['receivedAtUnixMs']==10000
    assert second['left']=={'triggerValue':0.2,'squeezeValue':None}
    assert second['right']=={'triggerValue':None,'squeezeValue':0.8}


def test_new_event_increments_receipt_and_invalid_values_stay_unknown():
    state=module.ControllerDiagnostics()
    state.accept({'leftState':{'triggerValue':float('nan'),'squeezeValue':False}},monotonic=10,unix_ms=10000)
    state.accept({'leftState':{'triggerValue':1.0},'rightState':{'triggerValue':2}},monotonic=12,unix_ms=12000)
    current=state.snapshot(now=12.1)
    assert current['sequence']==2 and current['sourceAgeSeconds']==pytest.approx(.1)
    assert current['right']['triggerValue'] is None
    assert current['left']['squeezeValue'] is None


def test_actual_callback_and_wrapper_mapping_share_one_receipt():
    # Execute only the production callback and final snapshot/return statements.
    # No Vuer server, DDS imports, images or hardware constructors are loaded.
    import ast
    import asyncio
    from multiprocessing import Value,Array
    from types import SimpleNamespace
    root=Path(__file__).parents[1]/'src/televuer'
    tree=ast.parse((root/'televuer.py').read_text())
    callback=next(node for node in ast.walk(tree) if isinstance(node,ast.AsyncFunctionDef) and node.name=='on_controller_move')
    namespace={};exec(compile(ast.Module(body=[callback],type_ignores=[]),'callback','exec'),namespace)
    tv=SimpleNamespace(controller_diagnostics=module.ControllerDiagnostics(),motion_data_ready_shared=Value('b',False))
    for side in ('left','right'):
        setattr(tv,side+'_arm_pose_shared',Array('d',16))
        for key in ('trigger','triggerValue','squeeze','squeezeValue','thumbstick','aButton','bButton'):
            setattr(tv,side+'_ctrl_'+key+'_shared',Value('d',0))
        setattr(tv,side+'_ctrl_thumbstickValue_shared',Array('d',2))
    event=SimpleNamespace(value={'left':[0]*16,'right':[0]*16,'leftState':{'triggerValue':.25,'squeezeValue':.4},'rightState':{'triggerValue':.75,'squeezeValue':.6}})
    asyncio.run(namespace['on_controller_move'](tv,event,None))
    assert tv.controller_diagnostics.snapshot()['sequence']==1
    for side in ('left','right'):
        for key in ('trigger','triggerValue','squeeze','squeezeValue','thumbstick','aButton','bButton'):
            setattr(tv,side+'_ctrl_'+key,getattr(tv,side+'_ctrl_'+key+'_shared').value)
        setattr(tv,side+'_ctrl_thumbstickValue',[0,0])
    tv.motion_data_ready=True
    wrapper=ast.parse((root/'tv_wrapper.py').read_text())
    atomic=next(node for node in ast.walk(wrapper) if isinstance(node,ast.With) and 'controller_diagnostics.lock' in ast.unparse(node.items[0].context_expr))
    returned=next(node for node in ast.walk(wrapper) if isinstance(node,ast.Return) and isinstance(node.value,ast.Call) and any(kw.arg=='controller_diagnostics' for kw in node.value.keywords))
    function=ast.parse('def sample(self):\n pass').body[0];function.body=[atomic,returned]
    context={'TeleData':lambda **kw:SimpleNamespace(**kw),'Brobot_world_head':None,'left_IPunitree_Brobot_waist_arm':None,'right_IPunitree_Brobot_waist_arm':None}
    import numpy as np
    import time
    tv.head_pose=tv.left_arm_pose=tv.right_arm_pose=np.eye(4)
    tv.head_received_monotonic=Value('d',time.monotonic())
    context.update({'np':np,'time':time,'CONST_HEAD_POSE':np.eye(4),'CONST_LEFT_ARM_POSE':np.eye(4),
                    'CONST_RIGHT_ARM_POSE':np.eye(4),'left_IPunitree_Brobot_world_arm':None,
                    'right_IPunitree_Brobot_world_arm':None})
    safe=next(node for node in wrapper.body if isinstance(node,ast.FunctionDef) and node.name=='safe_mat_update')
    exec(compile(ast.Module(body=[safe],type_ignores=[]),'safe','exec'),context)
    exec(compile(ast.fix_missing_locations(ast.Module(body=[function],type_ignores=[])),'wrapper','exec'),context)
    data=context['sample'](SimpleNamespace(tvuer=tv))
    assert data.left_ctrl_triggerValue==7.5 and data.right_ctrl_triggerValue==2.5
    assert data.left_ctrl_squeezeValue==.4 and data.right_ctrl_squeezeValue==.6
    assert data.controller_diagnostics['left']['triggerValue']==.25
    assert data.controller_diagnostics['sequence']==1


def test_invalidated_partial_event_cannot_certify_old_receipt():
    state=module.ControllerDiagnostics()
    state.accept({'leftState':{'triggerValue':.2}},monotonic=10,unix_ms=10000)
    state.invalidate();assert state.snapshot(now=11) is None
    state.accept({'leftState':{'triggerValue':.5}},monotonic=12,unix_ms=12000)
    assert state.snapshot(now=12)['sequence']==2
