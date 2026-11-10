"""Atomic, process-shared controller event receipts for wrapper/IPC diagnostics.

Sequence counts receiver callbacks, not headset emissions. No poses, images or
button identities are retained. Values absent/invalid in an event stay unknown.
"""
import math
from multiprocessing import Array, RLock
import time
from uuid import uuid4


class ControllerDiagnostics:
    def __init__(self):
        self.lock=RLock()
        self.stream_id=uuid4().hex
        self.values=Array('d',[0,0,0,float('nan'),float('nan'),float('nan'),float('nan')],lock=self.lock)

    def accept(self,event,*,monotonic=None,unix_ms=None):
        def number(side,key):
            value=event.get(side,{ }).get(key)
            return float(value) if type(value) in (int,float) and math.isfinite(value) and 0<=value<=1 else float('nan')
        with self.lock:
            self.values[:]=[self.values[0]+1,time.monotonic() if monotonic is None else monotonic,
                            time.time()*1000 if unix_ms is None else unix_ms,
                            *[number(side,key) for side in ('leftState','rightState') for key in ('triggerValue','squeezeValue')]]

    def invalidate(self):
        """A partially decoded callback cannot certify any coherent input pair."""
        with self.lock:
            self.values[1]=0

    def snapshot(self,*,now=None):
        with self.lock:
            sequence,received,unix_ms,*raw=self.values[:]
        if not sequence or not received:return None
        def hand(offset):
            return {key:raw[offset+i] if math.isfinite(raw[offset+i]) else None
                    for i,key in enumerate(('triggerValue','squeezeValue'))}
        return {'schemaVersion':1,'source':'vuer-controller-event','streamId':self.stream_id,
                'sequence':int(sequence),'receivedAtUnixMs':unix_ms,
                'sourceAgeSeconds':max(0,(time.monotonic() if now is None else now)-received),
                'left':hand(0),'right':hand(2)}
