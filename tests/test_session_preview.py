"""Session initialization test without constructing TeleVuer or controllers."""
import asyncio
import unittest
from televuer.televuer import TeleVuer


class SessionPreviewTest(unittest.TestCase):
    def test_each_session_disables_both_previews_before_streaming(self):
        owner=TeleVuer.__new__(TeleVuer)
        for unused in range(2):
            events=[]
            class Update:
                def __matmul__(self,node):events.append(node._serialize())
            class Session: update=Update()
            async def stream(session):events.append('stream')
            asyncio.run(owner._run_session(Session(),stream))
            self.assertEqual(events[-1],'stream')
            self.assertEqual(events[0]['key'],'perspective-camera')
            self.assertIs(events[0]['previewInScene'],False)
            self.assertIs(events[0]['previewInOverlay'],False)

if __name__=='__main__':unittest.main()
