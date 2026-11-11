from vuer import Vuer
from vuer.schemas import PerspectiveCamera, ImageBackground, Hands, MotionControllers, WebRTCVideoPlane, WebRTCStereoVideoPlane
# DIMENSO ADDITION: free-floating panels for the wrist cameras (DIM-547).
# ImageBackground, not Image: Image serialises the array as RAW `rgb` (~920KB per
# 640x480 frame, per panel, per tick), while ImageBackground compresses to `src`
# and honours format/quality. Both keep `matrix`, so camera-relative placement is
# unaffected by the choice -- bandwidth is the whole difference.
# Image, NOT ImageBackground: three ImageBackgrounds in one scene killed the Vuer
# child process outright (client alive, :8012 gone, leaked shared memory). Image
# ships raw `rgb` rather than compressed `src`, which costs bandwidth -- that is a
# trade to revisit, but a working client beats a smaller payload.
# Image, NOT ImageBackground: a *Background* is singular by design -- three of them
# put three payloads on the wire and the client keeps ONE, which is exactly what the
# operator saw. Image is a real plane, but world-space, so its position is computed
# from the LIVE head pose below to make it track the head anyway.
# The SAME element as the main frame. ImageBackground.position is an OFFSET FROM
# CAMERA, so all three panels ride the head together. Using `Image` (world-space)
# for the wings is what left them standing still in the room.
# ImageBackground for ALL THREE. Its `position` is an OFFSET FROM THE CAMERA, which is the
# only mechanism in this client that makes a panel ride the head -- and all three moving
# together is the thing that actually worked. `Image` was tried for the wings and is
# world-space: the wings stopped following the head and ended up on the floor of the room.
from vuer.schemas import ImageBackground as _DimensoImage
from multiprocessing import Value, Array, Process, shared_memory
# DIMENSO ADDITION (DIM-914): controller-only event provenance.
from .controller_diagnostics import ControllerDiagnostics
import numpy as np
import asyncio
import threading
import cv2
import os
from pathlib import Path
from typing import Literal


class TeleVuer:
    def __init__(self, use_hand_tracking: bool, binocular: bool=True, img_shape: tuple=None, display_fps: float=30.0,
                       display_mode: Literal["immersive", "pass-through", "ego"]="immersive", zmq: bool=False, webrtc: bool=False, webrtc_url: str=None, 
                       cert_file: str=None, key_file: str=None):
        """
        TeleVuer class for OpenXR-based XR teleoperate applications.
        This class handles the communication with the Vuer server and manages image and pose data.

        :param use_hand_tracking: bool, whether to use hand tracking or controller tracking.
        :param binocular: bool, whether the application is binocular (stereoscopic) or monocular.
        :param img_shape: tuple, shape of the head image (height, width).
        :param display_fps: float, target frames per second for display updates (default: 30.0).
        
        :param display_mode: str, controls the VR viewing mode. Options are "immersive", "pass-through", and "ego".
        :param zmq: bool, whether to use zmq for image transmission.
        :param webrtc: bool, whether to use webrtc for real-time communication.
        :param webrtc_url: str, URL for the webrtc offer. must be provided if webrtc is True.
        :param cert_file: str, path to the SSL certificate file.
        :param key_file: str, path to the SSL key file.

        Note:

        - display_mode controls what the VR headset displays:
            * "immersive": fully immersive mode; VR shows the robot's first-person view (zmq or webrtc must be enabled).
            * "pass-through": VR shows the real world through the VR headset cameras; no image from zmq or webrtc is displayed (even if enabled).
            * "ego": a small window in the center shows the robot's first-person view, while the surrounding area shows the real world.
        
        - Only one image mode is active at a time.
        - Image transmission to VR occurs only if display_mode is "immersive" or "ego" and the corresponding zmq or webrtc option is enabled.
        - If zmq and webrtc simultaneously enabled, webrtc will be prioritized.

        --------------              -------------------           --------------       -----------------                     -------
         display_mode       |        display behavior         |    image to VR     |      image source        |               Notes
        --------------              -------------------           --------------       -----------------                     ------- 
           immersive        |   fully immersive view (robot)  |     Yes (full)     |     zmq or webrtc        |   if both enabled, webrtc prioritized
        --------------              -------------------           --------------       -----------------                     -------
         pass-through       |       Real world view (VR)      |         No         |          N/A             |  even if image source enabled, don't display
        --------------              -------------------           --------------       -----------------                     -------
              ego           |      ego view (robot + VR)      |    Yes (small)     |     zmq or webrtc        |   if both enabled, webrtc prioritized
        --------------              -------------------           --------------       -----------------                     -------

        """
        self.use_hand_tracking = use_hand_tracking
        self.binocular = binocular
        if img_shape is None:
            raise ValueError("[TeleVuer] img_shape must be provided.")
        self.img_shape = (img_shape[0], img_shape[1], 3)
        self.display_fps = display_fps
        self.img_height = self.img_shape[0]
        if self.binocular:
            self.img_width  = self.img_shape[1] // 2
        else:
            self.img_width  = self.img_shape[1]
        self.aspect_ratio = self.img_width / self.img_height

        # SSL certificate path resolution
        env_cert = os.getenv("XR_TELEOP_CERT")
        env_key = os.getenv("XR_TELEOP_KEY")
        if cert_file is None or key_file is None:
            # 1.Try environment variables
            if env_cert and env_key:
                cert_file = cert_file or env_cert
                key_file = key_file or env_key
            else:
                # 2.Try ~/.config/xr_teleoperate/
                user_conf_dir = Path.home() / ".config" / "xr_teleoperate"
                cert_path_user = user_conf_dir / "cert.pem"
                key_path_user = user_conf_dir / "key.pem"

                if cert_path_user.exists() and key_path_user.exists():
                    cert_file = cert_file or str(cert_path_user)
                    key_file = key_file or str(key_path_user)
                else:
                    # 3.Fallback to package root (current logic)
                    current_module_dir = Path(__file__).resolve().parent.parent.parent
                    cert_file = cert_file or str(current_module_dir / "cert.pem")
                    key_file = key_file or str(current_module_dir / "key.pem")

        self.vuer = Vuer(host='0.0.0.0', cert=cert_file, key=key_file, queries=dict(grid=False), queue_len=3)
        self.controller_diagnostics = ControllerDiagnostics()
        self.vuer.add_handler("CAMERA_MOVE")(self.on_cam_move)
        if self.use_hand_tracking:
            self.vuer.add_handler("HAND_MOVE")(self.on_hand_move)
        else:
            self.vuer.add_handler("CONTROLLER_MOVE")(self.on_controller_move)

        self.display_mode = display_mode
        self.zmq = zmq
        self.webrtc = webrtc
        self.webrtc_url = webrtc_url

        if self.display_mode == "immersive":
            if self.webrtc:
                fn = self.main_image_binocular_webrtc if self.binocular else self.main_image_monocular_webrtc
            elif self.zmq:
                self.img2display_shm = shared_memory.SharedMemory(create=True, size=np.prod(self.img_shape) * np.uint8().itemsize)
                self.img2display = np.ndarray(self.img_shape, dtype=np.uint8, buffer=self.img2display_shm.buf)
                self.latest_frame = None
                self.new_frame_event = threading.Event()
                self.stop_writer_event = threading.Event()
                self.writer_thread = threading.Thread(target=self._xr_render_loop, daemon=True)
                self.writer_thread.start()
                # ---- DIMENSO ADDITION (DIM-547): wrist camera panels ------------------
                # Shared memory, exactly like the head buffer above, and for the same
                # reason: Vuer runs in a CHILD PROCESS, so a plain attribute written by the
                # parent would never be seen by the upsert loop.
                self._dimenso_init_wrist_buffers()
                # ---- END DIMENSO ADDITION --------------------------------------------
                fn = self.main_image_binocular_zmq if self.binocular else self.main_image_monocular_zmq
            else:
                raise ValueError("[TeleVuer] immersive mode requires zmq=True or webrtc=True.")
        elif self.display_mode == "ego":
            if self.webrtc:
                fn = self.main_image_binocular_webrtc_ego if self.binocular else self.main_image_monocular_webrtc_ego
            elif self.zmq:
                self.img2display_shm = shared_memory.SharedMemory(create=True, size=np.prod(self.img_shape) * np.uint8().itemsize)
                self.img2display = np.ndarray(self.img_shape, dtype=np.uint8, buffer=self.img2display_shm.buf)
                self.latest_frame = None
                self.new_frame_event = threading.Event()
                self.stop_writer_event = threading.Event()
                self.writer_thread = threading.Thread(target=self._xr_render_loop, daemon=True)
                self.writer_thread.start()
                fn = self.main_image_binocular_zmq_ego if self.binocular else self.main_image_monocular_zmq_ego
            else:
                raise ValueError("[TeleVuer] ego mode requires zmq=True or webrtc=True.")
        elif self.display_mode == "pass-through":
            fn = self.main_pass_through
        else:
            raise ValueError(f"[TeleVuer] Unknown display_mode: {self.display_mode}")
        
        async def session_handler(session):
            # Camera-only repair: auxiliary previews overwrite the immersive framebuffer.
            session.update @ PerspectiveCamera(
                key="perspective-camera", previewInScene=False, previewInOverlay=False)
            await fn(session)
        self.vuer.spawn(start=False)(session_handler)

        self.head_pose_shared = Array('d', 16, lock=True)
        self.left_arm_pose_shared = Array('d', 16, lock=True)
        self.right_arm_pose_shared = Array('d', 16, lock=True)
        self.motion_data_ready_shared = Value('b', False, lock=True)
        if self.use_hand_tracking:
            self.left_hand_position_shared = Array('d', 75, lock=True)
            self.right_hand_position_shared = Array('d', 75, lock=True)
            self.left_hand_orientation_shared = Array('d', 25 * 9, lock=True)
            self.right_hand_orientation_shared = Array('d', 25 * 9, lock=True)

            self.left_hand_pinch_shared = Value('b', False, lock=True)
            self.left_hand_pinchValue_shared = Value('d', 0.0, lock=True)
            self.left_hand_squeeze_shared = Value('b', False, lock=True)
            self.left_hand_squeezeValue_shared = Value('d', 0.0, lock=True)

            self.right_hand_pinch_shared = Value('b', False, lock=True)
            self.right_hand_pinchValue_shared = Value('d', 0.0, lock=True)
            self.right_hand_squeeze_shared = Value('b', False, lock=True)
            self.right_hand_squeezeValue_shared = Value('d', 0.0, lock=True)
        else:
            self.left_ctrl_trigger_shared = Value('b', False, lock=True)
            self.left_ctrl_triggerValue_shared = Value('d', 0.0, lock=True)
            self.left_ctrl_squeeze_shared = Value('b', False, lock=True)
            self.left_ctrl_squeezeValue_shared = Value('d', 0.0, lock=True)
            self.left_ctrl_thumbstick_shared = Value('b', False, lock=True)
            self.left_ctrl_thumbstickValue_shared = Array('d', 2, lock=True)
            self.left_ctrl_aButton_shared = Value('b', False, lock=True)
            self.left_ctrl_bButton_shared = Value('b', False, lock=True)

            self.right_ctrl_trigger_shared = Value('b', False, lock=True)
            self.right_ctrl_triggerValue_shared = Value('d', 0.0, lock=True)
            self.right_ctrl_squeeze_shared = Value('b', False, lock=True)
            self.right_ctrl_squeezeValue_shared = Value('d', 0.0, lock=True)
            self.right_ctrl_thumbstick_shared = Value('b', False, lock=True)
            self.right_ctrl_thumbstickValue_shared = Array('d', 2, lock=True)
            self.right_ctrl_aButton_shared = Value('b', False, lock=True)
            self.right_ctrl_bButton_shared = Value('b', False, lock=True)

        self.process = Process(target=self._vuer_run)
        self.process.daemon = True
        self.process.start()
    
    def _vuer_run(self):
        try:
            self.vuer.run()
        except KeyboardInterrupt:
            pass
        except Exception as e:
            print(f"Vuer encountered an error: {e}")
        finally:
            if hasattr(self, "stop_writer_event"):
                self.stop_writer_event.set()

    # ------------------------------------------------------------------ DIMENSO ADDITION
    # DIM-547: wrist camera views as FREE-FLOATING panels in the XR scene.
    #
    # Meet: "add wrist cameras, in the frame as well, so i can overcome occlusion" -- when the
    # hand is over an object the head camera cannot see the contact point. He chose floating
    # panels over insets burned into the head image, so each view can be placed in the room
    # and looked away from: "left wrist in left and right in right".
    #
    # Why this needs a fork at all: televuer draws exactly ONE stream. All eight of its
    # main_image_* paths upsert a single ImageBackground, and upstream issue #219 shows the
    # consequence -- a misconfigured head camera makes the UI fall back to a WRIST view
    # instead of the head view, never both. Simultaneous views are not a config flag.
    #
    # teleimager already receives the streams (get_left_wrist_frame / get_right_wrist_frame),
    # so nothing there changes; only the drawing does.
    DIMENSO_WRIST_SIDES = ("left", "right")
    # ScreenX-style side screens: a main view ahead with the wrist views canted in at the
    # edges of vision, ALL MOVING WITH THE HEAD. Metres, in CAMERA space -- the first version
    # placed these in the play-space and Meet reported exactly the consequence: "it does not
    # move with the head movement like the main frame does". Every value is env-tunable so the
    # geometry can be adjusted in the headset, which is the only place it can be judged.
    #
    # THE GEOMETRY KNOBS ARE DECLARED ONCE, in the block above `_dimenso_wing_matrix`.
    #
    # They used to be declared TWICE in this class body -- here AND there -- and in a class
    # body the LATER assignment silently wins. So this block was dead: it read
    # DIMENSO_WRIST_YAW_DEG 20.0 while the effective value was 50.0, and HEIGHT 0.62 while
    # the effective value was 0.80. Editing a value here changed nothing, with no error and
    # nothing in any log -- and these are precisely the values Meet tunes from inside the
    # headset, where a silent no-op costs a whole test cycle. Two of them (WRIST_DISTANCE,
    # WRIST_ASPECT) were read by nothing at all once `_dimenso_wing_matrix` superseded the
    # earlier placement, so they are gone rather than left looking load-bearing.

    def _dimenso_init_wrist_buffers(self):
        """One shared-memory RGB buffer per wrist, sized like the head frame.

        Never fatal: if the buffers cannot be created the head view must still work. A
        teleoperator losing the main view because a secondary panel failed to allocate would
        be a strictly worse trade than not having the panels.
        """
        self._dimenso_wrist_shm = {}
        self._dimenso_wrist = {}
        try:
            for side in self.DIMENSO_WRIST_SIDES:
                # A quarter turn swaps the frame's dimensions. Sizing the buffer landscape
                # and then resizing a portrait frame into it would squash the picture, so the
                # buffer takes the ROTATED shape.
                h, w, c = self.img_shape
                shape = (w, h, c) if self.DIMENSO_WRIST_ROTATE % 180 == 90 else (h, w, c)
                self._dimenso_wrist_shape = shape
                shm = shared_memory.SharedMemory(
                    create=True, size=int(np.prod(shape)) * np.uint8().itemsize)
                self._dimenso_wrist_shm[side] = shm
                self._dimenso_wrist[side] = np.ndarray(
                    shape, dtype=np.uint8, buffer=shm.buf)
                self._dimenso_wrist[side][:] = 0
        except Exception as e:
            print(f"[TeleVuer][dimenso] wrist panels unavailable ({e!r}); head view unaffected")
            self._dimenso_wrist = {}

    def render_wrist_to_xr(self, side, image):
        """Publish one wrist frame. `side` is 'left' or 'right'.

        Converted and copied inline rather than through a writer thread: these are secondary
        views at a fraction of the head camera's importance, and an extra thread per wrist
        would add two more places for the display to stall. A missing or wrong-shaped frame
        is DROPPED, never partially written -- a torn frame in the corner of the eye is worse
        than a stale one.
        """
        buf = getattr(self, "_dimenso_wrist", {}).get(side)
        if buf is None or image is None:
            return
        try:
            # Rotate FIRST, then fit. The wrist cameras are mounted turned relative to the
            # operator's view, so the raw frame reads sideways -- rotating it is what makes the
            # panel a POV of the hand rather than a picture lying on its side.
            rot = self.DIMENSO_WRIST_ROTATE % 360
            if rot == 90:
                image = cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE)
            elif rot == 270:
                image = cv2.rotate(image, cv2.ROTATE_90_COUNTERCLOCKWISE)
            elif rot == 180:
                image = cv2.rotate(image, cv2.ROTATE_180)
            if image.shape != buf.shape:
                image = cv2.resize(image, (buf.shape[1], buf.shape[0]))
            buf[:] = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        except Exception:
            return

    # --- ScreenX layout (DIM-547) -------------------------------------------------------
    # Centre frame is DIMENSO_MAIN_HEIGHT tall at DIMENSO_MAIN_DISTANCE, so its half-width is
    # (height * aspect) / 2. Each wing is the same height and is placed so its INNER EDGE
    # meets that outer edge, then canted back by YAW -- the cinema layout, not two floating
    # thumbnails.
    DIMENSO_MAIN_HEIGHT = float(os.environ.get("DIMENSO_MAIN_HEIGHT", "0.80"))
    DIMENSO_WRIST_YAW_DEG = float(os.environ.get("DIMENSO_WRIST_YAW_DEG", "50.0"))
    DIMENSO_WRIST_HEIGHT = float(os.environ.get("DIMENSO_WRIST_HEIGHT", "0.80"))
    # x offset: the seam knob. 0.77 left a visible gap at yaw 38.
    DIMENSO_WRIST_OFFSET_X = float(os.environ.get("DIMENSO_WRIST_OFFSET_X", "0.62"))
    # depth: the value that rendered. Larger pushes the wings behind the centre.
    DIMENSO_WRIST_DEPTH = float(os.environ.get("DIMENSO_WRIST_DEPTH", "0.82"))
    # The centre frame's distance. Was the literal `distanceToCamera=1` at each of its call
    # sites; named so ONE knob can move the whole rig.
    DIMENSO_MAIN_DISTANCE = float(os.environ.get("DIMENSO_MAIN_DISTANCE", "1.0"))

    # ---- 2026-08-11: zoom the whole three-frame layout out ---------------------------------
    # Meet, from inside the headset: "zoom out the whole 3 frame layout a bit, when i wear the
    # head set its a bit close, zoom out a bit so i can see wrist cameras as well".
    #
    # This multiplies the two DISTANCES ONLY (centre frame + wings) and deliberately leaves
    # the heights and the lateral offset alone. That is what "zoom out" means here, and the
    # arithmetic is the reason -- it is easy to get backwards:
    #
    #   angular size   of a panel = 2*atan((h/2) / d)      -> shrinks as d grows
    #   angular OFFSET of a wing  =   atan( x    / d)      -> shrinks as d grows
    #
    # so raising d alone pulls the wings in toward the centre AND makes every panel smaller,
    # which is the whole request. Scaling x and h by the same factor as d would cancel both
    # terms exactly and change nothing visible -- a uniform scale of a camera-relative rig is
    # angle-preserving. (That was the first thing tried on paper; it is why this knob touches
    # distance and nothing else.)
    #
    # Measured from the committed geometry, aspect 4:3, wrist frames rotated 270:
    #                       scale 1.0            scale 1.30
    #   centre half-height   21.8 deg             17.1 deg
    #   wing centre offset   37.1 deg             30.2 deg
    #   layout half-width   ~50    deg           ~41    deg   (Quest 3 is ~110 deg total)
    #
    # At 1.0 the outer edges sat within a few degrees of the panel's own FOV limit, which is
    # exactly what "a bit close" describes. Tunable without touching code and without a SIM
    # relaunch -- only the teleop client restarts -- because judging it needs the headset.
    DIMENSO_VIEW_SCALE = float(os.environ.get("DIMENSO_VIEW_SCALE", "1.30"))

    @property
    def _dimenso_main_distance(self):
        """Centre-frame distance, zoom applied. A property, not a class constant computed at
        import: the two are identical today, but a constant here would silently ignore any
        future runtime change to the scale, which is the `DIMENSO_DIAG` trap in this repo."""
        return self.DIMENSO_MAIN_DISTANCE * self.DIMENSO_VIEW_SCALE

    def _dimenso_wing_matrix(self, side):
        """Full TRS matrix for one wing, IN CAMERA SPACE, column-major.

        Read out of the client bundle rather than guessed:

            useLayoutEffect(... if (!ref.current || !fixed) return; ...position/rotation... )
            useFrame(... if (fixed) return;
                         if (matrix) { m.fromArray(matrix).premultiply(camera.matrixWorld) }
                         else camera.matrixWorld.decompose(mesh.position, ...) )

        So with the default `fixed=false`, `position`/`rotation` are IGNORED and every panel is
        placed exactly AT the camera -- which is why three panels looked like one. `matrix` is
        the only offset the component honours, and it is premultiplied by the camera, so it is
        camera-relative and rides the head.

        SCALE IS BAKED IN and `height`/`aspect` are NOT passed: those set mesh.scale earlier in
        the same useFrame, and `matrix.decompose()` then overwrites it. That collision is why
        the first matrix attempt rendered nothing usable.
        """
        import math
        sign = -1.0 if side == "left" else 1.0
        h = self.DIMENSO_WRIST_HEIGHT
        w = h * (self._dimenso_wrist_shape[1] / self._dimenso_wrist_shape[0])
        yaw = -sign * math.radians(self.DIMENSO_WRIST_YAW_DEG)
        c, sn = math.cos(yaw), math.sin(yaw)
        x = sign * self.DIMENSO_WRIST_OFFSET_X
        # 2026-08-11: zoom applies to the DEPTH only, never to x or h -- scaling all three is an
        # angle-preserving no-op. See DIMENSO_VIEW_SCALE.
        z = -self.DIMENSO_WRIST_DEPTH * self.DIMENSO_VIEW_SCALE
        # column-major: [ Rx*w | Ry*h | Rz | T ]
        return [ c * w, 0.0, -sn * w, 0.0,
                   0.0,   h,     0.0, 0.0,
                    sn, 0.0,       c, 0.0,
                     x, 0.0,       z, 1.0]

    # `_dimenso_wrist_matrix` was DELETED here, deliberately, rather than left in place.
    #
    # It was the earlier placement helper, fully superseded by `_dimenso_wing_matrix` above
    # (which bakes scale into the matrix). Nothing called it. It was also already BROKEN: its
    # body read `self.DIMENSO_WRIST_OFFSET_Y`, which is not defined anywhere in this class, so
    # the first call would have raised AttributeError -- and it would have been called from
    # the panel path, on a live session, in the headset.
    #
    # Kept-but-dead was the wrong choice for it: two same-shaped matrix helpers side by side,
    # one of them a landmine, is how the wrong one gets wired back in during a merge. The
    # retained-on-purpose fallbacks in this project (walk_mode, policy.onnx) are retained
    # because something still selects them; this had no such path.

    # The wrist cameras are mounted rotated relative to the operator's view, so the raw frame
    # reads sideways. 90 = clockwise, -90 = counter-clockwise, 180, or 0 to leave it alone.
    DIMENSO_WRIST_ROTATE = int(os.environ.get("DIMENSO_WRIST_ROTATE", "270"))
    DIMENSO_WRIST_PERIOD_S = float(os.environ.get("DIMENSO_WRIST_PERIOD_S", "0.2"))
    _dimenso_wrist_next = {}

    def _dimenso_wrist_panels(self, session):
        """The scene elements for the wrist views, or [] when they are unavailable.

        Emitted every Nth call, NOT every frame. The head view runs at display_fps; adding two
        more images at that rate is what killed the vuer child twice on a live session."""
        # PER SESSION, and time-based. The first version used a class-level tick counter and
        # a modulo, so with more than one session connected (headset + a probe, or two tabs)
        # the wings went only to whichever loop happened to hit the multiple -- the other
        # session never received them at all. That is why the panel count changed on reload.
        # A new session ALWAYS gets them immediately, then they refresh on a wall clock.
        import time as _t
        key = id(session)
        now = _t.monotonic()
        due = self._dimenso_wrist_next.get(key)
        if due is not None and now < due:
            return []
        self._dimenso_wrist_next[key] = now + self.DIMENSO_WRIST_PERIOD_S
        out = []
        for side in self.DIMENSO_WRIST_SIDES:
            buf = getattr(self, "_dimenso_wrist", {}).get(side)
            if buf is None:
                continue
            _dimenso_place = self._dimenso_wing_matrix(side)
            out.append(_DimensoImage(
                buf,
                key=f"dimenso-wrist-{side}",
                # CAMERA-RELATIVE, not world-placed. `position=`/`rotation=` put the panel at a
                # fixed spot in the ROOM, so it slides out of view the moment the operator turns
                # their head -- which is exactly what Meet saw: "it does not move with the head
                # movement like the main frame does".
                #
                # A `matrix` WITHOUT `fixed` is the camera-relative path in vuer's plane
                # component: it does `mesh.matrix.fromArray(matrix)` then
                # `.premultiply(camera.matrixWorld)`, so the transform is interpreted in the
                # camera's own frame and the panel rides with the head like the main view.
                # matrix ONLY. `position`/`rotation` are ignored unless `fixed` is true (the
                # component's own useLayoutEffect bails on `!fixed`), and `height`/`aspect`
                # set mesh.scale which `matrix.decompose()` then overwrites. Scale is baked
                # into the matrix instead.
                matrix=_dimenso_place,
                format="jpeg",
                quality=70,          # secondary views; the head stream keeps the bandwidth
                interpolate=True,
            ))
        return out
    # ---------------------------------------------------------------- END DIMENSO ADDITION

    def _xr_render_loop(self):
        while not self.stop_writer_event.is_set():
            if not self.new_frame_event.wait(timeout=0.1):
                continue
            self.new_frame_event.clear()
            if self.latest_frame is None:
                continue
            latest_frame = self.latest_frame
            latest_frame = cv2.cvtColor(latest_frame, cv2.COLOR_BGR2RGB)
            self.img2display[:] = latest_frame
    
    def render_to_xr(self, image):
        if self.webrtc or self.display_mode == "pass-through":
            print("[TeleVuer] Warning: render_to_xr is ignored when webrtc is enabled or pass_through is True.")
            return
        self.latest_frame = image
        self.new_frame_event.set()

    def close(self):
        self.process.terminate()
        self.process.join(timeout=0.5)
        if self.display_mode in ("immersive", "ego") and not self.webrtc:
            self.stop_writer_event.set()
            self.new_frame_event.set()
            self.writer_thread.join(timeout=0.5)
            try:
                self.img2display_shm.close()
                self.img2display_shm.unlink()
            except:
                pass

    async def on_cam_move(self, event, session, fps=60):
        try:
            with self.head_pose_shared.get_lock():
                self.head_pose_shared[:] = event.value["camera"]["matrix"]
        except:
            pass

    async def on_controller_move(self, event, session, fps=60):
        """https://docs.vuer.ai/en/latest/examples/20_motion_controllers.html"""
        # DIMENSO ADDITION (DIM-914): failure invalidation is atomic with writes.
        with self.controller_diagnostics.lock:
            try:
                # ControllerData
                with self.left_arm_pose_shared.get_lock():
                    self.left_arm_pose_shared[:] = event.value["left"]
                with self.right_arm_pose_shared.get_lock():
                    self.right_arm_pose_shared[:] = event.value["right"]
                # ControllerState
                left_controller = event.value["leftState"]
                right_controller = event.value["rightState"]

                def extract_controllers(controllerState, prefix):
                    # trigger
                    with getattr(self, f"{prefix}_ctrl_trigger_shared").get_lock():
                        getattr(self, f"{prefix}_ctrl_trigger_shared").value = bool(controllerState.get("trigger", False))
                    with getattr(self, f"{prefix}_ctrl_triggerValue_shared").get_lock():
                        getattr(self, f"{prefix}_ctrl_triggerValue_shared").value = float(controllerState.get("triggerValue", 0.0))
                    # squeeze
                    with getattr(self, f"{prefix}_ctrl_squeeze_shared").get_lock():
                        getattr(self, f"{prefix}_ctrl_squeeze_shared").value = bool(controllerState.get("squeeze", False))
                    with getattr(self, f"{prefix}_ctrl_squeezeValue_shared").get_lock():
                        getattr(self, f"{prefix}_ctrl_squeezeValue_shared").value = float(controllerState.get("squeezeValue", 0.0))
                    # thumbstick
                    with getattr(self, f"{prefix}_ctrl_thumbstick_shared").get_lock():
                        getattr(self, f"{prefix}_ctrl_thumbstick_shared").value = bool(controllerState.get("thumbstick", False))
                    with getattr(self, f"{prefix}_ctrl_thumbstickValue_shared").get_lock():
                        getattr(self, f"{prefix}_ctrl_thumbstickValue_shared")[:] = controllerState.get("thumbstickValue", [0.0, 0.0])
                    # buttons
                    with getattr(self, f"{prefix}_ctrl_aButton_shared").get_lock():
                        getattr(self, f"{prefix}_ctrl_aButton_shared").value = bool(controllerState.get("aButton", False))
                    with getattr(self, f"{prefix}_ctrl_bButton_shared").get_lock():
                        getattr(self, f"{prefix}_ctrl_bButton_shared").value = bool(controllerState.get("bButton", False))

                extract_controllers(left_controller, "left")
                extract_controllers(right_controller, "right")
                self.controller_diagnostics.accept(event.value)
                with self.motion_data_ready_shared.get_lock():
                    self.motion_data_ready_shared.value = True
            except:
                # Existing decoding failed, possibly after some shared fields changed.
                # Never pair those partial values with a previous receipt.
                self.controller_diagnostics.invalidate()

    async def on_hand_move(self, event, session, fps=60):
        """https://docs.vuer.ai/en/latest/examples/19_hand_tracking.html"""
        try:
            # HandsData
            left_hand_data = event.value["left"]
            right_hand_data = event.value["right"]
            left_hand = event.value["leftState"]
            right_hand = event.value["rightState"]
            # HandState
            def extract_hand_poses(hand_data, arm_pose_shared, hand_position_shared, hand_orientation_shared):
                with arm_pose_shared.get_lock():
                    arm_pose_shared[:] = hand_data[0:16]

                with hand_position_shared.get_lock():
                    for i in range(25):
                        base = i * 16
                        hand_position_shared[i * 3: i * 3 + 3] = [hand_data[base + 12], hand_data[base + 13], hand_data[base + 14]]

                with hand_orientation_shared.get_lock():
                    for i in range(25):
                        base = i * 16
                        hand_orientation_shared[i * 9: i * 9 + 9] = [
                            hand_data[base + 0], hand_data[base + 1], hand_data[base + 2],
                            hand_data[base + 4], hand_data[base + 5], hand_data[base + 6],
                            hand_data[base + 8], hand_data[base + 9], hand_data[base + 10],
                        ]

            def extract_hands(handState, prefix):
                # pinch
                with getattr(self, f"{prefix}_hand_pinch_shared").get_lock():
                    getattr(self, f"{prefix}_hand_pinch_shared").value = bool(handState.get("pinch", False))
                with getattr(self, f"{prefix}_hand_pinchValue_shared").get_lock():
                    getattr(self, f"{prefix}_hand_pinchValue_shared").value = float(handState.get("pinchValue", 0.0))
                # squeeze
                with getattr(self, f"{prefix}_hand_squeeze_shared").get_lock():
                    getattr(self, f"{prefix}_hand_squeeze_shared").value = bool(handState.get("squeeze", False))
                with getattr(self, f"{prefix}_hand_squeezeValue_shared").get_lock():
                    getattr(self, f"{prefix}_hand_squeezeValue_shared").value = float(handState.get("squeezeValue", 0.0))

            extract_hand_poses(left_hand_data, self.left_arm_pose_shared, self.left_hand_position_shared, self.left_hand_orientation_shared)
            extract_hand_poses(right_hand_data, self.right_arm_pose_shared, self.right_hand_position_shared, self.right_hand_orientation_shared)
            extract_hands(left_hand, "left")
            extract_hands(right_hand, "right")
            with self.motion_data_ready_shared.get_lock():
                self.motion_data_ready_shared.value = True

        except:
            pass
    
    ## immersive MODE
    async def main_image_binocular_zmq(self, session):
        if self.use_hand_tracking:
            session.upsert(
                Hands(
                    stream=True,
                    key="hands",
                    hideLeft=True,
                    hideRight=True
                ),
                to="bgChildren",
            )
        else:
            session.upsert(
                MotionControllers(
                    stream=True,
                    key="motionControllers",
                    left=True,
                    right=True,
                ),
                to="bgChildren",
            )
        while True:
            session.upsert(
                [
                    ImageBackground(
                        self.img2display[:, :self.img_width],
                        aspect=self.aspect_ratio,
                        height=self.DIMENSO_MAIN_HEIGHT,
                        # 2026-08-11. `height`+`aspect` are correct HERE and would be a bug on the
                        # wings: this is the upstream path and passes no `matrix`, so nothing
                        # overwrites mesh.scale. The wings bake scale in instead.
                        distanceToCamera=self._dimenso_main_distance,
                        # The underlying rendering engine supported a layer binary bitmask for both objects and the camera. 
                        # Below we set the two image planes, left and right, to layers=1 and layers=2. 
                        # Note that these two masks are associated with left eye’s camera and the right eye’s camera.
                        layers=1,
                        format="jpeg",
                        quality=80,
                        key="background-left",
                        interpolate=True,
                    ),
                    ImageBackground(
                        self.img2display[:, self.img_width:],
                        aspect=self.aspect_ratio,
                        height=self.DIMENSO_MAIN_HEIGHT,
                        # 2026-08-11. `height`+`aspect` are correct HERE and would be a bug on the
                        # wings: this is the upstream path and passes no `matrix`, so nothing
                        # overwrites mesh.scale. The wings bake scale in instead.
                        distanceToCamera=self._dimenso_main_distance,
                        layers=2,
                        format="jpeg",
                        quality=80,
                        key="background-right",
                        interpolate=True,
                    ),
                ],
                to="bgChildren",
            )
            # 'jpeg' encoding should give you about 30fps with a 16ms wait in-between.
            await asyncio.sleep(1.0 / self.display_fps)

    async def main_image_monocular_zmq(self, session):
        if self.use_hand_tracking:
            session.upsert(
                Hands(
                    stream=True,
                    key="hands",
                    hideLeft=True,
                    hideRight=True
                ),
                to="bgChildren",
            )
        else:
            session.upsert(
                MotionControllers(
                    stream=True, 
                    key="motionControllers",
                    left=True,
                    right=True,
                ),
                to="bgChildren",
            )

        while True:
            session.upsert(
                [
                    ImageBackground(
                        self.img2display,
                        aspect=self.aspect_ratio,
                        height=self.DIMENSO_MAIN_HEIGHT,
                        # 2026-08-11. `height`+`aspect` are correct HERE and would be a bug on the
                        # wings: this is the upstream path and passes no `matrix`, so nothing
                        # overwrites mesh.scale. The wings bake scale in instead.
                        distanceToCamera=self._dimenso_main_distance,
                        format="jpeg",
                        quality=80,
                        key="background-mono",
                        interpolate=True,
                    ),
                    # DIMENSO ADDITION (DIM-547): the wrist panels ride the SAME upsert as
                    # the head view. Sent together so a slow panel can never delay the main
                    # view, and so the two can never drift into different frames.
                    *self._dimenso_wrist_panels(session),
                ],
                to="bgChildren",
            )
            await asyncio.sleep(1.0 / self.display_fps)

    async def main_image_binocular_webrtc(self, session):
        if self.use_hand_tracking:
            session.upsert(
                Hands(
                    stream=True,
                    key="hands",
                    hideLeft=True,
                    hideRight=True
                ),
                to="bgChildren",
            )
        else:
            session.upsert(
                MotionControllers(
                    stream=True, 
                    key="motionControllers",
                    left=True,
                    right=True,
                ),
                to="bgChildren",
            )

        while True:
            session.upsert(
                WebRTCStereoVideoPlane(
                    src=self.webrtc_url,
                    iceServer=None,
                    iceServers=[], 
                    key="video-quad",
                    aspect=self.aspect_ratio,
                    height = 7,
                    layout="stereo-left-right"
                ),
                to="bgChildren",
            )
            await asyncio.sleep(1.0 / self.display_fps)

    async def main_image_monocular_webrtc(self, session):
        if self.use_hand_tracking:
            session.upsert(
                Hands(
                    stream=True,
                    key="hands",
                    hideLeft=True,
                    hideRight=True
                ),
                to="bgChildren",
            )
        else:
            session.upsert(
                MotionControllers(
                    stream=True, 
                    key="motionControllers",
                    left=True,
                    right=True,
                ),
                to="bgChildren",
            )

        while True:
            session.upsert(
                WebRTCVideoPlane(
                    src=self.webrtc_url,
                    iceServer=None,
                    iceServers=[],
                    key="video-quad",
                    aspect=self.aspect_ratio,
                    height = 7,
                ),
                to="bgChildren",
            )
            await asyncio.sleep(1.0 / self.display_fps)

    ## ego MODE
    async def main_image_binocular_zmq_ego(self, session):
        if self.use_hand_tracking:
            session.upsert(
                Hands(
                    stream=True,
                    key="hands",
                    hideLeft=True,
                    hideRight=True
                ),
                to="bgChildren",
            )
        else:
            session.upsert(
                MotionControllers(
                    stream=True,
                    key="motionControllers",
                    left=True,
                    right=True,
                ),
                to="bgChildren",
            )
        while True:
            session.upsert(
                [
                    ImageBackground(
                        self.img2display[:, :self.img_width],
                        aspect=self.aspect_ratio,
                        height=0.75,
                        distanceToCamera=2,
                        # The underlying rendering engine supported a layer binary bitmask for both objects and the camera. 
                        # Below we set the two image planes, left and right, to layers=1 and layers=2. 
                        # Note that these two masks are associated with left eye’s camera and the right eye’s camera.
                        layers=1,
                        format="jpeg",
                        quality=80,
                        key="background-left",
                        interpolate=True,
                    ),
                    ImageBackground(
                        self.img2display[:, self.img_width:],
                        aspect=self.aspect_ratio,
                        height=0.75,
                        distanceToCamera=2,
                        layers=2,
                        format="jpeg",
                        quality=80,
                        key="background-right",
                        interpolate=True,
                    ),
                ],
                to="bgChildren",
            )
            # 'jpeg' encoding should give you about 30fps with a 16ms wait in-between.
            await asyncio.sleep(1.0 / self.display_fps)

    async def main_image_monocular_zmq_ego(self, session):
        if self.use_hand_tracking:
            session.upsert(
                Hands(
                    stream=True,
                    key="hands",
                    hideLeft=True,
                    hideRight=True
                ),
                to="bgChildren",
            )
        else:
            session.upsert(
                MotionControllers(
                    stream=True, 
                    key="motionControllers",
                    left=True,
                    right=True,
                ),
                to="bgChildren",
            )

        while True:
            session.upsert(
                [
                    ImageBackground(
                        self.img2display,
                        aspect=self.aspect_ratio,
                        height=0.75,
                        distanceToCamera=2,
                        format="jpeg",
                        quality=80,
                        key="background-mono",
                        interpolate=True,
                    ),
                ],
                to="bgChildren",
            )
            await asyncio.sleep(1.0 / self.display_fps)

    async def main_image_binocular_webrtc_ego(self, session):
        if self.use_hand_tracking:
            session.upsert(
                Hands(
                    stream=True,
                    key="hands",
                    hideLeft=True,
                    hideRight=True
                ),
                to="bgChildren",
            )
        else:
            session.upsert(
                MotionControllers(
                    stream=True, 
                    key="motionControllers",
                    left=True,
                    right=True,
                ),
                to="bgChildren",
            )

        while True:
            session.upsert(
                WebRTCStereoVideoPlane(
                    src=self.webrtc_url,
                    iceServer=None,
                    iceServers=[], 
                    key="video-quad",
                    aspect=self.aspect_ratio,
                    height=3,
                    layout="stereo-left-right"
                ),
                to="bgChildren",
            )
            await asyncio.sleep(1.0 / self.display_fps)

    async def main_image_monocular_webrtc_ego(self, session):
        if self.use_hand_tracking:
            session.upsert(
                Hands(
                    stream=True,
                    key="hands",
                    hideLeft=True,
                    hideRight=True
                ),
                to="bgChildren",
            )
        else:
            session.upsert(
                MotionControllers(
                    stream=True, 
                    key="motionControllers",
                    left=True,
                    right=True,
                ),
                to="bgChildren",
            )

        while True:
            session.upsert(
                WebRTCVideoPlane(
                    src=self.webrtc_url,
                    iceServer=None,
                    iceServers=[],
                    key="video-quad",
                    aspect=self.aspect_ratio,
                    height=3,
                ),
                to="bgChildren",
            )
            await asyncio.sleep(1.0 / self.display_fps)

    ## pass-through MODE
    async def main_pass_through(self, session):
        if self.use_hand_tracking:
            session.upsert(
                Hands(
                    stream=True,
                    key="hands",
                    hideLeft=True,
                    hideRight=True
                ),
                to="bgChildren",
            )
        else:
            session.upsert(
                MotionControllers(
                    stream=True, 
                    key="motionControllers",
                    left=True,
                    right=True,
                ),
                to="bgChildren",
            )

        while True:
            await asyncio.sleep(1.0 / self.display_fps)

    # ==================== common data ====================
    @property
    def head_pose(self):
        """np.ndarray, shape (4, 4), head SE(3) pose matrix from Vuer (basis OpenXR Convention)."""
        with self.head_pose_shared.get_lock():
            return np.array(self.head_pose_shared[:]).reshape(4, 4, order="F")

    @property
    def left_arm_pose(self):
        """np.ndarray, shape (4, 4), left arm SE(3) pose matrix from Vuer (basis OpenXR Convention)."""
        with self.left_arm_pose_shared.get_lock():
            return np.array(self.left_arm_pose_shared[:]).reshape(4, 4, order="F")

    @property
    def right_arm_pose(self):
        """np.ndarray, shape (4, 4), right arm SE(3) pose matrix from Vuer (basis OpenXR Convention)."""
        with self.right_arm_pose_shared.get_lock():
            return np.array(self.right_arm_pose_shared[:]).reshape(4, 4, order="F")

    # ==================== Hand Tracking Data ====================
    @property
    def left_hand_positions(self):
        """np.ndarray, shape (25, 3), left hand 25 landmarks' 3D positions."""
        with self.left_hand_position_shared.get_lock():
            return np.array(self.left_hand_position_shared[:]).reshape(25, 3)

    @property
    def right_hand_positions(self):
        """np.ndarray, shape (25, 3), right hand 25 landmarks' 3D positions."""
        with self.right_hand_position_shared.get_lock():
            return np.array(self.right_hand_position_shared[:]).reshape(25, 3)

    @property
    def left_hand_orientations(self):
        """np.ndarray, shape (25, 3, 3), left hand 25 landmarks' orientations (flattened 3x3 matrices, column-major)."""
        with self.left_hand_orientation_shared.get_lock():
            return np.array(self.left_hand_orientation_shared[:]).reshape(25, 9).reshape(25, 3, 3, order="F")

    @property
    def right_hand_orientations(self):
        """np.ndarray, shape (25, 3, 3), right hand 25 landmarks' orientations (flattened 3x3 matrices, column-major)."""
        with self.right_hand_orientation_shared.get_lock():
            return np.array(self.right_hand_orientation_shared[:]).reshape(25, 9).reshape(25, 3, 3, order="F")

    @property
    def left_hand_pinch(self):
        """bool, whether left hand is pinching."""
        with self.left_hand_pinch_shared.get_lock():
            return self.left_hand_pinch_shared.value

    @property
    def left_hand_pinchValue(self):
        """float, pinch strength of left hand."""
        with self.left_hand_pinchValue_shared.get_lock():
            return self.left_hand_pinchValue_shared.value

    @property
    def left_hand_squeeze(self):
        """bool, whether left hand is squeezing."""
        with self.left_hand_squeeze_shared.get_lock():
            return self.left_hand_squeeze_shared.value

    @property
    def left_hand_squeezeValue(self):
        """float, squeeze strength of left hand."""
        with self.left_hand_squeezeValue_shared.get_lock():
            return self.left_hand_squeezeValue_shared.value

    @property
    def right_hand_pinch(self):
        """bool, whether right hand is pinching."""
        with self.right_hand_pinch_shared.get_lock():
            return self.right_hand_pinch_shared.value

    @property
    def right_hand_pinchValue(self):
        """float, pinch strength of right hand."""
        with self.right_hand_pinchValue_shared.get_lock():
            return self.right_hand_pinchValue_shared.value

    @property
    def right_hand_squeeze(self):
        """bool, whether right hand is squeezing."""
        with self.right_hand_squeeze_shared.get_lock():
            return self.right_hand_squeeze_shared.value

    @property
    def right_hand_squeezeValue(self):
        """float, squeeze strength of right hand."""
        with self.right_hand_squeezeValue_shared.get_lock():
            return self.right_hand_squeezeValue_shared.value

    # ==================== Controller Data ====================
    @property
    def left_ctrl_trigger(self):
        """bool, left controller trigger pressed or not."""
        with self.left_ctrl_trigger_shared.get_lock():
            return self.left_ctrl_trigger_shared.value

    @property
    def left_ctrl_triggerValue(self):
        """float, left controller trigger analog value (0.0 ~ 1.0)."""
        with self.left_ctrl_triggerValue_shared.get_lock():
            return self.left_ctrl_triggerValue_shared.value

    @property
    def left_ctrl_squeeze(self):
        """bool, left controller squeeze pressed or not."""
        with self.left_ctrl_squeeze_shared.get_lock():
            return self.left_ctrl_squeeze_shared.value

    @property
    def left_ctrl_squeezeValue(self):
        """float, left controller squeeze analog value (0.0 ~ 1.0)."""
        with self.left_ctrl_squeezeValue_shared.get_lock():
            return self.left_ctrl_squeezeValue_shared.value

    @property
    def left_ctrl_thumbstick(self):
        """bool, whether left thumbstick is touched or clicked."""
        with self.left_ctrl_thumbstick_shared.get_lock():
            return self.left_ctrl_thumbstick_shared.value

    @property
    def left_ctrl_thumbstickValue(self):
        """np.ndarray, shape (2,), left thumbstick 2D axis values (x, y)."""
        with self.left_ctrl_thumbstickValue_shared.get_lock():
            return np.array(self.left_ctrl_thumbstickValue_shared[:])

    @property
    def left_ctrl_aButton(self):
        """bool, left controller 'A' button pressed."""
        with self.left_ctrl_aButton_shared.get_lock():
            return self.left_ctrl_aButton_shared.value

    @property
    def left_ctrl_bButton(self):
        """bool, left controller 'B' button pressed."""
        with self.left_ctrl_bButton_shared.get_lock():
            return self.left_ctrl_bButton_shared.value

    @property
    def right_ctrl_trigger(self):
        """bool, right controller trigger pressed or not."""
        with self.right_ctrl_trigger_shared.get_lock():
            return self.right_ctrl_trigger_shared.value

    @property
    def right_ctrl_triggerValue(self):
        """float, right controller trigger analog value (0.0 ~ 1.0)."""
        with self.right_ctrl_triggerValue_shared.get_lock():
            return self.right_ctrl_triggerValue_shared.value

    @property
    def right_ctrl_squeeze(self):
        """bool, right controller squeeze pressed or not."""
        with self.right_ctrl_squeeze_shared.get_lock():
            return self.right_ctrl_squeeze_shared.value

    @property
    def right_ctrl_squeezeValue(self):
        """float, right controller squeeze analog value (0.0 ~ 1.0)."""
        with self.right_ctrl_squeezeValue_shared.get_lock():
            return self.right_ctrl_squeezeValue_shared.value

    @property
    def right_ctrl_thumbstick(self):
        """bool, whether right thumbstick is touched or clicked."""
        with self.right_ctrl_thumbstick_shared.get_lock():
            return self.right_ctrl_thumbstick_shared.value

    @property
    def right_ctrl_thumbstickValue(self):
        """np.ndarray, shape (2,), right thumbstick 2D axis values (x, y)."""
        with self.right_ctrl_thumbstickValue_shared.get_lock():
            return np.array(self.right_ctrl_thumbstickValue_shared[:])

    @property
    def right_ctrl_aButton(self):
        """bool, right controller 'A' button pressed."""
        with self.right_ctrl_aButton_shared.get_lock():
            return self.right_ctrl_aButton_shared.value

    @property
    def right_ctrl_bButton(self):
        """bool, right controller 'B' button pressed."""
        with self.right_ctrl_bButton_shared.get_lock():
            return self.right_ctrl_bButton_shared.value

    @property
    def motion_data_ready(self):
        """bool, whether at least one hand or controller motion data event has been received."""
        with self.motion_data_ready_shared.get_lock():
            return self.motion_data_ready_shared.value
