# /// script
# requires-python = ">=3.10"
# dependencies = [
#   "numpy",
#   "opencv-python",
#   "sounddevice",
#   "click",
#   "pupil-labs-realtime-api",
#   "glfw",
#   "moderngl",
# ]
# ///

import threading
import time
import sys
from typing import Optional, Tuple

import click
import cv2
import numpy as np
import sounddevice as sd
from pupil_labs.realtime_api.simple import Device, discover_one_device, SimpleVideoFrame
from pupil_labs.realtime_api.streaming.eye_events import BlinkEventData
from pupil_labs.realtime_api.streaming.eye_events import FixationEventData, FixationOnsetEventData

import glfw
import moderngl


def rgb_to_params(r: int, g: int, b: int) -> Tuple[float, float, str]:
    """
    Map ONLY the Red (R) channel to frequency; keep amplitude constant; waveform still selected by B.
    - Frequency from Red (R): 0–255 → 220–880 Hz (log scale)
    - Amplitude: constant 0.18 (independent of RGB)
    - Waveform from Blue (B): 0–85 sine, 86–170 square, 171–255 sawtooth
    """
    f_min, f_max = 220.0, 880.0
    r_norm = r / 255.0
    freq = f_min * ((f_max / f_min) ** r_norm)
    amp = 0.18  # fixed amplitude
    if b <= 85:
        waveform = "sine"
    elif b <= 170:
        waveform = "square"
    else:
        waveform = "sawtooth"
    return freq, amp, waveform


# ----------------------- Audio Synth (continuous stream) -----------------------

class ContinuousSine:
    """
    Low-latency continuous waveform generator whose frequency, amplitude,
    and waveform shape can be updated from another thread (our vision/gaze thread).
    """
    def __init__(self, samplerate: int = 44100):
        self.sr = samplerate
        self._freq = 440.0
        self._amp = 0.12
        self._waveform = "sine"
        self._phase = 0.0
        self._lock = threading.Lock()
        self._stream: Optional[sd.OutputStream] = None
        # one-shot drums
        self._oneshots = []  # list of dicts: {"buf": np.ndarray, "idx": int}

    def start(self):
        self._stream = sd.OutputStream(
            samplerate=self.sr,
            channels=1,
            dtype="float32",
            callback=self._callback,
            blocksize=0,  # let PortAudio choose for low latency
        )
        self._stream.start()

    def stop(self):
        if self._stream is not None:
            self._stream.stop()
            self._stream.close()
            self._stream = None

    def set_params(self, freq: float, amp: float, waveform: str):
        with self._lock:
            self._freq = float(max(1.0, freq))
            self._amp = float(np.clip(amp, 0.0, 1.0))
            if waveform in ("sine", "square", "sawtooth"):
                self._waveform = waveform
            else:
                self._waveform = "sine"

    def trigger_drum(self, kind: str = "kick"):
        """Schedule a short one-shot drum buffer to be mixed on the next callbacks."""
        sr = self.sr
        if kind == "kick":
            dur = 0.12
            n = int(sr * dur)
            t = np.linspace(0, dur, n, endpoint=False).astype(np.float32)
            # pitch sweep 150->60 Hz, exp amp decay
            f0, f1 = 150.0, 60.0
            f = f0 * (f1 / f0) ** (t / dur)
            phase = 2 * np.pi * np.cumsum(f) / sr
            env = np.exp(-t * 35.0)
            buf = 1.0 * env * np.sin(phase)
        elif kind == "snare":
            dur = 0.15
            n = int(sr * dur)
            t = np.linspace(0, dur, n, endpoint=False).astype(np.float32)
            noise = np.random.uniform(-1.0, 1.0, size=n).astype(np.float32)
            env = np.exp(-t * 25.0)
            buf = 0.4 * env * noise
        else:
            return
        with self._lock:
            self._oneshots.append({"buf": buf.astype(np.float32), "idx": 0})

    def _callback(self, outdata, frames, time_info, status):
        if status:
            # Print once per callback if glitches occur
            print("Audio status:", status)
        # Atomically read current params
        with self._lock:
            f = self._freq
            a = self._amp
            w = self._waveform
        t = (np.arange(frames, dtype=np.float32) + self._phase) / self.sr
        if w == "sine":
            buf = a * np.sin(2.0 * np.pi * f * t)
        elif w == "square":
            buf = a * np.sign(np.sin(2.0 * np.pi * f * t))
        elif w == "sawtooth":
            # Sawtooth from -1 to 1
            buf = a * (2.0 * (t * f - np.floor(0.5 + t * f)))
        else:
            buf = a * np.sin(2.0 * np.pi * f * t)
        self._phase = (self._phase + frames) % self.sr
        # base tone in "buf"
        mix = buf.copy()
        # mix any scheduled one-shots
        with self._lock:
            to_remove = []
            for i, ev in enumerate(self._oneshots):
                start = ev["idx"]
                remaining = ev["buf"].shape[0] - start
                take = min(frames, remaining)
                if take > 0:
                    mix[:take] += ev["buf"][start:start+take]
                    ev["idx"] += take
                if ev["idx"] >= ev["buf"].shape[0]:
                    to_remove.append(i)
            for i in reversed(to_remove):
                self._oneshots.pop(i)
        # write out
        outdata[:, 0] = mix


# ----------------------- Helpers: mapping + gaze extraction -----------------------

def avg_bgr_patch(img: np.ndarray, x: int, y: int, half: int = 2) -> Tuple[int, int, int]:
    """Average BGR in a small (2*half+1)^2 patch around (x, y)."""
    h, w = img.shape[:2]
    x0, x1 = max(0, x - half), min(w, x + half + 1)
    y0, y1 = max(0, y - half), min(h, y + half + 1)
    roi = img[y0:y1, x0:x1]
    b, g, r = roi.mean(axis=(0, 1)).astype(np.uint8)
    return int(b), int(g), int(r)

def hsv_from_bgr(b: int, g: int, r: int) -> Tuple[float, float, float]:
    """Return HSV as floats (H in [0,360), S,V in [0,1])."""
    bgr = np.array([[[b, g, r]]], dtype=np.uint8)
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)[0, 0]
    H = float(hsv[0]) * (360.0 / 179.0)  # OpenCV H range is 0..179
    S = float(hsv[1]) / 255.0
    V = float(hsv[2]) / 255.0
    return H, S, V

class EMA:
    """Simple exponential moving average for 1D or vector values."""
    def __init__(self, alpha: float, init: Optional[np.ndarray] = None):
        self.alpha = float(alpha)
        self.state = None if init is None else np.array(init, dtype=np.float32)

    def __call__(self, x: np.ndarray) -> np.ndarray:
        x = np.array(x, dtype=np.float32)
        if self.state is None:
            self.state = x
        else:
            self.state = self.alpha * x + (1.0 - self.alpha) * self.state
        return self.state

# def hue_to_frequency(
#     hue_deg: float,
#     f_min: float = 220.0,
#     f_max: float = 880.0,
#     n_octaves: Optional[float] = None,
# ) -> float:
#     """
#     Map hue (0..360) to frequency on a logarithmic (musical) scale.
#     Similar hues -> similar freqs (continuous).
#     - Option A: direct min..max mapping across some octaves.
#     - Option B: specify n_octaves to control exponential span.
#     """
#     h01 = np.clip(hue_deg / 360.0, 0.0, 1.0)
#     if n_octaves is not None:
#         return f_min * (2.0 ** (h01 * n_octaves))
#     # Default: interpolate exponentially between f_min and f_max
#     return f_min * ((f_max / f_min) ** h01)

# def brightness_to_amp(v01: float, min_amp: float = 0.05, max_amp: float = 0.25) -> float:
#     """Map Value (0..1) to amplitude."""
#     return float(np.interp(np.clip(v01, 0.0, 1.0), [0.0, 1.0], [min_amp, max_amp]))

def extract_norm_gaze_xy(gaze_sample) -> Optional[Tuple[float, float, float]]:
    """
    Try multiple common field names across RT API variants to get normalized gaze (x,y) and confidence.
    Returns (x01, y01, conf) or None if not available.
    """
    # Best-effort extraction (add/adjust to your exact dataclass)
    candidates = []
    if hasattr(gaze_sample, "gaze_point"):
        gp = getattr(gaze_sample, "gaze_point")
        if hasattr(gp, "x") and hasattr(gp, "y"):
            candidates.append((float(gp.x), float(gp.y)))
        if hasattr(gp, "on_display_normalized"):
            nd = gp.on_display_normalized
            if hasattr(nd, "x") and hasattr(nd, "y"):
                candidates.append((float(nd.x), float(nd.y)))
    if hasattr(gaze_sample, "point_on_display_normalized"):
        nd = getattr(gaze_sample, "point_on_display_normalized")
        if hasattr(nd, "x") and hasattr(nd, "y"):
            candidates.append((float(nd.x), float(nd.y)))
    if hasattr(gaze_sample, "x") and hasattr(gaze_sample, "y"):
        candidates.append((float(gaze_sample.x), float(gaze_sample.y)))

    conf = None
    if hasattr(gaze_sample, "confidence"):
        conf = float(getattr(gaze_sample, "confidence"))
    elif hasattr(gaze_sample, "gaze_confidence"):
        conf = float(getattr(gaze_sample, "gaze_confidence"))

    if not candidates:
        return None
    x, y = candidates[0]
    # Some APIs use top-left origin with y down. That’s what we want for image pixel mapping.
    return float(x), float(y), (conf if conf is not None else 1.0)



# ----------------------- Visuals (GL tunnel) -----------------------

VERT_SRC = """
#version 330
in vec2 in_pos;
void main(){
    gl_Position = vec4(in_pos, 0.0, 1.0);
}
"""

FRAG_SRC = """
#version 330
out vec4 fragColor;
uniform float u_time;      // seconds
uniform float u_speed;     // 0..~2: flow speed
uniform float u_distort;   // 0..~2: wobble/warp strength
uniform float u_color;     // 0..~2: hue/contrast shift
uniform vec2  u_res;       // viewport

vec3 pal(float t, vec3 a, vec3 b, vec3 c, vec3 d){
    return a + b * cos(6.28318 * (c * t + d));
}

void main(){
    vec2 uv = (gl_FragCoord.xy / u_res.xy) * 2.0 - 1.0;
    uv.x *= u_res.x / u_res.y;
    float r = length(uv);
    float a = atan(uv.y, uv.x);
    float t = u_time * mix(0.2, 3.0, clamp(u_speed, 0.0, 2.0));
    float z = 3.0 / (r + 0.12);
    float band    = sin(z + t * 2.1 + sin(a * 6.0 + t)) * 0.5 + 0.5;
    float rings   = smoothstep(0.25, 0.75, sin(z * 0.75 + t * 1.75));
    float stripes = smoothstep(0.2, 0.8,  sin(a * 10.0 + t * 2.0));
    float wobble = sin(a * 4.0 - t * 3.0) * 0.15 * (0.5 + u_distort);
    float warp   = sin((r + wobble) * 8.0 - t * 2.0);
    float m = mix(band, rings, 0.5) * 0.7 + stripes * 0.3;
    m = mix(m, warp * 0.5 + 0.5, 0.45 + 0.35 * clamp(u_distort, 0.0, 2.0));
    float hueShift = u_color * 0.2;
    vec3 col = pal(
        m + z * 0.02 + a * (0.08 + 0.12 * u_color),
        vec3(0.52, 0.35, 0.40) + vec3(0.05) * u_color,
        vec3(0.45, 0.55, 0.60),
        vec3(1.00, 0.80, 0.60),
        vec3(0.15 + hueShift, 0.35, 0.65)
    );
    float vig = smoothstep(1.25, 0.2, r);
    col *= vig;
    col = pow(col, vec3(0.9 + 0.3 * u_color));
    fragColor = vec4(col, 1.0);
}
"""

class VisualTunnelEmbedded:
    """Non-threaded GLFW+moderngl renderer meant to be driven from the main thread (needed on macOS)."""
    def __init__(self, title: str = "Psychedelic Shader Tunnel — 3 Signals", fullscreen: bool = False):
        self.title = title
        self.fullscreen = fullscreen
        self._inited = False
        self._ctx = None
        self._vao = None
        self._uniforms = {}
        self._window = None

    def init(self) -> bool:
        # Error callback
        def _glfw_err(err, desc):
            try:
                msg = desc.decode("utf-8", "ignore") if isinstance(desc, (bytes, bytearray)) else str(desc)
            except Exception:
                msg = str(desc)
            print(f"[GLFW] error {err}: {msg}")
        glfw.set_error_callback(_glfw_err)

        if not glfw.init():
            print("[visuals] Failed to init GLFW (embedded)")
            return False
        glfw.window_hint(glfw.CONTEXT_VERSION_MAJOR, 3)
        glfw.window_hint(glfw.CONTEXT_VERSION_MINOR, 3)
        glfw.window_hint(glfw.OPENGL_PROFILE, glfw.OPENGL_CORE_PROFILE)
        glfw.window_hint(glfw.DOUBLEBUFFER, glfw.TRUE)
        glfw.window_hint(glfw.OPENGL_FORWARD_COMPAT, glfw.TRUE)

        monitor = glfw.get_primary_monitor() if self.fullscreen else None
        width, height = (1280, 720)
        if self.fullscreen and monitor:
            mode = glfw.get_video_mode(monitor)
            width, height = mode.size.width, mode.size.height
        self._window = glfw.create_window(width, height, self.title, monitor, None)
        if not self._window:
            print("[visuals] Failed to create window (embedded)")
            glfw.terminate()
            return False
        glfw.show_window(self._window)
        glfw.make_context_current(self._window)
        glfw.swap_interval(0)
        try:
            self._ctx = moderngl.create_context()
        except Exception as e:
            print(f"[visuals] Failed to create moderngl context (embedded): {e}")
            glfw.terminate()
            return False
        vertices = np.array([-1.0, -1.0, 3.0, -1.0, -1.0, 3.0], dtype="f4")
        vbo = self._ctx.buffer(vertices.tobytes())
        prog = self._ctx.program(vertex_shader=VERT_SRC, fragment_shader=FRAG_SRC)
        self._vao = self._ctx.simple_vertex_array(prog, vbo, "in_pos")
        self._uniforms = {
            "u_time": prog["u_time"],
            "u_speed": prog["u_speed"],
            "u_distort": prog["u_distort"],
            "u_color": prog["u_color"],
            "u_res": prog["u_res"],
        }
        self._inited = True
        return True

    def render(self, t: float, speed: float, distort: float, color: float):
        if not self._inited or not self._window:
            return False
        if glfw.window_should_close(self._window):
            return False
        w, h = glfw.get_framebuffer_size(self._window)
        self._ctx.viewport = (0, 0, int(w), int(h))
        self._uniforms["u_time"].value = float(t)
        self._uniforms["u_speed"].value = float(np.clip(speed, 0.0, 2.0))
        self._uniforms["u_distort"].value = float(np.clip(distort, 0.0, 2.0))
        self._uniforms["u_color"].value = float(np.clip(color, 0.0, 2.0))
        self._uniforms["u_res"].value = (float(w), float(h))
        self._ctx.clear(0.0, 0.0, 0.0, 1.0)
        self._vao.render()
        glfw.swap_buffers(self._window)
        glfw.poll_events()
        return True

    def close(self):
        try:
            if self._window is not None:
                glfw.destroy_window(self._window)
        except Exception:
            pass
        glfw.terminate()

class VisualTunnel(threading.Thread):
    def __init__(self, title: str = "Psychedelic Shader Tunnel — 3 Signals", fullscreen: bool = False, target_fps: int = 120):
        super().__init__(daemon=True)
        self.title = title
        self.fullscreen = fullscreen
        self.target_fps = float(target_fps)
        self._lock = threading.Lock()
        self._speed = 0.0
        self._distort = 0.0
        self._color = 0.0
        self._running = threading.Event()
        self._running.set()

    def update(self, speed: float, distort: float, color: float):
        with self._lock:
            self._speed = float(np.clip(speed, 0.0, 2.0))
            self._distort = float(np.clip(distort, 0.0, 2.0))
            self._color = float(np.clip(color, 0.0, 2.0))

    def stop(self):
        self._running.clear()

    def run(self):
        # Log GLFW errors to console so failures are visible
        def _glfw_err(err, desc):
            try:
                msg = desc.decode("utf-8", "ignore") if isinstance(desc, (bytes, bytearray)) else str(desc)
            except Exception:
                msg = str(desc)
            print(f"[GLFW] error {err}: {msg}")
        glfw.set_error_callback(_glfw_err)

        if not glfw.init():
            print("[visuals] Failed to init GLFW")
            return
        glfw.window_hint(glfw.CONTEXT_VERSION_MAJOR, 3)
        glfw.window_hint(glfw.CONTEXT_VERSION_MINOR, 3)
        glfw.window_hint(glfw.OPENGL_PROFILE, glfw.OPENGL_CORE_PROFILE)
        glfw.window_hint(glfw.DOUBLEBUFFER, glfw.TRUE)
        # macOS often requires forward-compatible core profile
        glfw.window_hint(glfw.OPENGL_FORWARD_COMPAT, glfw.TRUE)

        monitor = glfw.get_primary_monitor() if self.fullscreen else None
        width, height = (1280, 720)
        if self.fullscreen and monitor:
            mode = glfw.get_video_mode(monitor)
            width, height = mode.size.width, mode.size.height
        window = glfw.create_window(width, height, self.title, monitor, None)
        if not window:
            glfw.terminate()
            print("[visuals] Failed to create window")
            return
        glfw.show_window(window)
        glfw.make_context_current(window)
        glfw.swap_interval(0)

        try:
            ctx = moderngl.create_context()
        except Exception as e:
            print(f"[visuals] Failed to create moderngl context: {e}")
            glfw.terminate()
            return
        try:
            info = ctx.info
            print(f"[visuals] GL renderer: {info.get('GL_RENDERER')}  version: {info.get('GL_VERSION')}  vendor: {info.get('GL_VENDOR')}")
        except Exception:
            pass
        vertices = np.array([-1.0, -1.0, 3.0, -1.0, -1.0, 3.0], dtype="f4")
        vbo = ctx.buffer(vertices.tobytes())
        prog = ctx.program(vertex_shader=VERT_SRC, fragment_shader=FRAG_SRC)
        vao = ctx.simple_vertex_array(prog, vbo, "in_pos")
        u_time = prog["u_time"]
        u_speed = prog["u_speed"]
        u_distort = prog["u_distort"]
        u_color = prog["u_color"]
        u_res = prog["u_res"]

        start = time.perf_counter()
        frame_cap = 1.0 / self.target_fps if self.target_fps > 0 else 0.0
        last = start

        while self._running.is_set() and not glfw.window_should_close(window):
            now = time.perf_counter()
            dt = now - last
            if frame_cap and dt < frame_cap:
                time.sleep(frame_cap - dt)
                now = time.perf_counter()
            last = now

            w, h = glfw.get_framebuffer_size(window)
            ctx.viewport = (0, 0, int(w), int(h))

            with self._lock:
                spd = self._speed
                dst = self._distort
                col = self._color

            u_time.value = now - start
            u_speed.value = spd
            u_distort.value = dst
            u_color.value = col
            u_res.value = (float(w), float(h))

            ctx.clear(0.0, 0.0, 0.0, 1.0)
            vao.render()
            glfw.swap_buffers(window)
            glfw.poll_events()

        glfw.terminate()

# ----------------------- Main loop -----------------------

@click.command()
@click.option("--ip", default='192.168.20.114', help="IP address of Neon Companion Device.")
@click.option("--port", default=8080, show_default=True, help="Neon Companion port.")
@click.option("--preview", is_flag=True, help="Show scene preview with gaze dot.")
@click.option("--debug-eye-events", is_flag=True, help="Print eye events (blink/fixation/saccade) to console")
@click.option("--visuals", is_flag=True, help="Show shader tunnel visuals driven by gaze & color")
@click.option("--fullscreen", is_flag=True, help="Open visuals in fullscreen")
@click.option("--visuals-embedded", is_flag=True, help="Create the shader window on the main thread (macOS fix)")
def main(ip: Optional[str], port: int, preview: bool, debug_eye_events: bool, visuals: bool, fullscreen: bool, visuals_embedded: bool):
    # Connect device
    if ip:
        device = Device(address=ip, port=port)
    else:
        device = discover_one_device(max_search_duration_seconds=10)
    if device is None:
        print("No Neon device found.")
        raise SystemExit(1)

    print(f"Connected: {device.address}:{port} (phone_id={device.phone_id})")
    if debug_eye_events:
        print("[eye-events] Debug printing enabled. Make sure 'Compute fixations' is ON in Companion settings.")

    # Audio engine
    synth = ContinuousSine(samplerate=44100)
    synth.start()

    # Smoothing for RGB & amplitude (less jitter)
    rgb_ema = EMA(alpha=0.25, init=np.array([128.0, 128.0, 128.0], dtype=np.float32))
    amp_ema = EMA(alpha=0.2, init=np.array([0.12], dtype=np.float32))

    last_update = 0.0
    throttle_hz = 60.0  # don’t over-update audio more than ~60x/sec
    min_conf = 0.4      # ignore very low-confidence gaze

    vis = None
    vis_emb = None
    if visuals:
        print("[visuals] Starting shader tunnel window...")
        if visuals_embedded or sys.platform == 'darwin':
            vis_emb = VisualTunnelEmbedded(fullscreen=fullscreen)
            if not vis_emb.init():
                print("[visuals] Failed to initialize embedded visuals.")
        else:
            vis = VisualTunnel(fullscreen=fullscreen)
            vis.start()

    gaze_prev = None
    gaze_speed_ema = EMA(alpha=0.2, init=np.array([0.0], dtype=np.float32))

    if preview:
        cv2.namedWindow("Scene", cv2.WINDOW_NORMAL)

    # For embedded visuals, need to drive clock
    start_time = time.perf_counter()

    try:
        while True:
            # Get latest gaze + scene frame (non-blocking, small timeout)
            gaze = device.receive_gaze_datum(timeout_seconds=0.1)
            scene: Optional[SimpleVideoFrame] = device.receive_scene_video_frame(timeout_seconds=0.1)

            if gaze is None or scene is None:
                # No fresh data; tiny sleep to avoid tight spin
                time.sleep(0.005)
                continue

            # Drain eye events queue (blinks/fixations/saccades) and optionally print
            while True:
                try:
                    ev = device.receive_eye_events(timeout_seconds=0.0)
                except Exception:
                    break  # sensor not available or not started
                if ev is None:
                    break  # no more events pending
                # Print event info if requested
                if debug_eye_events:
                    try:
                        name = type(ev).__name__
                        # Prefer datetime if available
                        ts = getattr(ev, "rtp_ts_unix_seconds", None)
                        if ts is None:
                            ts = getattr(ev, "timestamp_unix_ns", None)
                        print(f"[eye-events] {name}: {ev} @ {ts}")
                    except Exception:
                        print(f"[eye-events] {type(ev).__name__}")
                # Trigger drums on blink
                if isinstance(ev, BlinkEventData):
                    synth.trigger_drum("kick")

            # Extract normalized gaze (0..1)
            g = extract_norm_gaze_xy(gaze)
            if g is None:
                continue
            gx, gy, conf = g
            if conf < min_conf:
                # Optionally fade amplitude on low confidence
                synth.set_params(synth._freq, 0.0, synth._waveform)
                continue

            # Map normalized coords to pixel coords
            img = scene.bgr_pixels  # HxWx3, uint8
            h, w = img.shape[:2]
            x = int(np.clip(gx, 0.0, 1.0) * (w - 1))
            y = int(np.clip(gy, 0.0, 1.0) * (h - 1))

            # Sample a small patch and compute average BGR
            b, g_, r = avg_bgr_patch(img, x, y, half=2)
            # Smooth RGB values
            Rs, Gs, Bs = rgb_ema(np.array([r, g_, b], dtype=np.float32))

            # Map RGB to freq, amp, waveform
            freq, amp, waveform = rgb_to_params(int(Rs), int(Gs), int(Bs))
            amp = float(amp_ema(np.array([amp], dtype=np.float32))[0])

            # Throttle parameter pushes to audio engine
            now = time.time()
            if now - last_update >= (1.0 / throttle_hz):
                synth.set_params(freq, amp, waveform)
                last_update = now

            H, S, V = hsv_from_bgr(int(Bs), int(Gs), int(Rs))
            # Estimate normalized gaze speed in display space
            if gaze_prev is not None:
                dx = gx - gaze_prev[0]
                dy = gy - gaze_prev[1]
                speed = float(np.hypot(dx, dy))  # in 0..~1 range
            else:
                speed = 0.0
            gaze_prev = (gx, gy)
            speed_smooth = float(gaze_speed_ema(np.array([speed], dtype=np.float32))[0])

            # Map: speed -> u_speed (0..2), distort -> from S (0..2), color -> from hue (0..2)
            u_speed_val = float(np.clip(speed_smooth * 10.0, 0.0, 2.0))
            u_distort_val = float(np.clip(S * 2.0, 0.0, 2.0))
            u_color_val = float(np.clip((H % 360.0) / 180.0, 0.0, 2.0))
            if visuals and vis is not None:
                vis.update(u_speed_val, u_distort_val, u_color_val)
            if vis_emb is not None:
                t_now = time.perf_counter() - start_time
                ok = vis_emb.render(t_now, u_speed_val, u_distort_val, u_color_val)
                if not ok:
                    # window closed; disable further rendering
                    vis_emb.close()
                    vis_emb = None

            # Optional preview
            if preview:
                vis_img = img.copy()
                cv2.circle(vis_img, (x, y), 8, (0, 255, 255), 2)
                cv2.putText(
                    vis_img,
                    f"R={int(Rs)} G={int(Gs)} B={int(Bs)}  f={freq:6.1f}Hz  amp={(amp):0.2f}  wave={waveform}  patch={2}  spd={speed_smooth:0.3f}  S={S:0.2f}  H={H:0.1f}",
                    (10, 24),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (255, 255, 255),
                    1,
                    cv2.LINE_AA,
                )
                cv2.putText(
                    vis_img,
                    "Blink = Kick (one-shot)",
                    (10, 48),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (255, 255, 0),
                    1,
                    cv2.LINE_AA,
                )
                if debug_eye_events:
                    cv2.putText(
                        vis_img,
                        "Console: printing eye events",
                        (10, 72),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.6,
                        (0, 255, 255),
                        1,
                        cv2.LINE_AA,
                    )
                cv2.imshow("Scene", vis_img)
                # ESC to quit
                if (cv2.waitKey(1) & 0xFF) == 27:
                    break

    except KeyboardInterrupt:
        pass
    finally:
        synth.stop()
        device.close()
        if preview:
            cv2.destroyAllWindows()
        try:
            if 'vis' in locals() and vis is not None:
                vis.stop()
        except Exception:
            pass
        try:
            if 'vis_emb' in locals() and vis_emb is not None:
                vis_emb.close()
        except Exception:
            pass
        print("Bye.")


if __name__ == "__main__":
    main()