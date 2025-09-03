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
from pupil_labs.realtime_api.streaming.gaze import EyestateEyelidGazeData

import glfw
import moderngl


# ---- Musical helpers: quantize R to a musical scale (C major) ----
SCALE_SEMITONES = [0, 2, 4, 5, 7, 9, 11]  # C major degrees
BASE_MIDI = 60  # C4

def midi_to_freq(midi_note: float) -> float:
    return 440.0 * (2.0 ** ((float(midi_note) - 69.0) / 12.0))

def quantize_red_to_scale(r: int, base_midi: int = BASE_MIDI, scale=SCALE_SEMITONES, octaves: int = 3) -> float:
    """Map 0..255 -> notes of a scale across several octaves, return frequency in Hz.
    Builds a note list: base + scale + next octaves, then indexes by normalized R.
    """
    r = int(np.clip(r, 0, 255))
    # Build note pool across `octaves` starting at BASE_MIDI (inclusive)
    pool = []
    for o in range(octaves):
        base = base_midi + 12 * o
        for st in scale:
            pool.append(base + st)
    if not pool:
        return 440.0
    idx = int(np.interp(r, [0, 255], [0, len(pool) - 1]))
    midi = pool[idx]
    return float(midi_to_freq(midi))

def rgb_to_params(r: int, g: int, b: int) -> Tuple[float, float, str]:
    """
    Map RGB to sound with a more musical mapping:
      - R → frequency quantized to C major scale across 3 octaves around C4
      - B → amplitude (volume) in [0.10 .. 0.80]
      - waveform → fixed to sine (filter will shape tone)
    """
    freq = quantize_red_to_scale(r, base_midi=BASE_MIDI, scale=SCALE_SEMITONES, octaves=3)
    amp = float(np.interp(np.clip(b, 0, 255), [0, 255], [0.10, 0.80]))
    waveform = "sine"
    return float(freq), float(amp), waveform


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
        self._cutoff = 1200.0  # Hz, low-pass cutoff (G channel)
        self._lp_state = 0.0   # filter memory for one-pole LPF
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

    def set_filter_cutoff(self, cutoff_hz: float):
        with self._lock:
            self._cutoff = float(np.clip(cutoff_hz, 50.0, 8000.0))

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
            buf = 1.8 * env * np.sin(phase)
        elif kind == "snare":
            dur = 0.15
            n = int(sr * dur)
            t = np.linspace(0, dur, n, endpoint=False).astype(np.float32)
            noise = np.random.uniform(-1.0, 1.0, size=n).astype(np.float32)
            env = np.exp(-t * 25.0)
            buf = 0.6 * env * noise
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
        # --- One-pole low-pass filter controlled by _cutoff ---
        with self._lock:
            cutoff = self._cutoff
        # Compute smoothing coefficient per block using bilinear-like RC
        # alpha = dt / (RC + dt), RC = 1/(2*pi*fc), dt = 1/sr
        dt = 1.0 / float(self.sr)
        RC = 1.0 / (2.0 * np.pi * max(10.0, cutoff))
        alpha = dt / (RC + dt)
        # clamp alpha to (0,1)
        alpha = float(np.clip(alpha, 0.0001, 0.9999))
        y = np.empty_like(mix)
        s = self._lp_state
        for i in range(mix.shape[0]):
            s = s + alpha * (mix[i] - s)
            y[i] = s
        self._lp_state = float(s)
        mix = y
        # Safety clip after filtering
        mix = np.clip(mix, -0.99, 0.99)
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


# Helper: extract average pupil diameter (mm) from gaze sample, if available
def extract_avg_pupil_mm(gaze_sample) -> Optional[float]:
    """Return average pupil diameter (mm) across left/right if available, else None."""
    try:
        if isinstance(gaze_sample, EyestateEyelidGazeData):
            l = getattr(gaze_sample, "pupil_diameter_left", None)
            r = getattr(gaze_sample, "pupil_diameter_right", None)
            vals = [v for v in (l, r) if v is not None]
            if vals:
                return float(np.mean(vals))
    except Exception:
        pass
    # Try generic attributes
    l = getattr(gaze_sample, "pupil_diameter_left", None)
    r = getattr(gaze_sample, "pupil_diameter_right", None)
    vals = [v for v in (l, r) if v is not None]
    if vals:
        return float(np.mean(vals))
    return None



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

uniform float u_time;         // seconds
uniform vec2  u_res;          // viewport
uniform float u_speed;        // 0..~2: animation speed
uniform float u_distort;      // 0..~2: wobble/warp intensity
uniform vec3  u_col;          // main tint (0..1), strongly influences colors
uniform float u_col_strength; // 0..1: strength of color mapping
uniform float u_pupil;        // 0..1: external pupil radius (now stronger)
uniform float u_pulse_age;    // seconds since last pulse (0 at trigger)

const float PI = 3.14159265359;

// ================= helpers =================
float hash21(vec2 p){
    p = fract(p*vec2(123.34, 345.45));
    p += dot(p, p+34.345);
    return fract(p.x*p.y);
}
vec2 rot(vec2 p, float a){
    float s = sin(a), c = cos(a);
    return mat2(c,-s,s,c)*p;
}
float vnoise(vec2 p){
    vec2 i = floor(p);
    vec2 f = fract(p);
    float a = hash21(i);
    float b = hash21(i + vec2(1.0, 0.0));
    float c = hash21(i + vec2(0.0, 1.0));
    float d = hash21(i + vec2(1.0, 1.0));
    vec2  u = f*f*(3.0-2.0*f);
    return mix(mix(a, b, u.x), mix(c, d, u.x), u.y);
}
float fbm(vec2 p){
    float v = 0.0;
    float a = 0.5;
    vec2  s = p;
    for(int i=0;i<5;i++){
        v += a * vnoise(s);
        s  = rot(s*2.02, 0.5);
        a *= 0.55;
    }
    return v;
}
vec3 pal(float t, vec3 A, vec3 B, vec3 C, vec3 D){
    return A + B*cos(6.28318*(C*t + D));
}
vec3 toward(vec3 base, vec3 target, float k){
    return mix(base, target, clamp(k, 0.0, 1.0));
}
// --- HSV helpers: rgb2hsv/hsv2rgb for hue retinting ---
vec3 rgb2hsv(vec3 c){
    vec4 K = vec4(0.0, -1.0/3.0, 2.0/3.0, -1.0);
    vec4 p = mix(vec4(c.bg, K.wz), vec4(c.gb, K.xy), step(c.b, c.g));
    vec4 q = mix(vec4(p.xyw, c.r), vec4(c.r, p.yzx), step(p.x, c.r));
    float d = q.x - min(q.w, q.y);
    float e = 1.0e-10;
    return vec3(abs(q.z + (q.w - q.y) / (6.0 * d + e)), d / (q.x + e), q.x);
}
vec3 hsv2rgb(vec3 c){
    vec3 p = abs(fract(c.xxx + vec3(0.0, 1.0/3.0, 2.0/3.0)) * 6.0 - 3.0);
    vec3 rgb = c.z * mix(vec3(1.0), clamp(p - 1.0, 0.0, 1.0), c.y);
    return rgb;
}
// ===========================================

void main(){
    vec2 res = u_res;
    vec2 p = (gl_FragCoord.xy / res) * 2.0 - 1.0;
    p.x *= res.x / res.y;

    float spd = clamp(u_speed,   0.0, 2.0);
    float dst = clamp(u_distort, 0.0, 2.0);
    float pup = clamp(u_pupil,   0.0, 1.0);
    float cs  = clamp(u_col_strength, 0.0, 1.0);

    // ---- Pulse envelope (kept from MAX Pulse edition) ----
    float age   = u_pulse_age;
    float env   = exp(-age * 1.4);
    float vring = 1.35;
    float sigma = 0.09 + 0.06*dst;
    float r0    = vring * age;
    float t = (u_time + 0.20 * env) * mix(0.15, 3.0, spd + 0.35*env);

    // === NEW: eye zoom responds to pupil size ===
    // Larger pupil -> stronger zoom (blow-up). Multiplies with pulse zoom.
    float zoomPulse = 1.0 - 0.08 * env;
    float zoomPupil = 1.0 - 0.12 * pup;     // up to ~12% zoom for pup=1
    float zoom = max(0.65, zoomPulse * zoomPupil);  // safety floor
    p *= zoom;

    // short kaleido burst on pulse (unchanged)
    float kenv = smoothstep(0.0, 0.20, 0.20 - age) * env;
    if(kenv > 0.0){
        float seg = mix(6.0, 16.0, kenv);
        float ang0 = atan(p.y, p.x);
        float wedge = 2.0 * PI / max(seg, 1.0);
        float aFold = abs(mod(ang0, wedge) - 0.5 * wedge);
        float r0p = length(p);
        p = vec2(cos(aFold), sin(aFold)) * r0p;
    }

    // swirl
    float swirl = (0.15*dst + 0.25*env) * sin(t*0.8 + 3.0*env);
    p = rot(p, swirl);

    // domain warp
    vec2 pw = p;
    float w1 = fbm(pw*2.5 + vec2(0.0, t*0.5));
    float w2 = fbm(rot(pw, 1.2)*3.0 - vec2(t*0.4, 0.0));
    pw += 0.12*dst * vec2(w1 - 0.5, w2 - 0.5);

    float r = length(pw);
    float a = atan(pw.y, pw.x);

    // radial shockwave displacement on pulse
    float ring   = exp(-0.5 * pow((r - r0)/sigma, 2.0));
    float shock  = ring * env;
    pw += (0.22 * shock) * normalize(pw + 1e-4)
        + (0.06 * shock) * vec2(sin(a*12.0 + t*6.0), cos(a*10.0 - t*5.0));

    r = length(pw);
    a = atan(pw.y, pw.x);

    // ---------- pupil: STRONGER RANGE + prominence ----------
    // Expanded range: minR smaller, maxR larger visually prominent.
    float minR = 0.06;
    float maxR = 0.30;               // was ~0.22
    float pupilTarget = mix(minR, maxR, pup);

    // stronger pulse kick
    float pulseKick = mix(-0.040, 0.050, 0.5 + 0.5*sin(t*3.0)) * shock;
    // micro jitter
    float micro = 0.012*sin(t*1.9 + 3.0*fbm(pw*4.0)) - 0.007*dst*sin(t*0.6);
    float pupil = clamp(pupilTarget + micro + pulseKick, minR*0.70, maxR*1.20);

    // ---- iris & sclera adjust with pupil (to stay proportional) ----
    float irisInner = pupil + 0.018 - 0.010*shock;
    float irisBase  = 0.46 + 0.12 * pup;    // grows with pupil
    float irisOuter = irisBase + 0.02*sin(t*0.5) + 0.06*shock;
    float scleraStart = irisOuter + 0.03 + 0.01*shock;

    // masks
    float pupilMask  = smoothstep(pupil, pupil-0.012, r);
    float pupilEdge  = smoothstep(pupil+0.020, pupil-0.006, r) * 0.75;
    float irisMask   = smoothstep(irisInner, irisInner+0.022, r) * (1.0 - smoothstep(irisOuter-0.022, irisOuter+0.014, r));
    float scleraMask = smoothstep(irisOuter, scleraStart, r);

    // ---------- pupil color & halo ----------
    vec3 pupilCol = vec3(0.02, 0.03, 0.05);
    pupilCol = toward(pupilCol, max(u_col * 0.30, vec3(0.0)), cs*0.65);
    vec2 hlOff = vec2(0.12, 0.10);
    float highlight = smoothstep(0.07, 0.0, length(pw - hlOff)) * (0.7 + 0.6*env);
    // NEW: dilation glow—bright rim when pupil gets big
    float dilateGlow = smoothstep(0.22, 0.30, pupil) * 0.6; // kicks in for big pupils

    // ---------- iris fibers (amped near big pupils) ----------
    float fiberPhase = t*(1.0+0.65*spd) + 2.8*fbm(vec2(a*0.5, r*3.0));
    float radial = 0.6 + 0.4*sin(22.0*a + fiberPhase + 3.2*fbm(vec2(r*6.0, a*0.7)));
    float rings  = 0.5 + 0.5*sin(10.0*r*mix(6.0, 12.0, dst) - t*2.6);
    float vein   = fbm(pw*11.0 + vec2(0.0, t*0.9));
    float vein2  = fbm(rot(pw*8.0, 0.7) - vec2(t*0.65, 0.0));
    float caust  = pow(0.5 + 0.5*(vein*0.7 + vein2*0.3), 2.0);
    float mIris  = mix(0.0, 1.0, radial)*0.6 + rings*0.4;

    // palette seeded by u_col (kept strong)
    float lum   = dot(u_col, vec3(0.2126,0.7152,0.0722));
    float huep  = atan(u_col.g - u_col.b, u_col.r - u_col.g);
    float hue01 = huep / 6.28318 + 0.5;

    vec3 A = toward(vec3(0.28, 0.23, 0.20), u_col * 0.55, cs);
    vec3 B = toward(vec3(0.55, 0.60, 0.62), mix(vec3(0.4), u_col, 0.8), cs);
    vec3 C = toward(vec3(1.00, 0.85, 0.60), mix(vec3(0.8), u_col, 0.6), cs*0.6);
    vec3 D = toward(vec3(0.18 + 0.32*hue01, 0.35, 0.68),
                    mix(vec3(0.2,0.35,0.5), u_col, 0.9), cs);

    vec3 irisCol = pal(
        mIris + 0.32*caust + 0.03*sin(13.0*r + t),
        A, B, C, D
    );

    // pulse + dilation emphasis
    float spikes = pow(abs(sin(16.0*a + 4.5*fbm(vec2(a*2.2, t)))) , 7.0);
    irisCol += (0.55 + 0.35*dst) * (0.40*shock + 0.25*dilateGlow) * (0.65 + 0.45 * sin(a*10.0 + t*5.0));
    irisCol += spikes * (0.35 + 0.45*dst) * (0.30 + 0.50*env + 0.35*dilateGlow);

    // dispersion a bit stronger when pupil is large
    float offs = 0.006 * (0.3 + 0.7*dst) * (1.0 + 0.6*env + 0.5*pup);
    irisCol.r += offs * sin(a*3.0 + t*0.9);
    irisCol.b += -offs * cos(a*3.5 + t*1.0);

    // Retint iris by gaze hue without destroying texture detail
    // Keep Value from iris, steer Hue fully to gaze hue, blend Saturation by cs
    vec3 irisHSV = rgb2hsv(irisCol);
    vec3 gazeHSV = rgb2hsv(u_col);
    float kS = clamp(cs, 0.0, 1.0);
    irisHSV.x = gazeHSV.x;                     // take gaze hue (stable hue match)
    irisHSV.y = mix(irisHSV.y, gazeHSV.y, kS); // saturation follows cs
    irisCol = hsv2rgb(irisHSV);

    // ---------- sclera ----------
    float sclV = fbm(pw*6.0 + vec2(t*0.25, -t*0.18));
    vec3 scleraCol = vec3(0.96, 0.97, 0.985);
    scleraCol = toward(scleraCol, mix(vec3(1.0), u_col, 0.25 + 0.35*lum), cs*0.35);
    scleraCol += vec3(0.03, -0.01, 0.02)*(sclV-0.5);
    float edge = smoothstep(0.72, 1.0, r);
    // veins pop a bit more with big pupils
    scleraCol += vec3(0.10, 0.02, 0.06) * edge * (0.18 + 0.75*fbm(pw*12.0+2.0)) * (0.85 + 0.6*pup);

    // ---------- compose ----------
    vec3 col = scleraCol * scleraMask
             + irisCol   * irisMask
             + pupilCol  * pupilMask;

    // specular highlight, pupil edge, dilation glow halo
    col += vec3(1.0, 0.98, 0.95) * highlight * (0.7 + 0.5*env);
    col += vec3(0.06, 0.09, 0.14) * pupilEdge;
    col += vec3(1.0) * dilateGlow * 0.07; // subtle global lift on big pupils

    // vignette + flash
    float vig = smoothstep(1.22, 0.22, r);
    col *= vig;
    col += env * (0.10 * toward(vec3(1.0), normalize(u_col + 1e-3), 0.7));
    col = pow(col, vec3(0.90));

    fragColor = vec4(col, 1.0);
}
"""

# ---- Offscreen VisualTunnelOffscreen renderer ----
class VisualTunnelOffscreen:
    """Render the same shader to an offscreen framebuffer and return an RGB image."""
    def __init__(self, width: int = 640, height: int = 360):
        self.width = int(width)
        self.height = int(height)
        self._window = None
        self._ctx = None
        self._vao = None
        self._prog = None
        self._fbo = None
        self._uniforms = {}
        self._inited = False

    def init(self) -> bool:
        # Hidden window (needed for a GL context on many platforms)
        def _glfw_err(err, desc):
            try:
                msg = desc.decode("utf-8", "ignore") if isinstance(desc, (bytes, bytearray)) else str(desc)
            except Exception:
                msg = str(desc)
            print(f"[GLFW] error {err}: {msg}")
        glfw.set_error_callback(_glfw_err)
        if not glfw.init():
            print("[visuals-offscreen] Failed to init GLFW")
            return False
        glfw.window_hint(glfw.CONTEXT_VERSION_MAJOR, 3)
        glfw.window_hint(glfw.CONTEXT_VERSION_MINOR, 3)
        glfw.window_hint(glfw.OPENGL_PROFILE, glfw.OPENGL_CORE_PROFILE)
        glfw.window_hint(glfw.OPENGL_FORWARD_COMPAT, glfw.TRUE)
        glfw.window_hint(glfw.DOUBLEBUFFER, glfw.TRUE)
        glfw.window_hint(glfw.VISIBLE, glfw.FALSE)  # hidden
        self._window = glfw.create_window(self.width, self.height, "offscreen", None, None)
        if not self._window:
            print("[visuals-offscreen] Failed to create hidden window")
            glfw.terminate()
            return False
        glfw.make_context_current(self._window)
        glfw.swap_interval(0)
        try:
            self._ctx = moderngl.create_context()
        except Exception as e:
            print(f"[visuals-offscreen] moderngl context failed: {e}")
            glfw.terminate()
            return False
        # geometry
        vertices = np.array([-1.0, -1.0, 3.0, -1.0, -1.0, 3.0], dtype="f4")
        vbo = self._ctx.buffer(vertices.tobytes())
        self._prog = self._ctx.program(vertex_shader=VERT_SRC, fragment_shader=FRAG_SRC)
        self._vao = self._ctx.simple_vertex_array(self._prog, vbo, "in_pos")
        # uniforms
        self._uniforms = {
            "u_time": self._prog["u_time"],
            "u_res": self._prog["u_res"],
            "u_speed": self._prog["u_speed"],
            "u_distort": self._prog["u_distort"],
            "u_col": self._prog["u_col"],
            "u_col_strength": self._prog["u_col_strength"],
            "u_pupil": self._prog["u_pupil"],
            "u_pulse_age": self._prog["u_pulse_age"],
        }
        # FBO setup
        self._alloc_fbo(self.width, self.height)
        self._inited = True
        return True

    def _alloc_fbo(self, w: int, h: int):
        self.width, self.height = int(max(1, w)), int(max(1, h))
        color = self._ctx.texture((self.width, self.height), 3)
        depth = self._ctx.depth_renderbuffer((self.width, self.height))
        self._fbo = self._ctx.framebuffer(color, depth)

    def render_to_image(self, t: float, speed: float, distort: float, col_vec3: Tuple[float,float,float], col_strength: float, pupil: float, pulse_age: float, out_w: int, out_h: int) -> np.ndarray:
        if not self._inited:
            return None
        # Resize FBO if needed
        if out_w != self.width or out_h != self.height:
            self._alloc_fbo(out_w, out_h)
        self._fbo.use()
        self._ctx.viewport = (0, 0, self.width, self.height)
        self._uniforms["u_time"].value = float(t)
        self._uniforms["u_res"].value = (float(self.width), float(self.height))
        self._uniforms["u_speed"].value = float(np.clip(speed, 0.0, 2.0))
        self._uniforms["u_distort"].value = float(np.clip(distort, 0.0, 2.0))
        self._uniforms["u_col"].value = tuple(map(float, col_vec3))
        self._uniforms["u_col_strength"].value = float(np.clip(col_strength, 0.0, 1.0))
        self._uniforms["u_pupil"].value = float(np.clip(pupil, 0.0, 1.0))
        self._uniforms["u_pulse_age"].value = float(max(0.0, pulse_age))
        self._ctx.clear(0.0, 0.0, 0.0, 1.0)
        self._vao.render()
        # Read back RGB
        data = self._fbo.read(components=3, alignment=1)
        img = np.frombuffer(data, dtype=np.uint8).reshape(self.height, self.width, 3)
        # OpenGL origin is bottom-left; flip vertically for OpenCV
        img = np.flip(img, axis=0)
        return img

    def close(self):
        try:
            if self._window is not None:
                glfw.destroy_window(self._window)
        except Exception:
            pass
        glfw.terminate()

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
            "u_res": prog["u_res"],
            "u_speed": prog["u_speed"],
            "u_distort": prog["u_distort"],
            "u_col": prog["u_col"],
            "u_col_strength": prog["u_col_strength"],
            "u_pupil": prog["u_pupil"],
            "u_pulse_age": prog["u_pulse_age"],
        }
        self._inited = True
        return True

    def render(self, t: float, speed: float, distort: float, col_vec3: Tuple[float,float,float], col_strength: float, pupil: float, pulse_age: float):
        if not self._inited or not self._window:
            return False
        if glfw.window_should_close(self._window):
            return False
        w, h = glfw.get_framebuffer_size(self._window)
        self._ctx.viewport = (0, 0, int(w), int(h))
        self._uniforms["u_time"].value = float(t)
        self._uniforms["u_res"].value = (float(w), float(h))
        self._uniforms["u_speed"].value = float(np.clip(speed, 0.0, 2.0))
        self._uniforms["u_distort"].value = float(np.clip(distort, 0.0, 2.0))
        self._uniforms["u_col"].value = tuple(map(float, col_vec3))
        self._uniforms["u_col_strength"].value = float(np.clip(col_strength, 0.0, 1.0))
        self._uniforms["u_pupil"].value = float(np.clip(pupil, 0.0, 1.0))
        self._uniforms["u_pulse_age"].value = float(max(0.0, pulse_age))
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
        self._col_vec = (0.5, 0.5, 0.5)
        self._col_strength = 0.9
        self._pupil = 0.4
        self._pulse_age = 1e6
        self._running = threading.Event()
        self._running.set()

    def update(self, speed: float, distort: float, col_vec3: Tuple[float,float,float], col_strength: float, pupil: float, pulse_age: float):
        with self._lock:
            self._speed = float(np.clip(speed, 0.0, 2.0))
            self._distort = float(np.clip(distort, 0.0, 2.0))
            self._col_vec = tuple(np.clip(col_vec3, 0.0, 1.0))
            self._col_strength = float(np.clip(col_strength, 0.0, 1.0))
            self._pupil = float(np.clip(pupil, 0.0, 1.0))
            self._pulse_age = float(max(0.0, pulse_age))

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
        u_res = prog["u_res"]
        u_speed = prog["u_speed"]
        u_distort = prog["u_distort"]
        u_col = prog["u_col"]
        u_col_strength = prog["u_col_strength"]
        u_pupil = prog["u_pupil"]
        u_pulse_age = prog["u_pulse_age"]

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
                col_vec = self._col_vec
                col_strength = self._col_strength
                pupil = self._pupil
                pulse_age = self._pulse_age

            u_time.value = now - start
            u_res.value = (float(w), float(h))
            u_speed.value = spd
            u_distort.value = dst
            u_col.value = tuple(map(float, col_vec))
            u_col_strength.value = col_strength
            u_pupil.value = pupil
            u_pulse_age.value = pulse_age

            ctx.clear(0.0, 0.0, 0.0, 1.0)
            vao.render()
            glfw.swap_buffers(window)
            glfw.poll_events()

        glfw.terminate()

# ----------------------- Main loop -----------------------

@click.command()
@click.option("--ip", default='192.168.20.72', help="IP address of Neon Companion Device.")
@click.option("--port", default=8080, show_default=True, help="Neon Companion port.")
@click.option("--preview", is_flag=True, help="Show scene preview with gaze dot.")
@click.option("--debug-eye-events", is_flag=True, help="Print eye events (blink/fixation/saccade) to console")
@click.option("--visuals", is_flag=True, help="Show shader tunnel visuals driven by gaze & color")
@click.option("--fullscreen", is_flag=True, help="Open visuals in fullscreen")
@click.option("--visuals-embedded", is_flag=True, help="Create the shader window on the main thread (macOS fix)")
@click.option("--single-window", is_flag=True, help="Show scene (with gaze+eyes) on the left and shader visuals on the right in one OpenCV window")
@click.option("--pupil-min-mm", default=1.0, show_default=True, help="Pupil min (mm) mapped to 0.0")
@click.option("--pupil-max-mm", default=6.0, show_default=True, help="Pupil max (mm) mapped to 1.0")
@click.option("--pupil-gamma", default=1.5, show_default=True, help="Nonlinear emphasis for pupil: output=(norm^gamma)")
@click.option("--pupil-gain", default=1.5, show_default=True, help="Gain applied after gamma: output=gain*(norm^gamma)")
@click.option("--patch-half", default=50, show_default=True, help="Half-size of the sampling patch around gaze (pixels). Actual patch = (2*half+1)^2")
@click.option("--col-strength", default=0.9, show_default=True, help="How strongly the iris follows the sampled RGB (0..1)")
def main(ip: Optional[str], port: int, preview: bool, debug_eye_events: bool, visuals: bool, fullscreen: bool, visuals_embedded: bool, single_window: bool, pupil_min_mm: float, pupil_max_mm: float, pupil_gamma: float, pupil_gain: float, patch_half: int, col_strength: float):
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
    cutoff_ema = EMA(alpha=0.2, init=np.array([1200.0], dtype=np.float32))

    last_update = 0.0
    throttle_hz = 60.0  # don’t over-update audio more than ~60x/sec
    min_conf = 0.4      # ignore very low-confidence gaze

    vis = None
    vis_emb = None
    vis_off = None
    if visuals:
        print("[visuals] Starting shader tunnel window...")
        if visuals_embedded or sys.platform == 'darwin':
            vis_emb = VisualTunnelEmbedded(fullscreen=fullscreen)
            if not vis_emb.init():
                print("[visuals] Failed to initialize embedded visuals.")
        else:
            vis = VisualTunnel(fullscreen=fullscreen)
            vis.start()
    vis_off = None
    if (visuals and preview) and (single_window or sys.platform == 'darwin'):
        # Use offscreen so we can composite into a single cv2 window
        vis_off = VisualTunnelOffscreen(width=640, height=360)
        if not vis_off.init():
            print("[visuals-offscreen] Failed to init; falling back to separate window")
            vis_off = None

    gaze_prev = None
    gaze_speed_ema = EMA(alpha=0.2, init=np.array([0.0], dtype=np.float32))

    if preview:
        cv2.namedWindow("Scene", cv2.WINDOW_NORMAL)

    # For embedded visuals, need to drive clock
    start_time = time.perf_counter()
    last_blink_perf = time.perf_counter() - 1e6

    try:
        while True:
            # Get matched scene frame and gaze (timestamp-aligned)
            matched = None
            try:
                matched = device.receive_matched_scene_video_frame_and_gaze(timeout_seconds=0.1)
            except Exception:
                matched = None

            if matched is None:
                # Fallback to previous separate fetch if matched API not available
                gaze = device.receive_gaze_datum(timeout_seconds=0.1)
                scene: Optional[SimpleVideoFrame] = device.receive_scene_video_frame(timeout_seconds=0.1)
                if gaze is None or scene is None:
                    time.sleep(0.005)
                    continue
            else:
                scene, gaze = matched

            # Drain eye events queue (blinks/fixations/saccades) and optionally print
            while True:
                try:
                    ev = device.receive_eye_events(timeout_seconds=0.0)
                except Exception:
                    break  # sensor not available or not started
                if ev is None:
                    break  # no more events pending
                # Optional verbose print of all eye events
                if debug_eye_events:
                    try:
                        name = type(ev).__name__
                        ts = getattr(ev, "rtp_ts_unix_seconds", None)
                        if ts is None:
                            ts = getattr(ev, "timestamp_unix_ns", None)
                        print(f"[eye-events] {name}: {ev} @ {ts}")
                    except Exception:
                        print(f"[eye-events] {type(ev).__name__}")
                # Trigger drums on blink and always print a concise line
                if isinstance(ev, BlinkEventData):
                    synth.trigger_drum("kick")
                    print("[blink] detected -> KICK")
                    last_blink_perf = time.perf_counter()

            img = scene.bgr_pixels  # HxWx3, uint8
            h, w = img.shape[:2]

            # Prefer pixel-space gaze from matched API; fall back to normalized extraction
            if matched is not None and hasattr(gaze, 'x') and hasattr(gaze, 'y'):
                x = int(np.clip(float(getattr(gaze, 'x', 0.0)), 0, w - 1))
                y = int(np.clip(float(getattr(gaze, 'y', 0.0)), 0, h - 1))
                conf = float(getattr(gaze, 'confidence', 1.0)) if hasattr(gaze, 'confidence') else 1.0
                gx = x / float(max(1, w - 1))
                gy = y / float(max(1, h - 1))
            else:
                g = extract_norm_gaze_xy(gaze)
                if g is None:
                    continue
                gx, gy, conf = g
                x = int(np.clip(gx, 0.0, 1.0) * (w - 1))
                y = int(np.clip(gy, 0.0, 1.0) * (h - 1))

            if conf < min_conf:
                synth.set_params(synth._freq, 0.0, synth._waveform)
                continue

            # Sample single pixel at gaze coordinate (no patch averaging)
            b, g_, r = map(int, img[y, x])  # OpenCV is BGR order
            # Smooth RGB values (EMA) to reduce flicker
            Rs, Gs, Bs = rgb_ema(np.array([r, g_, b], dtype=np.float32))

            # Map RGB to freq, amp, waveform
            freq, amp, waveform = rgb_to_params(int(Rs), int(Gs), int(Bs))
            amp = float(amp_ema(np.array([amp], dtype=np.float32))[0])
            # Map G -> low-pass cutoff (300..3000 Hz) with EMA smoothing
            cutoff_raw = float(np.interp(np.clip(Gs, 0.0, 255.0), [0.0, 255.0], [300.0, 3000.0]))
            cutoff = float(cutoff_ema(np.array([cutoff_raw], dtype=np.float32))[0])

            # Throttle parameter pushes to audio engine
            now = time.time()
            if now - last_update >= (1.0 / throttle_hz):
                synth.set_params(freq, amp, waveform)
                synth.set_filter_cutoff(cutoff)
                print(
                    f"[color->sound] R={int(Rs)} G={int(Gs)} B={int(Bs)} | note_f={freq:.1f}Hz cutoff={cutoff:.0f}Hz vol={amp:.2f}"
                )
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

            # Prefer real pupil diameter from Neon if available
            pupil_mm = extract_avg_pupil_mm(gaze)
            if pupil_mm is not None and pupil_max_mm > pupil_min_mm:
                pupil_val = float(np.clip((pupil_mm - pupil_min_mm) / (pupil_max_mm - pupil_min_mm), 0.0, 1.0))
            else:
                # Fallback: use brightness as proxy
                brightness = (float(Rs) + float(Gs) + float(Bs)) / 3.0
                pupil_val = float(np.clip(brightness / 255.0, 0.0, 1.0))
            if pupil_mm is not None:
                print(f"[pupil] avg={pupil_mm:.2f} mm -> norm={pupil_val:.2f}")
            # Emphasize small changes: gamma then gain, finally clip
            if np.isfinite(pupil_gamma) and pupil_gamma > 0:
                pupil_val = float(np.clip((pupil_val ** float(pupil_gamma)) * float(pupil_gain), 0.0, 1.0))

            # Map: speed -> u_speed (0..2), distort -> from S (0..2)
            u_speed_val = float(np.clip(speed_smooth * 10.0, 0.0, 2.0))
            u_distort_val = float(np.clip(S * 2.0, 0.0, 2.0))
            u_speed_val = float(2.0)
            u_distort_val = float(2.0)

            # Color vector for shader (0..1)
            col_vec3 = (float(Rs)/255.0, float(Gs)/255.0, float(Bs)/255.0)
            col_strength = float(np.clip(col_strength, 0.0, 1.0))
            pulse_age = time.perf_counter() - last_blink_perf
            if visuals and vis is not None:
                # Threaded renderer
                vis.update(u_speed_val, u_distort_val, col_vec3, col_strength, pupil_val, pulse_age)
            if vis_emb is not None:
                # Embedded renderer (main thread)
                t_now = time.perf_counter() - start_time
                ok = vis_emb.render(t_now, u_speed_val, u_distort_val, col_vec3, col_strength, pupil_val, pulse_age)
                if not ok:
                    vis_emb.close()
                    vis_emb = None

            # -------- Combined Single Window (scene+eyes on left, visuals on right) --------
            if preview:
                # Start from the scene image with overlays
                vis_img = img.copy()

                # --- Red gaze ring (API snippet style) ---
                cv2.circle(
                    vis_img,
                    (int(x), int(y)),
                    radius=80,
                    color=(0, 0, 255),
                    thickness=15,
                )

                # --- HUD text (top-left) ---
                cv2.putText(
                    vis_img,
                    f"R={int(Rs)} G={int(Gs)} B={int(Bs)}  note={freq:6.1f}Hz  vol={amp:0.2f}  cutoff={cutoff:4.0f}Hz",
                    (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 1, cv2.LINE_AA)
                cv2.putText(
                    vis_img,
                    f"spd={speed_smooth:0.3f}  S={S:0.2f}  pupil={pupil_val:0.2f}",
                    (10, 48), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200,255,200), 1, cv2.LINE_AA)

                # --- Eye camera frame (if available) ---
                eye_frame = None
                try:
                    eye = device.receive_eyes_video_frame(timeout_seconds=0.0)
                    if eye is not None:
                        eye_frame = eye.bgr_pixels
                except Exception:
                    eye_frame = None

                # Layout sizes
                left_w = w
                left_h = h
                # Right visuals height matches left; width equals left_w (split 50/50)
                right_w = left_w
                right_h = left_h

                # Render shader offscreen image if available
                right_img = None
                if vis_off is not None:
                    t_now = time.perf_counter() - start_time
                    right_rgb = vis_off.render_to_image(t_now, u_speed_val, u_distort_val, col_vec3, col_strength, pupil_val, pulse_age, right_w, right_h)
                    if right_rgb is not None:
                        # convert RGB->BGR for OpenCV
                        right_img = right_rgb[:, :, ::-1]

                # If no offscreen, just make a black panel with text
                if right_img is None:
                    right_img = np.zeros((right_h, right_w, 3), dtype=np.uint8)
                    cv2.putText(right_img, "visuals window active separately", (20, right_h//2), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (180,180,180), 1, cv2.LINE_AA)

                # Compose left column (scene + optional eyes thumbnail at bottom-left)
                left_panel = vis_img
                if eye_frame is not None:
                    # scale eye frame to 1/3 height
                    thumb_h = max(80, left_h // 3)
                    scale = thumb_h / eye_frame.shape[0]
                    thumb_w = int(eye_frame.shape[1] * scale)
                    eye_thumb = cv2.resize(eye_frame, (thumb_w, thumb_h), interpolation=cv2.INTER_AREA)
                    # paste onto bottom-left with a white border
                    yb = left_h - thumb_h - 10
                    xb = 10
                    cv2.rectangle(left_panel, (xb-2, yb-2), (xb+thumb_w+2, yb+thumb_h+2), (255,255,255), 1)
                    left_panel[yb:yb+thumb_h, xb:xb+thumb_w] = eye_thumb

                # Final side-by-side canvas
                canvas = np.zeros((max(left_h, right_h), left_w + right_w, 3), dtype=np.uint8)
                canvas[0:left_h, 0:left_w] = left_panel
                canvas[0:right_h, left_w:left_w+right_w] = right_img

                cv2.imshow("Eyetrance — Live", canvas)
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
        try:
            if 'vis_off' in locals() and vis_off is not None:
                vis_off.close()
        except Exception:
            pass
        print("Bye.")


if __name__ == "__main__":
    main()