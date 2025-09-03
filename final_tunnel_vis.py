# pip install moderngl glfw numpy
import time
import numpy as np
import glfw
import moderngl

WINDOW_TITLE = "Psychedelic Eye — MAX Pulse"
START_FULLSCREEN = False
TARGET_FPS = 120

FRAG_SRC = """
#version 330

out vec4 fragColor;

uniform float u_time;         // seconds
uniform vec2  u_res;          // viewport
uniform float u_speed;        // 0..~2: animation speed
uniform float u_distort;      // 0..~2: wobble/warp intensity
uniform vec3  u_col;          // main tint (0..1), strongly influences colors
uniform float u_col_strength; // 0..1: strength of color mapping
uniform float u_pupil;        // 0..1: external pupil radius
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
// ===========================================

void main(){
    vec2 res = u_res;
    vec2 p = (gl_FragCoord.xy / res) * 2.0 - 1.0;
    p.x *= res.x / res.y;

    float spd = clamp(u_speed,   0.0, 2.0);
    float dst = clamp(u_distort, 0.0, 2.0);
    float pup = clamp(u_pupil,   0.0, 1.0);
    float cs  = clamp(u_col_strength, 0.0, 1.0);

    // ---- Pulse envelope (STRONG) ----
    float age   = u_pulse_age;
    float env   = exp(-age * 1.4);    // slower decay => longer punch
    float vring = 1.35;               // faster ring
    float sigma = 0.09 + 0.06*dst;    // thicker ring
    float r0    = vring * age;        // ring center radius

    // Time warp on pulse (everything speeds up briefly)
    float t = (u_time + 0.20 * env) * mix(0.15, 3.0, spd + 0.35*env);

    // Big zoom pop on pulse
    float zoom  = 1.0 - 0.08 * env;
    p *= zoom;

    // Kaleido fold during first ~200ms for extra craziness
    float kenv = smoothstep(0.0, 0.20, 0.20 - age) * env; // 0→1 near trigger
    if(kenv > 0.0){
        float seg = mix(6.0, 16.0, kenv);
        float ang = atan(p.y, p.x);
        float wedge = 2.0 * PI / max(seg, 1.0);
        float aFold = abs(mod(ang, wedge) - 0.5 * wedge);
        float r = length(p);
        p = vec2(cos(aFold), sin(aFold)) * r;
    }

    // subtle swirl + extra when pulsing
    float swirl = (0.15*dst + 0.25*env) * sin(t*0.8 + 3.0*env);
    p = rot(p, swirl);

    // domain warp (baseline)
    vec2 pw = p;
    float w1 = fbm(pw*2.5 + vec2(0.0, t*0.5));
    float w2 = fbm(rot(pw, 1.2)*3.0 - vec2(t*0.4, 0.0));
    pw += 0.12*dst * vec2(w1 - 0.5, w2 - 0.5);

    float r = length(pw);
    float a = atan(pw.y, pw.x);

    // Radial shockwave DISPLACEMENT (harder)
    float ring   = exp(-0.5 * pow((r - r0)/sigma, 2.0));
    float shock  = ring * env;
    // push pixels outwards + add angular ripple
    pw += (0.22 * shock) * normalize(pw + 1e-4)
        + (0.06 * shock) * vec2(sin(a*12.0 + t*6.0), cos(a*10.0 - t*5.0));

    // recompute r,a after displacement
    r = length(pw);
    a = atan(pw.y, pw.x);

    // ---------- pupil from signal with BIG pulse kick ----------
    float minR = 0.07;
    float maxR = 0.22;
    float pupilTarget = mix(minR, maxR, pup);
    float pulseKick = mix(-0.030, 0.038, 0.5 + 0.5*sin(t*3.0)) * shock; // constrict then dilate
    float micro = 0.012*sin(t*1.9 + 3.0*fbm(pw*4.0)) - 0.007*dst*sin(t*0.6);
    float pupil = clamp(pupilTarget + micro + pulseKick, minR*0.75, maxR*1.15);

    // iris gain/size also breathe with the pulse
    float irisInner = pupil + 0.016 - 0.010*shock;
    float irisOuter = (0.48 + 0.02*sin(t*0.5)) + 0.06*shock;
    float scleraStart = irisOuter + 0.02 + 0.01*shock;

    // ---------- masks (DECLARE ONCE) ----------
    float pupilMask  = smoothstep(pupil, pupil-0.012, r);
    float pupilEdge  = smoothstep(pupil+0.018, pupil-0.006, r) * 0.7;
    float irisMask   = smoothstep(irisInner, irisInner+0.02, r) * (1.0 - smoothstep(irisOuter-0.02, irisOuter+0.012, r));
    float scleraMask = smoothstep(irisOuter, scleraStart, r);

    // ---------- pupil ----------
    vec3 pupilCol = vec3(0.02, 0.03, 0.05);
    pupilCol = toward(pupilCol, max(u_col * 0.28, vec3(0.0)), cs*0.6);
    // bright sparkle grows with pulse
    vec2 hlOff = vec2(0.12, 0.10);
    float highlight = smoothstep(0.07, 0.0, length(pw - hlOff)) * (0.7 + 0.6*env);

    // ---------- iris structure (amped) ----------
    float fiberPhase = t*(1.0+0.65*spd) + 2.8*fbm(vec2(a*0.5, r*3.0));
    float radial = 0.6 + 0.4*sin(22.0*a + fiberPhase + 3.2*fbm(vec2(r*6.0, a*0.7)));
    float rings  = 0.5 + 0.5*sin(10.0*r*mix(6.0, 12.0, dst) - t*2.6);
    float vein   = fbm(pw*11.0 + vec2(0.0, t*0.9));
    float vein2  = fbm(rot(pw*8.0, 0.7) - vec2(t*0.65, 0.0));
    float caust  = pow(0.5 + 0.5*(vein*0.7 + vein2*0.3), 2.0);
    float mIris  = mix(0.0, 1.0, radial)*0.6 + rings*0.4;

    // palette seeded strongly by u_col
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

    // pulse glow band + fiber streaks
    irisCol += shock * (0.55 + 0.35*dst) * (0.65 + 0.45 * sin(a*10.0 + t*5.0));
    float spikes = pow(abs(sin(16.0*a + 4.5*fbm(vec2(a*2.2, t)))) , 7.0);
    irisCol += spikes * (0.35 + 0.45*dst) * (0.4 + 0.6*env);

    // dispersion stronger on pulse
    float offs = 0.006 * (0.3 + 0.7*dst) * (1.0 + 0.8*env);
    irisCol.r += offs * sin(a*3.0 + t*0.9);
    irisCol.b += -offs * cos(a*3.5 + t*1.0);

    // strong pull toward u_col
    irisCol = toward(irisCol, u_col, cs * (0.65 + 0.25*env));

    // ---------- sclera ----------
    float sclV = fbm(pw*6.0 + vec2(t*0.25, -t*0.18));
    vec3 scleraCol = vec3(0.96, 0.97, 0.985);
    scleraCol = toward(scleraCol, mix(vec3(1.0), u_col, 0.25 + 0.35*lum), cs*0.35);
    scleraCol += vec3(0.03, -0.01, 0.02)*(sclV-0.5);
    float edge = smoothstep(0.72, 1.0, r);
    scleraCol += vec3(0.10, 0.02, 0.06) * edge * (0.18 + 0.75*fbm(pw*12.0+2.0));

    // ---------- compose (NO REDECLARATION) ----------
    vec3 col = scleraCol * scleraMask
             + irisCol   * irisMask
             + pupilCol  * pupilMask;

    // specular highlight & inner edge
    col += vec3(1.0, 0.98, 0.95) * highlight * (0.7 + 0.5*env);
    col += vec3(0.06, 0.09, 0.14) * pupilEdge;

    // global pulse flash + vignette + curve
    float vig = smoothstep(1.22, 0.22, r);
    col *= vig;
    // bright flash biased toward u_col (so color remains faithful)
    col += env * (0.10 * toward(vec3(1.0), normalize(u_col + 1e-3), 0.7));
    col = pow(col, vec3(0.90));

    fragColor = vec4(col, 1.0);
}

"""

VERT_SRC = """
#version 330
in vec2 in_pos;
void main() {
    gl_Position = vec4(in_pos, 0.0, 1.0);
}
"""


def main():
    if not glfw.init():
        raise RuntimeError("Failed to init GLFW")

    glfw.window_hint(glfw.CONTEXT_VERSION_MAJOR, 3)
    glfw.window_hint(glfw.CONTEXT_VERSION_MINOR, 3)
    glfw.window_hint(glfw.OPENGL_PROFILE, glfw.OPENGL_CORE_PROFILE)
    glfw.window_hint(glfw.DOUBLEBUFFER, glfw.TRUE)

    monitor = glfw.get_primary_monitor() if START_FULLSCREEN else None
    width, height = (1280, 720)
    if START_FULLSCREEN:
        mode = glfw.get_video_mode(monitor)
        width, height = mode.size.width, mode.size.height

    window = glfw.create_window(width, height, WINDOW_TITLE, monitor, None)
    if not window:
        glfw.terminate()
        raise RuntimeError("Failed to create window")
    glfw.make_context_current(window)
    glfw.swap_interval(0)

    ctx = moderngl.create_context()

    # full-screen triangle
    import numpy as _np

    vertices = _np.array(
        [
            -1.0,
            -1.0,
            3.0,
            -1.0,
            -1.0,
            3.0,
        ],
        dtype="f4",
    )
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

    # mouse state (X->speed, Y->distort)
    win_w, win_h = glfw.get_framebuffer_size(window)
    mouse_x, mouse_y = win_w * 0.5, win_h * 0.5

    def on_cursor(win, xpos, ypos):
        nonlocal mouse_x, mouse_y
        mouse_x, mouse_y = xpos, ypos

    def on_resize(win, w, h):
        nonlocal win_w, win_h
        win_w, win_h = w, h

    glfw.set_cursor_pos_callback(window, on_cursor)
    glfw.set_window_size_callback(window, on_resize)

    # random RGB once (press R to reroll)
    color_rgb = np.random.rand(3).astype(np.float32)

    def maybe_reroll_color():
        nonlocal color_rgb
        if glfw.get_key(window, glfw.KEY_R) == glfw.PRESS:
            color_rgb = np.random.rand(3).astype(np.float32)

    # color influence strength (J/K)
    col_strength = 0.9

    def adjust_col_strength():
        nonlocal col_strength
        if glfw.get_key(window, glfw.KEY_J) == glfw.PRESS:
            col_strength = max(0.0, col_strength - 0.01)
        if glfw.get_key(window, glfw.KEY_K) == glfw.PRESS:
            col_strength = min(1.0, col_strength + 0.01)

    # pupil signal (0..1). Replace with your live signal.
    pupil_val = 0.4

    def read_pupil_signal():
        nonlocal pupil_val
        if glfw.get_key(window, glfw.KEY_S) == glfw.PRESS:
            pupil_val += 0.01
        if glfw.get_key(window, glfw.KEY_A) == glfw.PRESS:
            pupil_val -= 0.01
        pupil_val = float(np.clip(pupil_val, 0.0, 1.0))
        return pupil_val

    # Pulse rising-edge detection (replace read_pulse_bool with your boolean)
    pulse_bool = False
    prev_pulse_bool = False
    last_pulse_time = -1e9

    def read_pulse_bool():
        # Demo: SPACE triggers a pulse (True while held)
        return glfw.get_key(window, glfw.KEY_SPACE) == glfw.PRESS

    start = time.perf_counter()
    last = start
    frame_cap = 1.0 / float(TARGET_FPS)

    while not glfw.window_should_close(window):
        now = time.perf_counter()
        dt = now - last
        if dt < frame_cap:
            time.sleep(frame_cap - dt)
            now = time.perf_counter()
        last = now

        w, h = glfw.get_framebuffer_size(window)
        win_w, win_h = w, h
        ctx.viewport = (0, 0, w, h)

        # map mouse to speed/distortion
        speed_val = (
            0.0
            if win_w <= 0
            else 2.0 * float(np.clip(mouse_x / max(1, win_w), 0.0, 1.0))
        )
        distort_val = (
            0.0
            if win_h <= 0
            else 2.0 * float(np.clip(1.0 - (mouse_y / max(1, win_h)), 0.0, 1.0))
        )

        # signals
        pupil_sig = read_pupil_signal()
        maybe_reroll_color()
        adjust_col_strength()

        # rising-edge
        pulse_bool = read_pulse_bool()
        if (not prev_pulse_bool) and pulse_bool:
            last_pulse_time = now
        prev_pulse_bool = pulse_bool
        pulse_age = now - last_pulse_time

        # uniforms
        t = now - start
        u_time.value = t
        u_res.value = (float(w), float(h))
        u_speed.value = speed_val
        u_distort.value = distort_val
        u_col.value = tuple(map(float, color_rgb))
        u_col_strength.value = float(col_strength)
        u_pupil.value = pupil_sig
        u_pulse_age.value = float(max(0.0, pulse_age))

        ctx.clear(0.0, 0.0, 0.0, 1.0)
        vao.render()

        glfw.swap_buffers(window)
        glfw.poll_events()

        if glfw.get_key(window, glfw.KEY_ESCAPE) == glfw.PRESS:
            break

    glfw.terminate()


if __name__ == "__main__":
    main()
