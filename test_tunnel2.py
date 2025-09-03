import time
import numpy as np
import glfw
import moderngl

# ============== Config ==============
WINDOW_TITLE = "Psychedelic Shader Tunnel — 3 Signals"
START_FULLSCREEN = False
TARGET_FPS = 120
# ====================================

FRAG_SRC = """
#version 330

out vec4 fragColor;

uniform float u_time;      // seconds
uniform float u_speed;     // 0..~2: flow speed
uniform float u_distort;   // 0..~2: wobble/warp strength
uniform float u_color;     // 0..~2: hue/contrast shift
uniform vec2  u_res;       // viewport

// iq-style palette
vec3 pal(float t, vec3 a, vec3 b, vec3 c, vec3 d) {
    return a + b * cos(6.28318 * (c * t + d));
}

void main() {
    vec2 uv = (gl_FragCoord.xy / u_res.xy) * 2.0 - 1.0;
    uv.x *= u_res.x / u_res.y;

    float r = length(uv);
    float a = atan(uv.y, uv.x);

    // speed controls time scale
    float t = u_time * mix(0.2, 3.0, clamp(u_speed, 0.0, 2.0));

    // classic tunnel mapping
    float z = 3.0 / (r + 0.12);

    // rings / bands / stripes
    float band    = sin(z + t * 2.1 + sin(a * 6.0 + t)) * 0.5 + 0.5;
    float rings   = smoothstep(0.25, 0.75, sin(z * 0.75 + t * 1.75));
    float stripes = smoothstep(0.2, 0.8,  sin(a * 10.0 + t * 2.0));

    // distortion strength
    float wobble = sin(a * 4.0 - t * 3.0) * 0.15 * (0.5 + u_distort);
    float warp   = sin((r + wobble) * 8.0 - t * 2.0);

    float m = mix(band, rings, 0.5) * 0.7 + stripes * 0.3;
    m = mix(m, warp * 0.5 + 0.5, 0.45 + 0.35 * clamp(u_distort, 0.0, 2.0));

    // color modulation: shift palette params with u_color
    float hueShift = u_color * 0.2;
    vec3 col = pal(
        m + z * 0.02 + a * (0.08 + 0.12 * u_color),
        vec3(0.52, 0.35, 0.40) + vec3(0.05) * u_color,
        vec3(0.45, 0.55, 0.60),
        vec3(1.00, 0.80, 0.60),
        vec3(0.15 + hueShift, 0.35, 0.65)
    );

    // vignette + contrast (also color-modulated)
    float vig = smoothstep(1.25, 0.2, r);
    col *= vig;
    col = pow(col, vec3(0.9 + 0.3 * u_color)); // more pop with color

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

    # Full-screen triangle
    vertices = np.array(
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
    u_speed = prog["u_speed"]
    u_distort = prog["u_distort"]
    u_color = prog["u_color"]
    u_res = prog["u_res"]

    start = time.perf_counter()
    last = start
    frame_cap = 1.0 / float(TARGET_FPS)

    # --- Demo inputs (replace with your own three signals each frame) ---
    # Up/Down -> speed   | Left/Right -> distortion   | C/V -> color
    def demo_signals():

        x, y = get_mouse_pos(window)

        # normalize to [0..1]
        x /= width
        y /= height

        # speed
        s = np.clip(x, 0.0, 1.0) * 2.0

        # distortion
        d = np.clip(y, 0.0, 1.0) * 2.0

        # color
        c = np.clip((x + y) / 2.0, 0.0, 1.0) * 2.0

        # gentle idle “breathing” so it never looks dead
        t = time.perf_counter() - start
        s += 0.15 * (0.5 + 0.5 * np.sin(t * 0.7))
        d += 0.10 * (0.5 + 0.5 * np.sin(t * 0.9 + 1.0))
        c += 0.12 * (0.5 + 0.5 * np.sin(t * 0.6 + 2.0))

        # clamp to a sane range
        return (
            float(np.clip(s, 0.0, 2.0)),
            float(np.clip(d, 0.0, 2.0)),
            float(np.clip(c, 0.0, 2.0)),
        )

    # -------------------------------------------------------------------

    while not glfw.window_should_close(window):
        now = time.perf_counter()
        dt = now - last
        if dt < frame_cap:
            time.sleep(frame_cap - dt)
            now = time.perf_counter()
        last = now

        w, h = glfw.get_framebuffer_size(window)
        ctx.viewport = (0, 0, w, h)

        t = now - start

        # === FEED YOUR REAL SIGNALS HERE ===
        # Replace these with your own values each frame:
        speed_val, distort_val, color_val = demo_signals()
        # e.g.:
        # speed_val   = norm_speed_sensor()   # 0..2
        # distort_val = norm_distort_sensor() # 0..2
        # color_val   = norm_color_sensor()   # 0..2

        u_time.value = t
        u_speed.value = speed_val
        u_distort.value = distort_val
        u_color.value = color_val
        u_res.value = (float(w), float(h))

        ctx.clear(0.0, 0.0, 0.0, 1.0)
        vao.render()

        glfw.swap_buffers(window)
        glfw.poll_events()

        if glfw.get_key(window, glfw.KEY_ESCAPE) == glfw.PRESS:
            break

    glfw.terminate()


def get_mouse_pos(window):
    x, y = glfw.get_cursor_pos(window)
    return x, y


if __name__ == "__main__":
    main()
