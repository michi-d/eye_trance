import time
import numpy as np
import glfw
import moderngl

# ============== Config ==============
WINDOW_TITLE = "Psychedelic Shader Tunnel (ModernGL)"
START_FULLSCREEN = False
TARGET_FPS = 120
# ====================================

# ---- Fragment shader: tunnel effect with modulation ----
FRAG_SRC = """
#version 330

out vec4 fragColor;
in vec2 v_uv;

uniform float u_time;   // seconds
uniform float u_mod;    // your live signal (0..1+)
uniform vec2  u_res;    // viewport in pixels

// simple palette from iq
vec3 pal(float t, vec3 a, vec3 b, vec3 c, vec3 d) {
    return a + b * cos(6.28318 * (c * t + d));
}

void main() {
    // normalized coords, center at 0
    vec2 uv = (gl_FragCoord.xy / u_res.xy) * 2.0 - 1.0;
    uv.x *= u_res.x / u_res.y;

    // Polar coordinates
    float r = length(uv);
    float a = atan(uv.y, uv.x);

    // Time base; speed modulated by u_mod
    float t = u_time * mix(0.4, 2.0, clamp(u_mod, 0.0, 2.0));

    // Tunnel mapping: move "forward" along z by subtracting time
    float z = 3.0 / (r + 0.1);
    float band = sin(z + t * 2.0 + sin(a * 6.0 + t)) * 0.5 + 0.5;

    // Rings + angular stripes
    float rings = smoothstep(0.25, 0.75, sin(z * 0.75 + t * 1.75));
    float stripes = smoothstep(0.2, 0.8, sin(a * 10.0 + t * 2.0));

    // Distortion modulated by u_mod
    float wobble = sin(a * 4.0 - t * 3.0) * 0.15 * (0.5 + u_mod);
    float warp = sin((r + wobble) * 8.0 - t * 2.0);

    // Combine
    float m = mix(band, rings, 0.5) * 0.7 + stripes * 0.3;
    m = mix(m, warp * 0.5 + 0.5, 0.45);

    // Color palette reactive to z and a
    vec3 col = pal(
        m + z * 0.02 + a * 0.1,
        vec3(0.55, 0.35, 0.40),
        vec3(0.45, 0.55, 0.60),
        vec3(1.00, 0.80, 0.60),
        vec3(0.15, 0.35, 0.65) + u_mod * 0.1
    );

    // Vignette
    float vig = smoothstep(1.2, 0.2, r);
    col *= vig;

    // Punchy contrast
    col = pow(col, vec3(0.9));

    fragColor = vec4(col, 1.0);
}
"""

# ---- Vertex shader for a full-screen quad ----
VERT_SRC = """
#version 330
in vec2 in_pos;
out vec2 v_uv;
void main() {
    v_uv = in_pos * 0.5 + 0.5;
    gl_Position = vec4(in_pos, 0.0, 1.0);
}
"""


def main():
    if not glfw.init():
        raise RuntimeError("Failed to init GLFW")

    # Window hints
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
    glfw.swap_interval(0)  # vsync off; we’ll limit manually

    ctx = moderngl.create_context()

    # Full-screen triangle (faster than quad)
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

    # Uniform handles
    u_time = prog["u_time"]
    u_mod = prog["u_mod"]
    u_res = prog["u_res"]

    start = time.perf_counter()
    last_frame = start
    frame_cap = 1.0 / float(TARGET_FPS)

    # Example: keyboard-controlled modulation (hold keys to test)
    # Replace this with your real-time signal each frame.
    def sample_mod_signal():
        signal = get_mouse_pos(window)
        signal = signal % 1000  # keep in reasonable range
        signal = signal / 1000.0  # normalize to [0..1]
        return float(signal)
        # val = 0.0
        # if glfw.get_key(window, glfw.KEY_UP) == glfw.PRESS:
        #     val += 0.75
        # if glfw.get_key(window, glfw.KEY_DOWN) == glfw.PRESS:
        #     val += 0.25
        # # Mild breathing when idle so it never looks static
        # val += 0.25 * (0.5 + 0.5 * np.sin((time.perf_counter() - start) * 1.2))
        # return float(val)

    while not glfw.window_should_close(window):
        now = time.perf_counter()
        dt = now - last_frame
        if dt < frame_cap:
            time.sleep(frame_cap - dt)
            now = time.perf_counter()
        last_frame = now

        # Resize handling
        w, h = glfw.get_framebuffer_size(window)
        ctx.viewport = (0, 0, w, h)

        # Time & modulation
        t = now - start
        mod_val = sample_mod_signal()  # <--- replace with YOUR signal in [0..~2]

        u_time.value = t
        u_mod.value = mod_val
        u_res.value = (float(w), float(h))

        ctx.clear(0.0, 0.0, 0.0, 1.0)
        vao.render()

        glfw.swap_buffers(window)
        glfw.poll_events()

        # ESC quits
        if glfw.get_key(window, glfw.KEY_ESCAPE) == glfw.PRESS:
            break

    glfw.terminate()


def get_mouse_pos(window):
    x, y = glfw.get_cursor_pos(window)
    return x + y


if __name__ == "__main__":
    main()
