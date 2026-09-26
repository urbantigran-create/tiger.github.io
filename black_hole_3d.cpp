// ============================================================
//  Интерактивная 3D-модель чёрной дыры (Шварцшильд)
//  OpenGL 3.3 + GLFW + GLEW. Трассировка геодезических на GPU.
//
//  Сборка (PowerShell в папке проекта):
//    g++ -O2 -std=c++17 black_hole_3d.cpp -o black_hole_3d ^
//        -lglfw3 -lglew32 -lopengl32
// ============================================================

#include <GL/glew.h>
#include <GLFW/glfw3.h>

#include <cstdio>
#include <cmath>
#include <cstring>
#include <iostream>
#include <string>

// ------------------- Шейдеры ---------------------------------
static const char* VERT_SRC = R"GLSL(
#version 330 core
layout(location = 0) in vec2 aPos;
out vec2 vUV;
void main() {
    vUV = aPos * 0.5 + 0.5;
    gl_Position = vec4(aPos, 0.0, 1.0);
}
)GLSL";

static const char* FRAG_SRC = R"GLSL(
#version 330 core

in vec2 vUV;
out vec4 fragColor;

uniform vec2  uResolution;
uniform float uTime;
uniform vec3  uCamPos;
uniform vec3  uCamForward;
uniform vec3  uCamRight;
uniform vec3  uCamUp;
uniform float uFov;
uniform float uDiskIn;
uniform float uDiskOut;
uniform float uExposure;

const float RS     = 1.0;
const float B_CRIT = 2.598076211353316;

// ------- хэш / звёздный фон ---------------------------------
uint hashU(uint x) {
    x ^= x >> 16; x *= 0x7feb352dU;
    x ^= x >> 15; x *= 0x846ca68bU;
    x ^= x >> 16;
    return x;
}
float hash01(int a, int b, int c) {
    uint h = hashU(uint(a)*73856093u ^ uint(b)*19349663u ^ uint(c)*83492791u);
    return float(h) * (1.0 / 4294967296.0);
}

vec3 starField(vec3 d) {
    const float S = 60.0;
    vec3 p = d * S;
    ivec3 iv = ivec3(floor(p));
    float best = 0.0;
    for (int dx = -1; dx <= 1; ++dx)
    for (int dy = -1; dy <= 1; ++dy)
    for (int dz = -1; dz <= 1; ++dz) {
        ivec3 c = iv + ivec3(dx, dy, dz);
        if (hash01(c.x, c.y, c.z) > 0.08) continue;
        vec3 cc = vec3(c) + vec3(
            hash01(c.x + 11, c.y +  3, c.z +  5),
            hash01(c.x +  7, c.y + 13, c.z +  2),
            hash01(c.x +  1, c.y + 17, c.z + 19));
        float dist = length(p - cc);
        float rad  = 0.06 + 0.12 * hash01(c.x + 23, c.y + 29, c.z + 31);
        if (dist < rad) {
            float v = 1.0 - dist / rad;
            best = max(best, v * v * (0.15 + 0.85 * hash01(c.x + 37, c.y + 41, c.z + 43)));
        }
    }
    return vec3(best, best * 0.93, best * 0.85) + vec3(0.0008, 0.0010, 0.0020);
}

// ------- цвет чёрного тела ----------------------------------
vec3 blackbodyRGB(float T) {
    T = clamp(T, 1000.0, 40000.0);
    float t = T / 100.0;
    float r, g, b;
    if (t <= 66.0) {
        r = 255.0;
        g = 99.4708025861 * log(t) - 161.1195681661;
    } else {
        r = 329.698727446 * pow(t - 60.0, -0.1332047592);
        g = 288.1221695283 * pow(t - 60.0, -0.0755148492);
    }
    if      (t >= 66.0) b = 255.0;
    else if (t <= 19.0) b = 0.0;
    else                b = 138.5177312231 * log(t - 10.0) - 305.0447927307;
    r = clamp(r, 0.0, 255.0) / 255.0;
    g = clamp(g, 0.0, 255.0) / 255.0;
    b = clamp(b, 0.0, 255.0) / 255.0;
    return vec3(pow(r, 2.2), pow(g, 2.2), pow(b, 2.2));
}

// ------- трассировка фотона в метрике Шварцшильда ------------
vec3 traceRay(vec3 pos, vec3 dir) {
    float r0 = length(pos);
    vec3 Lv  = cross(pos, dir);
    float b  = length(Lv);

    if (b < 1e-5) {
        if (dot(dir, normalize(pos)) < 0.0) return vec3(0.0);
        return starField(dir);
    }

    vec3 e1 = pos / r0;
    vec3 e2 = cross(Lv / b, e1);
    float radial = dot(dir, e1);

    if (b < B_CRIT && radial < 0.0) return vec3(0.0);

    float u = 1.0 / r0;
    float w = sqrt(max(0.0, 1.0/(b*b) - u*u*(1.0 - u)));
    if (radial > 0.0) w = -w;

    float phi = 0.0;
    float yc  = (e1.y * cos(phi) + e2.y * sin(phi)) / u;
    const float dphi = 0.02;
    float uEsc = 1.0 / (2.0 * r0);

    for (int i = 0; i < 1500; ++i) {
        float k1u = w,                  k1w = -u + 1.5*u*u;
        float u2  = u + 0.5*dphi*k1u;
        float k2u = w + 0.5*dphi*k1w,   k2w = -u2 + 1.5*u2*u2;
        float u3  = u + 0.5*dphi*k2u;
        float k3u = w + 0.5*dphi*k2w,   k3w = -u3 + 1.5*u3*u3;
        float u4  = u + dphi*k3u;
        float k4u = w + dphi*k3w,       k4w = -u4 + 1.5*u4*u4;

        float un   = u   + (dphi/6.0)*(k1u + 2.0*k2u + 2.0*k3u + k4u);
        float wn   = w   + (dphi/6.0)*(k1w + 2.0*k2w + 2.0*k3w + k4w);
        float phiN = phi + dphi;

        if (un >= 1.0) return vec3(0.0);

        float yn = (e1.y * cos(phiN) + e2.y * sin(phiN)) / un;
        if (yc * yn < 0.0) {
            float t  = yc / (yc - yn);
            float uc = u + t * (un - u);
            float rc = 1.0 / uc;
            if (rc > uDiskIn && rc < uDiskOut) {
                float phc = phi + t * dphi;
                vec3 diskPos = (e1 * cos(phc) + e2 * sin(phc)) * rc;
                float wc = w + t * (wn - w);
                float drdphi = -wc / (uc * uc);
                float c1 = drdphi * cos(phc) - rc * sin(phc);
                float c2 = drdphi * sin(phc) + rc * cos(phc);
                vec3 diskDir = normalize(e1 * c1 + e2 * c2);

                float M     = 0.5;
                float beta  = min(sqrt(M / rc), 0.9);
                float gamma = 1.0 / sqrt(1.0 - beta*beta);
                vec3 axis   = vec3(0.0, 1.0, 0.0);
                vec3 vdir   = normalize(cross(axis, diskPos));
                vec3 khat   = -diskDir;
                float delta = 1.0 / (gamma * (1.0 - beta * dot(vdir, khat)));
                float grav  = sqrt(max(0.0, 1.0 - RS / rc));
                float g     = delta * grav;
                float T     = 11000.0 * pow(uDiskIn / rc, 0.75) * g;
                vec3  col   = blackbodyRGB(T);
                float I     = pow(uDiskIn / rc, 2.0) * pow(g, 4.0);
                return col * I;
            }
        }

        u = un; w = wn; phi = phiN; yc = yn;

        if (u < uEsc && w < 0.0) {
            float r      = 1.0 / u;
            float drdphi = -w / (u * u);
            float c1 = drdphi * cos(phi) - r * sin(phi);
            float c2 = drdphi * sin(phi) + r * cos(phi);
            return starField(normalize(e1 * c1 + e2 * c2));
        }
    }
    return vec3(0.0);
}

void main() {
    vec2 uv = vUV * 2.0 - 1.0;
    uv.x *= uResolution.x / uResolution.y;
    uv   *= uFov;

    vec3 dir = normalize(uCamForward + uCamRight * uv.x + uCamUp * uv.y);
    vec3 col = traceRay(uCamPos, dir);
    col *= uExposure;

    col = col / (1.0 + col);
    col = pow(col, vec3(1.0 / 2.2));

    fragColor = vec4(col, 1.0);
}
)GLSL";

// ------------------- вспомогательные -------------------------
static GLuint compileShader(GLenum type, const char* src) {
    GLuint s = glCreateShader(type);
    glShaderSource(s, 1, &src, nullptr);
    glCompileShader(s);
    GLint ok = 0;
    glGetShaderiv(s, GL_COMPILE_STATUS, &ok);
    if (!ok) {
        char log[4096];
        glGetShaderInfoLog(s, sizeof(log), nullptr, log);
        std::fprintf(stderr, "Ошибка компиляции шейдера:\n%s\n", log);
    }
    return s;
}

static GLuint makeProgram() {
    GLuint v = compileShader(GL_VERTEX_SHADER,   VERT_SRC);
    GLuint f = compileShader(GL_FRAGMENT_SHADER, FRAG_SRC);
    GLuint p = glCreateProgram();
    glAttachShader(p, v);
    glAttachShader(p, f);
    glLinkProgram(p);
    GLint ok = 0;
    glGetProgramiv(p, GL_LINK_STATUS, &ok);
    if (!ok) {
        char log[4096];
        glGetProgramInfoLog(p, sizeof(log), nullptr, log);
        std::fprintf(stderr, "Ошибка линковки программы:\n%s\n", log);
    }
    glDeleteShader(v);
    glDeleteShader(f);
    return p;
}

// ------------------- состояние камеры ------------------------
struct Camera {
    float pos[3]  = {0.0f, 3.0f, 24.0f};
    float yaw = 0.0f;
    float pitch   = -0.12f;
    float fov     = 0.30f;
    float speed   = 6.0f;
};

static Camera g_cam;

static void getBasis(const Camera& c, float fwd[3], float right[3], float up[3]) {
    float cp = std::cos(c.pitch), sp = std::sin(c.pitch);
    float cy = std::cos(c.yaw),   sy = std::sin(c.yaw);
    fwd[0] = cp * sy;
    fwd[1] = sp;
    fwd[2] = -cp * cy;

    float wx = 0.0f, wy = 1.0f, wz = 0.0f;
    right[0] =  fwd[1]*wz - fwd[2]*wy;
    right[1] =  fwd[2]*wx - fwd[0]*wz;
    right[2] =  fwd[0]*wy - fwd[1]*wx;
    float rl = std::sqrt(right[0]*right[0] + right[1]*right[1] + right[2]*right[2]);
    if (rl > 1e-6f) { right[0]/=rl; right[1]/=rl; right[2]/=rl; }

    up[0] = right[1]*fwd[2] - right[2]*fwd[1];
    up[1] = right[2]*fwd[0] - right[0]*fwd[2];
    up[2] = right[0]*fwd[1] - right[1]*fwd[0];
}

static bool  g_mouseDown = false;
static double g_lastX = 0.0, g_lastY = 0.0;

static void mouseButtonCB(GLFWwindow*, int button, int action, int) {
    if (button == GLFW_MOUSE_BUTTON_LEFT)
        g_mouseDown = (action == GLFW_PRESS);
}

static void cursorCB(GLFWwindow*, double x, double y) {
    if (g_mouseDown) {
        double dx = x - g_lastX;
        double dy = y - g_lastY;
        g_cam.yaw   += (float)dx * 0.003f;
        g_cam.pitch -= (float)dy * 0.003f;
        if (g_cam.pitch >  1.55f) g_cam.pitch =  1.55f;
        if (g_cam.pitch < -1.55f) g_cam.pitch = -1.55f;
    }
    g_lastX = x; g_lastY = y;
}

static void scrollCB(GLFWwindow*, double, double yoff) {
    g_cam.fov *= (yoff > 0) ? 0.92f : 1.08f;
    if (g_cam.fov < 0.05f) g_cam.fov = 0.05f;
    if (g_cam.fov > 1.20f) g_cam.fov = 1.20f;
}

static void updateCamera(GLFWwindow* w, float dt) {
    float fwd[3], right[3], up[3];
    getBasis(g_cam, fwd, right, up);

    float sp = g_cam.speed * dt;
    if (glfwGetKey(w, GLFW_KEY_LEFT_SHIFT) == GLFW_PRESS) sp *= 4.0f;

    if (glfwGetKey(w, GLFW_KEY_W) == GLFW_PRESS) for (int i=0;i<3;i++) g_cam.pos[i] += fwd[i]*sp;
    if (glfwGetKey(w, GLFW_KEY_S) == GLFW_PRESS) for (int i=0;i<3;i++) g_cam.pos[i] -= fwd[i]*sp;
    if (glfwGetKey(w, GLFW_KEY_D) == GLFW_PRESS) for (int i=0;i<3;i++) g_cam.pos[i] += right[i]*sp;
    if (glfwGetKey(w, GLFW_KEY_A) == GLFW_PRESS) for (int i=0;i<3;i++) g_cam.pos[i] -= right[i]*sp;
    if (glfwGetKey(w, GLFW_KEY_SPACE) == GLFW_PRESS)         g_cam.pos[1] += sp;
    if (glfwGetKey(w, GLFW_KEY_LEFT_CONTROL) == GLFW_PRESS)  g_cam.pos[1] -= sp;

    // не даём улететь внутрь горизонта
    float r = std::sqrt(g_cam.pos[0]*g_cam.pos[0] +
                        g_cam.pos[1]*g_cam.pos[1] +
                        g_cam.pos[2]*g_cam.pos[2]);
    if (r < 2.0f) {
        float k = 2.0f / (r + 1e-6f);
        g_cam.pos[0] *= k; g_cam.pos[1] *= k; g_cam.pos[2] *= k;
    }
}

// ------------------- main ------------------------------------
int main() {
    if (!glfwInit()) {
        std::fprintf(stderr, "Не удалось инициализировать GLFW\n");
        return 1;
    }

    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 3);
    glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);
    glfwWindowHint(GLFW_SAMPLES, 0);

    GLFWwindow* window = glfwCreateWindow(1280, 800,
        "Чёрная дыра — 3D (WASD + мышь, Shift — ускорение, колесо — зум)",
        nullptr, nullptr);
    if (!window) {
        std::fprintf(stderr, "Не удалось создать окно\n");
        glfwTerminate();
        return 1;
    }
    glfwMakeContextCurrent(window);
    glfwSwapInterval(1);

    glewExperimental = GL_TRUE;
    GLenum gerr = glewInit();
    if (gerr != GLEW_OK) {
        std::fprintf(stderr, "GLEW: %s\n", glewGetErrorString(gerr));
        return 1;
    }
    glGetError(); // сбрасываем безобидную ошибку GLEW

    glfwSetMouseButtonCallback(window, mouseButtonCB);
    glfwSetCursorPosCallback  (window, cursorCB);
    glfwSetScrollCallback     (window, scrollCB);

    std::printf("OpenGL: %s\n", glGetString(GL_VERSION));
    std::printf("Renderer: %s\n\n", glGetString(GL_RENDERER));
    std::printf("Управление:\n");
    std::printf("  ЛКМ + мышь — обзор\n");
    std::printf("  W A S D    — полёт\n");
    std::printf("  Space/Ctrl — вверх / вниз\n");
    std::printf("  Shift      — ускорение x4\n");
    std::printf("  Колесо     — зум\n");
    std::printf("  Esc        — выход\n");

    GLuint prog = makeProgram();
    glUseProgram(prog);

    // fullscreen quad
    float verts[] = {
        -1.0f, -1.0f,
         1.0f, -1.0f,
        -1.0f,  1.0f,
         1.0f,  1.0f,
    };
    GLuint vao, vbo;
    glGenVertexArrays(1, &vao);
    glBindVertexArray(vao);
    glGenBuffers(1, &vbo);
    glBindBuffer(GL_ARRAY_BUFFER, vbo);
    glBufferData(GL_ARRAY_BUFFER, sizeof(verts), verts, GL_STATIC_DRAW);
    glEnableVertexAttribArray(0);
    glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, 0, (void*)0);

    GLint locRes     = glGetUniformLocation(prog, "uResolution");
    GLint locTime    = glGetUniformLocation(prog, "uTime");
    GLint locCamPos  = glGetUniformLocation(prog, "uCamPos");
    GLint locCamFwd  = glGetUniformLocation(prog, "uCamForward");
    GLint locCamR    = glGetUniformLocation(prog, "uCamRight");
    GLint locCamU    = glGetUniformLocation(prog, "uCamUp");
    GLint locFov     = glGetUniformLocation(prog, "uFov");
    GLint locDiskIn  = glGetUniformLocation(prog, "uDiskIn");
    GLint locDiskOut = glGetUniformLocation(prog, "uDiskOut");
    GLint locExp     = glGetUniformLocation(prog, "uExposure");

    double prevTime = glfwGetTime();

    while (!glfwWindowShouldClose(window)) {
        double now = glfwGetTime();
        float dt = float(now - prevTime);
        prevTime = now;

        if (glfwGetKey(window, GLFW_KEY_ESCAPE) == GLFW_PRESS)
            glfwSetWindowShouldClose(window, GLFW_TRUE);

        updateCamera(window, dt);

        int fbw, fbh;
        glfwGetFramebufferSize(window, &fbw, &fbh);

        float fwd[3], right[3], up[3];
        getBasis(g_cam, fwd, right, up);

        glViewport(0, 0, fbw, fbh);
        glClear(GL_COLOR_BUFFER_BIT);

        glUseProgram(prog);
        glUniform2f(locRes, (float)fbw, (float)fbh);
        glUniform1f(locTime, (float)now);
        glUniform3f(locCamPos, g_cam.pos[0], g_cam.pos[1], g_cam.pos[2]);
        glUniform3f(locCamFwd, fwd[0],   fwd[1],   fwd[2]);
        glUniform3f(locCamR,   right[0], right[1], right[2]);
        glUniform3f(locCamU,   up[0],    up[1],    up[2]);
        glUniform1f(locFov,    g_cam.fov);
        glUniform1f(locDiskIn,  3.0f);
        glUniform1f(locDiskOut, 15.0f);
        glUniform1f(locExp,     1.0f);

        glBindVertexArray(vao);
        glDrawArrays(GL_TRIANGLE_STRIP, 0, 4);

        glfwSwapBuffers(window);
        glfwPollEvents();
    }

    glDeleteProgram(prog);
    glDeleteBuffers(1, &vbo);
    glDeleteVertexArrays(1, &vao);
    glfwDestroyWindow(window);
    glfwTerminate();
    return 0;
}