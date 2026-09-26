// ============================================================
//  Модель чёрной дыры (Шварцшильд) — обратная трассировка
//  нулевых геодезических + аккреционный диск.
//
//  Единицы: G = c = 1, радиус Шварцшильда r_s = 1
//
//  Сборка:  g++ -O2 -std=c++17 -fopenmp blackhole.cpp -o blackhole
//  Запуск:  ./blackhole   ->  blackhole.ppm
//  (открыть: GIMP, IrfanView, или  convert blackhole.ppm blackhole.png)
// ============================================================

#include <cstdio>
#include <cmath>
#include <cstdint>
#include <vector>
#include <algorithm>

#ifdef _OPENMP
#include <omp.h>
#endif

// ---------------------- векторы ------------------------------
struct V3 {
    double x, y, z;
    V3(double x_ = 0.0, double y_ = 0.0, double z_ = 0.0) : x(x_), y(y_), z(z_) {}
};

inline V3 operator+(V3 a, V3 b) { return {a.x + b.x, a.y + b.y, a.z + b.z}; }
inline V3 operator-(V3 a, V3 b) { return {a.x - b.x, a.y - b.y, a.z - b.z}; }
inline V3 operator*(V3 a, double s) { return {a.x * s, a.y * s, a.z * s}; }
inline V3 operator*(double s, V3 a) { return a * s; }
inline double dot(V3 a, V3 b) { return a.x*b.x + a.y*b.y + a.z*b.z; }
inline V3 cross(V3 a, V3 b) {
    return {a.y*b.z - a.z*b.y, a.z*b.x - a.x*b.z, a.x*b.y - a.y*b.x};
}
inline double len(V3 a) { return std::sqrt(dot(a, a)); }
inline V3 norm(V3 a) { double l = len(a); return l > 1e-15 ? a * (1.0/l) : V3{0,0,0}; }

// ---------------------- параметры ----------------------------
const double RS       = 1.0;         // радиус Шварцшильда
const double DISK_IN  = 3.0 * RS;    // ISCO = 6GM/c^2 = 3 r_s
const double DISK_OUT = 15.0 * RS;

const int    IMG_W   = 800;
const int    IMG_H   = 600;
const int    SAMPLES = 2;            // 2 => 2x2 суперсэмплинг (1 => вчетверо быстрее)
const double FOV_Y   = 0.30;         // tg половины вертикального угла обзора
const double CAM_D   = 24.0 * RS;    // расстояние камеры
const double CAM_H   = 3.0 * RS;     // высота камеры над плоскостью диска

// ============================================================
//                     Трассировка луча
// ============================================================
struct Hit {
    bool   captured = false;
    bool   disk     = false;
    double diskR    = 0.0;
    V3     diskPos;
    V3     diskDir;   // направление фотона в точке диска
    V3     outDir;    // направление вылета (для фона)
};

Hit traceRay(V3 pos, V3 dir)
{
    Hit h;

    V3     Lv = cross(pos, dir);
    double b  = len(Lv);
    double r0 = len(pos);

    // строго радиальный луч
    if (b < 1e-12) {
        if (dot(dir, norm(pos)) < 0.0) h.captured = true;
        else                            h.outDir   = dir;
        return h;
    }

    V3 e1 = pos * (1.0 / r0);          // базис плоскости орбиты
    V3 e2 = cross(Lv * (1.0 / b), e1);

    // Критический прицельный параметр: b_crit = (3*sqrt(3)/2) r_s
    const double B_CRIT = 2.598076211353316 * RS;
    double radial = dot(dir, e1);

    if (b < B_CRIT) {                  // такие лучи либо падают, либо улетают почти прямо
        if (radial < 0.0) h.captured = true;
        else              h.outDir   = dir;
        return h;
    }

    double u = 1.0 / r0;
    double w = std::sqrt(std::max(0.0, 1.0/(b*b) - u*u*(1.0 - RS*u)));
    if (radial > 0.0) w = -w;          // луч уходит наружу

    auto accel = [](double uu) { return -uu + 1.5 * RS * uu * uu; };

    const double dphi  = 0.02;
    const int    MAXST = 6000;
    const double uEsc  = 1.0 / (2.0 * r0);   // "улетел", когда r > 2*r0

    double phi = 0.0;
    double yc  = (e1.y * std::cos(phi) + e2.y * std::sin(phi)) / u;  // y-координата луча

    for (int step = 0; step < MAXST; ++step) {
        // --- шаг Рунге-Кутты 4 ---
        double k1u = w,                 k1w = accel(u);
        double k2u = w + 0.5*dphi*k1w,  k2w = accel(u + 0.5*dphi*k1u);
        double k3u = w + 0.5*dphi*k2w,  k3w = accel(u + 0.5*dphi*k2u);
        double k4u = w +     dphi*k3w,  k4w = accel(u +     dphi*k3u);

        double un   = u + (dphi/6.0)*(k1u + 2.0*k2u + 2.0*k3u + k4u);
        double wn   = w + (dphi/6.0)*(k1w + 2.0*k2w + 2.0*k3w + k4w);
        double phiN = phi + dphi;

        // --- упал под горизонт ---
        if (un >= 1.0 / RS) { h.captured = true; return h; }

        // --- пересечение плоскости диска ---
        double yn = (e1.y * std::cos(phiN) + e2.y * std::sin(phiN)) / un;
        if (yc * yn < 0.0) {
            double t  = yc / (yc - yn);
            double uc = u + t * (un - u);
            double rc = 1.0 / uc;
            if (rc > DISK_IN && rc < DISK_OUT) {
                double phc = phi + t * dphi;
                double r   = 1.0 / uc;

                h.disk    = true;
                h.diskR   = rc;
                h.diskPos = (e1 * std::cos(phc) + e2 * std::sin(phc)) * r;

                double wc     = w + t * (wn - w);
                double drdphi = -wc / (uc * uc);
                double c1 = drdphi * std::cos(phc) - r * std::sin(phc);
                double c2 = drdphi * std::sin(phc) + r * std::cos(phc);
                h.diskDir = norm(e1 * c1 + e2 * c2);
                return h;
            }
        }

        u = un; w = wn; phi = phiN; yc = yn;

        // --- улетел ---
        if (u < uEsc && w < 0.0) {
            double r      = 1.0 / u;
            double drdphi = -w / (u * u);
            double c1 = drdphi * std::cos(phi) - r * std::sin(phi);
            double c2 = drdphi * std::sin(phi) + r * std::cos(phi);
            h.outDir = norm(e1 * c1 + e2 * c2);
            return h;
        }
    }

    // закрутился у фотонной сферы — считаем захваченным
    h.captured = true;
    return h;
}

// ============================================================
//                    Цвет / фон / диск
// ============================================================
inline uint32_t hashU(uint32_t x) {
    x ^= x >> 16; x *= 0x7feb352dU;
    x ^= x >> 15; x *= 0x846ca68bU;
    x ^= x >> 16;
    return x;
}
inline double hash01(int a, int b, int c) {
    uint32_t h = hashU(uint32_t(a)*73856093u ^ uint32_t(b)*19349663u ^ uint32_t(c)*83492791u);
    return h * (1.0 / 4294967296.0);
}

// процедурное звёздное небо
V3 starField(V3 d)
{
    const double S = 60.0;
    V3 p = d * S;
    int ix = int(std::floor(p.x)), iy = int(std::floor(p.y)), iz = int(std::floor(p.z));

    double best = 0.0;
    for (int dx = -1; dx <= 1; ++dx)
    for (int dy = -1; dy <= 1; ++dy)
    for (int dz = -1; dz <= 1; ++dz) {
        int cx = ix + dx, cy = iy + dy, cz = iz + dz;
        if (hash01(cx, cy, cz) > 0.08) continue;

        V3 c{cx + hash01(cx + 11, cy +  3, cz +  5),
             cy + hash01(cx +  7, cy + 13, cz +  2),
             cz + hash01(cx +  1, cy + 17, cz + 19)};

        double dist = len(p - c);
        double rad  = 0.06 + 0.12 * hash01(cx + 23, cy + 29, cz + 31);
        if (dist < rad) {
            double v = 1.0 - dist / rad;
            best = std::max(best, v * v * (0.15 + 0.85 * hash01(cx + 37, cy + 41, cz + 43)));
        }
    }
    return V3{best, best*0.93, best*0.85} + V3{0.0008, 0.0010, 0.0020};
}

// приближение цвета абсолютно чёрного тела (T в кельвинах) -> линейный RGB
V3 blackbodyRGB(double T)
{
    T = std::clamp(T, 1000.0, 40000.0);
    double t = T / 100.0, r, g, b;

    if (t <= 66.0) {
        r = 255.0;
        g = 99.4708025861 * std::log(t) - 161.1195681661;
    } else {
        r = 329.698727446 * std::pow(t - 60.0, -0.1332047592);
        g = 288.1221695283 * std::pow(t - 60.0, -0.0755148492);
    }
    if      (t >= 66.0) b = 255.0;
    else if (t <= 19.0) b = 0.0;
    else                b = 138.5177312231 * std::log(t - 10.0) - 305.0447927307;

    r = std::clamp(r, 0.0, 255.0) / 255.0;
    g = std::clamp(g, 0.0, 255.0) / 255.0;
    b = std::clamp(b, 0.0, 255.0) / 255.0;

    // в линейное пространство
    return V3{std::pow(r, 2.2), std::pow(g, 2.2), std::pow(b, 2.2)};
}

// цвет пикселя (линейный)
V3 shade(V3 pos, V3 dir)
{
    Hit h = traceRay(pos, dir);

    if (h.captured) return V3{0,0,0};

    if (h.disk) {
        const double r = h.diskR;
        const double M = 0.5 * RS;                 // r_s = 2GM/c^2

        double beta  = std::min(std::sqrt(M / r), 0.9);   // кеплеровская скорость
        double gamma = 1.0 / std::sqrt(1.0 - beta*beta);

        V3 axis{0.0, 1.0, 0.0};                    // ось вращения диска
        V3 vdir = norm(cross(axis, h.diskPos));    // направление движения вещества
        V3 khat = h.diskDir * (-1.0);              // от диска к наблюдателю

        double delta = 1.0 / (gamma * (1.0 - beta * dot(vdir, khat)));  // доплер
        double grav  = std::sqrt(std::max(0.0, 1.0 - RS / r));          // гравитац. красное смещение
        double g     = delta * grav;

        double T   = 11000.0 * std::pow(DISK_IN / r, 0.75) * g;
        V3     col = blackbodyRGB(T);
        double I   = std::pow(DISK_IN / r, 2.0) * std::pow(g, 4.0);

        return col * I;
    }

    return starField(h.outDir) * 1.5;
}

// ============================================================
int main()
{
    const V3 camPos{0.0, CAM_H, CAM_D};
    const V3 target{0.0, 0.0, 0.0};

    V3 forward = norm(target - camPos);
    V3 right   = norm(cross(forward, V3{0,1,0}));
    V3 up      = cross(right, forward);

    const double aspect = double(IMG_W) / double(IMG_H);
    const double tanX = FOV_Y * aspect;
    const double tanY = FOV_Y;

    std::vector<unsigned char> img(size_t(IMG_W) * IMG_H * 3);

#ifdef _OPENMP
    #pragma omp parallel for schedule(dynamic, 4)
#endif
    for (int j = 0; j < IMG_H; ++j) {
        for (int i = 0; i < IMG_W; ++i) {
            V3 col{0,0,0};

            for (int sy = 0; sy < SAMPLES; ++sy)
            for (int sx = 0; sx < SAMPLES; ++sx) {
                double px = (2.0*(i + (sx + 0.5)/SAMPLES)/IMG_W - 1.0) * tanX;
                double py = (1.0 - 2.0*(j + (sy + 0.5)/SAMPLES)/IMG_H) * tanY;
                V3 dir = norm(forward + right*px + up*py);
                col = col + shade(camPos, dir);
            }
            col = col * (1.0 / (SAMPLES * SAMPLES));

            // тон-маппинг + гамма
            double rr = col.x / (1.0 + col.x);
            double gg = col.y / (1.0 + col.y);
            double bb = col.z / (1.0 + col.z);
            rr = std::pow(std::max(0.0, rr), 1.0/2.2);
            gg = std::pow(std::max(0.0, gg), 1.0/2.2);
            bb = std::pow(std::max(0.0, bb), 1.0/2.2);

            size_t idx = (size_t(j) * IMG_W + i) * 3;
            img[idx+0] = (unsigned char)(std::min(1.0, rr) * 255.0 + 0.5);
            img[idx+1] = (unsigned char)(std::min(1.0, gg) * 255.0 + 0.5);
            img[idx+2] = (unsigned char)(std::min(1.0, bb) * 255.0 + 0.5);
        }
    }

    FILE* f = std::fopen("blackhole.ppm", "wb");
    if (!f) { std::printf("Не удалось создать файл\n"); return 1; }
    std::fprintf(f, "P6\n%d %d\n255\n", IMG_W, IMG_H);
    std::fwrite(img.data(), 1, img.size(), f);
    std::fclose(f);

    std::printf("Готово: blackhole.ppm (%dx%d)\n", IMG_W, IMG_H);
    return 0;
}