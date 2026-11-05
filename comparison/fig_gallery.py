"""Глава 4, раздел 4.4, шаг 5 (23.10): характерные результаты семи методов, оценивавшихся в анкетном опросе.

Галерея собрана заново (решение 22.10: прежнего скрипта нет). Карты 256×256, бесшовные, каждая приведена к [0, 1] один раз.
Параметры шумов — как в замере времени (time_c.py, таблица времени синтеза): метод fBm — 5 октав, g = 0,5, L = 2,
крупнейшая октава — 8 периодов на карту (32 пикс.).
  а — числовой шум (значения в узлах решетки, интерполяция 6t^5 − 15t^4 + 10t^3), fBm;
  б — шум Перлина (градиентный), fBm;
  в — симплексный шум (Gustavson 2005), fBm; масштаб 0,6 от решетки шума Перлина (как в fig_c0.py), чтобы крупность была сопоставима;
  г — вейвлет-шум (Cook, DeRose 2005), fBm по 5 полосам;
  д — алгоритм DS, показатель Херста H = 0,8 (как в прежней галерее);
  е — шум Уорли, расстояние до ближайшего центра, n_c = 24 центра (как в прежней галерее);
  ж — АДСИ: сумма масштабов — свободная форма с плавными переходами (seed-массив 8×8) в узлах пяти уровней
      (пути AAAAA … A) с весами 1; 0,5; 0,25; 0,125; 0,0625, coif3, учет сдвига (отзыв на шаг 5: результат того же рода,
      что у шумов с fBm; прежний вариант — типовая задача «холмистая местность» при J = 5).
Шумы — реализация Claude (fig_c_plan.py, fig_c0.py, time_c.py этапа 2), обобщены на размер M.
Выход: out/methods_gallery.png (в главу — img/ch4_methods_gallery.png).
"""
import json, os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import LightSource
import fig_style as S
from c_lib import synth, align
from forms import FORMS

S.apply()
os.makedirs('out', exist_ok=True)
M, OCT, G0 = 256, 5, 8
FADE = lambda u: u ** 3 * (u * (u * 6 - 15) + 10)


def value_noise(G, rng):
    V = rng.uniform(-1, 1, (G, G))
    t = np.arange(M) * G / M
    i0 = np.floor(t).astype(int); f = FADE(t - i0)
    a = V[np.ix_(i0 % G, i0 % G)]; b = V[np.ix_(i0 % G, (i0 + 1) % G)]
    c = V[np.ix_((i0 + 1) % G, i0 % G)]; d = V[np.ix_((i0 + 1) % G, (i0 + 1) % G)]
    fy, fx = f[:, None], f[None, :]
    return (a * (1 - fx) + b * fx) * (1 - fy) + (c * (1 - fx) + d * fx) * fy


def perlin(G, rng):
    ang = rng.uniform(0, 2 * np.pi, (G, G)); gx, gy = np.cos(ang), np.sin(ang)
    t = np.arange(M) * G / M
    X, Y = np.meshgrid(t, t)
    x0, y0 = np.floor(X).astype(int), np.floor(Y).astype(int)
    fx, fy = X - x0, Y - y0
    def dot(ix, iy, dx, dy):
        ix, iy = ix % G, iy % G
        return gx[iy, ix] * dx + gy[iy, ix] * dy
    n00 = dot(x0, y0, fx, fy); n10 = dot(x0 + 1, y0, fx - 1, fy)
    n01 = dot(x0, y0 + 1, fx, fy - 1); n11 = dot(x0 + 1, y0 + 1, fx - 1, fy - 1)
    u, v = FADE(fx), FADE(fy)
    return (n00 * (1 - u) + n10 * u) * (1 - v) + (n01 * (1 - u) + n11 * u) * v


def simplex(scale, rng):
    p = rng.permutation(256); perm = np.concatenate([p, p, p])
    ang = np.arange(8) * np.pi / 4; grad = np.stack([np.cos(ang), np.sin(ang)], 1)
    X, Y = np.meshgrid(np.arange(M) * scale, np.arange(M) * scale)
    F2, G2 = 0.5 * (np.sqrt(3) - 1), (3 - np.sqrt(3)) / 6
    s = (X + Y) * F2
    i, j = np.floor(X + s).astype(int), np.floor(Y + s).astype(int)
    t = (i + j) * G2
    x0, y0 = X - (i - t), Y - (j - t)
    i1 = (x0 > y0).astype(int); j1 = 1 - i1
    x1, y1 = x0 - i1 + G2, y0 - j1 + G2
    x2, y2 = x0 - 1 + 2 * G2, y0 - 1 + 2 * G2
    ii, jj = i & 255, j & 255
    out = 0
    for dx, dy, gi in ((x0, y0, perm[ii + perm[jj]]), (x1, y1, perm[ii + i1 + perm[jj + j1]]),
                       (x2, y2, perm[ii + 1 + perm[jj + 1]])):
        g = grad[gi % 8]; tt = 0.5 - dx ** 2 - dy ** 2
        out = out + np.where(tt > 0, tt ** 4 * (g[..., 0] * dx + g[..., 1] * dy), 0)
    return 70 * out


A_COEF = np.array([0.000334, -0.001528, 0.000410, 0.003545, -0.000938, -0.008233, 0.002172, 0.019120,
                   -0.005040, -0.044412, 0.011655, 0.103311, -0.025936, -0.243780, 0.033979, 0.655340,
                   0.655340, 0.033979, -0.243780, -0.025936, 0.103311, 0.011655, -0.044412, -0.005040,
                   0.019120, 0.002172, -0.008233, -0.000938, 0.003546, 0.000410, -0.001528, 0.000334])
P_COEF = np.array([0.25, 0.75, 0.75, 0.25])


def _down(a, axis):
    a = np.moveaxis(a, axis, 0); n = a.shape[0]
    out = np.zeros((n // 2,) + a.shape[1:])
    for i in range(n // 2):
        for k in range(2 * i - 16, 2 * i + 16):
            out[i] += A_COEF[k - 2 * i + 16] * a[k % n]
    return np.moveaxis(out, 0, axis)


def _up(a, axis):
    a = np.moveaxis(a, axis, 0); h = a.shape[0]; n = 2 * h
    out = np.zeros((n,) + a.shape[1:])
    for i in range(n):
        for k in (i // 2, i // 2 + 1):
            out[i] += P_COEF[2 + i - 2 * k] * a[k % h]
    return np.moveaxis(out, 0, axis)


def wavelet_band(n, rng):
    """Одна полоса вейвлет-шума: тайл n×n (Cook, DeRose 2005), отсчеты квадратичным B-сплайном на карту M×M."""
    r = rng.standard_normal((n, n))
    nz = r - _up(_up(_down(_down(r, 0), 1), 1), 0)
    off = n // 2 + (1 if (n // 2) % 2 == 0 else 0)
    tile = nz + np.roll(nz, (off, off), axis=(0, 1))
    p = (np.arange(M) + 0.5) * n / M
    mid = np.ceil(p - 0.5).astype(int); t = mid - (p - 0.5)
    w = np.stack([t ** 2 / 2, 1 - t ** 2 / 2 - (1 - t) ** 2 / 2, (1 - t) ** 2 / 2])
    out = np.zeros((M, M))
    for a in range(3):
        for b in range(3):
            out += np.outer(w[a], w[b]) * tile[np.ix_((mid - 1 + a) % n, (mid - 1 + b) % n)]
    return out


def diamond_square(H, rng, n=M):
    a = np.zeros((n, n)); step = n; amp = 1.0
    a[0, 0] = rng.uniform(-1, 1)
    while step > 1:
        h = step // 2
        for y in range(0, n, step):
            for x in range(0, n, step):
                m = (a[y, x] + a[y, (x + step) % n] + a[(y + step) % n, x] + a[(y + step) % n, (x + step) % n]) / 4
                a[y + h, x + h] = m + rng.uniform(-amp, amp)
        for y in range(0, n, h):
            for x in range((y + h) % step, n, step):
                m = (a[(y - h) % n, x] + a[(y + h) % n, x] + a[y, (x - h) % n] + a[y, (x + h) % n]) / 4
                a[y, x] = m + rng.uniform(-amp, amp)
        step = h; amp *= 2 ** (-H)
    return a


def worley(nc, rng, n=M):
    c = rng.uniform(0, n, (nc, 2))
    Y, X = np.mgrid[0:n, 0:n] + 0.5
    d = np.full((n, n), np.inf)
    for cy, cx in c:
        dy = np.abs(Y - cy); dx = np.abs(X - cx)
        d = np.minimum(d, np.hypot(np.minimum(dy, n - dy), np.minimum(dx, n - dx)))
    return d


fbm = lambda f: sum(0.5 ** k * f(k) for k in range(OCT))
mm = lambda a: (a - a.min()) / (a.max() - a.min())


def adsi_scales(L=5, b='coif3'):
    """Сумма масштабов (отзыв на шаг 5, 23.10): свободная форма с плавными переходами (раздел 4.1) в узлах пяти уровней
    с весами 1, 1/2, ..., 1/16 — аналог метода fBm с 5 октавами; каждый уровень перенесен на место рисунка (учет сдвига)."""
    s = np.array(FORMS['other_smooth'][1], float)
    out = np.zeros((M, M))
    for k in range(L):
        part = synth(s, 'A' * (L - k), b); rep = M // part.shape[0]
        out += 0.5 ** k * align(np.tile(part, (rep, rep)), s, rep)[0]
    return out


rng = np.random.default_rng(20261023)
MAPS = [('а — числовой шум', fbm(lambda k: value_noise(G0 * 2 ** k, rng))),
        ('б — шум Перлина', fbm(lambda k: perlin(G0 * 2 ** k, rng))),
        ('в — симплексный шум', fbm(lambda k: simplex(0.6 * G0 / M * 2 ** k, rng))),
        ('г — вейвлет-шум', fbm(lambda k: wavelet_band(G0 * 2 ** k, rng))),
        ('д — алгоритм DS', diamond_square(0.8, rng)),
        ('е — шум Уорли', worley(24, rng)),
        ('ж — АДСИ', adsi_scales())]


def surf(ax, Z, title):
    st = 2                                     # 128×128 граней на 256×256 пикселей — для размера файла
    Z = Z[::st, ::st]; n = Z.shape[0]
    rgb = LightSource(315, 45).shade(Z, cmap=plt.get_cmap('Greys'), vmin=-0.6, vmax=1.5, vert_exag=12 * n / 64, blend_mode='soft')
    X, Y = np.meshgrid(np.arange(n), np.arange(n))
    ax.plot_surface(X, Y, Z, facecolors=rgb, rstride=1, cstride=1, linewidth=0, antialiased=False, shade=False)
    ax.view_init(**S.VIEW3D)
    ax.set_xlim(0, n - 1); ax.set_ylim(n - 1, 0); ax.set_zlim(-0.1, 1.6)
    ax.set_box_aspect((1, 1, 0.42)); ax.set_axis_off()
    ax.set_title(title, fontsize=12, pad=-2)


fig = plt.figure(figsize=(S.TEXT_W, S.TEXT_W * 0.56))
W4 = 0.25
for i, (title, m) in enumerate(MAPS):
    r, c = divmod(i, 4)
    x = c * W4 + (W4 / 2 if r == 1 else 0)          # нижний ряд из трех панелей — по центру
    ax = fig.add_axes([x, 0.49 - r * 0.49, W4, 0.44], projection='3d')
    surf(ax, mm(m), title)
fig.savefig('out/methods_gallery.png', dpi=S.DPI, facecolor='white')
print('out/methods_gallery.png')
