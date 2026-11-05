"""Замер времени синтеза для §4.3 (блок С): АДСИ и шумовые функции при одинаковом размере карты.

Запуск на ПК автора (нужны только numpy, scipy, PyWavelets):
    python time_c.py            — полный замер: 256, 512, 1024; прогрев + 15 замеров, медиана (перебор зёрен — 3 замера)
    python time_c.py --quick    — быстрая проверка: только 256, 3 замера
Результат: таблица в консоли и time_c.json (версии Python, NumPy, SciPy, PyWavelets и процессор — там же).

Что замеряется (одно изображение M×M, бесшовно):
  АДСИ, один запуск           — seed-массив 8×8, путь из одних A, J = log2(M/8), db8 (как в галерее рис. 4.11);
  АДСИ, сумма масштабов       — тот же рисунок в узлах уровней J, J−1, J−2 (повтор 1×1, 2×2, 4×4), веса 1, ½, ¼;
  АДСИ, рисунок и второй      — рисунок 8×8 на J и рисунок 16×16 на J−1 (вес 1);
  шум Перлина / симплексный шум, 1 октава и fBm 5 октав (g = 0,5, L = 2) — как в таблице §4.3;
  шум с маской                — рисунок задачи 8×8, сглаженный кубическим сплайном до M×M, + β·fBm шума Перлина (5 октав);
  перебор 100 зёрен           — 100 карт fBm шума Перлина (5 октав) и поиск лучшего циклического сдвига по корреляции
                                с маской (БПФ); так в С3 подбиралась карта, ближе всего к рисунку задачи.
Шумы реализованы средствами NumPy, ОПДВП — PyWavelets (скомпилированный код): результат характеризует порядок величин.
"""
import json, platform, sys, time
import numpy as np
import pywt
import scipy
from scipy.ndimage import zoom

FADE = lambda u: u ** 3 * (u * (u * 6 - 15) + 10)


def perlin(M, G, rng):
    """Бесшовный градиентный шум Перлина, решётка G×G на карту M×M."""
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


def simplex(M, scale, rng):
    """Симплексный шум 2D (Gustavson 2005), векторизовано."""
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


def fbm_perlin(M, rng, octaves=5, g=.5, G0=None):
    G0 = G0 or max(2, M // 32)
    return sum(g ** k * perlin(M, G0 * 2 ** k, rng) for k in range(octaves))


def fbm_simplex(M, rng, octaves=5, g=.5):
    s0 = 1 / 32
    return sum(g ** k * simplex(M, s0 * 2 ** k, rng) for k in range(octaves))


def synth(seed, J, w='db8'):
    """АДСИ: путь из одних A, J шагов ОПДВП (матрица — в узел A, остальные нулевые; periodization)."""
    x = np.asarray(seed, float)
    for _ in range(J):
        z = np.zeros_like(x)
        x = pywt.idwt2((x, (z, z, z)), w, mode='periodization')
    return x


def adsi_scales(seed, J, w='db8'):
    M = seed.shape[0] * 2 ** J
    out = synth(seed, J, w)
    for k, wgt in ((1, .5), (2, .25)):
        part = synth(seed, J - k, w)
        out = out + wgt * np.tile(part, (M // part.shape[0],) * 2)
    return out


def adsi_second(seed, seed16, J, w='db8'):
    return synth(seed, J, w) + synth(seed16, J - 1, w)


def mask_noise(drawing, M, rng, beta=.4):
    base = zoom(drawing, M // drawing.shape[0], order=3, mode='grid-wrap', grid_mode=True)
    return base + beta * fbm_perlin(M, rng)


def seed_search(drawing, M, n=100):
    target = zoom(drawing, M // drawing.shape[0], order=3, mode='grid-wrap', grid_mode=True)
    T = target - target.mean(); FT = np.conj(np.fft.fft2(T)); best = -2.0
    for k in range(n):
        m = fbm_perlin(M, np.random.default_rng(1000 + k)); A = m - m.mean()
        cc = np.real(np.fft.ifft2(np.fft.fft2(A) * FT)) / (np.linalg.norm(A) * np.linalg.norm(T))
        best = max(best, float(cc.max()))
    return best


def timed(fn, reps):
    fn()                                     # прогрев
    ts = []
    for _ in range(reps):
        t0 = time.perf_counter(); fn(); ts.append(time.perf_counter() - t0)
    return 1000 * float(np.median(ts))


def main():
    quick = '--quick' in sys.argv
    sizes = [256] if quick else [256, 512, 1024]
    reps = 3 if quick else 15
    rng0 = np.random.default_rng(1)
    seed = np.round(rng0.uniform(0, 1, (8, 8)) * 4) / 4
    seed16 = np.round(rng0.uniform(0, 1, (16, 16)) * 4) / 4
    rows = [('АДСИ, один запуск', lambda M, J: synth(seed, J)),
            ('АДСИ, сумма масштабов (3 узла)', lambda M, J: adsi_scales(seed, J)),
            ('АДСИ, рисунок и второй рисунок', lambda M, J: adsi_second(seed, seed16, J)),
            ('Шум Перлина, 1 октава', lambda M, J: perlin(M, max(2, M // 32), np.random.default_rng(2))),
            ('Симплексный шум, 1 октава', lambda M, J: simplex(M, 1 / 32, np.random.default_rng(3))),
            ('Шум Перлина, fBm, 5 октав', lambda M, J: fbm_perlin(M, np.random.default_rng(4))),
            ('Симплексный шум, fBm, 5 октав', lambda M, J: fbm_simplex(M, np.random.default_rng(5))),
            ('Шум с маской (fBm Перлина, 5 октав)', lambda M, J: mask_noise(seed, M, np.random.default_rng(6))),
            ('Перебор 100 зёрен (fBm Перлина, 5 октав)', lambda M, J: seed_search(seed, M))]
    res = {}
    for name, fn in rows:
        res[name] = {}
        for M in sizes:
            J = int(np.log2(M // 8))
            r = 3 if name.startswith('Перебор') else reps
            res[name][M] = round(timed(lambda: fn(M, J), r), 2)
            print(f'{name:45s} {M:5d}  {res[name][M]:10.2f} мс', flush=True)
    env = {'python': platform.python_version(), 'numpy': np.__version__, 'scipy': scipy.__version__,
           'pywavelets': pywt.__version__, 'processor': platform.processor() or platform.machine(),
           'platform': platform.platform(), 'reps': reps, 'reps_seed_search': 3, 'quick': quick}
    json.dump({'env': env, 'ms': res}, open('time_c.json', 'w'), ensure_ascii=False, indent=1)
    print('\nМетод'.ljust(46) + ''.join(f'{M}×{M}'.rjust(12) for M in sizes))
    for name in res:
        print(name.ljust(45) + ''.join(f'{res[name][M]:12.2f}' for M in sizes))
    print('\nСреда:', env)


if __name__ == '__main__':
    main()
