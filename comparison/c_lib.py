"""Блок С: функции, нужные для рисунков Р-С2 и Р-С3 (fig_c_work.py), собранные из скриптов блока С без изменений:
fig_c_plan.py (синтез, перенос, шум Перлина, fBm, mm, surf), fig_c1b.py (drawing, interp, corr), fig_c1s.py (radial_psd),
fig_c2.py (block, bands, spec_dist, fine_share, best_main, k1_details, fit_details), fig_c2d.py (fit_d2),
fig_c0b5.py (show_grid), fig_c3.py (a2, best_shift, N_SEEDS), fig_c3d.py (a2p, mask_half).
Полные скрипты с разметками — в проекте (claude/stage2_code/)."""
import numpy as np
import pywt
from scipy.ndimage import zoom
import matplotlib.pyplot as plt
from matplotlib.colors import LightSource
import fig_style as S

K, M, BASIS = 8, 64, 'coif3'
BASES = [f'coif{i}' for i in range(1, 7)] + [f'db{i}' for i in range(2, 11)] + [f'sym{i}' for i in range(2, 9)]
KS = (0.05, 0.1, 0.15, 0.2, 0.3, 0.4, 0.5, 0.75, 1.0, 1.5, 2.0)
N_SEEDS = 100


# --- fig_c_plan.py ---
def synth(seed, path, w=BASIS):
    """Путь записан от seed-массива к корню; результат — в уровнях seed-массива (× 2^J)."""
    x = np.asarray(seed, float)
    for ch in path:
        z = np.zeros_like(x)
        c = {'A': (x, z, z, z), 'H': (z, x, z, z), 'V': (z, z, x, z), 'D': (z, z, z, x)}[ch]
        x = pywt.idwt2((c[0], (c[1], c[2], c[3])), w, mode='periodization')
    return x * 2 ** len(path)


def align(img, seed, rep=1):
    """Перенос «где нарисовано»: максимум циклической корреляции с рисунком, растянутым блоками."""
    cell = img.shape[0] // (seed.shape[0] * rep)
    calque = np.kron(np.tile(seed, (rep, rep)), np.ones((cell, cell)))
    cc = np.real(np.fft.ifft2(np.fft.fft2(calque) * np.conj(np.fft.fft2(img - img.mean()))))
    dy, dx = np.unravel_index(np.argmax(cc), cc.shape)
    return np.roll(img, (dy, dx), axis=(0, 1)), (int(dy), int(dx))


def perlin(G, rng):
    ang = rng.uniform(0, 2 * np.pi, (G, G))
    gx, gy = np.cos(ang), np.sin(ang)
    t = np.arange(M) * G / M
    X, Y = np.meshgrid(t, t)
    x0, y0 = np.floor(X).astype(int), np.floor(Y).astype(int)
    fx, fy = X - x0, Y - y0
    fade = lambda u: u ** 3 * (u * (u * 6 - 15) + 10)
    def dot(ix, iy, dx, dy):
        ix, iy = ix % G, iy % G
        return gx[iy, ix] * dx + gy[iy, ix] * dy
    n00 = dot(x0, y0, fx, fy); n10 = dot(x0 + 1, y0, fx - 1, fy)
    n01 = dot(x0, y0 + 1, fx, fy - 1); n11 = dot(x0 + 1, y0 + 1, fx - 1, fy - 1)
    u, v = fade(fx), fade(fy)
    return (n00 * (1 - u) + n10 * u) * (1 - v) + (n01 * (1 - u) + n11 * u) * v


def perlin_fbm(rng, octaves=3, g=.5, L=2, G0=8):
    return sum(g ** k * perlin(int(G0 * L ** k), rng) for k in range(octaves))


def mm(a):
    return (a - a.min()) / (a.max() - a.min())


def surf(ax, Z, title):
    ls = LightSource(315, 45)
    rgb = ls.shade(Z, cmap=plt.get_cmap('Greys'), vmin=-0.6, vmax=1.5, vert_exag=12, blend_mode='soft')
    X, Y = np.meshgrid(np.arange(M), np.arange(M))
    ax.plot_surface(X, Y, Z, facecolors=rgb, rstride=1, cstride=1, linewidth=0, antialiased=False, shade=False)
    ax.view_init(**S.VIEW3D)
    ax.set_xlim(0, M - 1); ax.set_ylim(M - 1, 0); ax.set_zlim(-0.1, 1.6)
    ax.set_box_aspect((1, 1, 0.42))
    ax.set_axis_off()
    ax.set_title(title, fontsize=S.FS_SMALL, pad=-2, linespacing=1.15)


# --- fig_c1b.py, fig_c1s.py ---
def drawing(target, n):
    c = M // n
    d = target.reshape(n, c, n, c).mean(axis=(1, 3))
    d = (d - d.min()) / (d.max() - d.min())
    return np.round(d * 4) / 4


def interp(seed):
    return zoom(seed, M // seed.shape[0], order=3, mode='grid-wrap', grid_mode=True)


def corr(a, b):
    return float(np.corrcoef(a.ravel(), b.ravel())[0, 1])


def radial_psd(a):
    P = np.abs(np.fft.fft2(a - a.mean())) ** 2
    f = np.fft.fftfreq(M) * M
    R = np.hypot(*np.meshgrid(f, f))
    ks = np.arange(1, M // 2 + 1)
    p = np.array([P[(R >= k - .5) & (R < k + .5)].sum() for k in ks])
    return ks, p / p.sum()


# --- fig_c2.py ---
def block(t, n):
    c = M // n
    return t.reshape(n, c, n, c).mean(axis=(1, 3))


BANDS = ((1, 2), (2, 4), (4, 8), (8, 16), (16, 33))


def bands(a):
    p = radial_psd(mm(a))[1]
    return np.array([p[lo - 1:hi - 1].sum() for lo, hi in BANDS])


def spec_dist(a, t):
    return float(np.mean(np.abs(np.log10(bands(a) + 1e-8) - np.log10(bands(t) + 1e-8))))


def fine_share(a):
    return float(radial_psd(mm(a))[1][7:].sum())


def best_main(t, s):
    best = None
    for b in BASES:
        m = align(synth(s, 'AAA', b), s)[0]
        c = corr(m, t)
        if best is None or c > best[1]:
            best = (b, c, m)
    return best


def k1_details(s, b):
    out = np.zeros((M, M))
    for J, w in ((2, .5), (1, .25)):
        part = synth(s, 'A' * J, b)
        rep = M // part.shape[0]
        out += w * align(np.tile(part, (rep, rep)), s, rep)[0]
    return out


def fit_details(t, main, s, fn):
    best = None
    for b in BASES:
        det = fn(s, b)
        for k in KS:
            m = main + k * det
            d = spec_dist(m, t)
            if best is None or d < best[1]:
                best = ((b, k), d, m)
    return best


# --- fig_c2d.py ---
def d16_details(d16, b):
    out = .5 * align(synth(d16, 'AA', b), d16)[0]
    part = synth(d16, 'A', b)
    out += .25 * align(np.tile(part, (2, 2)), d16, 2)[0]
    return out


def fit_d2(t, main):
    r = t - main
    d16 = block(r, 16); d16 = (d16 - d16.min()) / (d16.max() - d16.min()); d16 = np.round(d16 * 4) / 4
    best = None
    for b in BASES:
        det = d16_details(d16, b)
        for k in KS:
            m = main + k * det
            d = spec_dist(m, t)
            if best is None or d < best[1]:
                best = ((b, k), d, m)
    return best, d16


# --- fig_c0b5.py ---
def show_grid(ax, img, title):
    n = img.shape[0]
    ax.imshow(img, cmap='Greys', vmin=0, vmax=1, interpolation='nearest')
    ax.set_xticks([]); ax.set_yticks([])
    if n <= 16:
        for k in range(1, n):
            ax.axhline(k - .5, color='#9db8d6', lw=0.3); ax.axvline(k - .5, color='#9db8d6', lw=0.3)
    for sp in ax.spines.values():
        sp.set_color(S.MUTED); sp.set_linewidth(0.6)
    ax.set_title(title, fontsize=S.FS_SMALL, linespacing=1.15)


# --- fig_c3.py, fig_c3d.py ---
def a2(d8, d16, b):
    out = align(synth(d8, 'AAA', b), d8)[0] + align(synth(d16, 'AA', b), d16)[0]
    return out + .5 * align(np.tile(synth(d16, 'A', b), (2, 2)), d16, 2)[0]


def a2p(d8, d16, b):
    return align(synth(d8, 'AAA', b), d8)[0] + align(synth(d16, 'AA', b), d16)[0]


def best_shift(pool, target):
    """Лучшее зерно и циклический сдвиг по корреляции Пирсона с target (сдвиг не меняет норм)."""
    T = target - target.mean(); FT = np.conj(np.fft.fft2(T)); nT = np.linalg.norm(T)
    best = None
    for k, m in enumerate(pool):
        A = m - m.mean()
        cc = np.real(np.fft.ifft2(np.fft.fft2(A) * FT)) / (np.linalg.norm(A) * nT)
        i = int(np.argmax(cc)); dy, dx = divmod(i, M)
        if best is None or cc.flat[i] > best[1]:
            best = (k, float(cc.flat[i]), (dy, dx))
    k, c, (dy, dx) = best
    return np.roll(pool[k], (-dy, -dx), axis=(0, 1)), {'seed_index': k, 'corr_full': round(c, 3), 'shift': [dy, dx]}


def mask_half(d8, ref_corr, octs):
    noise = sum(.5 ** k * o for k, o in enumerate(octs))
    base = interp(d8)
    cd = lambda m: corr(block(mm(m), 8), d8)
    lo, hi = 0.0, 5.0
    for _ in range(50):
        mid = (lo + hi) / 2
        lo, hi = (mid, hi) if cd(base + mid * noise) >= ref_corr else (lo, mid)
    return base + lo * noise, {'g': 0.5, 'beta': round(lo, 4), 'ref_corr': round(ref_corr, 3)}
