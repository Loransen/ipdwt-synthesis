"""Учет сдвига «форма там, где нарисована» — так же, как в программе и на стенде (stand_engine.js, compose):
основная карта переносится на сдвиг по модели S0 с поправкой на полэлемента (Δ0/2); дополнительные seed-массивы
до синтеза сдвигаются на целое число элементов e = round((Δn/2 − Δ0/2 + S0 − Sn)/Δn). Карта — сумма узлов в уровнях
seed-массива (вес × 2^L), без приведения к [0, 1]. Модель сдвига: δ = argmax|g| − (L/2 − 1) для ИХ синтеза rec_lo / rec_hi;
Δ_A = (δl, δl), Δ_H = (δl, δh), Δ_V = (δh, δl), Δ_D = (δh, δh) по (x, y); S = Σ 2^(J−j) Δ(p_j)."""
import math, numpy as np, pywt
M = 64
def deltas(w):
    wv = pywt.Wavelet(w); L = len(wv.rec_lo)
    d = lambda g: int(np.argmax(np.abs(g))) - (L // 2 - 1)
    return d(wv.rec_lo), d(wv.rec_hi)
def mshift(w, path):
    dl, dh = deltas(w); B = {'A': (dl, dl), 'H': (dl, dh), 'V': (dh, dl), 'D': (dh, dh)}
    L = len(path); sx = sy = 0
    for j, c in enumerate(path):
        f = 2 ** (L - 1 - j); sx += f * B[c][0]; sy += f * B[c][1]
    return sx, sy
def synth(s, path, w):
    x = np.asarray(s, float)
    for ch in path:                                  # первая буква — первый шаг от seed-массива
        z = np.zeros_like(x)
        c = {'A': (x, (z, z, z)), 'H': (z, (x, z, z)), 'V': (z, (z, x, z)), 'D': (z, (z, z, x))}[ch]
        x = pywt.idwt2(c, w, mode='periodization')
    return x * 2 ** len(path)                        # уровни seed-массива
jsround = lambda v: math.floor(v + 0.5)              # как Math.round в стенде
def compose(nodes, w, flag=True):
    """nodes: [(seed, path, weight), …]; первый — основной."""
    s0, p0, _ = nodes[0]; L0 = len(p0); D0 = 2 ** L0; S0 = mshift(w, p0)
    out = np.zeros((M, M))
    for i, (s, p, wt) in enumerate(nodes):
        s = np.asarray(s, float); K = s.shape[0]; L = len(p); Dn = 2 ** L
        if flag and i > 0:
            Sn = mshift(w, p)
            ex = jsround((Dn / 2 - D0 / 2 + S0[0] - Sn[0]) / Dn); ey = jsround((Dn / 2 - D0 / 2 + S0[1] - Sn[1]) / Dn)
            s = np.roll(s, (ey, ex), axis=(0, 1))
        x = synth(s, p, w)
        out += wt * np.tile(x, (M // x.shape[0], M // x.shape[1]))
    if not flag: return out
    rx = (-S0[0] + D0 // 2) % M; ry = (-S0[1] + D0 // 2) % M
    return np.roll(out, (ry, rx), axis=(0, 1))
