"""Синтез и модель накопленного сдвига (диссертация, разделы 3.1 – 3.2; синтез — как в ipdwt_synthesis_3_Lvl.py).

Синтез: на каждом шаге матрица ставится в один из узлов cA/cH/cV/cD, остальные — нули,
pywt.waverec2(..., mode='periodization'); путь — от seed-массива к корню.
Измерение: координаты пикселя с максимальным |значением| (argmax, как в analyze_multilevel_fixed.py).
Модель: delta_k = argmax|g_k[n]| при центрированной нумерации n = i - (L/2 - 1) (eq:delta_definition);
Delta_A = (dl, dl), Delta_H = (dl, dh), Delta_V = (dh, dl), Delta_D = (dh, dh) в порядке (x, y);
S_J = sum_j 2^(J-j) Delta(p_j) (eq:explicit_shift).
"""
import itertools
import numpy as np
import pywt

K = 8
COEFFS = ['cA', 'cH', 'cV', 'cD']
WAVELETS = ([f'coif{n}' for n in range(1, 7)] +
            [f'db{n}' for n in range(2, 11)] +
            [f'sym{n}' for n in range(2, 9)])  # 22 базиса §3.2


def step(mat, coeff, wavelet):
    z = np.zeros_like(mat)
    d = {'cA': z.copy(), 'cH': z.copy(), 'cV': z.copy(), 'cD': z.copy()}
    d[coeff] = mat.copy()
    return pywt.waverec2((d['cA'], (d['cH'], d['cV'], d['cD'])), wavelet, mode='periodization')


def synth(wavelet, path, row, col):
    m = np.zeros((K, K))
    m[row, col] = 1.0
    for c in path:
        m = step(m, c, wavelet)
    return m


def peak_xy(img):
    y, x = np.unravel_index(np.argmax(np.abs(img)), img.shape)
    return int(x), int(y)


def wrap(v, M):
    return ((np.asarray(v) + M // 2) % M) - M // 2


def deltas(wavelet):
    w = pywt.Wavelet(wavelet)
    out = []
    for g in (w.rec_lo, w.rec_hi):
        g = np.asarray(g)
        L = len(g)
        out.append(int(np.argmax(np.abs(g))) - (L // 2 - 1))
    return tuple(out)  # (delta_low, delta_high)


def model_shift(wavelet, path):
    dl, dh = deltas(wavelet)
    base = {'cA': (dl, dl), 'cH': (dl, dh), 'cV': (dh, dl), 'cD': (dh, dh)}
    J = len(path)
    S = np.zeros(2, dtype=int)
    for j, c in enumerate(path, start=1):
        S += (2 ** (J - j)) * np.array(base[c])
    return S  # (x, y)


def all_paths(J):
    return [tuple(p) for p in itertools.product(COEFFS, repeat=J)]
