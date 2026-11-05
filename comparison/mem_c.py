"""Блок 1 (25.10): пиковый объем памяти при синтезе одного изображения — к таблице времени синтеза (§4.4).

Те же конфигурации и функции, что в time_c.py. Память — наибольший объем, выделенный за время синтеза одного
изображения (tracemalloc; массивы NumPy учитываются), сверх памяти, занятой до запуска. От процессора не зависит,
поэтому замер можно выполнить на любом ПК с теми же версиями библиотек; время по-прежнему — из time_c.py.
Запуск: python mem_c.py → mem_c.json и таблица в МБ.
"""
import json, platform, tracemalloc
import numpy as np
import pywt, scipy
from time_c import synth, adsi_scales, adsi_second, perlin, simplex, fbm_perlin, fbm_simplex, mask_noise, seed_search


def peak_mb(fn):
    fn()                                    # прогрев: кэши и импорт не входят в замер
    tracemalloc.start(); tracemalloc.reset_peak()
    fn()
    peak = tracemalloc.get_traced_memory()[1]
    tracemalloc.stop()
    return peak / 2 ** 20


def main():
    sizes = [256, 512, 1024]
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
            ('Перебор 100 зёрен (fBm Перлина, 5 октав)', lambda M, J: seed_search(seed, M, n=10))]
    res = {}
    for name, fn in rows:
        res[name] = {}
        for M in sizes:
            J = int(np.log2(M // 8))
            res[name][M] = round(peak_mb(lambda: fn(M, J)), 1)
            print(f'{name:45s} {M:5d}  {res[name][M]:9.1f} МБ', flush=True)
    env = {'python': platform.python_version(), 'numpy': np.__version__, 'scipy': scipy.__version__,
           'pywavelets': pywt.__version__, 'platform': platform.platform(),
           'note': 'перебор зерен: 10 карт вместо 100 — пик не зависит от числа карт (карты строятся по одной)'}
    json.dump({'env': env, 'mb': res}, open('mem_c.json', 'w'), ensure_ascii=False, indent=1)


if __name__ == '__main__':
    main()
