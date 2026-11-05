"""Блок 1 (25.10): локальность изменения карты высот при изменении одного входного значения.

Протокол — `Блок1_критерии_ход.md`, шаги 2 – 3:
  карта 64×64; меняется по одному каждое значение, задаваемое пользователем;
  меры: доля площади, где высота изменилась больше 1 % и 5 % диапазона высот исходной карты,
        и наименьшая доля площади, на которую приходится 90 % суммы квадратов изменения;
  карты сравниваются до приведения к [0, 1].
Методы и параметры — код главы 4:
  шумы, DS, Уорли — функции fig_gallery.py (галерея, рисунок 4.17) при M = 64, крупнейшая октава — 32 пикс.
    (как в галерее и в замере времени), метод fBm — 5 октав, g = 0,5, L = 2; DS H = 0,8; Уорли 24 центра;
  маска и АДСИ — четыре типовые задачи блока «Сравнение по общему результату» (fig_c_work.py, c_work.json):
    маска — рисунок задачи, сглаженный до 64×64, + beta · fBm шума Перлина (3 октавы), beta — из c_work.json;
    АДСИ — основной seed-массив 8×8 (путь AAA) + второй 16×16 (путь AA), у гряды — копия второго (путь A, вес 0,5);
    базис по задаче (coif3, у гряды db2); перенос на место рисунка не применяется: он сдвигает обе карты одинаково.
Изменения: элемент seed-массива и клетка рисунка — +0,25 (у значения 1 — минус 0,25);
  seed — 10 пар; базовая частота — 2 → 3 периода на карту (ближайшее изменение, при котором карта остается
  бесшовной; у симплексного шума — то же отношение 1,5); число октав — минус 1; g — +0,1; L — 2 → 2,2;
  DS — H +0,1; Уорли — число центров +1, сдвиг одного центра на 8 пикс. (одна клетка 8×8 пикс.);
  АДСИ — путь (одна буква H на шаге 1, 2 или 3 основного пути), базис (соседний порядок того же семейства).
Запуск: python locality_c.py → locality_c.json и таблица в консоли.
"""
import json
import numpy as np
from c_lib import synth, interp, perlin as perlin64

M = 64
src = open('fig_gallery.py', encoding='utf-8').read().split('rng = np.random.default_rng(20261023)')[0]
G = {}
exec(src, G)          # функции галереи без построения рисунка
G['M'] = M            # те же функции на карте 64×64
G0 = 2                # 2 периода на карту = крупнейшая октава 32 пикс., как в галерее (8 на 256)
THR = (0.01, 0.05)
SEEDS = range(1, 11)


def measures(h0, h1):
    rng = h0.max() - h0.min()
    d = np.abs(h1 - h0)
    out = [float((d > t * rng).mean()) for t in THR]
    e = np.sort((d ** 2).ravel())[::-1]
    out.append(0.0 if e.sum() == 0 else float((np.searchsorted(np.cumsum(e) / e.sum(), 0.9) + 1) / e.size))
    return out


def even(x):
    return max(2, 2 * int(round(x / 2)))


def octave_list(kind, seed, g0=G0, lac=2.0, octaves=5):
    """Октавы шума в порядке fig_gallery.py (один генератор на все октавы)."""
    rng = np.random.default_rng(seed)
    if kind == 'value':
        return [G['value_noise'](int(round(g0 * lac ** k)), rng) for k in range(octaves)]
    if kind == 'perlin':
        return [G['perlin'](int(round(g0 * lac ** k)), rng) for k in range(octaves)]
    if kind == 'simplex':
        return [G['simplex'](0.6 * g0 / M * lac ** k, rng) for k in range(octaves)]
    if kind == 'wavelet':
        return [G['wavelet_band'](even(g0 * lac ** k), rng) for k in range(octaves)]


def fbm(octs, g=0.5):
    return sum(g ** k * o for k, o in enumerate(octs))


def bump(v):
    return v + 0.25 if v + 0.25 <= 1 + 1e-9 else v - 0.25


def worley_c(c, n=M):
    Y, X = np.mgrid[0:n, 0:n] + 0.5
    d = np.full((n, n), np.inf)
    for cy, cx in c:
        dy = np.abs(Y - cy); dx = np.abs(X - cx)
        d = np.minimum(d, np.hypot(np.minimum(dy, n - dy), np.minimum(dx, n - dx)))
    return d


res = {}
names = {'value': 'Числовой шум', 'perlin': 'Шум Перлина', 'simplex': 'Симплексный шум', 'wavelet': 'Вейвлет-шум'}
for kind, nm in names.items():
    r = {k: [] for k in ('seed', 'базовая частота 2 → 3 периода', 'число октав −1', 'g +0,1', 'L 2 → 2,2')}
    for s in SEEDS:
        o = octave_list(kind, s); h0 = fbm(o)
        r['seed'].append(measures(h0, fbm(octave_list(kind, s + 500))))
        r['базовая частота 2 → 3 периода'].append(measures(h0, fbm(octave_list(kind, s, g0=3))))
        r['число октав −1'].append(measures(h0, fbm(o[:-1])))
        r['g +0,1'].append(measures(h0, fbm(o, 0.6)))
        r['L 2 → 2,2'].append(measures(h0, fbm(octave_list(kind, s, lac=2.2))))
    res[nm + ' с применением fBm'] = {k: np.mean(v, 0).tolist() for k, v in r.items()}

r = {'seed': [], 'H +0,1': []}
for s in SEEDS:
    h0 = G['diamond_square'](0.8, np.random.default_rng(s), n=M)
    r['seed'].append(measures(h0, G['diamond_square'](0.8, np.random.default_rng(s + 500), n=M)))
    r['H +0,1'].append(measures(h0, G['diamond_square'](0.9, np.random.default_rng(s), n=M)))
res['Алгоритм DS'] = {k: np.mean(v, 0).tolist() for k, v in r.items()}

r = {'seed': [], 'число центров +1': [], 'сдвиг одного центра на 8 пикс.': []}
for s in SEEDS:
    c0 = np.random.default_rng(s).uniform(0, M, (24, 2)); h0 = worley_c(c0)
    assert np.allclose(h0, G['worley'](24, np.random.default_rng(s), n=M))
    r['seed'].append(measures(h0, worley_c(np.random.default_rng(s + 500).uniform(0, M, (24, 2)))))
    c1 = np.random.default_rng(s).uniform(0, M, (25, 2)); assert np.allclose(c1[:24], c0)
    r['число центров +1'].append(measures(h0, worley_c(c1)))
    for k in range(24):
        c2 = c0.copy(); c2[k, 1] = (c2[k, 1] + 8) % M
        r['сдвиг одного центра на 8 пикс.'].append(measures(h0, worley_c(c2)))
res['Шум Уорли'] = {k: np.mean(v, 0).tolist() for k, v in r.items()}
res['Шум Уорли']['сдвиг одного центра на 8 пикс., наибольшее'] = np.max(r['сдвиг одного центра на 8 пикс.'], 0).tolist()

# --- типовые задачи ---
tasks = {t['id']: t for t in json.load(open('c3_tasks.json', encoding='utf-8'))}
vk = json.load(open('c3_valley_kirill.json', encoding='utf-8'))
cw = {t['task']: t for t in json.load(open('c_work.json', encoding='utf-8'))['tasks']}
TASKS = []
for tid in ('hills', 'ridge', 'valley_k', 'plain_hill'):
    src_t = vk if tid == 'valley_k' else tasks[tid]
    TASKS.append((tid, np.array(src_t['d8'], float), np.array(src_t['d16'], float), cw[tid]['basis'], cw[tid]['mask']['beta']))


def mask_map(d8, beta, octs):
    return interp(d8) + beta * fbm(octs)


def adsi_map(d8, d16, b, tid, path='AAA'):
    out = synth(d8, path, b) + synth(d16, 'AA', b)
    if tid == 'ridge':
        out = out + 0.5 * np.tile(synth(d16, 'A', b), (2, 2))
    return out


NEXT = {'coif3': ('coif2', 'coif4'), 'db2': ('db3',)}
octs = [perlin64(4 * 2 ** k, np.random.default_rng(1000)) for k in range(3)]
rng = np.random.default_rng(1000); octs = [perlin64(4 * 2 ** k, rng) for k in range(3)]   # как в fig_c_work.py
rm = {'клетка рисунка задачи': [], 'seed шума': []}
ra = {'элемент основного seed-массива 8×8': [], 'элемент второго seed-массива 16×16': [], 'путь (одна буква H)': [], 'базис (соседний порядок)': []}
for tid, d8, d16, b, beta in TASKS:
    m0 = mask_map(d8, beta, octs)
    rng2 = np.random.default_rng(2000); octs2 = [perlin64(4 * 2 ** k, rng2) for k in range(3)]
    rm['seed шума'].append(measures(m0, mask_map(d8, beta, octs2)))
    a0 = adsi_map(d8, d16, b, tid)
    for i in range(8):
        for j in range(8):
            x = d8.copy(); x[i, j] = bump(x[i, j])
            rm['клетка рисунка задачи'].append(measures(m0, mask_map(x, beta, octs)))
            ra['элемент основного seed-массива 8×8'].append(measures(a0, adsi_map(x, d16, b, tid)))
    for i in range(16):
        for j in range(16):
            y = d16.copy(); y[i, j] = bump(y[i, j])
            ra['элемент второго seed-массива 16×16'].append(measures(a0, adsi_map(d8, y, b, tid)))
    for p in ('HAA', 'AHA', 'AAH'):
        ra['путь (одна буква H)'].append(measures(a0, adsi_map(d8, d16, b, tid, p)))
    for b2 in NEXT[b]:
        ra['базис (соседний порядок)'].append(measures(a0, adsi_map(d8, d16, b2, tid)))
res['Маска и шум Перлина с применением fBm'] = {k: np.mean(v, 0).tolist() for k, v in rm.items()}
res['Маска и шум Перлина с применением fBm']['клетка рисунка задачи, наибольшее'] = np.max(rm['клетка рисунка задачи'], 0).tolist()
res['АДСИ'] = {k: np.mean(v, 0).tolist() for k, v in ra.items()}
res['АДСИ']['элемент основного seed-массива, наибольшее'] = np.max(ra['элемент основного seed-массива 8×8'], 0).tolist()
res['АДСИ']['элемент второго seed-массива, наибольшее'] = np.max(ra['элемент второго seed-массива 16×16'], 0).tolist()

json.dump(res, open('locality_c.json', 'w', encoding='utf-8'), ensure_ascii=False, indent=1)
print(f"{'':60s} {'>1 %':>7s} {'>5 %':>7s} {'90 % изм.':>10s}")
for m, rows in res.items():
    print(m)
    for k, (a, b_, c) in rows.items():
        print(f"   {k:57s} {100*a:6.1f} {100*b_:7.1f} {100*c:9.1f}")
