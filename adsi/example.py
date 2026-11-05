"""Пример: карта высот 64 × 64 из нескольких узлов в одном запуске ОПДВП с учетом сдвига.
Долина (основной seed-массив, узел AAA) + два холма на дне (тот же узел, вес 0.4) + неровность по всей форме (узел DAA, вес 0.03).
Запуск: python example.py  ->  example.png"""
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from adsi_place import compose

valley = np.array([[1, 1, 1, 1, 1, 1, 1, 1],
                   [1, .75, .5, .5, .5, .5, .75, 1],
                   [.75, .5, .25, 0, 0, .25, .5, .75],
                   [.5, .25, 0, 0, 0, 0, .25, .5],
                   [.5, .25, 0, 0, 0, 0, .25, .5],
                   [.75, .5, .25, 0, 0, .25, .5, .75],
                   [1, .75, .5, .5, .5, .5, .75, 1],
                   [1, 1, 1, 1, 1, 1, 1, 1]], float)
hills = np.zeros((8, 8)); hills[3, 2] = hills[4, 5] = 1.0

W = 'coif3'                                   # вейвлет-базис (группа «небольшой вал с обеих сторон»)
H = compose([(valley + 0.4 * hills, 'AAA', 1.0),  # основной seed-массив: форма рельефа
             (valley, 'DAA', 0.03)], W)           # МуС с периодом 8 пикс.: бугры по всей форме
plt.imsave('example.png', H, cmap='Greys')       # темнее — выше
print('example.png', H.shape, round(float(H.min()), 3), round(float(H.max()), 3))
