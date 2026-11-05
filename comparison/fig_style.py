"""Общий стиль рисунков этапа 2 (размер под печать в диссертации).

Поле набора диссертации: A4, поля 25 / 10 мм слева / справа, 20 / 20 мм сверху / снизу
(preamble.tex) → ширина текста 175 мм, высота 257 мм. Основной текст — Times New Roman 14 пт.
Рисунки строятся сразу в конечном размере, поэтому кегль подписей ниже — это кегль на бумаге.
Править оформление — здесь; все скрипты рисунков берут значения отсюда.
"""
import matplotlib

MM = 1 / 25.4                       # мм → дюймы
TEXT_W = 175 * MM                   # ширина поля набора
PAGE_FIG_H = 225 * MM               # рисунок на страницу (оставлено место под подпись)

FONT_FAMILY = ['Times New Roman', 'Liberation Serif']   # Liberation Serif метрически совпадает с Times New Roman
FS_BASE = 11       # подписи осей, текст на рисунке
FS_TITLE = 12      # заголовки панелей
FS_TICK = 10       # числа на осях
FS_LEGEND = 11     # легенда
FS_MARK = 15       # номера профилей (жирные, на белой подложке, у начала профиля)
FS_SMALL = 9       # заголовки в мелких панелях (22 базиса)

DPI = 300          # для \includegraphics
DPI_PREVIEW = 110  # для просмотра в чате

INK = '#1d2429'      # текст
MUTED = '#5b6770'    # оси, вторичный текст
GRID = '#e3e8eb'     # сетка
FILL = '#d9e2e7'     # «как нарисовано» (заливка ступенями)
PROFILE = '#b3261e'  # линии профилей на карте и 3D
OLD = '#8a96a0'      # вариант 0/1 (пунктир) на сплошных профилях

# Представители групп базисов на Б + В: цвет + тип линии (читается и в ч/б печати); названия групп — по таблице К5 (21.10)
BASES = [('coif3', '#2a78d6', '-',  'coif3 — небольшой вал с обеих сторон'),
         ('db6',   '#eb6834', '--', 'db6 — вал у западного и северного краев'),
         ('db2',   '#1baf7a', ':',  'db2 — крутой край')]

# Порядок базисов на рисунках «сплошняком» и в таблицах — по группам (см. Этап2_графика.md)
GROUPS = [('Ровное плато', ['coif2', 'coif3', 'coif4', 'coif5', 'coif6', 'sym4', 'sym6', 'sym8']),
          ('Выброс с одной стороны', ['sym5', 'sym7']),
          ('Несимметричный склон', ['db4', 'db5', 'db6', 'db7', 'db8', 'db9', 'db10']),
          ('Близко к рисунку', ['coif1', 'db3', 'sym3', 'db2', 'sym2'])]
ORDER = [w for _, ws in GROUPS for w in ws]

# Группы по виду края — таблица К5 справочника (20.10); порядок базисов в главе 4 и ее приложениях (21.10).
# db2 = sym2, db3 = sym3 в PyWavelets — одни и те же фильтры.
K5_GROUPS = [('Небольшой вал с обеих сторон', ['coif2', 'coif3', 'coif4', 'coif5', 'coif6', 'sym4']),
             ('Вал у западного и северного краев', ['db3', 'sym3', 'db4', 'db5', 'db6', 'db7', 'db8', 'db9', 'db10',
                                                    'sym6', 'sym8']),
             ('Вал у восточного и южного краев', ['sym5', 'sym7']),
             ('Крутой край', ['db2', 'sym2', 'coif1'])]
K5_ORDER = [w for _, ws in K5_GROUPS for w in ws]

LW_PROFILE = 1.6   # линии профилей (карта, 3D)
LW_SERIES = 1.6    # кривые базисов
LW_REF = 0.6       # уровни 0 и 1

VIEW3D = dict(elev=40, azim=-58)    # ракурс 3D
ZLIM3D = (-0.8, 1.6)
CMAP3D = ('Greys', -1.2, 2.2)       # палитра поверхности и ее пределы
YLIM_PROFILE = (-0.6, 1.6)


def apply():
    matplotlib.rcParams.update({
        'font.family': 'serif', 'font.serif': FONT_FAMILY, 'font.size': FS_BASE,
        'axes.titlesize': FS_TITLE, 'axes.labelsize': FS_BASE, 'xtick.labelsize': FS_TICK,
        'ytick.labelsize': FS_TICK, 'legend.fontsize': FS_LEGEND, 'axes.edgecolor': MUTED,
        'axes.labelcolor': MUTED, 'xtick.color': MUTED, 'ytick.color': MUTED, 'text.color': INK,
        'axes.unicode_minus': True, 'savefig.dpi': DPI,
    })
