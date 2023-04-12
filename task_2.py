"""
В результате 10 независимых измерений некоторой величины X, выполненных с
одинаковой точностью, получены опытные данные:
6.9, 6.1, 6.2, 6.8, 7.5, 6.3, 6.4, 6.9, 6.7, 6.1
Предполагая, что результаты измерений подчинены нормальному закону
распределения вероятностей, оценить истинное значение величины X при помощи
доверительного интервала, покрывающего это значение с
доверительной вероятностью 0,95.
"""

import numpy as np
import scipy.stats as stats

arr = np.array([6.9, 6.1, 6.2, 6.8, 7.5, 6.3, 6.4, 6.9, 6.7, 6.1])
print(f'Среднее выборочное = {np.mean(arr): .2f},\n'
      f'Размер выборки n = {len(arr)},\n'
      f'Среднее квадратическое отклонение по выборке(несмещенное) = '
      f'{np.std(arr, ddof=1): .2f}'
      )


def t_from_table(confidens, len_array):
    alpha = (1 - confidens)
    return stats.t.ppf(1 - alpha / 2, len_array - 1)


print(
    f'Табличное значение t-критерия для 95%-го доверительного интервала '
    f'данной выборки = {t_from_table(0.95, len(arr)): .3f}')


def confidens_int(arr, confidens):
    return round(np.mean(arr) - t_from_table(confidens, len(arr)) *
                 np.std(arr, ddof=1) / len(arr) ** 0.5, 3), \
           round(np.mean(arr) + t_from_table(confidens, len(arr)) *
                 np.std(arr, ddof=1) / len(arr) ** 0.5, 3)


print(
    f'95%-й доверительный интервал для истинного значения Х = '
    f'{confidens_int(arr, 0.95)}')
