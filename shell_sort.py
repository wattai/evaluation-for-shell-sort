# -*- coding: utf-8 -*-
"""
Created on Wed Jan 18 15:23:49 2017

@author: wattai
"""

# ShellSortを実行時のステップ数と実行時間の観点から評価するプログラム．
# 実行にとても時間(10分くらい)がかかります．注意．

import numpy as np
import time
from random import shuffle
import matplotlib.pyplot as plt


def shellsort(a):
    n = len(a)
    n_step = 0

    h = 1
    while h < n/9:
        h = 3*h + 1

    while h > 0:
        print("h: ", h)
        for i in range(h, n):
            w = a[i]
            j = i-h
            while w < a[j] and j >= 0:
                a[j+h] = a[j]
                n_step += 1  # step数のカウント
                j -= h
            a[j+h] = w
        h = np.int(np.floor(h / 3))
    return a, n_step


def rmse(x, c):
    return np.sqrt(np.average((x - c)**2, axis=0))


if __name__ == '__main__':

    n_multiplier = 15  # 2 の n_multiplier [乗] までを検証
    n_iter = 1000  # n_iter [回] 繰り返し，その平均値をグラフにプロット

    time_consumptions = np.zeros([n_multiplier, n_iter])
    n_steps = np.zeros([n_multiplier, n_iter])
    for i in range(n_multiplier):
        for k in range(n_iter):

            alist = np.random.randint(low=0, high=2**(n_multiplier+1),
                                      size=(2**(i+1), 1))[:, 0]
            shuffle(alist)
            print('length: ', len(alist))
            print('Unsorted')
            print(alist)
            alist = alist.tolist()
            print('Sorted by Shell_sort')
            start = time.time()
            alist_sorted, n_step = shellsort(alist)
            time_consumption = time.time() - start
            print('Sorted: ', np.array(alist_sorted))
            print('n_step: ', n_step)
            print('%.3f [ms]' % (time_consumption * 1e3))

            time_consumptions[i, k] = time_consumption
            n_steps[i, k] = n_step
    time_consumptions *= 1e3  # [s] -> [ms] に単位変換

    r = [1.21, 1.23, 1.25, 1.27, 1.29]

    n = 2 ** np.arange(1, n_multiplier+1)

    a = np.average(n_steps, axis=1)
    plt.figure()
    plt.plot(n, n_steps.mean(1), label="experiment")
    plt.plot(n, (a[-1] / (n**r[0])[-1])*(n**r[0]),
             label="theory: $O(n^{1.21})$")
    plt.plot(n, (a[-1] / (n**r[1])[-1])*(n**r[1]),
             label="theory: $O(n^{1.23})$")
    plt.plot(n, (a[-1] / (n**r[2])[-1])*(n**r[2]),
             label="theory: $O(n^{1.25})$")
    plt.plot(n, (a[-1] / (n**r[3])[-1])*(n**r[3]),
             label="theory: $O(n^{1.27})$")
    plt.plot(n, (a[-1] / (n**r[4])[-1])*(n**r[4]),
             label="theory: $O(n^{1.29})$")
    plt.xscale("linear")
    plt.yscale("linear")
    plt.xlabel("length of array $n$ [number]", fontsize=12)
    plt.ylabel("number of step $n_{step}$ [time]", fontsize=12)
    plt.legend()
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0)
    plt.grid(True)
    plt.tight_layout()

    err_step1 = rmse(a, (a[-1] / (n**r[0])[-1])*(n**r[0]))
    err_step2 = rmse(a, (a[-1] / (n**r[1])[-1])*(n**r[1]))
    err_step3 = rmse(a, (a[-1] / (n**r[2])[-1])*(n**r[2]))
    err_step4 = rmse(a, (a[-1] / (n**r[3])[-1])*(n**r[3]))
    err_step5 = rmse(a, (a[-1] / (n**r[4])[-1])*(n**r[4]))
    print("RMSE: %.2f %.2f %.2f %.2f %.2f" % (
            err_step1, err_step2, err_step3, err_step4, err_step5))
    plt.title("RMSE $O(n^{1.25})$: %.2f" % err_step3)
    normalized_errs_step = [err_step1, err_step2, err_step3,
                            err_step4, err_step5] \
                            / np.sum([err_step1, err_step2, err_step3,
                                      err_step4, err_step5])

    b = np.average(time_consumptions, axis=1)
    plt.figure()
    plt.plot(n, time_consumptions.mean(1), label="experiment")
    plt.plot(n, (b[-1] / (n**r[0])[-1])*(n**r[0]),
             label="theory: $O(n^{1.21})$")
    plt.plot(n, (b[-1] / (n**r[1])[-1])*(n**r[1]),
             label="theory: $O(n^{1.23})$")
    plt.plot(n, (b[-1] / (n**r[2])[-1])*(n**r[2]),
             label="theory: $O(n^{1.25})$")
    plt.plot(n, (b[-1] / (n**r[3])[-1])*(n**r[3]),
             label="theory: $O(n^{1.27})$")
    plt.plot(n, (b[-1] / (n**r[4])[-1])*(n**r[4]),
             label="theory: $O(n^{1.29})$")
    plt.xscale("linear")
    plt.yscale("linear")
    plt.xlabel("length of array: $n$ [number]", fontsize=12)
    plt.ylabel("time consumption: $t$ [ms]", fontsize=12)
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0)
    plt.grid(True)
    plt.tight_layout()

    err_t1 = rmse(b, (b[-1] / (n**r[0])[-1])*(n**r[0]))
    err_t2 = rmse(b, (b[-1] / (n**r[1])[-1])*(n**r[1]))
    err_t3 = rmse(b, (b[-1] / (n**r[2])[-1])*(n**r[2]))
    err_t4 = rmse(b, (b[-1] / (n**r[3])[-1])*(n**r[3]))
    err_t5 = rmse(b, (b[-1] / (n**r[4])[-1])*(n**r[4]))
    print("RMSE: %.2f %.2f %.2f %.2f %.2f" % (
            err_t1, err_t2, err_t3, err_t4, err_t5))
    plt.title("RMSE $O(n^{1.25})$: %.2f" % err_t3)
    normalized_errs_t = [err_t1, err_t2, err_t3, err_t4, err_t5] \
                        / np.sum([err_t1, err_t2, err_t3, err_t4, err_t5])


    normalized_errs_total = (normalized_errs_step + normalized_errs_t) \
                            / np.sum(normalized_errs_step + normalized_errs_t)
    print("normalized_errs_step: ", normalized_errs_step)
    print("normalized_errs_t: ", normalized_errs_t)
    print("normalized_errs_total: ", normalized_errs_total)
    print("Order_on_min_err_step: %s" % r[np.argmin(normalized_errs_step)])
    print("Order_on_min_err_t: %s" % r[np.argmin(normalized_errs_t)])
    print("Order_on_min_err_total: %s" % r[np.argmin(normalized_errs_total)])

    """
    alist = [54,26,93,17,77,31,44,55,20]
    alist = np.random.randint(0, 2**(16), (2**(15),1))[:, 0]
    shellSort(alist)
    shellsort(alist)
    print(alist)
    """
