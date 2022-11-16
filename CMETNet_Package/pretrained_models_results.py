'''
 (c) Copyright 2022
 All rights reserved
 Programs written by Khalid A. Alobaid
 Department of Computer Science
 New Jersey Institute of Technology
 University Heights, Newark, NJ 07102, USA
 Permission to use, copy, modify, and distribute this
 software and its documentation for any purpose and without
 fee is hereby granted, provided that this copyright
 notice appears in all copies. Programmer(s) makes no
 representations about the suitability of this
 software for any purpose.  It is provided "as is" without
 express or implied warranty.
'''

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

def plot_results():
    materials = ['SVR','RF','GP','XGB','CNN','COMB','CMETNet']
    x_pos = np.arange(len(materials))

    CTEs = [17.77,11.67,13.22,12.54,17.48,10.32,9.75]
    error_org = [6.02,5.93,5.65,5.15,4.9,5.94,5.54]
    error = [1/3*x for x in error_org]

    colors = ['#4CB391','indianred','gray','lightsteelblue','gold','olive','blue']
    fig, ax = plt.subplots(1, 1, figsize=(8, 3))

    for i in range(0,len(x_pos)):
        ax.bar(x_pos[i], CTEs[i],color=colors[i] ,yerr=error[i], align='center', alpha=0.8, ecolor='black', capsize=3)

    ax.set_ylabel('MAE (Hours)')
    ax.set_xticks(x_pos)
    ax.set_xticklabels(materials)
    ax.yaxis.grid(True)

    plt.yticks(np.arange(0, 21, 5))
    plt.ylim([0, 20])
    plt.show()


    CTEs = [0.59,0.71,0.70,0.67,0.38,0.77,0.83]
    error_org = [0.33,0.25,0.23,0.28,0.26,0.21,0.15]
    error = [1/3*x for x in error_org]

    fig, ax = plt.subplots(1, 1, figsize=(8, 3))


    for i in range(0,len(x_pos)):
        ax.bar(x_pos[i], CTEs[i],color=colors[i] ,yerr=error[i], align='center', alpha=0.8, ecolor='black', capsize=3)

    ax.set_ylabel('PPMCC')
    ax.set_xticks(x_pos)
    ax.set_xticklabels(materials)
    ax.yaxis.grid(True)

    plt.ylim([0, 1])
    plt.show()




