import pandas as pd
import numpy as np
import sys

import matplotlib.pyplot as plt

fontsize = 13

def plot_mandragore(csv='mandragore.csv'):
    df = pd.read_csv(csv)
    xlabel, ylabel, data =
    minimum = min(data, key=lambda x: x[1])

    xs, ys = zip(*data)
    print(xs)
    print(ys)

    plt.xlabel(xlabel, fontsize=13)
    plt.ylabel(ylabel, fontsize=13)

    x_ticks_labels = [str(x) for x in xs]
    plt.xticks(xs, x_ticks_labels)
    # plt.xticklabels(x)

    plt.plot(xs, ys)
    plt.plot(minimum[0], minimum[1], marker="o", c="b")
    plt.show()

if __name__ == "__main__":
    print('Ploting mandragore')
    plot_mandragore()


