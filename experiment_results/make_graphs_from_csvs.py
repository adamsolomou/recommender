import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import csv
import sys

import matplotlib
from matplotlib.ticker import StrMethodFormatter, NullFormatter


fontsize=13

def read_file(filename):
    with open(filename) as f:
        reader = csv.reader(f, delimiter=",")
        # return [(float(r[0]), float(r[1])) for r in reader]
        d = []
        xlabel = ""
        ylabel = ""
        for i, r in enumerate(reader):
            if i == 0:
                xlabel = r[0]
                ylabel = r[1]
            else:
                try:
                    x = int(r[0])
                except ValueError:
                    x = float(r[0])
                y = float(r[1])
                d.append((x, y))
        return xlabel, ylabel, d


def autoencoder_graph():
    data = pd.read_csv('autoencoder_exp_layer.csv')
    data = data[(data.Layer_size >= 4) & (data.Layer_size <=16)]


    fig1, ax1 = plt.subplots()


    ax1.set_xlabel('Hidden Layer size')
    ax1.set_ylabel('RMSE')

    ax1.plot(data['Layer_size'], data['score'], '-o', markevery=[np.argmin(data['score'].values)])
    plt.savefig('graphs/autoenc_layer.pdf')

def svdplus_graph(csv_file):
    pdf_file = '.'.join([csv_file.split('.')[0], 'pdf'])
    pdf_file = "graphs/" + pdf_file
    xlabel, ylabel, data = read_file(csv_file)
    minimum = min(data, key=lambda x: x[1])

    xs, ys = zip(*data)
    print(xs)
    print(ys)

    plt.xlabel(xlabel, fontsize=fontsize)
    plt.ylabel(ylabel, fontsize=fontsize)

    x_ticks_labels = [str(x) for x in xs]
    plt.xticks(xs, x_ticks_labels)
    # plt.xticklabels(x)

    plt.plot(xs, ys)
    plt.plot(minimum[0], minimum[1], marker="o", c="b")
    plt.savefig(pdf_file)


if __name__ == "__main__":
    msg = "Plotting {} graphs..."

    print(msg.format("preprocessed/Autoencoder"))
    autoencoder_graph()

    print(msg.format("preprocessed/SVDplus"))
    svdplus_graph('preprocessed/hidden_size.csv')
    svdplus_graph('preprocessed/matrix_reg.csv')
    svdplus_graph('preprocessed/vector_reg.csv')


