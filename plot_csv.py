import csv
import sys

import matplotlib.pyplot as plt

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


if len(sys.argv) < 2:
    print("Provide a file name to plot")
    sys.exit(1)

# data = read_file("vector_reg.csv")
xlabel, ylabel, data = read_file(sys.argv[1])
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
