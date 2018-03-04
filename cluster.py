import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import sklearn.cluster as cluster
import time
import random


def plot_clusters(z, za, labelx, labely, badx, bady, algorithm, kwds):
    a = algorithm(**kwds)
    print('start1')
    x = z.tolist()
    y = za.tolist()
    print('x', [i for i in range(0, len(x)) if x[i] == badx])
    print('y', [i for i in range(0, len(y)) if y[i] == bady])
    inc = 0
    while inc < len(x):
        if x[inc] == badx or y[inc] == bady or x[inc] == str(badx) or y[inc] == str(bady) or x[inc] == ' ' or y[inc] == ' ':
            del x[inc]
            del y[inc]
            # x.pop(inc)
            # y.pop(i)
            inc -= 1

        inc += 1
    print('start2')
    print('x', [i for i in range(0, len(x)) if x[i] == badx])
    print('y', [i for i in range(0, len(y)) if y[i] == bady])
    passin = [[x[i], y[i]] for i in range(0, len(x))]
    labels = a.fit_predict(passin)
    print('start3')
    palette = sns.color_palette('pastel', np.unique(labels).max() + 1)
    colors = [palette[x] if x >= 0 else (0.0, 0.0, 0.0) for x in labels]
    print('x', [i for i in range(0, len(x)) if x[i] == badx])
    print('y', [i for i in range(0, len(y)) if y[i] == bady])
    plt.scatter(x, y, c=colors, **plot_kwds)
    plt.scatter(a.cluster_centers_[:,
                0], a.cluster_centers_[:, 1], c=palette, marker="x",
                **{'alpha': 1, 's': 80, 'linewidths': 0})
    frame = plt.gca()
    frame.axes.get_xaxis().set_visible(True)
    frame.set_xlabel(labelx)
    frame.axes.get_yaxis().set_visible(True)
    frame.set_ylabel(labely)
    plt.title(labely + ' vs ' + labelx, fontsize=24)
    plt.show()


sns.set_context('poster')
sns.set_color_codes()
plot_kwds = {'alpha': .01, 's': 80, 'linewidths': 0}  # alpha .01

# reads all the data from Pew Research
f = open('data.csv')
data = pd.DataFrame(pd.read_csv(f))
f.close()

input = [data['income'], data['q40jf1'], 'income', 'Immigration importance', 10, 9]
print('start')
plot_clusters(*input, cluster.KMeans, {'n_clusters': 4})

plot_clusters(*input, cluster.KMeans, {'n_clusters': 3})

plot_clusters(*input, cluster.KMeans, {'n_clusters': 2})
