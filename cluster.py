import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import sklearn.cluster as cluster
import time
import random

def plot_clusters(x, y, labelx, labely, data, algorithm, args, kwds):
    X = np.array([[4,0],
                  [0,4],
                  [2,2]],np.float64)
    a = algorithm(init=X, n_init=1, *args, **kwds)
    print(a.init)
    passin= [[data[x][a],data[y][a]] for a in range(0, len(data[x]))]
    labels = a.fit_predict(passin)
    palette = sns.color_palette('pastel', np.unique(labels).max() + 1)
    colors = [palette[x] if x >= 0 else (0.0, 0.0, 0.0) for x in labels]
    plt.scatter(data[x], data[y], c=colors, **plot_kwds)
    plt.scatter(a.cluster_centers_[:, 0], a.cluster_centers_[:, 1], marker="x", **plot_kwds)
    frame = plt.gca()
    frame.axes.get_xaxis().set_visible(True)
    frame.set_xlabel(labelx)
    frame.axes.get_yaxis().set_visible(True)
    frame.set_ylabel(labely)
    plt.title('Clusters found by {}'.format(str(algorithm.__name__)), fontsize=24)
    plt.show()

sns.set_context('poster')
sns.set_color_codes()
plot_kwds = {'alpha': 1, 's': 80, 'linewidths': 0} #alpha .01

#reads all the data from Pew Research
f = open('data.csv')
data = pd.DataFrame(pd.read_csv(f))
f.close()

#plt.scatter(data['q5'], data['q9'], c='b', **plot_kwds)
frame = plt.gca()
frame.axes.get_xaxis().set_visible(False)
frame.axes.get_yaxis().set_visible(False)

plot_clusters('q40e', 'q40f', 'Terrorism', 'Supreme court appiontments', data, cluster.KMeans, (), {'n_clusters': 3})