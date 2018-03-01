import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import sklearn.cluster as cluster
import time
import random

''' TODO
plot number of occerences of each response 
    number of 6,1,4, etc
pick a bunch of pairs of data to cluster
    graph
cluster using all the columns
use census data
'''

def occerence(check=None, sourceList=None):
    result = []
    for i in check:
        result.append(sourceList.count(i))
    return result
#reads all the data from Pew Research
f = open('data.csv')
data = pd.DataFrame(pd.read_csv(f))
for row in data:
    print(row)
f.close()

plot_kwds = {'alpha': .25, 's': 80, 'linewidths': 0}
data
print(data['q5'])
print(data['q9'])
weight = occerence([1,2,3,4,5,6,7,8,9], data)
plt.scatter(data['q5'], data['q9'], c='b', **plot_kwds)
plt.show()
