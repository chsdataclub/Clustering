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

def occerence(check=None, input=None):
    result = [0]*len(check)

    count =0
    for sourceList in input:
        if count < 18:
            count += 1
            continue
        result += [input[sourceList].count(i) for i in check]
    return result

#reads all the data from Pew Research
f = open('data.csv')
data = pd.DataFrame(pd.read_csv(f))
f.close()

weight = [0]*11
count = 0
for c in data:
    if count < 18:
        count += 1
        continue
    stuff = data[c].value_counts()
    print(stuff)
    for val in stuff.keys():
        if isinstance(val, float):
            break
        if isinstance(val, int) or val.isdigit():
            if int(val) > len(weight):
                break
            weight[int(val)] += stuff[val]
        else:
            weight[0] += stuff[val]

weight = [int(i/sum(weight)*1000) for i in weight]
plot_kwds = {'alpha': .25, 's': 80, 'linewidths': 0}

#weight = occerence([1,2,3,4,5,6,7,8,9], data)
#weight = [data.value_counts() for i in data.iterrows()]

#plt.scatter(data['q5'], data['q9'], c='b', **plot_kwds)
plt.scatter([i for i in range(1,10)], [1 for i in range(1,10)], s=weight)
plt.show()
