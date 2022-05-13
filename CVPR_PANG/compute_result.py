
import numpy as np
import torch
import time
import pandas as pd
import matplotlib.pyplot as plt

def kernal(xj, h):
    N = len(xj)
    a = []
    for index, value in enumerate(xj):

        # K1_cuda = 1 - torch.abs((value - xj) / h)

        K2_cuda = torch.exp(torch.square(value - xj) / -2 * np.square(h))

        px = (torch.sum(K2_cuda)) / (N * h)
        a.append(float(px))
        print(1)
    return a

def plot(x_data, y_data):
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    ax.plot(x_data, y_data, color="tab:blue")

start_time = time.time()
xj = torch.rand(10000)

data_path = r'/home/ouc/data_transformer/depth_0_softmax.csv'
data = pd.read_csv(data_path, sep=",", encoding="utf-8", engine='python', nrows=465707)
data1 = data.iloc[:, 1]
xj = torch.Tensor(data1.values)
result = kernal(xj, 0.5)
end_time = time.time()
cost = end_time - start_time

plot(xj, result)
plt.show()