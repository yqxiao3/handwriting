#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
@author: XIAO Yongqin
@contact: yongqin.xiao@united-imaging.com
@file: hand_writing_recog.py
@time: 2021/01/06 13:43
"""
import torch.nn as nn
import torch
from torch.autograd import Variable
import numpy as np
import matplotlib.pyplot as plt
import pickle
import gzip


data = []
data_size = 0
points = []
expects = []
training_data = []
validation_data = []
test_data = []

class Network(nn.Module):
    def __init__(self):
        super(Network, self).__init__()
        self.layer1 = nn.Linear(2, 1)
        self.layer2 = nn.Sigmoid()
        #self.sm = nn.ReLU()

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        return x


def read_data():
    global training_data, validation_data, test_data
    f = gzip.open('./mnist.pkl.gz', 'rb')
    training_data, validation_data, test_data = pickle.load(f, encoding="latin1")
    f.close()
    return

def build_plot(input_data):
    # distribute the points into 2 lists for '0' and '1'
    data_0 = list(filter(lambda i: 0 == i[-1], input_data))
    data_1 = list(filter(lambda i: 1 == i[-1], input_data))
    plot_x_0 = [i[0] for i in data_0]
    plot_y_0 = [i[1] for i in data_0]
    plot_x_1 = [i[0] for i in data_1]
    plot_y_1 = [i[1] for i in data_1]
    plt.plot(plot_x_0, plot_y_0, "ro")
    plt.plot(plot_x_1, plot_y_1, "go")
    #plt.show()


def ReArrangeData(input_data):
    global points, expects
    # 1. convert list to numpy array
    np_data = np.array(input_data, dtype='float32')
    # 2 split to 2 arrays and attach two tensors to the 2 arrays
    points = torch.from_numpy(np_data[:, 0:2])
    expects = torch.from_numpy(np_data[:, -1]).unsqueeze(1)


if __name__ == "__main__":
    print("Begin")
    logistic_model = Network()
    criterion = nn.BCELoss()
    #criterion = nn.MSELoss()
    #optimizer = torch.optim.SGD(logistic_model.parameters(), lr=1e-3, momentum=0.9)
    optimizer = torch.optim.Adam(logistic_model.parameters(), lr=1e-3)
    read_data()
    build_plot(data)
    ReArrangeData(data)
    # process the data
    for epoch in range(10000):
        x = Variable(points)
        y = Variable(expects)
        # =======  forward  ======
        out = logistic_model(x)
        loss = criterion(out, y)
        print_loss = loss.data
        mask = out.ge(0.5).float()
        correct = (mask == y).sum()
        acc = 1.0*correct / x.size(0)
        # =======  backward  ======
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if(epoch + 1) % 10000 == 0:
            print('*' * 10)
            print('epoch {}'.format(epoch+1))
            print('loss is {:.4f}'.format(print_loss))
            print('acc is {:.4f}'.format(acc))
    w0, w1 = logistic_model.layer1.weight[0]
    w0 = w0.data
    w1 = w1.data
    b = logistic_model.layer1.bias.data
    plot_x = torch.from_numpy(np.arange(0, 1, 0.001))
    plot_y = (-w0 * plot_x -b) / w1
    plt.plot(plot_x, plot_y)
    plt.show()


