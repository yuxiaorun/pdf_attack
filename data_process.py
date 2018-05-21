# -*- coding: UTF-8 -*-
import numpy as np
batch_size=128
def get_data():          #读文件，返回name, label, data数组
    f = open('format_matrix')
    data = []
    label = []
    name = []
    for line in f.readlines():
        line = line.split('|')
        if len(line) != 3:
            continue
        name.append(line[0])
        if line[1] == "1":
            label.append([1, 0])
        elif line[1] == "0":
            label.append([0, 1])
        data.append(line[2].split(','))
    return name, label, data


class input_data:  # 处理输入数据
    def __init__(self, test_num=1000):
        name, label, data = dt.get_data()
        all_data = []
        self.batch_i = 0
        for i in range(len(data)):
            all_data.append([data[i], label[i]])
        np.random.shuffle(all_data)
        self.train_data = [x[0] for x in all_data]
        self.test_data = [x[0] for x in all_data]
        self.train_label = [x[1] for x in all_data]
        self.test_label = [x[1] for x in all_data]
        self.test_i = 0
        self.train_data = np.array(self.train_data, dtype=np.float32)
        self.test_data = np.array(self.test_data, dtype=np.float32)
        self.train_label = np.array(self.train_label, dtype=np.float32)
        self.test_label = np.array(self.test_label, dtype=np.float32)

        self.train_num = self.train_data.shape[0]
        self.test_num = test_num

        print "train num:" + str(self.train_data.shape[0]) + " test num:" + str(self.test_data.shape[0])

    def get_train_data(self):  # 喂训练数据，每次调用返回一个batch
        i = self.batch_i
        if i == (self.train_data.shape[0] / batch_size):
            x = self.train_data[batch_size * i:][:]
            y = self.train_label[batch_size * i:][:]
            self.batch_i = 0
        else:
            x = self.train_data[batch_size * i:batch_size * (i + 1)][:]
            y = self.train_label[batch_size * i:batch_size * (i + 1)][:]
            self.batch_i += 1

        return (x, y)

    def get_test_data(self):  # 喂测试数据
        x = self.test_data
        y = self.test_label
        print y.shape
        return (x, y)


    def get_all_data(self):  # 每次读1000个
        i = self.test_i
        data = self.train_data
        label = self.train_label
        x = data[1000 * i:1000 * (i + 1)][:]
        y = label[1000 * i:1000 * (i + 1)][:]
        self.test_i += 1
        if self.test_i >= (data.shape[0] / 1000):
            self.test_i = 0
        return x, y




