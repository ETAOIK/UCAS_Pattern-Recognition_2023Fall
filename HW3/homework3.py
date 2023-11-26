# Pattern Recognition 2023 Fall
# Chenkai GUO
# Date: 2023.11.26

import numpy as np
import matplotlib.pyplot as plt

# transform origin data to homogeneous normalized form
def prepocessing(data, w_1, w_2, norm=True):
    # Data structure
    # x_1 | x_2 | label
    # ------------------
    # 0.1 | 1.1 |   1

    # Select corresponding categories
    data1 = data[data[:, 2] == w_1]
    data2 = data[data[:, 2] == w_2]
    # print(data1)

    # Homogeneous representation
    insert = np.ones(len(data1))
    data1 = np.insert(data1, 2, insert, axis=1)
    insert = np.ones(len(data2))
    data2 = np.insert(data2, 2, insert, axis=1)

    # Normalization
    if norm is True:
        data2[:, 0:3] = np.negative(data2[:, 0:3])

    output = np.concatenate((data1, data2), axis=0)

    return output



class batch_perceptron():
    def __init__(self, data, w_1, w_2):
        self.data = data
        self.w_1 = w_1
        self.w_2 = w_2
        self.step = 0

    def train(self, a, lr, cut_off):
        data_train = prepocessing(data, self.w_1, self.w_2)[:, 0:3]
        self.data_train = data_train
        loss = []
        self.a = np.array(a)
        self.lr = lr

        for epoch in range(100):
            y = []
            # print('------epoch:{epoch}---------')
            self.step += 1
            for idx in range(len(self.data_train)):
                if np.dot(self.a, self.data_train[idx]) <= 0:
                    y.append(self.data_train[idx])
            # print(len(y))
            # print('-----')
            self.a = self.a + (self.lr * np.sum(y, axis=0))
            loss.append(self.lr * np.linalg.norm(y, ord=2))

            if loss[-1] < cut_off:
                self.loss = loss
                return '{}(lr={}, final_a={}, training_steps={})'.format(self.__class__.__name__,
                                                                         self.lr, self.a, self.step)

    def visualization(self):
        data_origin = prepocessing(data, self.w_1, self.w_2, norm=False)[:, 0:3]
        self.origin = data_origin

        plt.figure(figsize=(11, 5))

        plt.subplot(1, 2, 1)
        self.data_train[0:10]
        plt.scatter(self.origin[0:10, 0], self.origin[0:10, 1], label=str(self.w_1))
        plt.scatter(self.origin[10:20, 0], self.origin[10:20, 1], label=str(self.w_2))
        plt.title('Data ditribution and boundary of ' + str(self.w_1) + ' and ' + str(self.w_2))
        plt.xlabel('x1')
        plt.ylabel('x2')
        x_min = np.min(self.origin[:, 0]) - 1
        x_max = np.max(self.origin[:, 0]) + 1
        x_bound = np.arange(x_min, x_max, (x_max - x_min) / 100)
        y_bound = -(self.a[0] * x_bound + self.a[2]) / self.a[1]
        plt.legend()
        plt.plot(x_bound, y_bound, linewidth=2, label='boundary')

        plt.subplot(1, 2, 2)
        x = range(len(self.loss))
        plt.plot(x, self.loss)
        plt.xlabel('epoch')
        plt.ylabel('loss')
        plt.title('Training Loss(' + str(self.w_1) + ',' + str(self.w_2) + ', step=' + str(self.step) + ')')
        img = 'batch perceptron training' + str(self.w_1) + ',' + str(self.w_2) + '.png'
        plt.savefig(img)
        plt.show


class Ho_Kashyap():
    def __init__(self, data, w_1, w_2):
        self.data = data
        self.w_1 = w_1
        self.w_2 = w_2
        self.step = 0

    def train(self, a, b, lr, cut_off):
        data_train = prepocessing(data, self.w_1, self.w_2)[:, 0:3]
        self.data_train = data_train
        loss = []
        self.a = np.array(a)
        self.b = np.array(b)
        self.lr = lr

        for epoch in range(1500):
            self.step += 1
            e = np.subtract(np.matmul(data_train, self.a), self.b)
            e_2 = 0.5 * (e + abs(e))
            # print(e_2)
            self.b = self.b + 2 * self.lr * e_2
            self.a = np.matmul(np.linalg.pinv(data_train), self.b)
            loss.append(np.linalg.norm(e, ord=2))

            if abs(e).max() < cut_off:
                self.loss = loss
                return '{}(lr={}, final_a={}, final_b={}, training_steps={})'.format(self.__class__.__name__,
                                                                                     self.lr, self.a, self.b, self.step)

        self.loss = loss
        print('No solution found!')

    def visualization(self):
        data_origin = prepocessing(data, self.w_1, self.w_2, norm=False)[:, 0:3]
        self.origin = data_origin
        plt.figure(figsize=(11, 5))
        plt.subplot(1, 2, 1)
        self.data_train[0:10]
        plt.scatter(self.origin[0:10, 0], self.origin[0:10, 1], label=str(self.w_1))
        plt.scatter(self.origin[10:20, 0], self.origin[10:20, 1], label=str(self.w_2))
        plt.title('Data ditribution and boundary of ' + str(self.w_1) + ' and ' + str(self.w_2))
        plt.xlabel('x1')
        plt.ylabel('x2')
        x_min = np.min(self.origin[:, 0]) - 1
        x_max = np.max(self.origin[:, 0]) + 1
        x_bound = np.arange(x_min, x_max, (x_max - x_min) / 100)
        y_bound = -(self.a[0] * x_bound + self.a[2]) / self.a[1]
        plt.legend()
        plt.plot(x_bound, y_bound, linewidth=2, label='boundary')

        plt.subplot(1, 2, 2)
        x = range(len(self.loss))
        plt.plot(x, self.loss)
        plt.xlabel('epoch')
        plt.ylabel('loss')
        plt.title('Training Loss(' + str(self.w_1) + ',' + str(self.w_2) + ', step=' + str(self.step) + ')')
        img = 'Ho_Kashyap training' + str(self.w_1) + ',' + str(self.w_2) + '.png'
        plt.savefig(img)
        plt.show


class MSE_Expand():
    def __init__(self, data):
        # homogeneous representation
        self.data = np.insert(data, 2, np.ones(len(data)), axis=1)
        # ensure the dim of data and label is not wrong
        # core formula: y = W^T * x + b
        # y(class*num), W(class*dim), x(dim*num), b(class*num)
        # training set: y(4*32), W(4*3), x(3*32), b(4*32)
        self.total_class = 4
        self.dim = 3
        self.num = len(data) - 2 * self.total_class
        self.data_train = np.zeros([3, 32])
        self.label_train = np.zeros([4, 32])
        self.data_test = np.zeros([3, 8])
        self.label_test = np.zeros([4, 8])

    def preprocess(self):
        # construct origin train and test data
        self.data_train = np.concatenate((self.data[0:8, 0:3], self.data[10:18, 0:3], self.data[20:28, 0:3], self.data[30:38, 0:3]),
                                    axis=0).T
        self.label_train = np.tile((np.arange(4)).reshape(4,1), 8).reshape(32)
        self.label_train = np.eye(4)[self.label_train].T
        self.data_test = np.concatenate((self.data[8:10, 0:3], self.data[18:20, 0:3], self.data[28:30, 0:3], self.data[38:40, 0:3]),
                                   axis=0).T
        self.label_test = np.tile((np.arange(4)).reshape(4, 1), 2).reshape(8)
        self.label_test = np.eye(4)[self.label_test].T

        # print(self.data_train.shape, self.data_test.shape)
        # print(self.label_train.shape, self.label_test.shape)

    def train(self, lamda):
        self.lamda = lamda
        # w = (X * X.T + lamda * I)^{-1} * X * Y.T
        inv = np.linalg.inv(np.dot(self.data_train, self.data_train.T) + self.lamda * np.eye(self.dim))
        w = np.dot(np.dot(inv, self.data_train), self.label_train.T)
        self.test(w)

    def test(self, w):
        pred = np.dot(w.T, self.data_test)
        class_index = np.argmax(pred, axis=0)
        pred = np.eye(self.total_class)[class_index].T  # class*test_num

        wrong_num = 0.5 * sum(sum(np.absolute(pred - self.label_test)))
        acc = ((self.label_test.shape[1] - wrong_num) / self.label_test.shape[1]) * 100
        print('the MSE-expand model acc is：%.2f%% ' % acc)


if __name__ == '__main__':
    data = np.loadtxt('./data.csv', delimiter=',', skiprows=1)

    # batch perceptron between w_1 and w_2
    model1 = batch_perceptron(data, 1, 2)
    model1.train(a=[0, 0, 0], lr=0.5, cut_off=0.01)
    model1.visualization()

    # batch perceptron between w_2 and w_3
    model2 = batch_perceptron(data, 2, 3)
    model2.train(a=[0, 0, 0], lr=0.5, cut_off=0.01)
    model2.visualization()

    # Ho-Kashyap classification between w_1 and w_3
    model3 = Ho_Kashyap(data, 1, 3)
    model3.train(a=np.zeros(3), b=np.ones(20), lr=0.5, cut_off=0.01)
    model3.visualization()

    # Ho-Kashyap classification between w_1 and w_3
    model4 = Ho_Kashyap(data, 2, 4)
    model4.train(a=np.zeros(3), b=np.ones(20), lr=0.5, cut_off=0.01)
    model4.visualization()

    model5 = MSE_Expand(data)
    model5.preprocess()
    model5.train(lamda=1e-5)
    # the MSE-expand model acc is：100.00%