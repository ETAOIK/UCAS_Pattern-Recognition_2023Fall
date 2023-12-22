# Pattern Recognition 2023 Fall
# Chenkai GUO
# Date: 2023.12.21

import time
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score


# PCA
def PCA(data, dim):
    # calculate mean and covariance
    mean = np.mean(data, axis=0)
    X_scale = np.subtract(data, mean)
    cov = 1/len(data) * np.dot(X_scale.T, X_scale)

    # eigenvalue decomposition
    eig, W = np.linalg.eig(cov)

    eig_index = np.absolute(np.argsort(-eig))[:dim]
    W_red = W[:, eig_index]

    # projection
    y = np.dot(data, W_red)

    return y

def PCA_eigVec(data):
    # zero-mean and calculate covariance
    mean = np.mean(data, axis=0)
    X_scale = np.subtract(data, mean)
    cov = 1/len(data) * np.dot(X_scale.T, X_scale)

    # eigenvalue decomposition for W
    eig, W = np.linalg.eig(cov)
    eig_index = np.argsort(-eig)

    return W, eig_index


# LDA
def LDA(data, dim):
    # zero-mean
    mean = np.mean(data, axis=0)
    X_scale = np.subtract(data, mean)

    # calculate S_w
    S_w = 0
    for i in range(40):
        class_data = data[10*i, 10*i+10]
        class_mean = np.mean(class_data, axis=0)
        class_X_scale = np.subtract(class_data, class_mean)
        S_w += np.dot(class_X_scale.T, class_X_scale)

    # calculate S_b = S_t - S_w
    S_t = np.dot(X_scale.T, X_scale)
    S_b = S_t - S_w

    # eigenvalue decomposition for W
    S = np.linalg.inv(S_w) * S_b
    eig, W = np.linalg.eig(S)
    eig_index = np.argsort(-eig)[:dim]
    W_red = W[:, eig_index]

    # projection
    y = np.dot(data, W_red)

    return y

def LDA_eigVec(data):
    # zero-mean
    mean = np.mean(data, axis=0)
    X_scale = np.subtract(data, mean)

    # calculate S_w
    S_w = 0
    for i in range(40):
        class_data = data[10*i:10*i+10]
        class_mean = np.mean(class_data, axis=0)
        class_X_scale = np.subtract(class_data, class_mean)
        S_w += np.dot(class_X_scale.T, class_X_scale)

    # calculate S_b = S_t - S_w
    S_t = np.dot(X_scale.T, X_scale)
    S_b = S_t - S_w

    # eigenvalue decomposition for W
    S = np.linalg.inv(S_w) * S_b
    eig, W = np.linalg.eig(S)
    eig_index = np.argsort(-eig)

    return W, eig_index


if __name__ == '__main__':
    # ORL dataset
    # change ORL and dim_lists for vehicle results
    dim_list = [2, 5, 10, 15, 20, 25, 30, 40, 50, 75, 100, 150, 200, 400]
    acc_list = []
    time_list = []

    ORL = np.loadtxt('./ORLData.csv', skiprows=0, delimiter=',').T
    ORL_train = ORL[:, :-1]
    ORL_label = ORL[:, -1]

    # PCA reduction results, replace PCA_eigVec to LDA_eigVec for LDA reduction results
    W, eig_index = PCA_eigVec(ORL_train)

    for dim in dim_list:
        if dim < 400:
            red_train = np.real(np.dot(ORL_train, W[:, eig_index[:dim]]))
            x_train, x_test, y_train, y_test = train_test_split(red_train, ORL_label, test_size=0.2, random_state=1126)
        else:
            x_train, x_test, y_train, y_test = train_test_split(ORL_train, ORL_label, test_size=0.2, random_state=1127)

        knn = KNeighborsClassifier(n_neighbors=1)  # use 1-NN model to train
        start = time.time()
        knn.fit(x_train, y_train)
        training_time = time.time() - start

        y_pred = knn.predict(x_test)
        acc = accuracy_score(y_pred, y_test)
        print(f"Accuracy:{acc}, Training time:{training_time}")
        acc_list.append(acc)
        time_list.append(training_time)
        del knn

    fig = plt.figure(figsize=(8, 5))
    plt.plot(dim_list[:8], acc_list[:8], marker="o")

    for x, y in zip(dim_list[:8], acc_list[:8]):
        plt.text(x, y + 0.005, '%.3f' % y, ha='center', va='bottom', fontsize=7.5)  # y_axis_data1加标签数据

    plt.legend = ('lower right')
    plt.xlabel('Feature Dimensions')
    plt.ylabel('Accuracy')
    plt.title('Accuracy of PCA with Different Dimensions')
    plt.show()

    # LDA
    # fig = plt.figure(figsize=(8, 5))
    # plt.plot(dim_list, acc_list, marker="o")
    # # plt.plot(dim_list, time_list, c='g')
    #
    # for x, y in zip(dim_list, acc_list):
    #     plt.text(x, y + 0.0018, '%.3f' % y, ha='center', va='bottom', fontsize=7.5)  # y_axis_data1加标签数据
    #
    # plt.legend = ('lower right')
    # plt.xticks(np.arange(1, 19, 1))
    # plt.xlabel('Feature Dimensions')
    # plt.ylabel('Accuracy')
    # plt.title('Accuracy of LDA with Different Dimensions (UCI_entropy, seed=3074)')
    # plt.show()




