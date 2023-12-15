# Pattern Recognition 2023 Fall
# Chenkai GUO
# Date: 2023.12.10

import random
import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.utils.data as Data
import torch.optim as optim


# Linear neural network
class Linear_network(torch.nn.Module):

    def __init__(self, in_size, hid_size, out_size):
        super(Linear_network, self).__init__()
        self.hid_size = hid_size

        # Architecture
        # Input -> Linear(Tanh activation) -> Output(sigmoid)
        self.linear1 = nn.Linear(in_size, hid_size)
        self.tanh = nn.Tanh()
        self.linear2 = nn.Linear(hid_size, out_size)
        self.sigmoid = nn.Sigmoid()

    def init_params(self):
        self.linear1.reset_parameters()
        self.linear2.reset_parameters()

    def forward(self, x):
        x = self.linear1(x)
        x = self.tanh(x)
        x = self.linear2(x)
        x = self.sigmoid(x)
        return x

# initial params
def init_params(model):
    if isinstance(model, nn.Linear):
        nn.init.xavier_normal_(model.weight)
        nn.init.constant_(model.bias, 0)

# dataloader
def dataloader(data, labels, batch_size, is_train=True):
    dataset = Data.TensorDataset(data, labels)
    return Data.DataLoader(dataset, batch_size, shuffle=is_train)

# set all seeds together
def seed_all(seed):
    if not seed:
        seed = 0

    log("[ Using Seed : ", seed, " ]")

    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

# visualize training process
def Loss_viz(model, viz_list1, viz_list2):
    epoch_list = list(range(1,num_epochs+1))
    model_list = ['model1 (h=3)', 'model2 (h=10)', 'model3 (h=25)', 'model4 (h=50)', 'model5 (h=100)']
    plt.suptitle('Training results of models with different hid_size')
    plt.figure(figsize=(12, 6))
    plt.subplot(1,2,1)
    for i in range(5):
        plt.plot(epoch_list, viz_list1[i])
    plt.xlabel('epoch')
    plt.ylabel('loss')
    plt.legend(loc='upper right', labels=model_list)
    plt.title('Training Loss')

    plt.subplot(1,2,2)
    for i in range(5):
        plt.plot(epoch_list, viz_list2[i])
    plt.xlabel('epoch')
    plt.ylabel('loss')
    plt.legend(loc='upper right', labels=model_list)
    plt.title('Testset MSE error')

    img = 'Linear network training results.png'
    plt.savefig(img)
    # plt.show()

# visualize origin dataset:
def TriD_plot(dataset):
    colors = ['#c72e29', '#098154', '#fb832d']
    markers = ['s', 'o', '^']
    labels = ['class 1', 'class 2', 'class 3']

    fig = plt.figure(figsize=(12, 12))
    # fig.suptitle('Data Distribution', fontsize=20)

    ax = fig.add_subplot(221, projection='3d')
    for i in range(3):
        ax.scatter(dataset[0 + 10 * i: 10 + 10 * i, 0],
                   dataset[0 + 10 * i: 10 + 10 * i, 1],
                   dataset[0 + 10 * i: 10 + 10 * i, 2],
                   s=50,
                   c=colors[i],
                   marker=markers[i],
                   alpha=0.8,
                   facecolors='r',
                   edgecolors='none',
                   linewidths=1,
                   label=labels[i])
    plt.legend(loc='upper right')
    ax.view_init(elev=30,  # elevation
                 azim=30  # azimuth
                 )
    plt.title('A. 3D distribution of dataset')

    plt.subplot(2, 2, 2)
    for i in range(3):
        plt.scatter(dataset[0 + 10 * i: 10 + 10 * i, 0],
                    dataset[0 + 10 * i: 10 + 10 * i, 1],
                    s=50,
                    c=colors[i],
                    marker=markers[i],
                    alpha=0.8,
                    facecolors='r',
                    edgecolors='none',
                    linewidths=1,
                    label=labels[i])
    plt.legend(loc='upper right')
    plt.title('B. x-y 2D distribution of dataset')

    plt.subplot(2, 2, 3)
    for i in range(3):
        plt.scatter(dataset[0 + 10 * i: 10 + 10 * i, 0],
                    dataset[0 + 10 * i: 10 + 10 * i, 2],
                    s=50,
                    c=colors[i],
                    marker=markers[i],
                    alpha=0.8,
                    facecolors='r',
                    edgecolors='none',
                    linewidths=1,
                    label=labels[i])
    plt.legend(loc='upper right')
    plt.title('C. x-z 2D distribution of dataset')

    plt.subplot(2, 2, 4)
    for i in range(3):
        plt.scatter(dataset[0 + 10 * i: 10 + 10 * i, 1],
                    dataset[0 + 10 * i: 10 + 10 * i, 2],
                    s=50,
                    c=colors[i],
                    marker=markers[i],
                    alpha=0.8,
                    facecolors='r',
                    edgecolors='none',
                    linewidths=1,
                    label=labels[i])
    plt.legend(loc='upper right')
    plt.title('D. y-z 2D distribution of dataset')

    img = 'origin_data_distribution.png'
    plt.savefig(img)
    # plt.show()


# Define loss function
loss = nn.MSELoss()
# hid_list = [3,5,10,25,50]
num_epochs = 50
# lr_list = [0.01, 0.05, 0.1]
train_loss_list = []
test_loss_list = []
viz_list1 = []
viz_list2 = []

# Training model
if __name__ == '__main__':
    # set my birthday as random seed
    seed_all(1127)

    # choose 70% origin data for training
    trainset = np.loadtxt('./hw4_data.txt', skiprows = 0, delimiter=',')
    # choose 30% origin data for testing
    testset = np.loadtxt('./hw4_test.txt', skiprows=0, delimiter=',')

    TriD_plot(trainset+testset)

    train_data = torch.Tensor(trainset[:, 0:3])
    train_labels = torch.Tensor(trainset[:, 3]).reshape(-1, 1)
    test_data = torch.Tensor(testset[:, 0:3])
    test_labels = torch.Tensor(testset[:, 3]).reshape(-1, 1)

    data_iter = dataloader(train_data, train_labels, batch_size=4)

    model1 = Linear_network(3, 3, 1)
    model2 = Linear_network(3, 10, 1)
    model3 = Linear_network(3, 25, 1)
    model4 = Linear_network(3, 50, 1)
    model5 = Linear_network(3, 100, 1)
    model_list = [model1, model2, model3, model4, model5]

    for model in model_list:
        model.apply(init_params)
        trainer = optim.SGD(model.parameters(), lr=0.03)

        for epoch in range(num_epochs):
            for x, y in data_iter:
                l = loss(model(x), y)
                trainer.zero_grad()
                l.backward()
                trainer.step()
            # calculate loss
            train_loss = loss(model(train_data), train_labels)
            test_loss = loss(model(test_data), test_labels)
            train_loss_list.append(train_loss.detach().numpy())
            test_loss_list.append(test_loss.detach().numpy())
            # print(np.array(train_loss_list).shape)
        viz_list1.append(train_loss_list)
        viz_list2.append(test_loss_list)
        train_loss_list = []
        test_loss_list = []
        # print(f'epoch {epoch + 1}, loss {l:f}')
    # print(loss_list)
    print(np.array(viz_list1).shape)
    print(np.array(viz_list2).shape)
    Loss_viz(model1, viz_list1, viz_list2)

    print('Success! Congrats!')

