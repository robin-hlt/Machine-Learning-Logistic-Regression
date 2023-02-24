import numpy as np
import random
import copy


def trans(s):  # 用字典将鸢尾花类别映射成数字
    dict = {"Iris-setosa": 0, "Iris-versicolor": 1, "Iris-virginica": 2}
    value = dict[s]
    return value


def read_file(filename):  # 读取鸢尾花数据
    random.seed(255)
    f = open(filename, "r")
    dataset = []
    for line in f.readlines():
        line = line.strip().split(",")
        dataset.append([float(line[1]), float(line[2]), float(line[3]), float(line[4]), trans(line[-1])])
    f.close()
    dataset = np.array(dataset)
    np.random.shuffle(dataset)
    return dataset


def data_label_split(dataset, catagory):  # 分离数据与标签，并将3类标签转换为2类，catagory表示分类目标是第几类别
    data = dataset[:, 0:-1]
    label = dataset[:, -1].reshape(150, 1)
    label = np.where(label == catagory, 1, 0)
    return data, label


def logistic_model(z):
    return 1.0 / (1 + np.exp(-z))


def ten_fold_train(data, label, max_iter, index):
    print("trainning")
    label_test = copy.deepcopy(label[0+index*15: 15+index*15])  #
    label_train = copy.deepcopy(np.delete(label, [x for x in range(0+index*15,15+index*15)],axis=0))
    data_test = copy.deepcopy(data[0+index*15: 15+index*15])
    data_train = copy.deepcopy(np.delete(data, [x for x in range(0+index*15,15+index*15)],axis=0))
    w = np.ones((data_train.shape[1]+1, 1))  # 初始化w
    a = np.ones((data_train.shape[0], 1))
    data = np.c_[data_train, a]
    # 梯度下降法求w
    n = 0.001  # 步长
    iteration = 0
    # 每次迭代计算一次正确率
    acc = 0
    while iteration <= max_iter:
        # 计算当前参数w下的预测值
        pred = logistic_model(np.dot(data, w))
        # 梯度下降
        loss = pred - label_train.reshape(data_train.shape[0],1)
        grad = np.dot(np.transpose(data), loss)
        w = w - grad * n
        # 预测，更新正确率
        rightrate = calculate(data_train, label_train, w)
        print("iter:{} train_accuracy:{}".format(iteration,rightrate))
        rightrate2 = calculate(data_test, label_test, w)
        print("        test_accuracy:{}".format(rightrate2))
        acc = rightrate2
        iteration += 1
    return acc


def calculate(data, label, w):
    a = np.ones((data.shape[0], 1))
    data = np.c_[data, a]
    # 使用训练好的参数w进行计算
    y = logistic_model(np.dot(data, w))
    row, col = np.shape(y)
    # 记录预测正确的个数，用于计算正确率
    rightcount = 0
    for i in range(row):
        # 预测标签
        flag = -1
        # 大于0.5的为正例
        if y[i] > 0.5:
            flag = 1
        # 小于等于0.5的为反例
        else:
            flag = 0
        # 记录预测正确的个数
        if label[i] == flag:
            rightcount += 1
    # 正确率
    rightrate = rightcount / data.shape[0]
    return rightrate


if __name__ == '__main__':
    filename = "Iris.csv"
    dataset = read_file(filename)
    ls = []
    for index in range(10):
        data, label = data_label_split(dataset, 2)
        acc = ten_fold_train(data, label, 30, index)
        ls.append(acc)
    average_acc = sum(ls)/len(ls)
    print("ten fold cross validation accuracy:{} ".format(average_acc))


