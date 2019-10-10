import numpy as np
import pickle
import os
import random

random.seed(1)


class KNearestNeighbor(object):
    def __init__(self):
        pass

    def train(self, X, y):
        """ X is N x D where each row is an example. Y is 1-dimension of size N """
        # the nearest neighbor classifier simply remembers all the training data
        self.Xtr = X
        self.ytr = y

    def predict(self, X, k=1):
        """ X is N x D where each row is an example we wish to predict label for """
        """ k is the number of nearest neighbors that vote for the predicted labels."""
        # num_test = X.shape[0] # 使用validation
        num_test = 1

        # lets make sure that the output type matches the input type
        Ypred = np.zeros(num_test, dtype=self.ytr.dtype)
        # print('num_test', num_test)
        # print('type of num_test',type(num_test))

        # loop over all test rows
        for i in range(num_test):
            # using the L1 distance (sum of absolute value differences)
            distances = np.sum(np.abs(self.Xtr - X[i, :]), axis=1)
            # L2 distance:
            # distances = np.sqrt(np.sum(np.square(self.Xtr - X[i, :]), axis=1))
            # argsort()函数是将x中的元素从小到大排列，提取其对应的index(索引)，然后输出到y
            indexes = np.argsort(distances)
            Yclosest = self.ytr[indexes[:k]]
            # np.bincount()返回一个数组,这个数组比他的参数的最大值大1,返回的数组是0-max每个数的频率
            cnt = np.bincount(Yclosest)
            # 取出cnt中元素最大值所对应的索引
            Ypred[i] = np.argmax(cnt)

        return Ypred

# 返回两个数组X-->10000*32*32*3; Y-->10000*1
def load_CIFAR_batch(file):
    """ load single batch of cifar """
    with open(file, 'rb') as f:
        datadict = pickle.load(f, encoding='latin1')# 字典
        X = datadict['data'] #数组:10000*3072
        # print('X', X, X.shape)
        Y = datadict['labels'] #列表:10000
        X = X.reshape(10000, 3, 32, 32).transpose(0, 2, 3, 1).astype("float")
        # 原来的顺序是0,1,2,3；现在是0,2,3,1；相当于把1-->维度等于三的那一列放到最后;使10000*32*3*32变成10000*32*32*3
        Y = np.array(Y)
        # print('x',X,X.shape,'y', Y.shape)
    return X, Y


def load_CIFAR10(ROOT):
    """ load all of cifar """
    xs = []
    ys = []
    # 根据cifar10数据集的特点 将data_batch_1-5
    for b in range(1, 6):# 前开后闭
        f = os.path.join(ROOT, 'data_batch_%d' % (b, ))
        X, Y = load_CIFAR_batch(f)
        xs.append(X)
        ys.append(Y)
    # 把batch1-5的所有数据加载到Xtr和Ytr中
    Xtr = np.concatenate(xs)  # 拼接, 默认axis=0-->拼接到原数组的下方变成50000*32*32*3==>使第一维增加;若axis=1-->使第二维增加....以此类推
    Ytr = np.concatenate(ys)  # 一维的,只能在后面拼.变成50000

    del X, Y
    Xte, Yte = load_CIFAR_batch(os.path.join(ROOT, 'test_batch'))
    return Xtr, Ytr, Xte, Yte

if __name__ == "__main__":
    Xtr, Ytr, Xte, Yte = load_CIFAR10('./data/cifar-10-batches-py')  # a magic function we provide
    # Xtr(50000*32*32*3)是训练集中所有的图像,Ytr(50000*1)是图像对应的标签(0-9)
    # flatten out all images to be one-dimensional 将每一张图片变成行向量
    # Xtr.shape[0] = 50000
    Xtr_rows = Xtr.reshape(Xtr.shape[0], 32 * 32 * 3)  # Xtr_rows becomes 50000 x 3072
    Xte_rows = Xte.reshape(Xte.shape[0], 32 * 32 * 3)  # Xte_rows becomes 10000 x 3072


    # assume we have Xtr_rows, Ytr, Xte_rows, Yte as before
    # recall Xtr_rows is 50,000 x 3072 matrix
    # 拆出验证集计算准确率0-1000验证集,,1000-50000训练集,,50000-60000测试集
    # 把训练集分成训练集和验证集。使用验证集来对所有超参数调优。最后只在测试集上跑一次并报告结果。
    Xval_rows = Xtr_rows[:1000, :]  # take first 1000 for validation
    Yval = Ytr[:1000]
    Xtr_rows = Xtr_rows[1000:, :]  # keep last 49,000 for train
    Ytr = Ytr[1000:]

    # find hyperparameters that work best on the validation set
    # 尝试k值[1, 3, 5, 10, 20, 50, 100]用validation测试哪一个的准确率最高
    validation_accuracies = []
    for k in [1, 3, 5, 10, 20, 50, 100]:
        # use a particular value of k and evaluation on validation data
        knn = KNearestNeighbor()
        knn.train(Xtr_rows, Ytr)
        # here we assume a modified NearestNeighbor class that can take a k as input
        Yval_predict = knn.predict(Xval_rows, k=k)
        acc = np.mean(Yval_predict == Yval)
        # if Yval_predict == Yval 取1,if Yval_predict =! Yval 取零 加起来取平均值
        print('accuracy: %f' % (acc,))

        # keep track of what works on the validation set
        validation_accuracies.append((k, acc))

    # 使用validation，分析出哪个k值表现最好，然后用这个k值来跑真正的测试集，并作出对算法的评价。
    print('k= ', k,'accuracies',validation_accuracies)


    # 测试集test
    knn = KNearestNeighbor() # create a Nearest Neighbor classifier class
    knn.train(Xtr_rows, Ytr) # train the classifier on the training images and labels
    Yte_predict = knn.predict(Xte_rows) # predict labels on the test images
    # and now print the classification accuracy, which is the average number
    # of examples that are correctly predicted (i.e. label matches)
    print ('test_accuracy: %f' % (np.mean(Yte_predict == Yte)))
