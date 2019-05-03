import numpy as np 
import scipy.io as scio
import matplotlib.pyplot as plt

class Perceptron():
    """
    data: 训练数据，二维矩阵，行为数据个数；列为数据特征，（N x M） 维
    label： 训练标签，取值为1，-1，（1 x N） 维

    """
    def __init__(self, data, label, learningRate):
        self.data = data
        self.label = label
        self.learningRate = learningRate

    def train(self):
        data_num = self.data.shape[0] # 训练集长度
        data_width = self.data.shape[1] #实例维度
        print(data_num,data_width)
        a = np.zeros(data_num)
        b = 0
        inner_product = np.zeros([data_num, data_num]) # 实列的内积表
        for i in range(data_num):
            for j in range(data_num):
                inner_product[i,j] = sum(self.data[i,:] * self.data[j,:])
        count = 0 # 记录误分类实例数
        while True:
            for i in range(data_num):
                loss = 0
                for j in range(data_num):
                    loss  += a[j] * self.label[j] * inner_product[j,i]
                loss = self.label[i] * (loss + b)
                if loss <= 0:
                    a[i] = a[i] + self.learningRate
                    b = b + self.label[i] * self.learningRate
                    count = 0
            count += 1
            if count == data_num:
                break
        w = np.zeros(data_width)
        b = 0
        for i in range(data_num):
            w += a[i] * self.label[i] * self.data[i,:]
            b += a[i] * self.label[i]
        return w, b


if __name__ == '__main__':
    D1 = scio.loadmat('')
    data = D1['data']
    #myPerceptron = Perceptron()
    Data = data[:,0:2]
    label = data[:,2]
    #print(Data,label)
    myPerceptron = Perceptron(Data, label, 0.1)
    w, b = myPerceptron.train()
    test = np.zeros(100)
    for i in range(100):
        if (w[0]*Data[i,0]+w[0]*Data[i,1]+b)>0:
            test[i] = 1
        else:
            test[i] = -1
    count = 0
    for i in range(100):
        if test[i] != label[i]:
            count += 1
    print('count:',count)

    print('w:',w,'b:',b)
    for i in range(100):
        if data[i,2] == 1:
            plt.plot(data[i, 0], data[i, 1], 'or')
        if data[i,2] == -1:
            plt.plot(data[i, 0], data[i, 1], 'ob')
    x = np.arange(0,1,0.05)
    y = -(w[0]/w[1]) * x - b/w[1]
    plt.plot(x,y,'-g')
    plt.show()







