import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.datasets import load_digits


class SoftmaxNN:
    """softmax多分类神经网络实现"""
    def __init__(self, alpha, times, input_size, hidden_size, output_size):
        """初始化函数
        
        parameters
        -----
        alpha: float
            学习率
        times: int
            迭代次数
        input_size: int
            输入特征数
        hidden_size: int
            隐藏层特征数
        output_size: int
            输出特征数
        """
        self.alpha = alpha
        self.times = times
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.loss_ = []

        self.w1_ = np.random.randn(self.hidden_size, self.input_size) * self.alpha
        self.b1_ = np.zeros((self.hidden_size, 1))
        self.w2_ = np.random.randn(self.output_size, self.hidden_size) * self.alpha
        self.b2_ = np.zeros((self.output_size, 1))
    
    def sigmoid(self, z):
        """激活函数sigmoid"""
        return 1.0 / (1.0 + np.exp(-z))
    
    def sigmoid_deriv(self, a):
        """sigmoid导数实现"""
        return a * (1 - a)

    def softmax(self, z):
        """softmax实现"""
        exp = np.exp(z - np.max(z, axis=0, keepdims=True))
        return exp / np.sum(exp, axis=0, keepdims=True)
    
    def forward(self, X):
        """前向传播实现
        
        parameters
        -----
        X: 数组类型。[输入特征数量，样本数量]
            训练数据
        """
        Z1 = np.dot(self.w1_, X) + self.b1_
        A1 = self.sigmoid(Z1)

        Z2 = np.dot(self.w2_, A1) + self.b2_
        A2 = self.softmax(Z2)

        return Z1, A1, Z2, A2
    
    def compute_loss(self, y, A2):
        """损失函数实现
        
        parameters
        -----
        y: 数组类型。[输出层特征数量， 样本数量]
            真实值
        A3: 数组类型。[输出层特征数量， 样本数量]
            预测值
        
        Returns
        -----
        result: float
            损失值"""
        return -np.mean(np.sum(y * np.log(A2), axis=0))
    
    def backward(self, X, y, Z1, A1, A2):
        """反向传播实现
        
        parameters
        -----
        X: 数组类型。[输入特征数量，样本数量]
            训练数据
        y: 数组类型。[输出特征数量， 样本数量]
            真实值
        Z1: 数组类型。[隐藏层特征数量，样本数量]
            隐藏层
        A1：数组类型。[隐藏层特征数量，样本数量]

        """
        dZ2 = A2 - y
        dw2 = np.dot(dZ2, A1.T)
        db2 = np.sum(dZ2, axis=1, keepdims=True)

        dA1 = np.dot(self.w2_.T, dZ2)
        dZ1 = dA1 * self.sigmoid_deriv(A1)
        dw1 = np.dot(dZ1, X.T)
        db1 = np.sum(dZ1, axis=1, keepdims=True)

        self.w1_ -= self.alpha * dw1
        self.b1_ -= self.alpha * db1
        self.w2_ -= self.alpha * dw2
        self.b2_ -= self.alpha * db2


    def fit(self, X, y):
        """训练模型
        
        parameters
        -----
        X: 数组类型。[输入特征数量，样本数量]
            训练数据
        y: 数组类型。[输出特征数量， 样本数量]
            真实值
        """
        for i in range(self.times):
            Z1, A1, Z2, A2 = self.forward(X)
            loss = self.compute_loss(y, A2)
            self.loss_.append(loss)
            self.backward(X, y, Z1, A1, A2)
            print(f"epoch {i}, loss: {loss:4f}")


    def predict(self, X):
        """根据输入数据进行预测"""
        _, _, _, A2 = self.forward(X)
        return np.argmax(A2, axis=0)


if __name__ =="__main__":
    # 加载数据集
    data = load_digits()
    
    # 数据预处理
    x = data.data / 16
    y = data.target.reshape(1, -1).T

    # 将真实值转化为onehot编码
    transform = OneHotEncoder(sparse_output=False)

    # 分割数据集
    train_x, test_x, train_y, test_y = train_test_split(x, y, random_state=22)

    # 模型训练
    snn = SoftmaxNN(alpha=0.0002, times=4000, input_size=64, hidden_size=200, output_size=10)
    snn.fit(train_x.T, transform.fit_transform(train_y).T)
    result = snn.predict(test_x.T)

    # 可视化
    mpl.rcParams["font.family"] = "SimHei"
    mpl.rcParams["axes.unicode_minus"] = False

    # plt.subplot(1, 2, 1)
    plt.plot(result.flatten(), "ro-", label="预测值")
    plt.plot(test_y.flatten(), "go-", label="真实值")
    plt.legend(loc="best")
    plt.title("手写数字识别")
    plt.xlabel("节点序号")
    plt.ylabel("识别数字")
    plt.subplot(1, 2, 2)
    plt.plot(np.arange(0, snn.times), snn.loss_, label="损失值")
    plt.title("损失函数")
    plt.xlabel("迭代次数")
    plt.ylabel("损失值")
    plt.legend(loc="best")
    plt.show()
