from sklearn.datasets import load_breast_cancer
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt

class SNN:
    """利用python实现一个浅层神经网络"""

    def __init__(self, input_size, hidden_size, times, alpha=0.01):
        """初始化实现
        
        parameters
        -----
        input_size: int
            输入大小
        hidden_size: int
            隐藏层大小
        alpha: float
            xuexil
        """
        self.alpha = alpha
        self.times = times
        self.W1_ = np.random.randn(hidden_size, input_size) * self.alpha
        self.b1_ = np.zeros((hidden_size, 1))
        self.W2_ = np.random.randn(1, hidden_size) * self.alpha
        self.b2_ = np.zeros((1, 1))
        self.loss_ = []



    def sigmoid(self, z):
        """实现sigmoid函数
        
        parameters
        -----
        z: float或数组类型
        
        Returns
        -----
        result: float或数组类型
            计算结果
        """

        return 1.0 / (1.0 + np.exp(-z))
    
    def sigmoid_deriv(self, a):
        """sigmoid导数实现
        
        parameters
        -----
        a: float
            sigmoid的输出

        Returns
        -----
        result: float
            导数
        """
        return a * (1 - a)
    
    def relu(self, z):
        """relu函数实现
        
        parameters
        -----
        z: float
        
        Returns
        -----
        result: int
        """

        return np.maximum(0, z)
    
    def relu_deriv(self, z):
        """relu导数实现"""

        return (z > 0).astype(float)
    
    def forward(self, X):
        """前向传播实现
        
        parameters
        -----
        X: 类数组类型
            输入样本
        """

        Z1 = np.dot(self.W1_, X) + self.b1_
        A1 = self.relu(Z1)
        Z2 = np.dot(self.W2_, A1) + self.b2_
        A2 = self.sigmoid(Z2)

        return Z1, A1, Z2, A2
    
    def compute_binary_loss(self, y, A2):
        """计算损失函数
        
        parameters
        -----
        y: 类数组类型。形状[1, 样本数量]
            真实值
        A2: 类数组类型。形状为[1, 样本数量]
            概率值
        
        Returns
        -----
        loss: float
            损失值"""

        loss = -np.mean(y * np.log(A2) + (1 - y) * np.log(1 - A2))
        return loss
    
    def backword(self, X, y, Z1, A1, A2):
        """计算梯度
        
        parameters
        -----
        X: 数组类型。[特征数量， 样本数量]
            输入
        y: 数组类型。[1, 样本数量]
            真实值
        Z1: 数组类型。[隐藏层特征数量，样本数量]
            隐藏层输入
        A1: 数组类型。[隐藏层特征数量，样本数量]
            隐藏层输出
        A2: 数组类型。[1, 样本数量]
        """
        # 样本数量
        m = X.shape[1]  

        dZ2 = A2 - y
        dW2 = np.dot(dZ2, A1.T) / m
        db2 = np.sum(dZ2, axis=1, keepdims=True)

        dA1 = np.dot(self.W2_.T, dZ2)
        dZ1 = dA1 * self.relu_deriv(Z1)
        dW1 = np.dot(dZ1, X.T)
        db1 = np.sum(dZ1, axis=1, keepdims=True) / m

        self.W1_ -= self.alpha * dW1
        self.b1_ -= self.alpha * db1
        self.W2_ -= self.alpha * dW2
        self.b2_ -= self.alpha * db2

    def fit(self, X, y, verbose=True):
        """训练模型"""

        for i in range(self.times):
            Z1, A1, Z2, A2 = self.forward(X)
            loss = self.compute_binary_loss(y, A2)
            self.loss_.append(loss)
            self.backword(X, y, Z1, A1, A2)
            if verbose and i % 100 == 0:
                print(f"epoch {i}, loss: {loss: 4f}")


    def predict(self, X, threshold):
        """对数据进行预测"""
        _, _, _, A2 = self.forward(X)
        return (A2 >= threshold).astype(int)

if __name__ == "__main__":
    # 加载数据集
    data = load_breast_cancer()
    y = data.target.reshape(1, -1)
    # 标准化
    transform = StandardScaler()
    x = transform.fit_transform(data.data)
    # 分割数据集
    train_x, test_x, train_y, test_y = train_test_split(x, y.T, random_state=23)
    # 模型训练
    snn = SNN(data.data.shape[1], 20, times=100000, alpha=0.001)
    snn.fit(train_x.T, train_y.T, verbose=True)
    # 预测
    result = snn.predict(test_x.T, 0.5)
    # 可视化
    mpl.rcParams["font.family"] = "SimHei"
    mpl.rcParams["axes.unicode_minus"] = False
    plt.figure(figsize=(8, 4))
    plt.subplot(1, 2, 1)
    plt.plot(test_y.flatten(), "ro", label="真实值", ms=8)
    plt.plot(result.flatten(), "go", label="预测值")
    plt.title("神经网络初体验")
    plt.legend(loc="best")
    plt.xlabel("节点序号")
    plt.ylabel("目标值")
    plt.subplot(1, 2, 2)
    plt.plot(np.arange(0, snn.times), snn.loss_)
    plt.xlabel("迭代次数")
    plt.ylabel("损失值")
    plt.title("损失函数")
    plt.show()
    

