import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt

class standardscaler:
    """对数据进行标准化处理"""

    def fit(self, X):
        """根据传递的样本，计算每个特征列的均值和标准差
        
        parameters
        -----
        X: 类数组类型
            计算均值和标准差
        
        """
        X = np.asarray(X)
        self.std_ = np.std(X, axis=0)
        self.mean_ = np.mean(X, axis=0)

    def transform(self, X):
        """对给定的数据X， 进行标准化处理
        
        parameters
        -----
        X: 类数组类型。
            待转化的数据
        """
        return (X - self.mean_) / self.std_
    
    def fit_transform(self, X):
        """对数据训练转化，返回转化之后的结果
        
        parameters
        -----
        X: 类数组类型
            待转化的数据
            
        Returns
        -----
        result: 数组类型
            转化后的数据
        """
        self.fit(X)
        return self.transform(X)

class LinearRegression:
    """使用python实现梯度下降"""

    def __init__(self, alpha, times):
        """初始化方法
        
        parameters
        -----
        alpha: float
            学习率。控制步长
        times: int
            循环迭代次数
            
        """
        self.alpha = alpha
        self.times = times

    def fit(self, X, y):
        """根据提供训练数据，对模型训练
        
        parameters
        -----
        X: 类数组类型。形状为[样本数量,特征数量]
            训练数据
        y: 类数组类型。形状为[样本数量]
            目标
        
        """
        X = np.asarray(X)
        y = np.asarray(y)

        # 创建权重的向量
        self.w_ = np.zeros(1 + X.shape[1])
        # 创建损失列表
        self.loss_ = []
        for i in range(self.times):
            # 计算预测值
            y_hat= np.dot(X, self.w_[1:]) + self.w_[0]
            # 计算真实值与预测值的差距
            error = y - y_hat
            # 将损失值加入损失列表
            self.loss_.append(np.sum(error ** 2) / 2)
            # 根据差距调整权重
            self.w_[0] += self.alpha * np.sum(error)
            self.w_[1:] += self.alpha * np.dot(X.T, error)
            print(i)

    def predict(self, X):
        """根据传递参数，进行预测
        
        parameters
        -----
        X: 类数组类型。形状[样本数量，特征数量]
            预测数据
        Returns
        -----
        result: 数组类型
            预测结果
        """
        X = np.asarray(X)
        result = np.dot(X, self.w_[1:]) + self.w_[0]
        return result


if __name__ == "__main__":
    data = pd.read_csv("house_data.csv")
    data = data.sample(len(data), random_state=0)
    transform = standardscaler()

    lr = LinearRegression(alpha=0.0005, times=2000)

    train_x = data.iloc[:400, 5:6]
    train_y = data.iloc[:400:, -1]
    test_x = data.iloc[400:, 5:6]
    test_y = data.iloc[400:, -1]
    train_x = transform.fit_transform(train_x)
    test_x = transform.transform(test_x)

    lr.fit(train_x, train_y)
    result = lr.predict(test_x)

    mpl.rcParams["font.family"] = "SimHei"
    mpl.rcParams["axes.unicode_minus"] = False
    plt.figure(figsize=(8, 4))
    plt.subplot(1, 3, 1)
    plt.plot(result, "ro-", label="预测值", marker="*")
    plt.plot(test_y.values, "go--", label="真实值", marker=">")
    plt.title("波士顿房价预测")
    plt.xlabel("节点序号")
    plt.ylabel("房价")
    plt.legend(loc="best")
    plt.subplot(1, 3, 2)
    plt.plot(range(1, lr.times + 1), lr.loss_, "o-", label="loss")
    plt.title("loss")
    plt.xlabel("time")
    plt.ylabel("loss")
    plt.legend(loc="best")
    plt.subplot(1, 3, 3)
    plt.scatter(train_x, train_y, color="r", label="实际数据")
    x = np.arange(-5, 5, 0.1)
    y = lr.w_[0] + lr.w_[1] * x
    plt.plot(x, y, label="训练结果")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.title("模型训练拟合")
    plt.legend(loc="best")
    plt.show()