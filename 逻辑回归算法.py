import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt

class logisticregression:
    """使用python实现逻辑回归算法"""

    def __init__(self, alpha, times):
        """初始化方法
        
        parameters
        -----
        alpha: float
            学习率
        times: int
            迭代次数
        """
        self.alpha = alpha
        self.times = times

    def sigmoid(self, z):
        """sigmoid函数实现
        
        parameters:
        -----
        z: float
            自变量
        
        Returns 
        -----
        p: float
            概率值
        """
        return 1.0 / (1.0 + np.exp(-z))
    
    def fit(self, X, y):
        """根据提供数据，对模型训练
        
        parameters
        -----
        X: 类数组类型。形状[样本数量， 特征数量]
            训练样本
        y: 类数组类型。形状[样本数量]
            目标值
        """

        X = np.asarray(X)
        y = np.asarray(y)

        self.w_ = np.zeros(1 + X.shape[1])

        self.loss_ = []

        for i in range(self.times):
            z = np.dot(X, self.w_[1:]) + self.w_[0]
            # 计算概率值
            p = self.sigmoid(z)
            # 计算损失函数
            cost = -np.sum(y * np.log(p) + (1 - y) * np.log(1 - p))
            self.loss_.append(cost)

            # 调整权重值
            self.w_[0] += self.alpha * np.sum(y - p)
            self.w_[1:] += self.alpha * np.dot(X.T, y - p)

    def predict_proba(self, X):
        """根据传递参数，进行预测
        
        parameters
        -----
        X: 类数组类型。[样本数量，特征数量]
            测试样本
        
        Returns 
        -----
        result: 类数组类型。[样本数量]
            目标值
        """
        X = np.asarray(X)
        z = np.dot(X, self.w_[1:]) + self.w_[0]
        p = self.sigmoid(z)
        # 将预测结果转化为二维结构
        p = p.reshape(-1, 1)
        # 拼接
        return np.concatenate([1 - p, p], axis=1)
    
    def predict(self, X):
        """根据传递参数，进行预测
        
        parameters
        -----
        X: 类数组类型。[样本数量，特征数量]
            测试样本
        
        Returns 
        -----
        result: 类数组类型。[样本数量]
            目标值
        """

        return np.argmax(self.predict_proba(X), axis=1)


if __name__ == "__main__":
    data = pd.read_csv("iris.csv")
    # 删除鸢尾花数据集中重复数据
    data.drop_duplicates(inplace=True)
    data["species"] = data["species"].map({"setosa":0, "versicolor":1, "virginica":2})
    # 删除species==2的记录
    data = data[data["species"] != 2]
    # 对数据进行洗牌
    t1 = data[data["species"] == 0]
    t2 = data[data["species"] == 1]
    t1 = t1.sample(len(t1), random_state=0)
    t2 = t2.sample(len(t2), random_state=0)
    # 拼接并分割数据集
    train_x = pd.concat([t1.iloc[:40, :-1], t2.iloc[:40, :-1]], axis=0)
    train_y = pd.concat([t1.iloc[:40, -1], t2.iloc[:40, -1]], axis=0)
    test_x = pd.concat([t1.iloc[40:, :-1], t2.iloc[40:, :-1]], axis=0)
    test_y = pd.concat([t1.iloc[40:, -1], t2.iloc[40:, -1]], axis=0)
    # 创建预测器
    lr = logisticregression(0.0005, 2000)
    lr.fit(train_x, train_y)
    result = lr.predict(test_x)
    right = test_x[result == test_y]
    wrong = test_x[result != test_y]
    # 可视化
    mpl.rcParams["font.family"] = "SimHei"
    mpl.rcParams["axes.unicode_minus"] = False
    plt.figure(figsize=[5, 3])
    plt.subplot(1, 2, 1)
    plt.plot(result, "ro", label="预测值", ms=8)
    plt.plot(test_y.values, "go", label="真实值")
    plt.title("预测值真实值对比")
    plt.xlabel("节点序号")
    plt.ylabel("种类")
    plt.legend(loc="best")
    plt.subplot(1, 2, 2)
    plt.plot(np.arange(1, lr.times + 1), lr.loss_, label="损失值")
    plt.title("损失迭代")
    plt.xlabel("times")
    plt.ylabel("loss")
    plt.legend(loc="best")
    plt.show()



