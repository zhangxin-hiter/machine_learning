import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt

class perceptron:
    """利用python实现感知器"""

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

    def step(self, z):
        """阶跃函数
        
        parameters
        -----
        z: 数组类型
            阶跃函数参数
            
        Returns
        -----
        value: int
            如果z>=0, 返回1，否则返回-1
        """

        return np.where(z >= 0, 1, -1)
    
    def fit(self, X, y):
        """根据训练数据，对模型进行训练
        
        parameters
        -----
        X: 类数组类型。形状[样本数量，特征数量]
            待训练样本
            
        y: 类数组类型。形状[样本数量]
            目标
        """
        X = np.asarray(X)
        y = np.asarray(y)
        # 创建权重向量
        self.w_ = np.zeros(1 + X.shape[1])
        # 创建损失列表
        self.loss_ = []

        for i in range(self.times):
            loss = 0
            for x, target in zip(X, y):
                # 计算预测值
                y_hat = self.step(np.dot(x, self.w_[1:] + self.w_[0]))
                loss += y_hat != target
                # 更新权重
                self.w_[0] += self.alpha * (target - y_hat)
                self.w_[1:] += self.alpha * (target - y_hat) * x

            self.loss_.append(loss)

    def predict(self, X):
        """根据参数，进行预测
        
        parameters
        -----
        X: 类数组类型。形状[样本数量，特征数量]
            要预测的数据
        
        Returns
        -----
        result: 数组类型
            预测结果
            
        """
        return self.step(np.dot(X, self.w_[1:]) + self.w_[0])



if __name__ == "__main__":
    data = pd.read_csv("iris.csv")
    data.drop_duplicates(inplace=True)
    data["species"] = data["species"].map({"setosa":-1, "versicolor":0, "virginica":1})
    data = data[data["species"] != 0]

    t1 = data[data["species"] == 1]
    t2 = data[data["species"] == -1]
    t1 = t1.sample(len(t1), random_state=0)
    t2 = t2.sample(len(t2), random_state=0)

    train_x = pd.concat([t1.iloc[:40, :-1], t2.iloc[:40, :-1]], axis=0)
    train_y = pd.concat([t1.iloc[:40, -1], t2.iloc[:40, -1]], axis=0)
    test_x = pd.concat([t1.iloc[40:, :-1], t2.iloc[40:, :-1]], axis=0)
    test_y = pd.concat([t1.iloc[40:, -1], t2.iloc[40:, -1]], axis=0)

    lr = perceptron(0.1, 20)
    lr.fit(train_x, train_y)
    result = lr.predict(test_x)
    
    mpl.rcParams["font.family"] = "SimHei"
    mpl.rcParams["axes.unicode_minus"] = False

    plt.plot(test_y.values, "go", label="真实值", ms=8)
    plt.plot(result, "ro", label="预测值")
    plt.title("鸢尾花预测")
    plt.xlabel("节点序号")
    plt.ylabel("预测结果")
    plt.legend(loc="best")
    plt.show()