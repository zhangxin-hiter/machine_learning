import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt

class KMeans:
    """使用python实现KMeans"""

    def __init__(self, k, times):
        """初始化方法
        
        parameters
        -----
        k: int
            聚类的个数
        times: int
            迭代的次数
        """

        self.k = k
        self.times = times

    def fit(self, X):
        """根据提供的数据，对模型进行训练
        
        parameters
        -----
        X: 类数组类型。形状为[样本数量，特征数量]
            待训练的数据
        """

        X = np.asarray(X)
        # 设置随机数种子
        np.random.seed(0)
        self.cluster_centers_ = X[np.random.randint(0, len(X), self.k)]
        self.labels_ = np.zeros(len(X))

        for i in range(self.times):
            for index, x in enumerate(X):
                # 计算每个样本到聚类中心的距离
                dis = np.sqrt(np.sum((x - self.cluster_centers_) ** 2, axis=1))
                self.labels_[index] = dis.argmin()
            # 循环遍历每一个簇
            for t in range(self.k):
                self.cluster_centers_[t] = np.mean(X[self.labels_ == t], axis=0)

    def predict(self, X):
        """对传入的数据进行预测
        
        parameters
        -----
        X: 类数组类型。[样本数量，特征数量]
            要预测的数据
        Returns
        -----
        result: 数组类型
            预测结果，所属簇
        """
        X = np.asarray(X)
        result = np.zeros(len(X))
        for index, i in enumerate(X):
            dis = np.sqrt(np.sum((i - self.cluster_centers_) ** 2, axis=1))
            result[index] = dis.argmin()
        return result


if __name__ == "__main__":
    data = pd.read_csv("house_data.csv")
    # data.drop("species", axis=1, inplace=True)
    data.drop_duplicates(inplace=True)
    data = data.iloc[:, :2]
    kmeans = KMeans(3, 50)
    kmeans.fit(data)
    mpl.rcParams["font.family"] = "SimHei"
    mpl.rcParams["axes.unicode_minus"] = False
    plt.scatter(data[kmeans.labels_ == 0].iloc[:, 0], data[kmeans.labels_ == 0].iloc[:, 1], label="类别1")
    plt.scatter(data[kmeans.labels_ == 1].iloc[:, 0], data[kmeans.labels_ == 1].iloc[:, 1], label="类别2")
    plt.scatter(data[kmeans.labels_ == 2].iloc[:, 0], data[kmeans.labels_ == 2].iloc[:, 1], label="类别3")
    plt.title("聚类分析")
    plt.xlabel("sepal_length")
    plt.ylabel("sepal_width")
    plt.legend(loc="best")
    plt.show()
