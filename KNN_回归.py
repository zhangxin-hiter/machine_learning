import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt

class KNN:
    """使用python实现KNN回归算法
    
    """

    def __init__(self, k):
        """初始化方法
        
        parameters
        -----
        k: int
            邻居的个数
            
        """
        self.k = k

    def fit(self, X, y):
        """训练方法
        
        parameters
        -----
        X: 类数组类型。形状为[样本数量,特征数量]
            待训练的样本特征
        
        y: 类数组类型。形状为[样本数量]
            每个样本的目标值
            
        """
        
        # 将数据转化为numpy数组形式
        self.X = np.asarray(X)
        self.y = np.asarray(y)

    def predict(self, X):
        """根据传递的数据，对数据进行预测
        
        parameters
        -----
        X: 类数组类型。形状为[样本数据， 特征数量]
            要预测的样本特征
        
        Returns
        -----
        result: 数组类型
            预测结果
        """
        # 将X转化为numpy数组
        X = np.asarray(X)
        result = []
        for x in X:
            # 计算测试样本与训练集中所有样本的距离
            dis = np.sqrt(np.sum((x - self.X) ** 2, axis=1))
            # 对数组进行排序
            index = dis.argsort()
            index = index[:self.k]
            # 取均值
            result.append(np.sum(self.y[index] * dis[index] / np.sum(dis[index])))
        
        return np.asarray(result)

if __name__ == "__main__":
    data = pd.read_csv("iris.csv")
    # 删除不需要的字段
    data.drop(["species"], axis=1, inplace=True)
    # 删除重复字段
    data.drop_duplicates(inplace=True)
    # 洗牌数据集
    data = data.sample(len(data), random_state=0)
    # 分割数据集
    train_x = data.iloc[:120, :-1]
    train_y = data.iloc[:120, -1]
    test_x = data.iloc[120:, :-1]
    test_y = data.iloc[120:, -1]
    # 初始化KNN预测器
    knn = KNN(k=3)
    # 训练模型
    knn.fit(train_x, train_y)
    # 预测
    result = knn.predict(test_x)
    # 设置中文支持
    mpl.rcParams["font.family"] = "SimHei"
    mpl.rcParams["axes.unicode_minus"] = False
    # 绘制
    plt.plot(result, "ro-", label="预测值")
    plt.plot(test_y.values, "go--",label="真实值", marker=">")
    plt.legend(loc="best")
    plt.xlabel("节点序号")
    plt.ylabel("花瓣宽度")
    plt.title("花瓣宽度预测")
    plt.show()


