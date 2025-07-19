import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt

class LinearRegression:
    """使用python实现线性回归（最小二乘法）"""

    def fit(self, X, y):
        """根据提供的训练数据，对模型进行训练
        
        parameters
        -----
        X: 类数组类型。形状[样本数量，特征数量]
            样本特征
        y: 类数组类型。形状[样本数量]
            
        """

        # 将特征数组转化为特征矩阵
        X = np.asmatrix(X.copy())
        y = np.asmatrix(y.copy()).reshape(-1, 1)
        self.w_ = (X.T * X).I * X.T * y

    def predict(self, X):
        """根据参数X，对样本进行预测
        
        parameters
        -----
        X: 类数组类型。形状为[样本数量，特征数量]
            样本特征
            
        Returns
        -----
        result: 数组类型
            预测结果
        
        """
        X = np.asmatrix(X)
        """
        matrix.T 返回矩阵的转置
        matrix.I 返回矩阵的逆
        matrix.ravel() 返回矩阵展平后的一维数组
        """
        return np.asarray(X * self.w_).ravel()

if __name__ == "__main__":
    data = pd.read_csv("house_data.csv")
    # 分割数据集
    train_x = data.iloc[:400, :-1]
    train_y = data.iloc[:400, -1]
    test_x = data.iloc[400:, :-1]
    test_y = data.iloc[400:, -1]
    # 生成最小二乘线性回归预测器
    lr = LinearRegression()
    # 训练模型
    lr.fit(train_x, train_y)
    # 预测
    result = lr.predict(test_x)

    mpl.rcParams["font.family"] = "SimHei"
    mpl.rcParams["axes.unicode_minus"] = False
    plt.plot(result, "ro-", label="预测值")
    plt.plot(test_y.values, "go--", label="真实值")
    plt.title("波士顿房价预测")
    plt.xlabel("节点序号")
    plt.ylabel("房价")
    plt.legend(loc="best")
    plt.show()

    # # 考虑截距
    # t = data.sample(len(data), random_state=0)
    # new_columns = t.columns.insert(0, "intercept")
    # t.reindex(new_columns, fill_value=1)



    