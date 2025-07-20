import pandas as pd
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt

class KNN:
    """使用python实现k近邻算法(分类)"""
    
    def __init__(self, k):
        """
        初始化方法
        
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
        X: 类数组类型, 形状为：[样本数量，特征数量]
            待训练的样本特征
            
        y: 类数组类型，形状为[样本数量]
            每个样本的目标值
        """

        # np.asarray()将类数组类型转换为数组类型
        self.X = np.asarray(X)
        self.y = np.asarray(y)

    def predict(self, X):
        """根据参数传递的样本，对样本数据进行预测
        
        parameters
        -----
        X： 类数组类型，形状为[样本数量，特征数量]
            待训练的样本特征
        
        Returns
        -----
        result: 数组类型
                预测结果
        """
        
        X = np.asarray(X)
        result = []

        # 对ndarray数组遍历，每次取数组一行, array对位运算
        for x in  X:

            """
            np.sum() 数组求和函数，axis用于设置求和方向
            np.sqrt() 数组开方函数
            np.array() 创建一个ndarray数组对象
            np.bincount() 对数组中的元素进行计数，数组元素非负
            array.argsort() 返回数组排序后，每个元素在原数组的索引
            array.argmax() 返回值最大的索引
            array[start:end] 对array进行截断，返回[start,end)
            array1[array2] 高级索引方法，对array2中的值在array1中进行索引
            """

            # 对测试集中的每一个样本，与训练集中的所有样本求距离
            dis = np.sqrt(np.sum((x - self.X) ** 2, axis=1))
            index = dis.argsort()
            # 进行截断，仅取前k个元素
            index = index[0: self.k]
            # 返回数组中每个元素出现的次数，元素非负
            count = np.bincount(self.y[index])
            # 返回数组中值最大的索引
            result.append(count.argmax())

        return np.asarray(result)
    
    def predict2(self, X):
        """根据参数传递的样本，对样本数据进行预测（考虑权重）
        
        parameters
        -----
        X： 类数组类型，形状为[样本数量，特征数量]
            待训练的样本特征
        
        Returns
        -----
        result: 数组类型
                预测结果
        """
        
        X = np.asarray(X)
        result = []

        # 对ndarray数组遍历，每次取数组一行, array对位运算
        for x in  X:

            """
            np.sum() 数组求和函数，axis用于设置求和方向
            np.sqrt() 数组开方函数
            np.array() 创建一个ndarray数组对象
            np.concatenate() 将二维数组进行拼接
            np.random.randint() 生成一个在指定范围中的随机数
            np.zeros() 创建全0数组
            np.bincount() 对数组中的元素进行计数，数组元素非负
            array.shape 返回一个一维数组
            array.argsort() 返回数组排序后，每个元素在原数组的索引
            array.argmax() 返回值最大的索引
            array[start:end] 对array进行截断，返回[start,end)
            array1[array2] 高级索引方法，对array2中的值在array1中进行索引
            """

            # 对测试集中的每一个样本，与训练集中的所有样本求距离
            dis = np.sqrt(np.sum((x - self.X) ** 2, axis=1))
            index = dis.argsort()
            # 进行截断，仅取前k个元素
            index = index[0: self.k]
            # 返回数组中每个元素出现的次数，元素非负
            count = np.bincount(self.y[index], weights=1 / dis[index])
            # 返回数组中值最大的索引
            result.append(count.argmax())

        return np.asarray(result)



if __name__ == "__main__":

    # pandas.read_csv() 读取csv格式文件返回一个dataframe对象
    # pandas.concat() 可将多个dataframe或series进行拼接，使用axis规定方向
    data = pd.read_csv("iris.csv")
    # dataframe.head() 显示前n行数据
    # dataframe.columns dataframe的成员变量是一个index类
    # index.insert() 插入
    # dataframe.reindex() 重新修改索引
    # dataframe.tail() 显示后n行数据
    # dataframe.sample() 随机抽取n行数据
    # dataframe["name"] 返回一个series对象
    # dataframe[表达式] 筛选符合条件表格
    # dataframe["name"]= 添加列
    # dataframe.drop() 删除特定列
    # dataframe.iloc() 取出特定位置的数据
    # dataframe.duplicated() 用于判断dataframe中是否存在重复，subset用于设置判断的列，keep用于重复提示信息的设置，返回一个布尔序列
    # dataframe.drop_duplicated() 用于删除dataframe中的重复行， subset用于指定判断的列，inplace用于设置是否在原对象中修改    
    # series.any() 专门用与判断一个布尔序列是否有true
    # series.value_counts() 返回序列，索引为原序列的值，值为在原series中出现的次数
    # series.map() 值映射函数，以字典方式将特定的值映射为设定值
    data["species"] = data["species"].map({"setosa": 0, "virginica": 1, "versicolor": 2})
    data.drop_duplicates(inplace=True)
    # 提取每个类比的鸢尾花数据
    t0 = data[data["species"] == 0]
    t1 = data[data["species"] == 1]
    t2 = data[data["species"] == 2]
    # 通过随机取样将顺序打乱
    t0 = t0.sample(len(t0), random_state=0)
    t1 = t1.sample(len(t1), random_state=0)
    t2 = t2.sample(len(t2), random_state=0)
    # 构建训练集和测试集
    train_x = pd.concat([t0.iloc[:40, :-1], t1.iloc[:40, :-1], t2.iloc[:40, :-1]], axis=0)
    train_y = pd.concat([t0.iloc[:40, -1], t1.iloc[:40, -1], t2.iloc[:40, -1]], axis=0)
    test_x = pd.concat([t0.iloc[40:, :-1], t1.iloc[40:, :-1], t2.iloc[40:, :-1]], axis=0)
    test_y = pd.concat([t0.iloc[40:, -1], t1.iloc[40:, -1], t2.iloc[40:, -1]], axis=0)
    # 开始训练
    knn = KNN(k=3)
    knn.fit(train_x, train_y)
    predict_y = knn.predict(test_x)
    # print(predict_y==test_y)
    # 设置字体为黑体，支持中文显示
    mpl.rcParams["font.family"] = "SimHei"
    # 设置中文显示时，能够显示负号
    mpl.rcParams["axes.unicode_minus"] = False
    # 绘制训练集数据
    plt.scatter(x=t0["sepal_length"][:40], y=t0["petal_length"][:40], color="r", label="setosa")
    plt.scatter(x=t1["sepal_length"][:40], y=t1["petal_length"][:40], color="b", label="virginica")
    plt.scatter(x=t2["sepal_length"][:40], y=t2["petal_length"][:40], color="g", label="versicolor")
    # 绘制测试集数据
    right = test_x[predict_y == test_y]
    wrong = test_x[predict_y != test_y]
    plt.scatter(x=right["sepal_length"], y=right["petal_length"], color="c", marker="x", label="right")
    plt.scatter(x=wrong["sepal_length"], y=wrong["petal_length"], color="m", marker=">", label="wrong")
    plt.title("KNN分类结果显示")
    plt.xlabel("sepal_length")
    plt.ylabel("petal_length")
    plt.legend(loc="best")
    plt.show()





    