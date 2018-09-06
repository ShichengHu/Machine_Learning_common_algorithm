import numpy as np
from math import pi, exp

sqrt_pi = (2 * pi) ** 0.5


class NBFunctions(object):
    # 定义正态分布的概率密度函数
    @staticmethod
    def gaussian(x, mu, sigma):
        return exp(-(x - mu) ** 2 / (2 * sigma)) / (sqrt_pi * sigma**0.5)

    # 定义极大似然估计函数
    @staticmethod
    def gaussian_maximum_likelihood(labelled_x, n_category, dim):
        # 定义各属性的均值和方差
        mu = [np.sum(
            labelled_x[c][dim]) / len(labelled_x[c][dim]) for c in range(n_category)]
        sigma = [np.sum(
            (labelled_x[c][dim] - mu[c]) ** 2) / len(labelled_x[c][dim]) for c in range(n_category)]
        return [mu, sigma]


# 连续型高斯分布的属性，应该都是数字了，还需要数值化吗,而且也不再需要平滑
class GaussianNB(object):
    """
    高斯型朴素贝叶斯，处理连续数据

    假设数据分布服从高斯分布
    """
    def __init__(self, _data=None):
        """
            self._x,self._y: 对y数值化，但对x没有数值化
            self._data: 核心数组，存储实际使用的条件概率的相关信息
            self._func: 核心函数，根据输入的x,y输出对应的后验概率
            self._n_possibility: 记录各个维度特征取值个数的数组：[s1,s2,...,sn]
            self._labelled_x: 记录按类别分开后的输入数据的数组
            self._label_zip: 记录类别相关信息的数组，视具体算法，定义会有所不同,先改为_labels
            self._cat_counter:  核心数组，记录第i类数据的个数
            self._con_counter: 核心数组，记录数据条件概率的原始极大似然估计self._con_counter[d][c][p]=p(x(d)=p|y=c)
            self.label_dic: 核心字典，用于记录数值化类别时的转换关系（数值化前与数值化后取值的对应关系，keys是数值化前）
            self._feat_dics: 核心字典，用于记录数值化各维度特征（feat）的转换关系
        """
        self._data = _data
        self._x = self._y = None
        self._func = None
        self._labelled_x = self._labels = None
        self._cat_counter = None
        self.label_dic = None

        # 重载__getitem__运算符以避免定义大量property
    def __getitem__(self, item):
        if isinstance(item, str):
            return getattr(self, "_" + item)

    # 数据预处理
    def feed_data(self, x, y, sample_weight=None):
        # 输入的数据是连续的，需要调用python自带的float方法将输入数据数值化,连转置都不需要了
        x = np.array([list(map(lambda c: float(c), sample)) for sample in x])
        # 数值化，从0开始,数值化用列表更快？
        label_dics = {v: k for k, v in enumerate(set(y))}
        y = np.array([label_dics[char] for char in y])  # y可以数值化
        labels = [y == label for label in range(len(label_dics))]
        labelled_x = [x[label].T for label in labels]
        print("labelled_x: {}\n".format(labelled_x))
        cat_counter = np.bincount(y)
        # 更新各个模型参数
        self._x, self._y = x, y
        self._labelled_x = labelled_x
        self._labels = labels
        self.label_dic = {v: k for k, v in label_dics.items()}
        self._cat_counter = cat_counter
        # 调用处理样本权重的函数，以更新记录条件概率的数组
        self.feed_sample_weight(sample_weight)

    # 这个对于加权为0的无效，所以先不管
    def feed_sample_weight(self, sample_weight):
        if sample_weight is not None:
            local_weight = sample_weight * len(sample_weight)
            for i, label in enumerate(self._labels):
                self._labelled_x[i] *= local_weight[label]  # 样本加权

    # 获取先验概率
    def get_prior_probability(self, lb):
        return [(label + lb) / (len(self._y) + lb * len(self.label_dic)) for label in self._cat_counter]

        # 留下抽象核心算法让子类定义（核心训练函数）（调用与整合预处理时记录下来的信息的过程

    def _fit(self, lb):
        n_dim = self._x.shape[1]
        n_category = len(self._cat_counter)
        p_category = self.get_prior_probability(lb)
        if self._data is None:
            data = [
                NBFunctions.gaussian_maximum_likelihood(self._labelled_x, n_category, dim=dim) for dim in range(n_dim)]
            self._data = data
        print("_data: {}".format(self._data))

        def func(input_x, tar_category):
            res = 1
            for d, xx in enumerate(input_x):
                p = NBFunctions.gaussian(xx, mu=self._data[d][0][tar_category], sigma=self._data[d][1][tar_category])
                res *= p  # type: int
            return res * p_category[tar_category]

        return func

        # 定义具有普适性的训练函数
    def fit(self, x=None, y=None, sample_weight=None, lb=1):
        # 如果有传入的x,y就用传入的x,y初始化模型
        if x is not None and y is not None:
            y = y.flatten()  # 把y变成一维的，实现降维，如果本来就是一维，就不变
            self.feed_data(x, y, sample_weight)
        # 改用核心算法得到决策函数
        self._func = self._fit(lb)

    # 对输入的数据用特征字典转换，数值化，对于连续取值无需数值化
    @staticmethod
    def _transfer_x(xx):
        xx = list(map(lambda c: float(c), xx))
        return xx

    def predict_one(self, x, get_raw_result=False):
        # 在预测之前要将新的输入数据数值化
        # 如果输入的是Numpy数组，要将他转换成python的数组，因为python数组在数值化这个操作上要更快
        if type(x) is np.ndarray:  # if isinstance(x, np.ndarray):
            x = x.tolist()  # 把数组变成列表
            # 否则对数组进行拷贝
        else:
            x = x[:]  # 所有行
        # 调用相关方法进行数值化，该方法随具体模型的不同而不同
        x = self._transfer_x(x)
        m_arg, m_probability = 0, 0
        # 遍历各类别、找到能使后验概率最大化的类别
        for i in range(len(self._cat_counter)):
            p = self._func(x, i)
            if p > m_probability:
                m_arg, m_probability = i, p
        if not get_raw_result:
            return self.label_dic[m_arg]
        return m_probability

    def predict(self, x, get_raw_result=False):
        return np.array([self.predict_one(xx, get_raw_result) for xx in x])

    def estimate(self, x, y):
        predict = self.predict(x)
        return "accuracy is : {:8.6} %".format(100*+np.sum(predict == y) / len(y))

    def acc(self, x, y):
        predict = self.predict(x)
        return np.sum(predict == y) / len(y)
