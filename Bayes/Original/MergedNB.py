import numpy as np
from math import pi, exp
from Util.Util import quantize_data

sqrt_pi = (2 * pi) ** 0.5


class NBFunctions(object):
    # 定义正态分布的概率密度函数
    @staticmethod
    def gaussian(x, mu, sigma):
        return exp(-(x - mu) ** 2 / (2 * sigma ** 2)) / (sqrt_pi * sigma)

    # 定义极大似然估计函数
    @staticmethod
    def gaussian_maximum_likelihood(labelled_x, n_category, dim):
        # 定义各属性的均值和方差
        # 返回当前属性的均值，方差各个类别组成的列表
        # [[mu],[siama]]  -> mu=[mu[label0],[mu[label1],...]
        # data=[[[mu],[sigma]],[],[]...]
        mu = [np.sum(
            labelled_x[c][dim]) / len(labelled_x[c][dim]) for c in range(n_category)]
        sigma = [np.sum(
            (labelled_x[c][dim] - mu[c]) ** 2) / len(labelled_x[c][dim]) for c in range(n_category)]
        return [mu, sigma]


# self._data: 对于离散特征，对应后验概率矩阵，连续特征，对应相应的mu, sigma矩阵
class MergedNB(object):
    """
    混合型朴素贝叶斯，可以处理连续和离散数据

       self._x,self._y:
       self._data: 核心数组，存储实际使用的条件概率的相关信息
       self._func: 核心函数，根据输入的x,y输出对应的后验概率
       self._n_possibility: 记录各个维度特征取值个数的数组：[s1,s2,...,sn]
       self._labelled_x: 记录按类别分开后的输入数据的数组
       self._label_zip: 记录类别相关信息的数组，视具体算法，定义会有所不同
       self._cat_counter:  核心数组，记录第i类数据的个数
       self._con_counter: 核心数组，记录数据条件概率的原始极大似然估计self._con_counter[d][c][p]=p(x(d)=p|y=c)
       self.label_dic: 核心字典，用于记录数值化类别时的转换关系（数值化前与数值化后取值的对应关系，keys是数值化前）
       self._feat_dics: 核心字典，用于记录数值化各维度特征（feat）的转换关系
       self._whether_continuous: 记录各个维度的变量是否是连续型变量
       self._whether_discrete： 记录各个维度的变量是否是离散型的
    """

    def __init__(self, whether_continuous=None):
        if whether_continuous is not None:
            self._whether_continuous = whether_continuous
            self._whether_discrete = ~whether_continuous
        else:
            self._whether_discrete = None
            self._whether_continuous = None
        self._x = self._y = None
        self._data = self._func = None
        self._n_possibilities = None
        self._labelled_x = self._label_zip = None
        self._labels = None
        self._cat_counter = self._con_counter = None
        self.label_dic = self._feat_dics = None

    # 重载__getitem__运算符以避免定义大量property
    def __getitem__(self, item):
        if isinstance(item, str):
            return getattr(self, "_" + item)

    def feed_data(self, x=None, y=None, sample_weight=None):
        if sample_weight is not None:
            sample_weight = np.array(sample_weight)
        x, y, wc, features, feat_dicts, label_dict = quantize_data(x, y, wc=self._whether_continuous)
        if self._whether_continuous is None:
            self._whether_continuous, self._whether_discrete = wc, ~wc
        labels = [y == tar for tar in range(len(set(y)))]
        # 更新参数
        self._x, self._y = x, y
        self._feat_dics, self.label_dic = feat_dicts, {k: v for v, k in label_dict.items()}
        self._cat_counter = np.bincount(y)
        self._labelled_x = [x[label].T for label in labels]
        self._label_zip = zip(labels, self._labelled_x)  # 离散取值
        self._labels = labels  # 用于连续
        self._n_possibilities = [len(feats) for feats in features]
        self.feed_sample_weight(sample_weight)

    def feed_sample_weight(self, sample_weight=None):
        self._con_counter = [[] for _ in range(len(self._feat_dics))]
        # 得到的self._con_counter在连续取值对于属性为空，在离散值进行统计
        # 而且对连续取值的特征列都进行了加权
        # 我的目的是得到一个矩阵，对于离散，我要进行拉普拉斯平滑，而对连续取值，我直接计算均值方差矩阵
        if sample_weight is not None:
            for idx, _p in enumerate(self._n_possibilities):
                if self._whether_continuous[idx]:  # 连续
                    local_weight = sample_weight * len(sample_weight)
                    for i, label in enumerate(self._labels):
                        self._labelled_x[i] *= local_weight[label]  # 样本加权
                        self._con_counter[i].append(None)
                else:  # 离散
                    for label, xx in enumerate(self._labelled_x):
                        local_weight = sample_weight * len(sample_weight)
                        local_weight = local_weight[self._labels[label]]
                        self._con_counter[idx].append(np.bincount(xx[idx], local_weight, minlength=_p))
        else:
            for idx, _p in enumerate(self._n_possibilities):  # 特征
                if self._whether_continuous[idx]:  # 连续
                    self._con_counter[idx].append(None)
                else:  # 离散
                    for xx in self._labelled_x:  # 类别
                        xx = np.array(xx, dtype=np.int)
                        self._con_counter[idx].append(np.bincount(xx[idx], minlength=_p))

    def get_prior_probability(self, lb=1):
        return [(_c_num + lb) / (len(self._y) + lb * len(self._cat_counter)) for _c_num in self._cat_counter]

    # c=[(x**2 for x in range(3))if True else 1]  这里生成的是生成器
    # c=[[x**2 for x in range(3)]if True else 1]  这里生成的是列表
    def _fit(self, lb):
        n_category = len(self._cat_counter)
        p_category = self.get_prior_probability(lb)
        data = [
            [(self._con_counter[dim][c] + lb) / (lb * n_possibility + self._cat_counter[c]) for c in range(n_category)]
            if not self._whether_continuous[dim]
            else NBFunctions.gaussian_maximum_likelihood(self._labelled_x, n_category, dim)
            for dim, n_possibility in enumerate(self._n_possibilities)]
        self._data = data  # 特征，类别，

        def func(input_x, tar_category):
            rs = 1
            for dim, char in enumerate(input_x):
                p = NBFunctions.gaussian(char, mu=self._data[dim][0][tar_category],
                                         sigma=self._data[dim][1][tar_category]) \
                     if self._whether_continuous[dim] else data[dim][tar_category][char]
                rs *= p
            return rs * p_category[tar_category]
        return func

    def fit(self, x=None, y=None, sample_weight=None, lb=1):
        if x is not None and y is not None:
            self.feed_data(x, y, sample_weight)
        self._func = self._fit(lb)

    def x_transfer(self, x):
        for i, xx in enumerate(x):
            if isinstance(xx, int):
                xx = str(xx)
            char = self._feat_dics[i][xx] if not self._whether_continuous[i] else float(xx)
            x[i] = int(char)
        return x

    def predict_one(self, x, get_raw_result=False):
        if type(x) is np.ndarray:  # if isinstance(x, np.ndarray):
            x = x.tolist()  # 把数组变成列表
            # 否则对数组进行拷贝
        else:
            x = x[:]  # 所有行
        # 调用相关方法进行数值化，该方法随具体模型的不同而不同
        x = self.x_transfer(x)
        m_arg, m_probability = 0, 0
        for i in range(len(self._labels)):
            p = self._func(x, i)
            if p > m_probability:
                m_probability = p
                m_arg = i
        if not get_raw_result:
            return self.label_dic[m_arg]
        return m_probability

    def predict(self, x, get_raw_result=False):
        return np.array([self.predict_one(xx, get_raw_result) for xx in x])

    def eva(self, x, y):
        y_predict = self.predict(x)
        return "accuracy is     : {:.2%}".format(np.sum(y_predict == y) / len(x))
