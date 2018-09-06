import numpy as np
from math import pi
from Util.Bases import ClassifierBase

sqrt_pi = (2 * pi) ** 0.5


class NBFunctions:
    @staticmethod
    def gaussian(x, mu, sigma):
        return np.exp(-(x - mu) ** 2 / (2 * sigma ** 2)) / (sqrt_pi * sigma)

    @staticmethod
    def gaussian_maximum_likelihood(labelled_x, n_category, dim):
        #
        mu = [np.sum(
            labelled_x[c][dim]) / len(labelled_x[c][dim]) for c in range(n_category)]
        sigma = [np.sum(
            (labelled_x[c][dim] - mu[c]) ** 2) / len(labelled_x[c][dim]) for c in range(n_category)]
        # return [mu, sigma]

        def func(_c):
            def sub(x):
                return NBFunctions.gaussian(x, mu[_c], sigma[_c])
            return sub

        return [func(_c=c) for c in range(n_category)]


class NaiveBayes(ClassifierBase):
    # 定义参数字典
    _params = {
        "sample_weight": None
    }

    def __init__(self, **kwargs):
        super(NaiveBayes, self).__init__(**kwargs)
        self._x = self._y = None
        self._n_possibilities = self._p_category = None
        self._labelled_x = self._label_zip = None
        self._cat_counter = self._con_counter = None
        self.label_dict = self._feat_dicts = None
        # self._params = dict()   self._params = {}
        self._data = kwargs.get("data", None)
        self._params["lb"] = kwargs.get("lb", 1)

    def feed_data(self, x, y, sample_weight=None):
        pass

    def feed_sample_weight(self, sample_weight=None):
        pass

    def get_prior_probability(self, lb=1):
        return [(c_num + lb) / (len(self._y) + lb * len(self._cat_counter))
                for c_num in self._cat_counter]

    def fit(self, x=None, y=None, sample_weight=None, lb=None):
        """
        feed_data and _fit() 获取各类字典和 计算后验概率矩阵（离散），求解概率密度函数（连续）

        :param x: 未转置的数据！！！！
        :param y:
        :param sample_weight: 集成学习调用
        :param lb: 高斯平滑项
        :return:
        """
        if sample_weight is None:
            sample_weight = self._params["sample_weight"] if self._params["sample_weight"] is not None \
                else np.ones(len(y))
        if lb is None:
            lb = self._params["lb"]
        if x is not None and y is not None:
            y = y.flatten()  #
            self.feed_data(x, y, sample_weight)
        self._fit(lb)

    def _fit(self, lb):
        pass

    def _func(self, x, i):
        pass

    def predict(self, x, get_raw_result=False, **kwargs):
        if isinstance(x, np.ndarray):
            x = x.tolist()
        else:
            x = [xx[:] for xx in x]
        x = self._transfer_x(x)
        m_arg, m_probability = np.zeros(len(x), dtype=np.int8), np.zeros(len(x))
        # 怎么理解这个循环：显然在循环中对于大于0的部分会得到更新
        # 预测是在于寻找最大的后验概率
        for i in range(len(self._cat_counter)):
            p = self._func(x, i)  # 按类别i计算后验概率
            mask = p > m_probability  # 把后验概率和之前的后验概率比较，并更新后验概率增大的部分，只要这样循环最后得到的m_probability一定是最大的
            m_arg[mask], m_probability[mask] = i, p[mask]
        if not get_raw_result:
            return np.array([self.label_dict[arg] for arg in m_arg])
        return m_probability

    def _transfer_x(self, x):
        return x
