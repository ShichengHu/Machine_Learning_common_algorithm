import numpy as np


class MultinomialNB(object):
    """
    朴素贝叶斯，只处理离散数据

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
    """
    def __init__(self):
        self._x = self._y = None
        self._data = self._func = None
        self._n_possibilities = None
        self._labelled_x = self._label_zip = None
        self._cat_counter = self._con_counter = None
        self.label_dic = self._feat_dics = None

    # 重载__getitem__运算符以避免定义大量property
    def __getitem__(self, item):
        if isinstance(item, str):
            return getattr(self, "_" + item)

    # 留下抽样方法让子类定义，这里的tar_idx参数和self.tar_idx的意义一致
    # 数据预处理的方法
    def feed_data(self, x, y, sample_weight):
        # 列表转置，或者矩阵转置
        if isinstance(x, list):
            features = map(list, zip(*x))  # features = list(zip(*x))
            # map返回的是迭代器，而list(zip(*x))返回转置后的列表。zip()函数返回迭代器，需要用list(zip)才能返回具体的结果
        else:
            features = x.T
        # 数值化，从0开始,数值化用列表更快？
        features = [set(feature) for feature in features]  # 用for循环可以读出迭代器中的值
        feat_dics = [{v: k for k, v in enumerate(idx)} for idx in features]
        # feat_dics = [{key : val for key, val in enumerate(features[i])} for i in range(len(features))]
        label_dics = {v: k for k, v in enumerate(set(y))}  # k是转换后取值，v转换前取值
        #  数值化x,y，并表示为矩阵形式
        x = np.array([[feat_dics[i][key] for i, key in enumerate(sample)] for sample in x])
        # work good,得到就是样本数*特征数
        y = np.array([label_dics[i] for i in y])
        cat_counter = np.bincount(y)
        # 各特征维度取值个数
        n_possibilities = [len(feats) for feats in features]  # 各个维度可能取值的个数
        # 获取输出类别对应的索引，并且记录类别分开后的输入数据的数组
        labels = [y == label for label in range(len(cat_counter))]
        labelled_x, labelled_y = [x[label].T for label in labels], [y[label] for label in labels]
        # 更新各个模型参数
        self._x, self._y = x, y
        self._labelled_x = labelled_x
        self._feat_dics, self._label_zip = feat_dics, zip(labels, labelled_x)
        self.label_dic = {v: k for k, v in label_dics.items()}
        # self.label_dic = {k: v for k, v in enumerate(set(y))}  # 数值化后的值与数值化前值的对应关系
        self._cat_counter, self._n_possibilities = cat_counter, n_possibilities
        # 调用处理样本权重的函数，以更新记录条件概率的数组
        self.feed_sample_weight(sample_weight)

    # 留下抽象方法让子类定义，这里的sample_weight参数代表着样本权重
    # 定义处理样本权重的函数
    def feed_sample_weight(self, sample_weight=None):
        self._con_counter = [[] for _ in range(len(self._feat_dics))]
        # 利用bincount获取带权重的条件概率的最大似然估计
        for dim, _p in enumerate(self._n_possibilities):
            if sample_weight is None:
                for xx in self._labelled_x:
                    self._con_counter[dim].append(np.bincount(xx[dim], minlength=_p))  # for xx in self._labelled_x
                # xx[dim]就取的是这一类别中一个特征
                # _con_counter[i][j][k] i对应的类别，j是对应的属性，k是对应的取值
                # 返回的是迭代器是什么鬼
            else:
                local_weight = sample_weight * len(sample_weight)  # sample_weight[label] / sample_weight[label].mean()
                for label, xx in self._label_zip:
                    self._con_counter[dim].append(
                        np.bincount(xx[dim], local_weight[label], minlength=_p))
                    # 要明确xx[dim]和local_weight的长度是相同的，这样加权就是对这个类别中的各个样本加权

    # 定义计算先验概率的函数，lb就是各个估计中的平滑项
    # lb的默认值是1，也就是说默认采取拉普拉斯平滑
    def get_prior_probability(self, lb=1):
        return [(_c_num + lb) / (len(self._y) + lb * len(self._cat_counter)) for _c_num in self._cat_counter]

    # 定义具有普适性的训练函数
    def fit(self, x=None, y=None, sample_weight=None, lb=1):
        # 如果有传入的x,y就用传入的x,y初始化模型
        if x is not None and y is not None:
            self.feed_data(x, y, sample_weight)
        # 改用核心算法得到决策函数
        self._func = self._fit(lb)

    # 留下抽象核心算法让子类定义（核心训练函数）（调用与整合预处理时记录下来的信息的过程
    def _fit(self, lb):
        n_dim = len(self._n_possibilities)
        n_category = len(self._cat_counter)
        p_category = self.get_prior_probability(lb)
        # data存储加了平滑项后后验概率数组
        data = [[] for _ in range(n_dim)]  # [[None] * n_dim]
        # 我这样处理的目的在于避免generator出现，而且generator是不能和数直接运算的，没法把它当成矩阵
        for dim, n_possibility in enumerate(self._n_possibilities):
            # data[dim].append((self._con_counter[dim][c] + lb) / (lb * n_possibility + self._cat_counter[c]
            # ) for c in range(n_categeroy) 就不是一个东西，它就是一个生成器，没法运算
            for c in range(n_category):
                data[dim].append((self._con_counter[dim][c] + lb) / (lb * n_possibility + self._cat_counter[c]))
        self._data = [np.array(dim_info) for dim_info in data]  # 对data属性下，各个类别的小数组合并

        def func(input_x, tar_category):
            res = 1
            for d, xx in enumerate(input_x):
                res *= data[d][tar_category][xx]
            return res * p_category[tar_category]
        return func

    def _transfer_x(self, x):
        for j, char in enumerate(x):
            x[j] = self._feat_dics[j][char]
        return x

    # _con_counter[i][j][k] i对应的类别，j是对应的属性，k是对应的取值
    # 定义预测单一样本的函数
    # 参数get_raw_result=False控制该函数是输出预测的类别还是输出相应的后验概率
    # get_raw_result=False 则输出类别，True则输出后验概率
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

    # 定义多样本的预测，本质是对predict_one的重复调用
    def predict(self, x, get_raw_result=False):
        return np.array([self.predict_one(sample, get_raw_result) for sample in x])

    # 定义评估方法，预测精度函数
    def eva(self, x, y):
        y_predict = self.predict(x)
        return "accuracy is {:.2%}".format(np.sum(y_predict == y) / len(y))
