# -*- coding:utf-8 -*-
# Decision Tree Algorithm
# Cluster用于实现计算各种信息量
# 条件熵，信息增益针对某个属性而言的
import numpy as np
from math import log


class Cluster(object):
    """
    self._x,self._y: 记录数据集的变量
    self._counters: 类别向量的计数器，记录第i类数据的个数
    self._sample_weight: 记录样本权重的属性
    self._con_chaos_cache,self._ent_cache,self._gini_cache: 记录中间结果的属性
    self._base :记录对数的底
    """

    def __init__(self, x, y, sample_weight=None, base=2):
        self._x, self._y = x.T, y
        # 利用样本权重对类别向量y进行计数
        # np.bincount没有Counter好用，Counter可统计字符,但是没法用样本权重
        # 在运行速度上np.bincount更快
        if sample_weight is None:
            self._counters = np.bincount(self._y)  # 要提前把Target的元素变成数，不然会报错
            # self._counters = Counter(self._y)
        else:
            self._counters = np.bincount(self._y, weights=sample_weight)
            # self._counters = Counter(self._y)
        self._sample_weight = sample_weight  # 样本数*1
        self._con_chaos_cache = self._ent_cache = self._gini_cache = None
        self._base = base

    def __str__(self):
        return "Cluster"

    __repr__ = __str__

    # 定义信息熵
    def ent(self, ent=None, eps=1e-12):
        # ent是分类情况
        # 我的想法是这里都是要求ent 为空，不然都没法进行
        # 若已计算过熵（self._ent_cache），且这次调用时没有给定个类别的数目（即ent），就直接调用上次计算的结果
        if self._ent_cache is not None and ent is None:
            return self._ent_cache
        _len = len(self._y)
        # if self._ent_cache is None and ent is None,并调用时没有给定各类别数目，就需要自己计算
        if ent is None:
            ent = self._counters
        # 使用eps提高算法稳定度
        # 实际上，在对数据集分割时，有些类别的数据可能就不可避免的没有了，这时候需要注意这点
        _ent_cache = max(eps, -sum([_c/_len * log(_c/_len, self._base) if _c != 0 else 0 for _c in ent]))
        # 如果调用时没有给各类别样本数，就将计算的熵存下，这两个if语句是可以合并的
        if ent is None:
            self._ent_cache = _ent_cache
        return _ent_cache

    # 定义基尼系数
    def gini(self, p=None, eps=1e-12):
        # p是分类情况
        # 若已经有计算过gini_cache，则直接返回gini_cache
        if self._gini_cache is not None and p is None:
            return self._gini_cache
        _len = len(self._y)
        if p is None:
            p = self._counters
        _gini_cache = max(eps, (1 - sum((_p / _len)**2 if _p != 0 else 0 for _p in p)))
        if p is None:
            self._gini_cache = _gini_cache
        return _gini_cache

    # 定义计算H(y|A)和 gini(y|A)
    def con_chaos(self, idx, criterion="ent", features=None):
        # features是当前选择的类别
        # label_lst: 存放各个类别的集合
        # tmp_labels: 存放各个类别的数据的集合
        # chaos_lst: 存放各个类别的条件熵
        # 根据不同准则调用不同方法
        if criterion == "ent":
            _method = lambda cluster: cluster.ent()  # lambda可以将类实例化
        elif criterion == "gini":
            _method = lambda cluster: cluster.gini()
        else:
            raise NotImplementedError("Conditional info criterion '{} not defined".format(criterion))
        # 根据输入获取相应维度的向量,获取某个属性的所有值
        data = self._x[idx]
        # 如果调用时没有给该维度的取值空间features，就调用set函数得到对应的取值空间
        # 调用set函数耗时，最好预先给定
        if features is None:
            features = set(data)
        # feature = Counter(data).keys()
        # 返回的是一个列表，列表中的每个元素都是以矩阵，矩阵长度是data的长度
        # 矩阵中的元素是布尔运算的结果，True or False。列表长度为feature个数
        # 所以我不懂这个tmp_labels是干啥的iris中是35*150
        # 获取该维度不同取值对于的数据
        tmp_labels = [data == feature for feature in features]
        self._con_chaos_cache = [np.sum(_label) for _label in tmp_labels]  # 返回不同类别数目,如果对他求信息量
        # self._con_chaos_cache = Counter(data).values()
        # 获得各个类别
        label_lst = [self._y[label] for label in tmp_labels]  # _y[label]是一个一维数组，是label中为True值
        # 以tmp_labels[0]为例，于第一个类别对应的是第三和第三十个样本，因此y[tmp_label[0]]返回第3和第30对应
        # 的输出0 0
        res, chaos_lst = 0, []
        # 遍历各下标和对应的类别向量
        # data_label is callable? of course
        for data_label, tar_label in zip(tmp_labels, label_lst):
            # 获取相应的数据
            tmp_data = self._x.T[data_label]
            # data_label对应的样本,因为data_label是一个布尔符号组成的矩阵
            # 以iris为例，data=x[:,0]有150个元素，tmp_labels[0]中第三个和第三十个是True
            # 因此返回x中满足True的样本x[2],x[30]
            # 根据相应数据、类别向量和样本权重计算出不确定性
            if self._sample_weight is None:
                # _chaos存放当前类别的熵，可是实际计算ent没有用上tmp_data，只是Cluster初始化有
                _chaos = _method(Cluster(tmp_data, tar_label, base=self._base))
            else:
                _new_weight = self._sample_weight[data_label]
                _chaos = _method(Cluster(tmp_data, tar_label, _new_weight, base=self._base))
            # 依概率加权，同时把各个初始条件不确定性记录下来
            # res 存条件熵，chaos_lst存这个属性中各个类别的熵
            res += len(tmp_data)/len(data) * _chaos
            chaos_lst.append(_chaos)
        return res, chaos_lst

    # 定义计算信息增益的函数，参数get_chaos_lst用于控制输出
    def info_gain(self, idx, criterion="ent", get_chaos_lst=False, features=None):
        # 根据不同的准则，获取相应的“条件不确定性”
        if criterion in ("ent", "ratio"):
            _con_chaos, _chaos_lst = self.con_chaos(idx, "ent", features)
            _gain = self.ent() - _con_chaos
            if criterion == "ratio":
                _gain /= self.ent(self._con_chaos_cache)
        elif criterion == "gini":  # 应该有ratio
            _con_chaos, _chaos_lst = self.con_chaos(idx, "gini", features)
            _gain = self.gini() - _con_chaos
        else:
            raise NotImplementedError("info_gain criterion '{} not defined".format(criterion))
        return (_gain, _chaos_lst) if get_chaos_lst else _gain

    # 定义二分类问题条件的不确定性
    # 参数tar即是二分标准，参数continuous则告诉我们该维度的特征是否连续
    def bin_con_chaos(self, idx, tar, criterion="gini", continuous=False):
        """根据驶入的属性和划分的阈值，返回条件熵，和属性下各取值的熵列表"""
        # 根据不同准则调用不同方法
        if criterion == "ent":
            _method = lambda cluster: cluster.ent()  # lambda可以将类实例化
        elif criterion == "gini":
            _method = lambda cluster: cluster.gini()
        else:
            raise NotImplementedError("Conditional info criterion '{} not defined".format(criterion))
        data = self._x[idx]
        # 根据二分标准划分数据，注意要分离离散和连续两种情况讨论
        tar = data == tar if not continuous else data < tar
        # if not continuous data == tar,else data < tar ,return the res to tar.
        # if not continuous --> cart
        #  if continuous --> element is continuous
        tmp_labels = [tar, ~tar]  # 二分类，两个类别是不相容的，一个是True到另一个就是False
        self._con_chaos_cache = [np.sum(label) for label in tmp_labels]
        label_lst = [self._y[label] for label in tmp_labels]
        res = 0
        chaos_lst = []
        for data_label, tar_label in zip(tmp_labels, label_lst):
            tmp_data = self._x.T[data_label]
            if self._sample_weight is None:
                _chaos = _method(Cluster(tmp_data, tar_label, base=self._base))

            else:
                _new_weight = self._sample_weight[data_label]  # _new_weight还需要归一化
                _chaos = _method(
                    Cluster(tmp_data, tar_label, _new_weight / np.sum(_new_weight), base=self._base))
            res += len(tar_label)/len(data) * _chaos
            chaos_lst.append(_chaos)
            # print("_chaos:{}, data_label:{}\n".format(_chaos, tmp_labels))
        return res, chaos_lst

    # 定义计算二类问题信息增益的函数，参数get_chaos_lst用于控制输出,就是要不要chaos_lst要就True，else False
    def bin_info_gain(self, idx, tar, criterion="gini", get_chaos_lst=False, continuous=False):
        # 根据不同的准则，获取相应的“条件不确定性”
        if criterion in ("ent", "ratio"):
            _con_chaos, _chaos_lst = self.bin_con_chaos(idx, tar, criterion='ent', continuous=continuous)
            _gain = self.ent() - _con_chaos
            if criterion == "ratio":
                _gain /= self.ent(self._con_chaos_cache)
        elif criterion == "gini":
            _con_chaos, _chaos_lst = self.bin_con_chaos(idx, tar, criterion="gini", continuous=continuous)
            _gain = self.gini() - _con_chaos
        else:
            raise NotImplementedError("bin_info_gain criterion '{} not defined".format(criterion))
        return (_gain, _chaos_lst) if get_chaos_lst else _gain
