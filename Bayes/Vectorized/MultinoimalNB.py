import matplotlib.pyplot as plt
from Bayes.Vectorized.Basic import *
from Util.Util import quantize_data


class MultinomialNB(NaiveBayes):
    """
    self._labels: 记录target分类信息
    self._labelled_x: 记录data分类信息
    self._cat_counter: 记录各类别样本数
    self._con_counter: 存放，各类别中各特征取值数的统计结果，用来求解后验概率
    self._data: 核心矩阵，记录高斯平滑后的后验概率矩阵
    self.label_dict: 类别字典
    self._feat_dicts: 信息字典
    self._n_possibilities: 各特征的取值数
    self._p_category: 先验概率
    """
    def feed_data(self, x, y, sample_weight=None):
        """

        :param x: 未转置的数据
        :param y:
        :param sample_weight: 集成学习中的adaboost调整权重
        :return: 分类信息，信息字典，类别字典，样本权重
        """
        if sample_weight is not None:
            sample_weight = np.asarray(sample_weight)
        x, y, _, features, feat_dicts, label_dict = quantize_data(x, y, wc=np.array([False] * len(x[0])))
        cat_counter = np.bincount(y)
        n_possibilities = [len(feats) for feats in features]

        labels = [y == value for value in range(len(cat_counter))]
        labelled_x = [x[ci].T for ci in labels]

        self._x, self._y = x, y
        self._labelled_x, self._label_zip = labelled_x, list(zip(labels, labelled_x))
        self._cat_counter, self._feat_dicts, self._n_possibilities = cat_counter, feat_dicts, n_possibilities
        self.label_dict = label_dict
        self.feed_sample_weight(sample_weight)

    def feed_sample_weight(self, sample_weight=None):
        self._con_counter = []
        for dim, p in enumerate(self._n_possibilities):
            if sample_weight is None:
                self._con_counter.append([
                    np.bincount(xx[dim], minlength=p) for xx in self._labelled_x])
            else:
                self._con_counter.append([
                    np.bincount(xx[dim], weights=sample_weight[label] / sample_weight[label].mean(), minlength=p)
                    for label, xx in self._label_zip])

    # 核心矩阵
    def _fit(self, lb):
        n_dim = len(self._n_possibilities)
        n_category = len(self._cat_counter)
        self._p_category = self.get_prior_probability(lb)

        data = [[] for _ in range(n_dim)]
        for dim, n_possibilities in enumerate(self._n_possibilities):
            data[dim] = [
                [(self._con_counter[dim][c][p] + lb) / (self._cat_counter[c] + lb * n_possibilities)
                 for p in range(n_possibilities)] for c in range(n_category)]
        self._data = [np.asarray(dim_info) for dim_info in data]

    # 计算第i个类别的后验概率数组
    def _func(self, x, i):
        x = np.atleast_2d(x).T
        rs = np.ones(x.shape[1])
        for d, xx in enumerate(x):
            rs *= self._data[d][i][xx]
        return rs * self._p_category[i]

    def _transfer_x(self, x):
        for i, sample in enumerate(x):
            for j, char in enumerate(sample):
                x[i][j] = self._feat_dicts[j][char]
        return x

    def visualize(self, save=False):
        colors = plt.cm.Paired([i / len(self.label_dict) for i in range(len(self.label_dict))])
        colors = {cat: color for cat, color in zip(self.label_dict.values(), colors)}
        rev_feat_dicts = [{val: key for key, val in feat_dict.items()} for feat_dict in self._feat_dicts]
        for j in range(len(self._n_possibilities)):
            rev_dict = rev_feat_dicts[j]
            sj = self._n_possibilities[j]
            tmp_x = np.arange(1, sj + 1)
            title = "$j = {}; S_j = {}$".format(j + 1, sj)
            plt.figure()
            plt.title(title)
            for c in range(len(self.label_dict)):
                plt.bar(tmp_x - 0.35 * c, self._data[j][c, :], width=0.35,
                        facecolor=colors[self.label_dict[c]], edgecolor="white",
                        label=u"class: {}".format(self.label_dict[c]))
            plt.xticks([i for i in range(sj + 2)], [""] + [rev_dict[i] for i in range(sj)] + [""])
            plt.ylim(0, 1.0)
            plt.legend()
            if not save:
                plt.show()
            else:
                plt.savefig("d{}".format(j + 1))
