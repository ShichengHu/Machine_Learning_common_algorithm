from Bayes.Vectorized.Basic import *
import matplotlib.pyplot as plt


class GaussianNB(NaiveBayes):

    def feed_data(self, x, y, sample_weight=None):
        if sample_weight is not None:
            sample_weight = np.asarray(sample_weight)
        x = np.array([list(map(lambda c: float(c), sample)) for sample in x])
        labels = list(set(y))
        label_dict = {label: i for i, label in enumerate(labels)}
        y = np.array([label_dict[yy] for yy in y])
        cat_counter = np.bincount(y)
        labels = [y == value for value in range(len(cat_counter))]
        labelled_x = [x[label].T for label in labels]

        self._x, self._y = x.T, y
        self._labelled_x, self._label_zip = labelled_x, labels
        self._cat_counter, self.label_dict = cat_counter, {i: l for l, i in label_dict.items()}
        self.feed_sample_weight(sample_weight)

    def feed_sample_weight(self, sample_weight=None):
        if sample_weight is not None:
            local_weights = sample_weight * len(sample_weight)
            for i, label in enumerate(self._label_zip):
                self._labelled_x[i] *= local_weights[label]

    def _fit(self, lb):
        lb = 0
        n_category = len(self._cat_counter)
        self._p_category = self.get_prior_probability(lb)
        data = [
            NBFunctions.gaussian_maximum_likelihood(
                self._labelled_x, n_category, dim) for dim in range(len(self._x))]
        self._data = data

    def _func(self, x, i):
        x = np.atleast_2d(x).T
        rs = np.ones(x.shape[1])
        for d, xx in enumerate(x):
            rs *= self._data[d][i](xx)
        return rs * self._p_category[i]

    def fit(self, x=None, y=None, sample_weight=None, lb=None):
        """
        feed_data and _fit() 获取各类字典和 计算后验概率矩阵（离散），求解概率密度函数（连续）

        :param x: 未转置的数据！！！！
        :param y:
        :param sample_weight: 集成学习调用
        :param lb: 高斯平滑项
        :return:
        """
        if lb is None:
            lb = self._params["lb"]
        if x is not None and y is not None:
            y = y.flatten()  #
            self.feed_data(x, y, sample_weight)
        self._fit(lb)

    def visualize(self, save=False):
        colors = plt.cm.Paired([i / len(self.label_dict) for i in range(len(self.label_dict))])
        colors = {cat: color for cat, color in zip(self.label_dict.values(), colors)}
        for j in range(len(self._x)):
            tmp_data = self._x[j]
            x_min, x_max = np.min(tmp_data), np.max(tmp_data)
            gap = x_max - x_min
            tmp_x = np.linspace(x_min-0.1*gap, x_max+0.1*gap, 200)
            title = "$j = {}$".format(j + 1)
            plt.figure()
            plt.title(title)
            for c in range(len(self.label_dict)):
                plt.plot(tmp_x, self._data[j][c](tmp_x),
                         c=colors[self.label_dict[c]], label="class: {}".format(self.label_dict[c]))
            plt.xlim(x_min-0.2*gap, x_max+0.2*gap)
            plt.legend()
            if not save:
                plt.show()
            else:
                plt.savefig("d{}".format(j + 1))
