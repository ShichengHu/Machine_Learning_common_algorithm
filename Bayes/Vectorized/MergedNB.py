from Bayes.Vectorized.Basic import *
from Bayes.Vectorized.MultinoimalNB import MultinomialNB
from Bayes.Vectorized.GaussianNB import GaussianNB
from Util.Util import quantize_data


class MergedNB(NaiveBayes):

    def __init__(self, **kwargs):
        super(MergedNB, self).__init__(**kwargs)
        self._multinomial, self._gaussian = MultinomialNB(), GaussianNB()

        wc = kwargs.get("whether_continuous")
        if wc is None:
            self._whether_discrete = self._whether_continuous = None
        else:
            self._whether_continuous = np.asarray(wc)
            self._whether_discrete = ~self._whether_continuous

    def feed_data(self, x, y, sample_weight=None):
        if sample_weight is not None:
            sample_weight = np.asarray(sample_weight)
        x, y, wc, features, feat_dicts, label_dict = quantize_data(
            x, y, wc=self._whether_continuous, separate=True)
        if self._whether_continuous is None:
            self._whether_continuous = wc
            self._whether_discrete = ~self._whether_continuous
        self.label_dict = label_dict

        (discrete_x, continuous_x) = x

        cat_counter = np.bincount(y)
        self._cat_counter = cat_counter

        labels = [y == value for value in range(len(cat_counter))]
        labelled_x = [discrete_x[ci].T for ci in labels]

        self._multinomial._x, self._multinomial._y = discrete_x.T, y  # x, y
        self._multinomial._labelled_x, self._multinomial._label_zip = labelled_x, list(zip(labels, labelled_x))
        self._multinomial._cat_counter = cat_counter
        self._multinomial._feat_dicts = [dic for i, dic in enumerate(feat_dicts) if self._whether_discrete[i]]
        self._multinomial._n_possibilities = [len(feats) for i, feats in enumerate(features)
                                              if self._whether_discrete[i]]
        self._multinomial.label_dict = label_dict

        labelled_x = [continuous_x[label].T for label in labels]

        self._gaussian._x, self._gaussian._y = continuous_x.T, y  # self._x是转置过的
        self._gaussian._labelled_x, self._gaussian._label_zip = labelled_x, labels
        self._gaussian._cat_counter, self._gaussian.label_dict = cat_counter, label_dict

        self.feed_sample_weight(sample_weight)

    def feed_sample_weight(self, sample_weight=None):
        self._multinomial.feed_sample_weight(sample_weight)
        self._gaussian.feed_sample_weight(sample_weight)

    def _fit(self, lb):  # lb 是高斯平滑项
        # 因为self._x都已经定义过了，所以_fit()就可以计算分割过的数据各自的后验概率矩阵
        self._multinomial["fit"](lb)  # 之前是 self._multinomial._fit(lb),当然现在意义是相同的
        self._gaussian["fit"](lb)
        self._p_category = self._multinomial["p_category"]

    def _func(self, x, i):
        x = np.atleast_2d(x)
        return self._multinomial["func"](
            x[:, self._whether_discrete].astype(np.int), i) * self._gaussian["func"](
            x[:, self._whether_continuous], i) / self._p_category[i]

    def _transfer_x(self, x):
        feat_dicts = self._multinomial["feat_dicts"]
        idx = 0
        for d, discrete in enumerate(self._whether_discrete):
            for i, sample in enumerate(x):
                if not discrete:
                    x[i][d] = float(x[i][d])
                else:
                    x[i][d] = feat_dicts[idx][sample[d]]
            if discrete:
                idx += 1
        return x
