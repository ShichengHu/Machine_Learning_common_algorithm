from DecisionTree.CVDNode import *
from copy import deepcopy
import numpy as np
import cv2

# warnings.simplefilter(action="ignore", category="FutureWarning")
# 更新 在draw中，为了能可视化特征类别，因为我的数据事先就数值化了，在295行加入了判决self.rev_feat_dic[node.parent[self.feature_dim][node.prev_feat]来使得可以可视化
#  self.root.label_dict = self.label_dict  # 在feed_data中加了这句话


class Base:
    """
    self.nodes: 记录所有node的列表
    self.roots: 主要用于CART剪枝的属性
    self.max_depth: 记录决策树最大深度的属性
    self.root,self.feature_sets: 根节点和记录可选特征维度的列表
    self.label_dic: 类别的转换字典
    self.prune_alpha，self.layers: 主要用于ID3和C4.5剪枝的两个属性，可先按下不表示
    self.prune_alpha: 惩罚因子，正则化处理最小化代价函数，是一个超参数
    self.layers: 记录每一层的Node
    self.whether_continuous: 记录各个维度特征是否连续的列表
    self.visualized: 绘制框图

    """
    def __init__(self, criterion='ent', label_dict=None, max_depth=None, whether_continuous=None, rev_feat_dict=None,
                 is_cart=False, visualized=False):

        self.nodes, self.layers, self.roots = [], [], []
        self.max_depth = max_depth
        self.feature_sets = []
        self.label_dict = label_dict
        self.rev_feat_dic = rev_feat_dict
        self.prune_alpha = 1
        self.whether_continuous = np.array(whether_continuous) if whether_continuous is not None else None
        self.is_cart = is_cart
        self.criterion = criterion
        self.visualized = visualized  #
        self.root = Node(tree=self)

    def __str__(self):
        return "Tree ({})".format(self.root.height)

    __repr__ = __str__

    # 判断各特征的连续型并记录在whether_continuous中，及其他数据预处理
    def feed_data(self, x, continuous_rate=0.2):
        # continuous_rate 也算是超参数，需要提前给定
        # x = np.atleast_2d(x)
        assert x.ndim == 2, "数据x是二维的"
        # 利用set获得各个维度特征的所有可能取值
        self.feature_sets = [set(dimension) for dimension in x.T]
        data_len, data_dim = x.shape
        # 判断是否连续
        # 如果这个特征中的可能取值个数大于样本个数的continuous倍，判定为连续，continuous的值应该随样本个数变化
        if self.whether_continuous is None:
            self.whether_continuous = np.array([len(feat) > continuous_rate * data_len for feat in self.feature_sets])
        self.root.feats = [i for i in range(data_dim)]
        self.root.label_dict = self.label_dict  #
        self.root.feed_tree(self)  # 在根节点有无意义呢

    # Grow
    # 考虑到剪枝
    # 将类别向量数值化
    # 将数据集切分成训练集和交叉验证集，同时处理好样本权重
    # 对根节点调用决策树生成算法
    # 调用剪枝算法
    # 参数α和剪枝有关，cv_rate用于控制交叉验证集的大小，train_only则控制程序是否进行数据集的切分
    def fit(self, x, y, alpha=None, sample_weight=None, eps=1e-8, cv_rate=0.2, train_only=False, feature_bound="log",
            rf=False):
        x = np.atleast_2d(x)
        # 数值化类别向量
        _dic = {c: i for i, c in enumerate(set(y))}  # 类别的取值及类别的索引
        if self.label_dict is None:
            self.label_dict = {i: c for c, i in _dic.items()}  # 数值化后与数值化前之字典
        y = np.array([_dic[yy] for yy in y])  # 因为类别与类别在set中的位置是一一对应的，因此可以用类别位置代替类别名
        # 感觉就是对类别的进行了简化，计算更方便矩阵对数运算方便
        # 也就是再颠倒回来 self.label_dic={key:value for key, value in enumerate(set(y))}
        # dict.items() 以列表形式返回可以遍历的元组(key,value)数据
        # 根据特征个数定出alpha
        self.prune_alpha = alpha if alpha is not None else x.shape[1] / 2
        # 如果需要划分数据集,根节点是CART
        if not train_only:  # is_cart
            # 根据cv_rate将数据集随机分成训练集和交叉验证集
            # 实现的核心思想是利用下标来进行各种切分
            _train_num = int(len(x) * (1 - cv_rate))
            indices = np.random.permutation(range(x.shape[0]))
            _train_indices = indices[:_train_num]
            _test_indices = indices[_train_num:]
            if sample_weight is not None:
                _train_weight = sample_weight[_train_indices]
                _test_weight = sample_weight[_test_indices]
                _train_weight /= np.sum(_train_weight)
                _test_weight /= np.sum(_test_weight)
            else:
                _test_weight = _train_weight = None
            x_train, y_train = x[_train_indices], y[_train_indices]
            x_cv, y_cv = x[_test_indices], y[_test_indices]
        else:
            x_train, y_train, _train_weight = x, y, sample_weight
            x_cv, y_cv, _test_weight = None, None, None
        self.feed_data(x_train)
        # 调用根节点的生成算法,已经可以用来生成树了
        self.root.fit(x_train, y_train, _train_weight, feature_bound, eps)
        # 调用对Node剪枝算法的封装
        if not rf:  # 如果不是随机森林才可以调用剪枝方法
            self.prune_(x_cv, y_cv, _test_weight)  # wrong? 问题在于实现随机森林不允许调用剪枝方法，这样的话我还需要加点东西
        # 是否需要绘图
        if self.visualized:
            self.draw()

    # 将被剪掉的Node从nodes中删除，从后往前剪枝
    def reduce_nodes(self):
        for i in range(len(self.nodes) - 1, -1, -1):
            if self.nodes[i].pruned:  # 对所有节点遍历，如果当前节点被剪枝过了，那么就删除该节点
                self.nodes.pop(i)

    # prune
    def _update_layers(self):
        # 根据整棵决策树的高度，在self.layers里面放相应数量的列表
        self.layers = [[] for _ in range(self.root.height)]  # create an empty list.len(list) is the tree's height
        self.root.update_layers()

    # 更新树种所有节点存放在_tmp_nodes中，计算各节点损失函数，比较新旧损失大小，然后将结果存入mask,对于需要剪掉的节点
    # 要将节点从layers和_tmp_nodes删除，并对被影响的节点更新

    def _prune(self):
        """
        ID3 & C45 剪枝操作

        计算各个节点的损失函数 = 求和（叶子节点样本数*叶节点的熵）+ 惩罚因子*叶节点数 ，存好在old
        对所有节点计算剪枝后的损失函数 = 当前节点样本数*节点的熵 + 惩罚因子 ， 因为只有一个节点了 存于 new
        从最底层比较其损失函数变化，若剪枝后的损失函数少就剪枝。更新old， new，并将剪掉的节点从树中删除
        循环直到根节点为止
        :return:
        """
        # 得到的是old，new,mask，_tmp_nodes这些都是经过处理的结果
        # 第一步是更新每一层的节点,因为地下是_tmp_nodes=nodes,所以不明白这两个_tmp_nodes是否一样
        self._update_layers()
        _tmp_nodes = []
        # 更新完每层的Node后，从后往前
        for _node_lst in self.layers[::-1]:  # 最后一层，最后一个节点开始
            for _node in _node_lst[::-1]:
                if _node.category is None:  # _tmp_nodes存放的是非叶节点
                    _tmp_nodes.append(_node)  # _tmp_nodes中元素和layers中元素存放是相反顺序
        # old = np.array([_node.cost() + self.prune_alpha * len(_node.leafs)] for _node in _tmp_nodes)
        # 生成的是生成器，就没法比较了
        old = []
        for _node in _tmp_nodes:
            old.append(_node.cost() + self.prune_alpha*len(_node.leafs))
        old = np.array(old)
        new = np.array([_node.cost(pruned=True) + self.prune_alpha for _node in _tmp_nodes])
        mask = old >= new  # 比较剪枝前后的损失函数
        while True:
            # if only root,stop the loop
            if self.root.height == 1:
                break   # return is wrong
            p = np.argmax(mask)  # type: int
            # p 返回第一个出现True的位置，所以mask[p]肯定是有值的
            # mask[p]表面对于Node的剪枝是从后往前剪的，否则试想从根节点剪，那肯定会出问题
            if mask[p]:
                _tmp_nodes[p].prune()  # 把p对应位置的节点剪掉，变成叶节点
                for i, _node in enumerate(_tmp_nodes):  # 所以这里遍历是从树的最后节点往前递归
                    if _node.affected:  # 被影响的节点有当前节点的子节点和各父节点，node中定义了一个affected变量
                        # 对被影响的Node需要更新old，new,mask对应位置
                        old[i] = _node.cost() + self.prune_alpha * len(_node.leafs)
                        mask[i] = old[i] >= new[i]
                        _node.affected = False
                # 根据被剪掉的Node，将各个变量对应的位置除去（注意从后往前遍历）（对树而言是从根部开始遍历）
                for i in range(len(_tmp_nodes) - 1, -1, -1):
                    if _tmp_nodes[i].pruned:
                        _tmp_nodes.pop(i)
                        old = np.delete(old, i)  # np.delete(arr, obj) delete arr[obj] and return new arr
                        # np.delete() 会改变矩阵的形状，但由于这里的old是一维的，所以叶没有影响
                        new = np.delete(new, i)
                        mask = np.delete(mask, i)
            else:
                break
        self.reduce_nodes()

    def _cart_prune(self):
        """
        CART Tree 剪枝

        用不同阈值剪枝，直到根节点
        :return:
        """
        # 得到_threshold,_tmp_nodes
        # 暂时将所有节点记录所属的Tree的属性置为None，为什么呢
        # 这里的tmp_nodes每次循环都会改变，删除一些节点，而且这个过程每次是不独立的pop了就真的没有了
        self.root.cut_tree()  # 所有节点中的self.tree=None了， 但是nodes依然存在
        tmp_nodes = [node for node in self.nodes if node.category is None]  # node没有经过排序吧
        _thresholds = np.array([node.get_threshold() for node in tmp_nodes])
        # 对所有阈值进行遍历
        while True:
            # 利用deepcopy对当前的根节点进行深拷贝，存入self.roots列表
            # 如果前面没有记录Tree的属性置为None，那么这里也会对整个Tree做深拷贝。
            # 可以想象，这样会引发严重的内存问题，速度也会拖慢很多
            root_copy = deepcopy(self.root)
            self.roots.append(root_copy)  # 每次循环好像都会往里面加入一个self.root
            if self.root.height == 1:
                break
            # 从阈值最小的开始，逐渐递增，问题在于这种方法必然会执行到最后一个阈值，对于只有根节点，并没有选择最优解
            p = np.argmin(_thresholds)  # type: int
            tmp_nodes[p].prune()
            for i, node in enumerate(tmp_nodes):
                # 更新被影响的阈值
                if node.affected:
                    _thresholds[i] = node.get_threshold()
                    node.affected = False
            for i in range(len(tmp_nodes) - 1, - 1, -1):  # -1
                # 除去各列表相应位置的元素
                if tmp_nodes[i].pruned:
                    tmp_nodes.pop(i)
                    _thresholds = np.delete(_thresholds, i)
                    # _prune()的对象是_tmp_nodes,_threshold，通过遍历的方式对所有可能节点进行剪枝
        self.reduce_nodes()

    @staticmethod
    def acc(y, y_pred, weights):  # 静态方法，无需实例化即可调用
        if weights is not None:
            return np.sum((np.array(y) == np.array(y_pred) * weights)) / len(y)
        return np.sum(np.array(y) == np.array(y_pred)) / len(y)

    def prune_(self, x_cv, y_cv, weights=None):
        if self.root.is_cart:
            # 如果该node使用cart剪枝，那么只有在确实传入交叉验证集的情况下才能调用相关函数，否则没有意义，选择最佳阈值
            if x_cv is not None and y_cv is not None:
                self._cart_prune()
                _arg = np.argmax([Base.acc(y_cv, tree.predict(x_cv), weights) for tree in self.roots])  # type: int
                _tar_root = self.roots[_arg]  # 对于这里存的信息搞不懂元素是self.root然后呢
                self.nodes = []
                # 调用cart_prune()会先清空node 中的tree属性，所以剪完了还要再feed_tree
                #
                _tar_root.feed_tree(self)
                self.root = _tar_root
        else:
            self._prune()

    def predict_one(self, x):
            return self.label_dict[self.root.predict_one(x)]

    def predict(self, x):
            return np.array([self.predict_one(xx) for xx in x])  # 尚且可以优化提示运行速度

    def estimate(self, x, y, get_raw_result=False):
        y = np.array(y)
        if not get_raw_result:
            print("Acc: {:8.6} %".format(100 * np.sum(self.predict(x) == y) / len(y)))
        else:
            return np.average(self.predict(x) == y)

    def view(self):
            self.root.view()

    # 1200, 800
    def draw(self, radius=24, width=1800, height=1000, height_padding_ratio=0.2, width_padding=30, title="CvDTree"):
        """
        CV2 绘制决策树框图

        :param radius: 决策树中的每个节点的圆半径
        :param width: 框图的宽度
        :param height: 框图的高度
        :param height_padding_ratio: 圆的分布
        :param width_padding:
        :param title: 标题
        :return: 返回框图
        """
        self._update_layers()
        n_units = [len(layer) for layer in self.layers]

        img = np.ones((height, width, 3), np.uint8) * 255
        height_padding = int(
            height / (len(self.layers) - 1 + 2 * height_padding_ratio)  # (800/层数-1+2*0.2)*0.2+30
        ) * height_padding_ratio + width_padding
        height_axis = np.linspace(
            height_padding, height - height_padding, len(self.layers), dtype=np.int)
        # 800/层数, 800-800/层数, 层数
        width_axis = [
            np.linspace(width_padding, width - width_padding, unit + 2, dtype=np.int)
            # np.linspace(30, 1200-30, 每层节点数+2)
            for unit in n_units
        ]
        width_axis = [axis[1:-1] for axis in width_axis]  # 从第二个到倒数第二个(去掉首尾)

        for i, (y, xs) in enumerate(zip(height_axis, width_axis)):
            # i表示当前层
            # y当前y轴坐标，xs对应当前位置，j对应当前图在该层的第几个位置，x对应x轴坐标
            for j, x in enumerate(xs):
                if i == 0:  # 第一层和其他层颜色不同
                    # cv2.circle(img, center, radius, color, thickness=1,lineType=8, shift=0)
                    # img要画圆的矩形，center圆心位置，radius圆半径，color边框颜色，thickness 正值表示圆边框宽度. 负值表示画一个填充圆形
                    # lineType圆边框线形
                    cv2.circle(img, (x, y), radius, (225, 100, 125), 1)
                else:
                    cv2.circle(img, (x, y), radius, (125, 100, 225), 1)
                node = self.layers[i][j]
                if node.feature_dim is not None:  # 非叶节点
                    text = str(node.feature_dim + 1)
                    color = (0, 0, 255)
                else:
                    text = str(self.label_dict[node.category])
                    color = (0, 255, 0)
                cv2.putText(img, text, (x - 7 * len(text) + 2, y + 3), cv2.LINE_AA, 0.6, color, 1)
        # img显示文字所在的图片,text放置的文字，org文本左下角位置，fontFace字体格式，字体大小，color文本颜色，thickness线条粗细

        for i, y in enumerate(height_axis):
            if i == len(height_axis) - 1:   # 最后一层不管
                break
            for j, x in enumerate(width_axis[i]):
                new_y = height_axis[i + 1]  # height_axis[1]->height_axis[倒数第二] 也就是去掉首尾两行
                dy = new_y - y - 2 * radius
                for k, new_x in enumerate(width_axis[i + 1]):
                    dx = new_x - x
                    length = np.sqrt(dx ** 2 + dy ** 2)
                    ratio = 0.5 - min(0.4, 1.2 * 24 / length)
                    if self.layers[i + 1][k] in self.layers[i][j].children.values():
                        # cv2.line(img,pt1,pt2,color，thickness,lineType=8,shift=0)
                        # pt1线条的起点，pt2线条的终点，color线条颜色
                        cv2.line(img, (x, y + radius), (x + int(dx * ratio), y + radius + int(dy * ratio)),
                                 (125, 125, 125), 1)
                        node = self.layers[i + 1][k]
                        if self.rev_feat_dic is not None:
                            cv2.putText(img,
                                        str(self.rev_feat_dic[node.parent.feature_dim][node.prev_feat]
                                            if node.prev_feat is not "+" and self.rev_feat_dic[
                                            node.parent.feature_dim] is not None
                                            else node.prev_feat),
                                        (x + int(dx * 0.5) - 6, y + radius + int(dy * 0.5)),
                                        cv2.LINE_AA, 0.6, (0, 0, 0), 1)
                        else:
                            cv2.putText(img=img, text=str(node.prev_feat), org=(x+int(dx*0.5)-6, y+radius+int(dy*0.5)),
                                        fontFace=cv2.LINE_AA, fontScale=0.6, color=(0, 0, 0), thickness=1)
                        cv2.line(img, (new_x - int(dx * ratio), new_y - radius - int(dy * ratio)),
                                 (new_x, new_y - radius), (125, 125, 125), 1)

        cv2.imshow(title, img)
        cv2.imwrite(
            "C:\\Users\\C\\PycharmProjects\\Machine_Learning_1\\Classifier\\Tree\\"
            "test_pred_ent_cart_standardize.jpg", img)
        cv2.waitKey(0)
        return img


class ID3Tree(Base):
    def __init__(self, *args, **kwargs):
        # , whether_continuous=np.array([False, False, False])
        Base.__init__(self, criterion="ent", is_cart=False, *args, **kwargs)


class C45Tree(Base):
    def __init__(self, *args, **kwargs):
        # , whether_continuous = np.array([True, True, True])
        Base.__init__(self, criterion="ratio", is_cart=False, *args, **kwargs)


class CartTree(Base):
    def __init__(self, *args, **kwargs):
        # , whether_continuous=np.array([True, True, True])
        Base.__init__(self, criterion="gini", is_cart=True,
                      *args, **kwargs)
