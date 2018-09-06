from DecisionTree.Cluster import Cluster
import numpy as np

# affected是向上移动，pruned是下移（也就是如果当前节点要剪枝，当前节点的父节点affected=True, 而其子节点pruned=True)
#  更新_parent=self.parent 会导致，self.root.height==1这显然是错误的，也会导致draw(),list index out of range
# 删除了self.tree.y_transformer
# return 1+ max(_child.height)总是出错
# 在property height 中加了max_depth判别，防止出现ValueError: max() arg is an empty sequence
# 总是有解决办法，我记得解决过这个问题
# return "Node ({})({} -> {})".
# format(self._depth, self.tree.rev_feat_dic[self.parent.feature_dim][self.prev_feat], self.feature_dim)
# self.left_child is not None and self.right_child is not None -> self.left_child.category is not None
# stop 2加入hand_termainate()
# 285 Node.stop2 -> self.stop2之前怎么没发现
# self.label_dict不能没有，否则当tree中cut_tree时，tree没了，就不能通过self.label_dict数值化结果


# 这个类用于生成决策树本身,也就是一个个节点
# 这里处理的数据是数值化后的
class Node:
    """
    self._x,self._y：存放数据
    self.base：对数的基底
    self.chaos：当前的不确定度
    self.criterion : 记录该节点用来计算信息增益所用的方法
    self.category: 记录该节点所属的类别
    self.left_child,self.right_child：记录节点的左右子节点
    self._children,self.leafs: 记录该节点的所有子节点和叶节点(大字典里存小字典)字典可以用pop删除任意元素
    self.sample_weight: 记录样本权重
    self.wc: 记录各个维度的特征是否是连续的列表(whether continuous)
    self.tree: 记录该节点所属的树
    self.feature_dim: 记录作为划分标准的特征的维度（作为划分标准的特征）
    self.tar: 针对连续型特征和cart，记录二分标准
    self.feats: 记录该节点所能进行选择的作为划分标准的特征的维度(self._x[feat]说明指定就是特征)
    obviously, on one road the same feature can't be choose twice.因此生成新节点时需要减去上一个被选属性
    self.feats最初是所有属性，self.depth+1,self.feats数目减一
    self.parent: 记录该节点的父节点
    self.is_root: 该节点是否为根节点
    self._depth: 记录节点的深度
    self.prev_feat: 当前节点的类别标签，只有根节点记录的是root
    self.is_cart: 是否使用CART算法,这里的CART是以gini指数作为评估指标，而不是最小均方差MSI
    self.is_continuous: 记录该节点选择的划分标准对应的特征是否连续
    self.pruned: 记录该节点是否已经被剪掉，局部剪枝算法
    self.affected: 当前节点是否被剪枝影响,如果当前节点的子节点，则将affected标记为True
    """
    def __init__(self, tree=None, base=2, chaos=None, depth=0,
                 parent=None, is_root=True, prev_feats="Root"):
        self._x = self._y = None
        self.base, self.chaos = base, chaos
        self.criterion = self.category = None
        self.left_child = self.right_child = None
        self._children, self.leafs = {}, {}  # _children 和leafs存在dict中的，可以用pop弹出不需要的信息。来剪枝,{0:{},1:{}}
        self.sample_weight = None
        self.wc = None
        self.tree = tree  # label_dict,y_transformer,max_depth,feature_sets,reduce_nodes,layers,nodes,max_depth
        # 如果传入了TREE就进行相应的初始化
        # print(self.tree)
        if self.tree is not None:
            # 由于数据的预处理是
            # y由于数据的预处理是由Tree完成的，所有各个维度的特征是否是连续型随机变量也是有Tree记录的
            self.criterion = self.tree.criterion
            self.wc = self.tree.whether_continuous
            self.is_cart = self.tree.is_cart
            self.label_dict = self.tree.label_dict  #
            # nodes是tree中记录的所有Node的列表
            tree.nodes.append(self)  # 用的是Tree而不是self.tree， 但是应当是一样的，因为生成子节点用到是self.tree
        self.feature_dim, self.tar, self.feats = None, None, []
        self.parent, self.is_root = parent, is_root
        self._depth, self.prev_feat = depth, prev_feats
        self.is_continuous = self.pruned = False
        self.affected = False

    def __getitem__(self, item):
        # isinstance(obj,class_or_tuple) 判断这个实例是不是后面的类的实例，obj是不是class的实例
        if isinstance(item, str):  # 这个判断我觉得没有必要
            return getattr(self, "_" + item)  # 访问带有_的变量

    # 重载__lt__方法，使得Node之间可以比较谁更小、进而方便调试和可视化
    def __lt__(self, other):
        return self.prev_feat < other.prev_feat

    # 重载__str__和__repr__方法，同样是为了方便调试和可视化
    # 对非叶子节点，我们希望得到的信息包括：深度，先前节点划分的特征，当前节点划分的特征
    # 对叶子节点我们希望得到：深度，先前节点划分属性，当前节点标记的类别。这样可以实现可视化
    def __str__(self):
        # __str__用print或者str出发，返回return的结果
        if self.category is None:  # 非叶子节点
            # return Node(self._depth)(self.prev_feat "->" self.feature_dim) format的优点在于不需要格式化对象的类
            # self._depth的格式是%d,self._prev_feat的格式是%s,self.feature_dim的格式是%d,但是很麻烦容易出错。
            return "Node ({})({} -> {})".format(self._depth, self.prev_feat, self.feature_dim)
        # 叶子节点
        return "Node ({})({} ->{})".format(self._depth, self.prev_feat, self.tree.label_dict[self.category])

    __repr__ = __str__  # repr魔法在实例化的时候会被调用
    # 定义children属性，主要是区分开连续和CART的情况和其余情况
    # 有了该属性后，想要获得所有子节点时就不用分情况讨论
    # 类下定义装饰器相当于把children中的self传给装饰器

    @property
    def info(self):
        if self.category is None:
            return "Node ({})({} -> {})".format(self._depth, self.prev_feat, self.feature_dim)
        return "Node ({})({} -> class: {})".format(
            self._depth, self.prev_feat, self.tree.label_dict[self.category])

    @property
    def children(self):  # 这三个property都是只有getter,也就是只能输出，不能设置
        return {
            "left": self.left_child, "right": self.right_child
        } if (self.is_cart or self.is_continuous) else self._children
    # if is_cart or continuous return left and right child,else return _children

    # 递归定义height属性
    # 叶节点高度都定义为1，其余节点高度定义为最高的子节点的高度+1
    @property
    def height(self):
        # 如果当前节点的标记类别不为空，也就是叶节点，叶节点高度都为1，返回1
        # print("children: {}\n".format(self.children))
        if self.category is not None:
            return 1
        # 否则返回1+最大子节点的高度
        # if self.tree.max_depth is not None:  # AttributeError: 'NoneType' object has no attribute 'max_depth'
            # 如果cut_tree那么哪里还有tree呢，无论max_depth
            # return self.tree.max_depth
        else:
            return 1 + max([_child.height if _child is not None else 0 for _child in self.children.values()])

    # 定义信息字典属性，记录该节点的主要信息
    # 在更新各个Node的叶节点时，被记录进各个self.leafs属性的就是该字典
    # property 的优点在于调用的时候不需要加后面的括号
    # info_dic store the leaf node's message % _parent.leafs[id(self)] = self.info_dic
    @property
    def info_dic(self):
        return {"chaos": self.chaos, "y": self._y}

    # 实现生成算法的准备工作，定义停止生成的准则，定义停止后该节点的行为
    # 定义停止准则1：当特征维度(样本数)为0或当前Node的数据几属于同一类别
    # 同时，如果用户指定了决策树的最大深度
    # 那么当该Node的深度太深时也停止
    # 若满足了停止条件，该函数会返回True，否则会返回False

    def stop1(self, eps):
        if(
            self._x.shape[1] == 0 or len(self.feats) == 1 or(self.chaos is not None and self.chaos <= eps)
                or (self.tree.max_depth is not None and self._depth >= self.tree.max_depth)
        ):
            # 定义处理停止情况的方法，核心思想就是把该Node转化为一个叶节点
            self._handle_terminate()
            return True
        return False

    # 定义第二种停止条件,如果信息增益仍小于阈值时停止
    def stop2(self, max_gain, eps):
        if max_gain <= eps:
            self._handle_terminate()
            return True
        return False

    # 生成该节点所属的类
    # 类别标记为训练样例数最多的类别
    # 利用bincount方法定义根据数据生成该Node所属类别的方法
    def get_category(self):
        # np.argmax(a,axis=None)返回最大值对应的索引,np.bincount是从0开始，所以argmax对应的也就是位置了，即类别
        return np.argmax(np.bincount(self._y))

    def _handle_terminate(self):
        # 首先要生成该Node所属的类
        self.category = self.get_category()
        # 然后一路回溯，更新父节点，等等，记录叶节点的的属性leafs
        # id(o)返回地址
        _parent = self.parent  # _parent = self -> _parent=self.parent
        while _parent is not None:
            _parent.leafs[id(self)] = self.info_dic
            _parent = _parent.parent

    # 局部剪枝
    # 定义一个方法使其能将一个有子节点的Node转化为叶节点（局部剪枝）
    # 定义一个方法使其能挑选出最好的划分标准
    # 定义一个方法使其能根据划分标准进行生成
    def prune(self):
        """把当前节点变成叶节点，当前节点的子节点，叶节点清空
        重新定义当前属性值，并通过递归对父节点，父节点的父节点的叶节点更新
        但是当前节点还是存在，并没有从self.tree.nodes中删除
        """
        # 调用相应方法计算该Node所属类别
        self.category = self.get_category()
        # 记录由于该Node转化为叶节点而剪去的、下属的叶节点
        _pop_lst = [key for key in self.leafs]
        # 然后一路回溯，更新各个parent的属性leafs（使用id作为key以避免重复）
        # _parent = self
        _parent = self.parent  # 对所有当前节点的父节点标记
        while _parent is not None:
            _parent.affected = True
            for _k in _pop_lst:
                # 删去由于局部剪枝而被剪掉的叶节点
                _parent.leafs.pop(_k)  # KeyError
            _parent.leafs[id(self)] = self.info_dic
            _parent = _parent.parent  # self.parent -> _parent.parent
            # _parent.affected = True  # AttributeError: 'NoneType' object has no attribute 'affected'
        #     根节点的parent=None,这就是问题所在
        # 调用mark_pruned方法将自己所有的子节点、子节点的子节点
        # 的pruned的属性置为True，因为他们都被’剪掉‘了
        self.mark_pruned()
        # 重置各个属性
        self.feature_dim = None
        self.left_child = self.right_child = None
        self._children = {}
        self.leafs = {}

    def mark_pruned(self):
        self.pruned = True
        # 遍历各个子节点
        for _child in self.children.values():
            # 如果当前的子节点不是None的话，递归调用mark_pruned方法
            # 连续型特征和CART算法有可能导致children中出现None，因为此时children有left_child和right_child组成
            # 他们有可能是None
            if _child is not None:
                _child.mark_pruned()

    def fit(self, x, y, sample_weight, feature_bound=None, eps=1e-8):
        """
        fit 是寻找最佳增益的特征维度和划分值，更新相关属性，并且判断是否局部剪枝,把当前节点变成叶节点
        然后调用gen_children对数据分割，然后重复调用fit，实现生成整个决策树。
        """
        self._x, self._y = np.atleast_2d(x), np.array(y)
        self.sample_weight = sample_weight
        # 若满足第一停止准则，退出函数体
        if self.stop1(eps):
            return
        # 用该节点的数据实例化Cluster类以计算各种信息量
        _cluster = Cluster(self._x, self._y, self.sample_weight, self.base)
        if self.is_root:
            if self.criterion == 'gini':
                self.chaos = _cluster.gini()
            else:
                self.chaos = _cluster.ent()
        _max_gain, _chaos_lst = 0, []  # 最佳增益，熵表
        _max_feature = None  # 最大增益的属性维度,返回给feature_dim
        _max_tar = None  # 最佳划分二分类分割点,返回给self.tar
        # 遍历还能选择的特征

        # 为了实现随机森林，加入feature_bound，实现随机的作用

        feat_len = len(self.feats)
        if feature_bound is None:
            indices = range(0, feat_len)
        elif feature_bound == "log":
            # np.random.permutation(n) 对0，1，...n-1打乱循序，返回打乱后的结果。当然不同于shuffle
            indices = np.random.permutation(feat_len)[: max(1, int(np.log(feat_len)))]
        else:
            indices = np.random.permutation(feat_len)[: feature_bound]
        tmp_feat = [self.feats[i] for i in indices]

        # tmp_feat 存的是原x中的一个列，对应同一个属性
        # update: for feat in self.feats:
        for feat in tmp_feat:
            # feat就是取这一列中的元素，现在来找分割点集
            # 我觉得直接对这列元素排序，然后求和取平均就可以了
            # 如果是连续型特征或者是CART算法，需要另外计算二分标准的取舍集合
            if self.wc[feat]:
                _features = self.tree.feature_sets[feat]  # set(self._x.T[feat])
                _features = np.array(list(_features), dtype=float)
                _set = (_features[:-1] + _features[1:]) * 0.5
            else:  # 不连续且是cart时
                if self.is_cart:
                    _set = self.tree.feature_sets[feat]  # set(self._x.T[feat])
                else:
                    _set = None
            # 然后遍历这些二分标准并且调用二类问题相关的计算信息量的方法
            if self.is_cart or self.wc[feat]:
                for tar in _set:
                    _tmp_gain, _tmp_chaos_lst = _cluster.bin_info_gain(
                        idx=feat, tar=tar, criterion=self.criterion, get_chaos_lst=True, continuous=self.wc[feat])
                    if _tmp_gain > _max_gain:
                        (_max_gain, _chaos_lst) = (_tmp_gain, _tmp_chaos_lst)
                        _max_feature = feat
                        _max_tar = tar
            # 离散的ID3和C4.5调用一般计算信息量的算法，这时就没有tar了
            else:
                _tmp_gain, _tmp_chaos_lst = _cluster.info_gain(
                    idx=feat, criterion=self.criterion, get_chaos_lst=True)  # , features=self.tree.feature_sets[feat]
                if _tmp_gain > _max_gain:
                    (_max_gain, _chaos_lst) = (_tmp_gain, _tmp_chaos_lst)
                    _max_feature = feat

        # 若满足第二停止条件，就退出函数体
        if self.stop2(max_gain=_max_gain, eps=eps):
            return
        # 更新相关属性
        self.feature_dim = _max_feature
        if self.is_cart or self.wc[_max_feature]:
            self.tar = _max_tar

            # 调用根据划分标准进行生成的方法，之前是条件熵的生成子节点后就称为了信息熵，计算方式一致，集合大小变了而已
            self._gen_children(_chaos_lst, feature_bound)
            # prev_feat:Root
            # category: left:None, right: None  AttributeError: 'NoneType' object has no attribute 'affected'
            # 如果左右孩子都是叶节点且所属类别一样，那就将他们合并，亦进行局部剪枝
            if (
                    self.left_child.category is not None
            )and (
                    self.left_child.category == self.right_child.category):

                self.prune()
                # 调用tree的相关方法，将被剪掉的该Node的左右子节点，从Tree的记录所有Nodes中出去
                self.tree.reduce_nodes()
        else:

            # 调用根据划分标准进行生成的方法
            self._gen_children(_chaos_lst, feature_bound)

    # Create Node,realize recursion
    # 生成子节点，不考虑局部修剪，然后递归调用fit生成
    def _gen_children(self, chaos_lst, feature_bound=None):
        feat, tar = self.feature_dim, self.tar
        self.is_continuous = continuous = self.wc[feat]
        # sample = self._x[..., feat]  # x[:,1]=x[...,1]
        features = self._x.T[feat]
        #  features改成sample是不是更好,因为这个语句相当于取了x中的一个样本
        new_feats = self.feats.copy()  # new_feats是原feats的复制，new_feats的变化不影响原feats
        # 特征取值连续
        if continuous:
            mark = features < tar  # mark 样本中小于tar的部分
            marks = [mark, ~mark]  # 互补,masks存放二分类结果
        else:  # cart决策树，在特征取值离散时亦二分
            if self.is_cart:
                mark = features == tar
                marks = [mark, ~mark]
                self.tree.feature_sets[feat].discard(tar)  # 弃牌？
            else:  # 一般情况，特征取值离散，ID3，C4.5
                marks = None
        if self.is_cart or continuous:
            # CART and continuous 是二分类
            # if not continuous,feats=[tar,"+"] else feats=["tar-", "tar+"]
            # [tar,"+"]怎么匹配啊，左边是tar,右边是+
            feats = [tar, "+"] if not continuous else ["{:6.4}-".format(tar), "{:6.4}+".format(tar)]
            for feat, side, chaos in zip(feats, ["left_child", "right_child"], chaos_lst):
                # 用可选的特征维度喂给各新的Node，生成当前节点子节点
                new_node = self.__class__(
                    self.tree, self.base, chaos, depth=self._depth + 1,
                    parent=self, is_root=False, prev_feats=feat)
                new_node.criterion = self.criterion
                setattr(self, side, new_node)  # 将当前Node中的side（也就是将左右孩子的）属性值设为new_node
                # setattr 从内部赋值
                # self._children[side] = new_node  self.left_child = new_node
            for node, feat_mark in zip([self.left_child, self.right_child], marks):
                # node 对应当前所取节点的子节点，feat_mark对应当前所取的子集[True,False]
                # feat_mark 对应当前类别的分类结果
                if self.sample_weight is None:
                    local_weights = None
                else:
                    local_weights = self.sample_weight[feat_mark]
                    local_weights /= np.sum(local_weights)  # 对样本权重归一化
                tmp_data, tmp_label = self._x[feat_mark, ...], self._y[feat_mark]
                # x[1, ...] = x[1,:] 第二行 x[...,1]=x[:,1]第二列
                if len(tmp_label) == 0:
                    # 当前节点样本数为0
                    continue
                node.feats = new_feats
                node.fit(tmp_data, tmp_label, local_weights, feature_bound)
        # 划分标准是离散特征，需要将ID3或者C4.5,需将该特征对应的维度从新Node的self.feats属性中除去
        # 若算法是CART，需要将二分标准从新Node的二分标准取值集合中除去
        # 最后对新Node调用fit方法，完成递归
        else:
            # ID3 and C4.5 may not create binary tree,we don't have left and right child.
            new_feats.remove(self.feature_dim)  # from current feats remove feature_dim
            for feat, chaos in zip(self.tree.feature_sets[self.feature_dim], chaos_lst):
                feat_mark = features == feat
                tmp_x = self._x[feat_mark, ...]  # 该特征维度取值为feat_mark对应的样本
                tmp_y = self._y[feat_mark]
                if len(tmp_x) == 0:
                    continue
                # new node will marked with feat as leaf node's label.
                new_node = self.__class__(
                    self.tree, self.base, chaos, depth=self._depth + 1,
                    parent=self, is_root=False, prev_feats=feat)
                new_node.feats = new_feats
                self.children[feat] = new_node
                if self.sample_weight is None:
                    local_weights = None
                else:
                    local_weights = self.sample_weight[feat_mark]
                    local_weights = local_weights / np.sum(local_weights)
                new_node.fit(tmp_x, tmp_y, local_weights, feature_bound)

    # if the children of current node exist,update current layer,add this node to this layer
    def update_layers(self):
        # 在当前深度中记录当前节点
        self.tree.layers[self._depth].append(self)  # append(self) 实际上会调用__str__，也就得以将信息存到layers中
        for _node in sorted(self.children):
            # 对字典排序得到的是字典的keys，可以用self.children[_node]遍历所有子节点,type(sorted(dic))=list
            # sorted()函数返回排序后的结果，而且被排序的对象本身不变，but how to sort,感觉没有作用
            # but children store left child and right child how func sort work?
            # why child node at the same layer
            # 对所有节点递归
            _node = self.children[_node]
            # for _node_index, _node in enumerate(sorted(self.children)):
            # _node = self.children[_node_index]
            if _node is not None:
                _node.update_layers()

    # 定义当前节点的损失函数（用于ID3和C4.5的剪枝）
    def cost(self, pruned=False):
        if not pruned:  # 有叶节点
            return sum([leaf["chaos"] * len(leaf["y"]) for leaf in self.leafs.values()])
        return self.chaos * len(self._y)  # 无叶节点

    # 剪枝阈值，希望叶子节点少，剪枝前后的代价变化大（CART剪枝）
    def get_threshold(self):
        return (self.cost(pruned=True) - self.cost()) / (len(self.leafs) - 1)

    def cut_tree(self):
        # make current node's children self.tree=None
        self.tree = None
        for child in self.children.values():
            if child is not None:
                child.cut_tree()

    def feed_tree(self, tree):
        # make every child of current node self.tree = tree,append child
        self.tree = tree
        self.tree.nodes.append(self)
        self.wc = tree.whether_continuous
        for child in self.children.values():
            if child is not None:
                child.feed_tree(tree)

    # 不管有多少个return，只要符合条件就返回，往后的return就不看了。走到第一个return就结束了
    def predict_one(self, x):
        # 对输入的样本进行预测
        if self.category is not None:  # 若只含叶节点，返回叶节点的类别作为预测结果
            return self.category
        if self.is_continuous:  # else
            if x[self.feature_dim] < self.tar:
                # 再当前节点决策，选择当前节点的划分特征比较
                # 但是这种预测只是单层的决策，并没法得到预测的输出类别，还需要重复调用
                return self.left_child.predict_one(x)
            return self.right_child.predict_one(x)
        if self.is_cart:
            if x[self.feature_dim] == self.tar:
                return self.left_child.predict_one(x)
            return self.right_child.predict_one(x)
        else:
            try:
                return self.children[x[self.feature_dim]].predict_one(x)
            except KeyError:
                return self.get_category()

    # 对输入数据预测，返回预测结果
    def predict(self, x):  # 在tree.prune_中调用了计算acc，因此我为了形式的一致对它结果数值化
        return np.array([self.label_dict[self.predict_one(xx)] for xx in x])

    def view(self, indent=4):
        print(" " * indent * self._depth, self.info)  # 会调用__str__()
        for node in sorted(self.children):  # 对于怎么sort我就不知道了，实际没有sort也可以吧
            node = self.children[node]
            if node is not None:
                node.view()
