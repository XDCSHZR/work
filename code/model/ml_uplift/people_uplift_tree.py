"""
Describe :
Time :
Author: liwuzhuang@didiglobal.com
"""

import numpy as np
from collections import defaultdict
import pydotplus

class TreeNode:
    def __init__(self, value=None, branch_left=None, branch_right=None, summary=None):
        self.value = value
        self.branch_left = branch_left
        self.branch_right = branch_right
        self.summary = summary

class PeopleUpliftTree:
    def __init__(self, max_depth=6, min_samples_leaf=40000, function_score="default", function_gain="default"):
        self.max_depth = max_depth
        self.min_samples_leaf = min_samples_leaf

        if function_score == "default":
            self.get_score = self.function_score_default
        elif function_score == "user-defined":
            self.get_score = self.function_score_user_defined

        if function_gain == "default":
            self.get_gain = self.function_gain_default
        elif function_gain == "user-defined":
            self.get_gain = self.function_gain_user_defined

    def fit(self, x, treatment, y, cost=None):
        assert len(x) == len(y) and len(x) == len(treatment), 'Data length must be equal for X, treatment, and y.'
        assert set(treatment) == set([0, 1]), '0 means control name, 1 means treatment, they must be 0 and 1.'

        tree_fitted = self.grow_tree(x, treatment, y, cost, max_depth=self.max_depth, min_samples_leaf=self.min_samples_leaf, depth=1)
        return tree_fitted

    def grow_tree(self, x, treatment, y, cost=None, max_depth=10, min_samples_leaf=100, depth=1):

        if len(x) == 0:
            return TreeNode()

        # 节点的数据统计
        score_current = self.get_score(treatment, y, cost)
        summary = {}
        summary['n_samples'] = len(x)
        summary['n_samples_treatment'] = (treatment == 1).sum()
        summary['n_samples_control'] = (treatment == 0).sum()
        summary['score'] = score_current

        # 获取特征的分割点
        values_unique = list(set(x))
        if isinstance(values_unique[0], int) or isinstance(values_unique[0], float):
            if len(values_unique) > 10:
                values_percentile = np.percentile(x, [3, 5, 10, 20, 30, 50, 70, 80, 90, 95, 97])
                values_unique = np.unique(values_percentile)

        # 寻找最优分割点
        gain_best = 0.0
        value_best = None
        data_best_left = [[], [], [], []]
        data_best_right = [[], [], [], []]
        for value_split in values_unique:
            x_left, x_right, w_left, w_right, y_left, y_right, cost_left, cost_right = self.split_data(x, treatment, y, value_split, cost)

            if len(x_left) < min_samples_leaf or len(x_right) < min_samples_leaf:
                continue
            if set(w_left) != set(w_right):
                continue

            score_left = self.get_score(w_left, y_left, cost_left)
            score_right = self.get_score(w_right, y_right, cost_right)

            gain = self.get_gain(score_left, score_right, score_current)

            if gain > gain_best:
                gain_best = gain
                value_best = value_split
                data_best_left = [x_left, w_left, y_left, cost_left]
                data_best_right = [x_right, w_right, y_right, cost_right]

        if gain_best > 0 and depth < max_depth:
            branch_left = self.grow_tree(*data_best_left, max_depth, min_samples_leaf, depth + 1)
            branch_right = self.grow_tree(*data_best_right, max_depth, min_samples_leaf, depth + 1)
            return TreeNode(value=value_best, branch_left=branch_left, branch_right=branch_right, summary=summary)
        else:
            return TreeNode(summary=summary)

    @staticmethod
    def function_score_default(treatment, y, cost=None):
        g_t = treatment == 1
        g_c = treatment == 0

        if cost is not None:
            cost_delta = cost[g_t].mean() - cost[g_c].mean()
            y_delta = y[g_t].mean() - y[g_c].mean()
            score = y_delta / cost_delta
        else:
            y_delta = y[g_t].mean() - y[g_c].mean()
            score = y_delta

        return score


    @staticmethod
    def function_score_user_defined(treatment, y, cost=None):
        pass

    @staticmethod
    def function_gain_default(score_left, score_right, score_parent):
        return max(score_left, score_right) - score_parent
    @staticmethod
    def function_gain_user_defined(score_left, score_right, score_parent=None):
        pass

    @staticmethod
    def split_data(x, treatment, y, value, cost=None):
        if isinstance(value, int) or isinstance(value, float):
            flag = x >= value
        else:  # for strings
            flag = x == value
        if cost is not None:
            return x[flag], x[~flag], treatment[flag], treatment[~flag], y[flag], y[~flag], cost[flag], cost[~flag]
        else:
            return x[flag], x[~flag], treatment[flag], treatment[~flag], y[flag], y[~flag], None, None

    @staticmethod
    def plot_tree(tree, x_name, score_name):

        nodes_data_tree = defaultdict(list)

        def to_string(is_split, tree, bBranch, szParent="null", indent='', indexParent=0, x_name="cnt"):
            if tree.value is None:
                nodes_data_tree[is_split].append(['leaf', "leaf", szParent, bBranch,
                                                  str(round(float(tree.summary['score']), 2)),
                                                  str(round(float(tree.summary['n_samples']) / 100, 1)) + "hundred",
                                                  indexParent])
            else:
                if isinstance(tree.value, int) or isinstance(tree.value, float):
                    decision = '%s >= %s' % (x_name, tree.value)
                else:
                    decision = '%s == %s' % (x_name, tree.value)

                indexOfLevel = len(nodes_data_tree[is_split])
                to_string(is_split + 1, tree.branch_left, True, decision, indent + '\t\t', indexOfLevel, x_name)
                to_string(is_split + 1, tree.branch_right, False, decision, indent + '\t\t', indexOfLevel, x_name)
                nodes_data_tree[is_split].append([is_split + 1, decision, szParent, bBranch,
                                                  str(round(float(tree.summary['score']), 2)),
                                                  str(round(float(tree.summary['n_samples']) / 100, 1)) + "hundred",
                                                  indexParent])

        to_string(0, tree, None, x_name=x_name)

        dots = ['digraph Tree {',
                'node [shape=box, style="filled, rounded", fontname=helvetica] ;',
                'edge [fontname=helvetica] ;'
                ]
        i_node = 0
        dcParent = {}
        for nSplit in range(len(nodes_data_tree.items())):
            lsY = nodes_data_tree[nSplit]
            indexOfLevel = 0
            for lsX in lsY:
                iSplit, decision, szParent, bBranch, score, n_samples, indexParent = lsX

                if type(iSplit) is int:
                    szSplit = '%d-%d' % (iSplit, indexOfLevel)
                    dcParent[szSplit] = i_node
                    dots.append('%d [label=< %s : %s<br/> n_samples : %s<br/> %s<br/>> ] ;' % (
                        i_node, score_name, score, n_samples, decision.replace('>=', '&ge;').replace('?', '')
                    ))
                else:
                    dots.append('%d [label=< %s : %s<br/> n_samples : %s<br/>>, fillcolor="%s"] ;' % (
                        i_node, score_name, score, n_samples, "green"
                    ))

                if szParent != 'null':
                    if bBranch:
                        szAngle = '45'
                        szHeadLabel = 'True'
                    else:
                        szAngle = '-45'
                        szHeadLabel = 'False'
                    szSplit = '%d-%d' % (nSplit, indexParent)
                    p_node = dcParent[szSplit]
                    if nSplit == 1:
                        dots.append('%d -> %d [labeldistance=2.5, labelangle=%s, headlabel="%s"] ;' % (p_node,
                                                                                                       i_node, szAngle,
                                                                                                       szHeadLabel))
                    else:
                        dots.append('%d -> %d ;' % (p_node, i_node))
                i_node += 1
                indexOfLevel += 1
        dots.append('}')
        dot_data = '\n'.join(dots)
        graph = pydotplus.graph_from_dot_data(dot_data)
        return graph