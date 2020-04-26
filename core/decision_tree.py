import pandas as pd
import numbers as np
import numpy as np
import matplotlib.pyplot as plt
from tools.logger import Logger, logged
from types import FunctionType
from core.data_handler import DataHandler
from core.data_loader import DataLoader
import os

class Judge:
    def __init__(self, features_df):
        self.features_df = features_df

    def find_total_abs_deviation(self, cols=None):
        if cols is None:
            eval_set_df = self.features_df
        else:
            eval_set_df = self.features_df.loc[:, cols]
        departure_time = self.find_departure_time(eval_set_df)
        actual_arrivals = eval_set_df.loc[eval_set_df.index == departure_time, :].values
        total_deviation = np.sum(np.abs(actual_arrivals))
        return total_deviation, departure_time

    @staticmethod
    def find_departure_time(eval_set_df) -> float:
        """
        finds the 90th percentile lateness for each minute on the time window(row)
        :param eval_set_df:
        :return:
        """
        lateness = eval_set_df.quantile(q=0.9, axis=1)
        # find the departure time that corresponds to a lateness of 0
        # i.e the one that get us there on time 90%
        lateness[lateness > 0] = -120
        i_dep = np.argmax(lateness.values)
        return eval_set_df.index[i_dep]


class TreeNode:
    def __init__(self, features: list = None, parent=None,
                 recommendation: float = None, split_feature: int = 0):
        """
        :param features: The feature values corresponding to this node 1 or 0, None if that feature hasn't been split.
        :param parent: TreeNode
        :param recommendation: float
        :param split_feature: The index of the feature on which this node's children are split
        """
        self.low = None
        self.high = None
        self.is_leaf = True
        self.split_features = split_feature
        self.features = features
        self.recommendation = recommendation
        self.parent = parent

    def attempt_split(self, data: pd.DataFrame, error_fun: FunctionType, n_min: int) -> bool:
        """
        This method try to split this node into two child nodes
        :param data: Features for each of the data points
        :param error_fun: determines the fitness of a split
        Choose a split that minimizes the combined error of the resulting branches
        :param n_min: The number of data points that need to remain in a node to make it viable
        :return: True if a split happened and node has been updated with low high nodes
        """
        success = False
        n_features = len(data.columns)
        if self.features is None:
            self.features = [None] * n_features
        best_feature = -1
        # huge initial score
        best_score = 1e10
        best_hi_recommendation = self.recommendation
        best_lo_recommendation = self.recommendation

        node_data = self.filter_data(data)
        feature_candidates = [idx for idx, feature in enumerate(self.features) if feature is None]
        if feature_candidates:
            np.random.shuffle(feature_candidates)
            for i_feature in feature_candidates:
                # find the data for positive/negative features
                hi_data = node_data.loc[node_data.iloc[:, i_feature] == 1, :]
                low_data = node_data.loc[node_data.iloc[:, i_feature] == 0, :]
                if hi_data.shape[0] >= n_min and low_data.shape[0] >= n_min:

                    hi_score, hi_recommendation = error_fun(list(hi_data.index))
                    low_score, low_recommendation = error_fun(list(low_data.index))
                    split_score = hi_score + low_score
                    if split_score < best_score:
                        best_score = split_score
                        best_feature = i_feature
                        best_hi_recommendation = hi_recommendation
                        best_lo_recommendation = low_recommendation
                        success = True
            if success:
                self.update_child(best_feature=best_feature,
                                  hi_recommendation=best_hi_recommendation,
                                  low_recommendation=best_lo_recommendation)
                self.is_leaf = False
                self.split_features = best_feature

        return success

    def update_child(self, best_feature, hi_recommendation, low_recommendation) -> None:
        """
        Update the current node with  a successfully split with the hi and low TreeNode
        :param best_feature:
        :param hi_recommendation:
        :param low_recommendation:
        """
        hi_features = list(self.features)
        hi_features[best_feature] = 1
        lo_features = list(self.features)
        lo_features[best_feature] = 0
        self.high = TreeNode(parent=self, features=hi_features, recommendation=hi_recommendation)
        self.low = TreeNode(parent=self, features=lo_features, recommendation=low_recommendation)

    def filter_data(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        this method filters the data based on active  feature(0 or 1) of this node
        Initial value of self.features = None * size of features
        :param data:
        :return:
        """
        member_data = data
        for i_feature, feature in enumerate(self.features):
            if self.features[i_feature] is not None:
                member_data = member_data.loc[member_data.iloc[:, i_feature] == feature]
        return member_data


class DecisionTree:
    """
    a Decision tree for data with binary features where each feature has a value of 0 or 1.
    """

    def __init__(self, error_function, n_min=10, debug=False):
        """

        :param error_function: This is the function that will be used to judge the fitness of each leaf of the tree
        :param n_min: The minimum number of members a leaf node is permitted to have
        :param debug:
        """

        self.n_min = n_min
        self.debug = debug
        self.root = TreeNode()
        self.feature_names = None
        if error_function is None:
            raise ValueError("An error function must be supplied")
        else:
            self.err_fun = error_function

    @logged
    def train(self, training_features: pd.DataFrame) -> None:
        """
        Split nodes of the tree until  they can't be split any more.
        :param training_features:
        :return:
        """
        self.feature_names = training_features.columns
        nodes_to_check = [self.root]
        while len(nodes_to_check) > 0:
            current_node = nodes_to_check.pop()
            success = current_node.attempt_split(data=training_features, error_fun=self.err_fun, n_min=self.n_min)
            if success:
                nodes_to_check.append(current_node.low)
                nodes_to_check.append(current_node.high)

    def predict(self, features):
        """
        Make a prediction for each day  - features
        :param features:
        :return:
        """
        if len(features) != len(self.root.features):
            if self.debug:
                Logger.logger.debug(f"The feature you are asking to predict has a different"
                                    f" number of features than the tree.")
                return None

        current_node = self.root
        while True:
            if current_node.is_leaf:
                return current_node.recommendation
            if features[current_node.split_features] == 0:
                current_node = current_node.low
            elif features[current_node.split_features] == 1:
                current_node = current_node.high
            else:
                Logger.logger.debug(f"Feature: {current_node.split_feature} is not 0 or 1")
                return None

    def render(self, results_dir: str):
        """
        Create a plot that describes the tree
        :return:
        """
        plt.figure(34857)
        plt.clf()
        initial_level = 0
        initial_x = 0

        def plot_node(node, level, x):
            """
            Render the important information about a single node.
            :param node:
            :param level:
            :param x:
            :return:
            """
            recommendation = node.recommendation
            feature_name = self.feature_names[node.split_features]
            if node.is_leaf:
                node_text = f"at :{recommendation}\n"
            else:
                node_text = f"at: {recommendation}\n {feature_name}?\n" \
                            f"no    yes"
            plt.text(x, -level, node_text, horizontalalignment='center',
                     verticalalignment='center')

        def plot_branches(level: int, x0: float, y_delta: float = .2):
            """
            Draw the branches between the current node and it's children
            :param level:
            :param x0:
            :param y_delta: The amount of y-tail to be hidden
            :return:
            """
            y0 = - level
            y3 = -level - 1
            y1 = y0 - y_delta
            y2 = y3 + y_delta
            x3_lo = x0 - 2 ** y3
            x3_hi = x0 + 2 ** y3
            slope_lo = 1 / (x3_lo - x0)
            slope_hi = 1 / (x3_hi - x0)
            x_lo_delta = y_delta / slope_lo
            x_hi_delta = y_delta / slope_hi
            x1_lo = x0 + x_lo_delta
            x1_hi = x0 + x_hi_delta
            x2_lo = x3_lo - x_lo_delta
            x2_hi = x3_hi - x_hi_delta

            plt.plot([x1_lo, x2_lo], [y1, y2], color='black')
            plt.plot([x1_hi, x2_hi], [y1, y2], color='black')

            return x3_lo, x3_hi

        def recurse(node, level, x):
            plot_node(node, level, x)
            if node.is_leaf:
                return
            x_lo, x_hi = plot_branches(level, x)
            recurse(node.low, level + 1, x_lo)
            recurse(node.high, level + 1, x_hi)

        recurse(node=self.root, level=initial_level, x=initial_x)
        plt.title("Decision tree")
        plt.savefig(os.path.join(results_dir, "decision_tree.png"),
                    format='png')
        plt.show()


if __name__ == "__main__":
    data_loader = DataLoader()
    data_handler = DataHandler(trips=data_loader.data,
                               config=data_loader.config,
                               saves_dir=data_loader.saves_dir,
                               results_dir=data_loader.results_dir)
    train, test = data_handler.data_sets
    judge = Judge(features_df=train)
    training_features_df = data_handler.create_features(list(train.columns))
    tree = DecisionTree(error_function=judge.find_total_abs_deviation)
    tree.train(training_features=training_features_df)
    tree.render(results_dir=data_loader.results_dir)
    testing_score = data_handler.evaluate(tree=tree, data_set=test)
    print(f"testing score is: {testing_score}")
