import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs, make_moons
import random
import math
import seaborn as sns
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn import tree
import graphviz


def entropy(rows):  # labels=y in the main.
    """Calculates entropy of the given nodes."""
    labels = []
    for row in rows:
        labels.append(row[2])
    # labels = rows[2]  # labels are in the 2th index of the row.

    counts = {}  # {label: count}
    for sample_label in labels:
        if sample_label not in counts:
            counts[sample_label] = 0
        counts[sample_label] += 1

    entropy_ = 0
    for lbl in counts:
        prob_of_lbl = counts[lbl] / len(labels)
        if prob_of_lbl == 0:
            entropy_ -= 0
        else:
            entropy_ -= prob_of_lbl*math.log(prob_of_lbl,2)

    return entropy_


def calculate_gini(rows):  # labels=y in the main.
    """Calculates gini of the given nodes."""

    labels = rows[2]  # labels are in the 2th index of the row.

    counts = {}  # {label: count}
    for sample_label in labels:
        if sample_label not in counts:
            counts[sample_label] = 0
        counts[sample_label] += 1

    impurity = 1
    for lbl in counts:
        prob_of_lbl = counts[lbl] / len(labels)
        impurity -= prob_of_lbl**2
    return impurity


def info_gain(left, right, node_uncertainty):  # left, right = rows of left children and right children
    """Information Gain = (impurity of the node) - (weighted impurity of the children nodes.)"""
    p = len(left) / (len(left) + len(right))  # ratio of left child to whole data. 1-p is for the right part.
    return node_uncertainty - p * entropy(left) - (1 - p) * entropy(right)


def find_boundary_values(X):

    list_of_boundary_values = [[], []]  # for x and y, two lists
    max_value_x = max(X.transpose()[0])
    min_value_x = min(X.transpose()[0])
    max_value_y = max(X.transpose()[1])
    min_value_y = min(X.transpose()[1])

    range_x = max_value_x - min_value_x
    range_y = max_value_y - min_value_y
    # increment_value = min(range_x, range_y)/20
    i = min_value_x
    for _ in range(int(range_x*100)):
        i += 0.05
        list_of_boundary_values[0].append(i)

    i = min_value_y
    for _ in range(int(range_y*100)):
        i += 0.05
        list_of_boundary_values[1].append(i)

    return list_of_boundary_values


def partition(rows, boundary_value, axis):
    """Partitions a dataset. Less and more than a value."""
    less_than, more_than = [], []
    if axis == 0:  # X axis
        for i in range(len(rows)):
            if float(rows[i][0]) < boundary_value:
                less_than.append(rows[i])
            else:
                more_than.append(rows[i])

    else:  # Y axis
        for i in range(len(rows)):
            if float(rows[i][1]) < boundary_value:
                less_than.append(rows[i])
            else:
                more_than.append(rows[i])

    return less_than, more_than


def find_best_split(rows, X, y):
    """Find the best split. It is a univariate decision tree. We iterate for x and y, find the best split.
    Maximum information gain implies the most beneficial split in terms of impurity decrease."""
    max_info_gain = 0  # keep track of the best information gain
    best_split = 0  # coordinate of the decision boundary.
    split_axis = 0  # 0 => x, 1 => y. If 0, x=best_split is the decision boundary returned; x is the axis.
    current_uncertainty = entropy(rows)

    list_of_boundary_values = find_boundary_values(X)
    boun_val_x = list_of_boundary_values[0]
    boun_val_y = list_of_boundary_values[1]

    for i in range(len(boun_val_x)):
        less_than, more_than = partition(rows, boun_val_x[i], 0)
        gain_i = info_gain(less_than, more_than, current_uncertainty)
        if gain_i > max_info_gain:
            max_info_gain, split_axis, best_split = gain_i, 0, boun_val_x[i]

    for i in range(len(boun_val_y)):
        less_than, more_than = partition(rows, boun_val_y[i], 1)
        gain_i = info_gain(less_than, more_than, current_uncertainty)
        if gain_i > max_info_gain:
            max_info_gain, split_axis, best_split = gain_i, 1, boun_val_y[i]

    return max_info_gain, split_axis, best_split


decision_nodes = {}
def build_tree(rows, level):
    # my tree is a dictionary. decision_nodes = {id: [level, axis, boundary_value,
    #                                               [entropy_less, entropy_more, entropy_weighted]]}
    # program should return the decision_nodes dictionary in a report format for each id key, i.e. decision boundary).
    # partition the data set first with the best split.
    max_gain, split_axis, split_value = find_best_split(rows, X, y)  # the best split
    less_than, more_than = partition(rows, split_value, split_axis)  # did split the data.
    entropy_less = entropy(less_than)
    entropy_more = entropy(more_than)
    p = len(less_than) / (len(less_than) + len(more_than))
    entropy_weighted = p * entropy_less + (1 - p) * entropy_more

    node_id = len(decision_nodes)
    decision_nodes[node_id] = [level, split_axis, split_value, [entropy_less, entropy_more, entropy_weighted]]
    # look whether the partition derives a pure child node, if it derives a pure child node,
    # do not split that sub dataset.
    less_than_nomore = False
    more_than_nomore = False

    if entropy_less== 0:
        less_than_nomore = True

    if entropy_more == 0:
        more_than_nomore = True

    # if level = 1 calculate and go to level 2, else if level = 2, calculate and stop.
    # because we have max_depth = 2

    if level == 1:
        if less_than_nomore and not more_than_nomore:
            build_tree(more_than, 2)
        elif not less_than_nomore and more_than_nomore:
            build_tree(less_than, 2)
        else:
            build_tree(less_than, 2)
            build_tree(more_than, 2)
    elif level == 2:  # in this case the max_depth = 2.
        return 0

random.seed(22)

def generate_dataset():
    dataset = []
    for i in range(100):
        x = str(random.uniform(0, 6))
        y = str(random.uniform(0, 4))
        label = "1"
        data_point = [x, y, label]
        dataset.append(data_point)

    for i in range(50):
        x = str(random.uniform(0, 2))
        y = str(random.uniform(4, 8))
        label = "2"
        data_point = [x, y, label]
        dataset.append(data_point)

    for i in range(100):
        x = str(random.uniform(2, 8))
        y = str(random.uniform(4, 8))
        label = "1"
        data_point = [x, y, label]
        dataset.append(data_point)

    for i in range(50):
        x = str(random.uniform(6, 8))
        y = str(random.uniform(0, 4))
        label = "2"
        data_point = [x, y, label]
        dataset.append(data_point)

    return dataset


def print_decision_boundary(decision_dict):
    print("Printing the report:\n")
    for i in range(len(decision_dict)):
        id = i
        level = decision_dict[i][0]
        if decision_dict[i][1] == 0:
            axis = "X"
        else:
            axis = "Y"
        boundary_value = decision_dict[i][2]
        left = decision_dict[i][3][0]
        right = decision_dict[i][3][1]
        weight = decision_dict[i][3][2]
        print("id:",id," level:",level, " axis:", axis, "boundary value", boundary_value, " left, right, weighted entropys:", [left, right, weight])

    return 0


if __name__ == '__main__':
    with open("data.txt", 'r') as f:
        data = f.read()

    data = data.split("\n")
    X = []
    y = []
    rows = []
    for row in data:
        row = row.split(",")
        X.append([float(row[0]), float(row[1])])
        y.append(int(row[2]))
        rows.append(row)

    X = np.array(X)
    y = np.array(y)
    # print(X, y)
    # print(X.transpose())
    # print(rows)

    plt.scatter(X[y == 1, 0], X[y == 1, 1], color="blue", s=9, label="Cluster1")
    plt.scatter(X[y == 2, 0], X[y == 2, 1], color="orange", s=9, label="Cluster2")
    plt.xlabel("X-axis")
    plt.ylabel("Y-axis")
    plt.title("Initial distribution of data points generated by the professor")
    plt.show()

    build_tree(rows, 1)
    # my tree is a dictionary. decision_nodes = {id: [level, axis, boundary_value,
    #                                               [entropy_less, entropy_more, entropy_weighted]]}
    # program should return the decision_nodes dictionary in a report format for each id key, i.e. decision boundary).
    # print("manual for reading the decision boundaries through the decision_nodes dictionary:")
    # print("decision_nodes = {id: [level, axis, boundary_value, [entropy_less, entropy_more, entropy_weighted]]} \n")
    print("level=1, axis=1 so, y = 4.00 is the root node")
    print("level=2, axis=0 so, x = 3.00 is the level-2 node")
    print_decision_boundary(decision_nodes)

    clf = tree.DecisionTreeClassifier(criterion='entropy', max_depth=2)
    clf = clf.fit(X, y)
    result = clf.predict([[4., 2.]])
    proba_Result = clf.predict_proba([[4., 2.]])
    tree.plot_tree(clf)

    plt.show()

    # print("answer:", result, proba_Result)
    decision_nodes = {}

    dataset = generate_dataset()

    X = []
    y = []
    rows = []
    for row in dataset:
        X.append([float(row[0]), float(row[1])])
        y.append(int(row[2]))
        rows.append(row)

    X = np.array(X)
    y = np.array(y)
    # print(X, y)
    # print(X.transpose())
    # print(rows)

    plt.scatter(X[y == 1, 0], X[y == 1, 1], color="blue", s=9, label="Cluster3")
    plt.scatter(X[y == 2, 0], X[y == 2, 1], color="orange", s=9, label="Cluster4")
    plt.xlabel("X-axis")
    plt.ylabel("Y-axis")
    plt.title("Initial distribution of data points generated manually")
    plt.show()

    build_tree(dataset,1)
    print_decision_boundary(decision_nodes)

    clf_2 = tree.DecisionTreeClassifier(criterion='entropy', max_depth=2)
    clf_2 = clf_2.fit(X, y)
    result = clf_2.predict([[4., 2.]])
    proba_Result_2 = clf_2.predict_proba([[4., 2.]])
    tree.plot_tree(clf_2)

    plt.show()

    # my tree is a dictionary. decision_nodes = {id: [level, axis, boundary_value,
    #                                               [entropy_less, entropy_more, entropy_weighted]]}
    # program should return the decision_nodes dictionary in a report format for each id key, i.e. decision boundary).



