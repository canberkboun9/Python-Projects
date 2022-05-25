# This is a sample Python script.

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs, make_moons
import random
import seaborn as sns
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans


def calculate_objective_function(cluster_centers, X, y, K):  # Total within-class variation.
    k_ = 0  # cluster counter for the while loop
    counter = 0
    dist = 0  # j calculator
    # We know the mean is cluster center. I will use the fast method, the second one. In the slides, 9/17 page.
    while k_ < K:
        cluster_k_x, cluster_k_y = X[y == k_, 0], X[y == k_, 1]
        cluster_k_arr = np.array([cluster_k_x, cluster_k_y]).transpose()
        for i in range(len(cluster_k_x)):
            dist += (np.linalg.norm(cluster_k_arr[i] - cluster_centers[k_]))**2  # We take the sq. of the distance.
        # We sum all the squares of the distances to a cluster center. Also other clusters' too, in the dist variable.
        k_ += 1

    j = dist
    return j


def plot_ya(cluster_centers, X, y, K, step, is_final):
    title_ = "Step "+str(step)+" - "+str(K)+" Clusters"
    if is_final:
        title_ = "Final Step and Situation of Clusters"

    cluster_centers = np.array(cluster_centers)
    cluster_centers = cluster_centers.transpose()
    cmap_ = ["red", "blue", "green", "#9467bd", "#e377c2", "#bcbd22", "#17becf", "#008080"]
    # I put 8 colors as I will test at most 8 clusters in the main function.

    for k in range(K):
        label_ = "Cluster" + str(k)
        plt.scatter(X[y == k, 0], X[y == k, 1], color=cmap_[k], s=9, label=label_)

    plt.scatter(cluster_centers[0], cluster_centers[1], color="orange", marker="X", s=55, label="Clustercent")
    plt.xlabel("X-axis")
    plt.ylabel("Y-axis")
    plt.title(title_)
    plt.show()

    cluster_centers = cluster_centers.transpose()

def generate_clusters_blobs(centers, cluster_std):
    X, y = make_blobs(n_samples=300, cluster_std=cluster_std, centers=centers, n_features=2, random_state=931599)
    return X, y


def K_means(K, blob_or_elongated):
    # generate clusters which are observations
    if blob_or_elongated == 1:  # For the first Kmeans trial with circular clusters
        centers = [(2, 4), (5, 5), (3, 6)]
        cluster_std = [0.36, 0.35, 0.3]
        X, y = generate_clusters_blobs(centers, cluster_std)
    elif blob_or_elongated == 2:  # For the second Kmeans trial with elongated shape clusters
        X, y = make_moons(300, noise=.05, random_state=0)

    # X: The generated samples, y: The integer labels for cluster membership of each sample.
    # We will use X and implement our own KMeans algorithm to find y, or better membership.
    # As the data points may not be in the correct cluster according to our generated values, the y may change.

    # Choose k random points(k is a parameter, num of clusters)
    cluster_centers = []
    # I will choose 3 random observations.
    for k in range(K):
        cluster_centers.append(X[int(random.random() * 300)])

    cluster_centers = np.array(cluster_centers)
    cluster_centers = cluster_centers.transpose()

    plt.scatter(X[y == 0, 0], X[y == 0, 1], color="red", s=9, label="Cluster1")
    plt.scatter(X[y == 1, 0], X[y == 1, 1], color="blue", s=9, label="Cluster2")
    if blob_or_elongated == 1:
        plt.scatter(X[y == 2, 0], X[y == 2, 1], color="green", s=9, label="Cluster3")

    plt.scatter(cluster_centers[0], cluster_centers[1], color="orange", marker="X", s=55, label="Clustercent")
    plt.xlabel("X-axis")
    plt.ylabel("Y-axis")
    plt.title("Initial distribution of data points")
    plt.show()

    cluster_centers = cluster_centers.transpose()

    counter = 0  # a counter for plotting the iterations, and its changes.
    # we want to show first 3 iterations and the last one.
    variation_list_obj_func = []  # Holds the within class variation values for each iteration.
    alpha = 40000  # random alpha initialized.
    while alpha >= 0.0005:
        counter += 1  # counts the steps of the KMeans algo
        is_final = False

        # assignment to clusters, minimum euclidean distance to the centroids.
        for i in range(300):
            dist = 0  # arbitrary initialization
            min_dist = 100000  # arbitrary initialization
            cluster_label = K  # arbitrary initialization

            for k in range(K):  # calculate distance K times, find the correct cluster for i^th observation.
                dist = np.linalg.norm(X[i] - cluster_centers[k])
                # used euclidean distance algorithm, that is built-in, from numpy.
                if dist < min_dist:
                    min_dist = dist
                    cluster_label = k  # We label this point as the K^th cluster's member.

            # Assignment Part, first assignment into random cluster centroids.
            y[i] = cluster_label

        # Cluster Center Update Step. Now calculate the new centroids.

        # Calculate the alpha: Step-1.
        # Copy the old centers into a variable.
        old_cluster_centers = cluster_centers.copy()
        change_in_distances_of_centers = 0

        for k in range(K):  # there are K clusters
            cluster_k_x = X[y == k, 0]
            cluster_k_y = X[y == k, 1]
            cluster_k_arr = np.array([cluster_k_x, cluster_k_y]).transpose()
            cluster_centroid_k = cluster_k_arr.mean(axis=0)
            cluster_centers[k] = cluster_centroid_k  # New centroid is found.

        # Calculate the variation. We may have a graph of variance per iteration like in the slides.
        # So put them in a list. Plot a graph at the end.
        variation_i = calculate_objective_function(cluster_centers, X, y, K)
        variation_list_obj_func.append(variation_i)  # Calculated the objective function value for i^th iteration/step.

        # Calculate the alpha: Step-2.
        for v in range(K):
            change_dist = np.linalg.norm(old_cluster_centers[v] - cluster_centers[v])
            # print(change_dist)
            change_in_distances_of_centers += change_dist

        print(change_in_distances_of_centers)
        alpha = change_in_distances_of_centers

        if counter < 4 or alpha < 0.0005:  # I want to plot the 1st, 2nd, and 3rd iteration, and the last one.
            if alpha < 0.0005:
                is_final = True
                plt.plot(variation_list_obj_func, marker="*")
                plt.title("Objective function value for the steps")
                plt.xlabel("Steps")
                plt.ylabel("Objective Function Value")
                plt.show()

            plot_ya(cluster_centers, X, y, K, counter, is_final)

    # out of the while loop. K means iterations ended.

    return variation_i, cluster_centers, counter;


if __name__ == '__main__':
    list_of_variations = []
    # ATTENTION: Blob_or_not is important!
    Blob_or_not = 2  # It can be 1 or 2, it will create circle or half moon shaped datasets.
    K_means(4-Blob_or_not, Blob_or_not)  # 3 for circles, 2 for half moon shapes.
    # Kmeans do not work properly with non-convex clusters.
    K_means(5,Blob_or_not)
    for i in range(2, 9):
        variation_i, cluster_centers_found_i, number_of_steps_i = K_means(i,Blob_or_not)
        list_of_variations.append(variation_i)  # I collect final within class variations to use them in
        # Elbow Method, to find the optimal cluster number
        print("Cluster centers found for " + str(i) + " clusters: ", cluster_centers_found_i)
        print("Found in " + str(number_of_steps_i) + " steps.")
        print("Final objective function value: ", variation_i)

    plt.plot([2,3,4,5,6,7,8], list_of_variations, marker="*")
    plt.title("Objective function value for each number of clusters")
    plt.xlabel("Number of Clusters (K)")
    plt.ylabel("Objective Function Value")
    plt.show()

    # I will use scikitlearn's KMeans with 3 clusters, as it is the intended value.

    if Blob_or_not == 1:
        centers = [(2, 4), (5, 5), (3, 6)]
        cluster_std = [0.36, 0.35, 0.3]
        X, y = generate_clusters_blobs(centers, cluster_std)
    elif Blob_or_not == 2:
        X, y = make_moons(300, noise=.05, random_state=0)
    else:
        print("please initialize the Blob_or_not variable with 1 or 2")

    X = X.transpose()
    X_x = X[0].transpose()
    X_y = X[1].transpose()
    X_X_X = np.array([X_x, X_y]).transpose()

    data_ = dict()
    clustering = KMeans(n_clusters=4 - Blob_or_not)  # 3 for circles, 2 for half moon shapes.
    clusters = clustering.fit_predict(X_X_X)
    cluster_centers_from_built_in = clustering.cluster_centers_.transpose()
    data_['Cluster'] = clusters
    data_['X'] = X_x
    data_['y'] = X_y
    sns.scatterplot('X', 'y', data=data_, hue='Cluster')
    plt.scatter(cluster_centers_from_built_in[0], cluster_centers_from_built_in[1], color="orange", marker="X", s=55, label="Clustercent")
    plt.title("Built-in KMeans Algorithm results for 3 clusters")
    plt.show()


