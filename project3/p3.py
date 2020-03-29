from matplotlib.colors import LogNorm
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.mixture import GaussianMixture
import matplotlib.pyplot as plt
import numpy as np


def pca_analysis(data):
    """
    Perform principal components analysis on supplied data for
    each of the first k principal components. Function will print
    the fraction of the total variance in the training data that is
    explained my the first k components as well as lineplots of
    fraction of total variance vs. number of principal components.
    """

    # define various k values for number of principal components
    k = [1, 2, 3, 4, 5, 10, 20, 30, 40, 50]

    # set values for plots
    fig, axes = plt.subplots(figsize=(10, 20))
    fig.tight_layout()

    cell = 1
    # iterate through each k value
    for n_comp in k:
        # build the PCA model
        pca_mod = PCA(n_components=n_comp)
        pca_mod.fit(data)
        var_ratio = pca_mod.explained_variance_ratio_
        print(f'k={n_comp}\n{var_ratio}')

        # plot the explained variance vs. the number of components
        plt.subplot(5, 2, cell, title=f'k={n_comp}')
        plt.plot(var_ratio)
        cell += 1


def poison_nopoison(model, data, labels):
    """
    Get ndarrays of poisonous versus non-poisonous mushrooms using PCA model
    """

    # create poisonous and non-poisonous nparrays
    nopoison = model[labels == 0]
    poison = model[labels != 0]

    return poison, nopoison


def plot_poison_nopoison(data, labels):
    """
    Visualize a two dimensional plot of poisonous versus non-poisonous mushrooms
    """

    # build the model with k components
    pca_mod = PCA(n_components=2)
    pca_fit = pca_mod.fit_transform(data)

    # get nparrays for poisonous and nopoisonous mushrooms
    p, n = poison_nopoison(pca_fit, data, labels)

    # draw the plot
    plt.figure(figsize=(15, 10))
    plt.plot(p[:, 0], p[:, 1], 'ro', lw=1, mec='k')
    plt.plot(n[:, 0], n[:, 1], 'go', lw=1, mec='k')
    plt.title('2D plot of poisonous vs. non-poisonous mushrooms')
    plt.text(-1.5, 2, 'green: non-poisonous\nred: poisonous')


def plot_kmeans_cluster(data, labels):
    """
    Visualize a two dimensional plot of poisonous versus non-poisonous mushrooms
    using KMeans to cluster results into 6 clusters.
    """

    clusters = 6
    plt.figure(figsize=(15, 10))

    # build the model with two components
    pca_mod = PCA(n_components=2)
    pca_fit = pca_mod.fit_transform(data)

    # generate KMeans model with 6 clusters and predict
    km = KMeans(n_clusters=clusters).fit(pca_fit)
    x_transform = km.transform(pca_fit)
    y_pred = km.predict(pca_fit)
    centroids = km.cluster_centers_

    # set create an array of distances to be used for plotting the cluster circle
    distances = np.zeros((clusters, x_transform.shape[0]))
    for num in range(x_transform.shape[0]):
        prediction = y_pred[num]
        distances[prediction][num] = x_transform[num][prediction]

    # plot the cluster circle for each cluster
    for num in range(clusters):
        circle = plt.Circle((centroids[num, 0], centroids[num, 1]),
                            max(distances[num]),
                            alpha=0.2)
        plt.gcf().gca().add_artist(circle)

    # get nparrays for poisonous and nopoisonous mushrooms
    p, n = poison_nopoison(pca_fit, data, labels)

    # draw plot
    plt.plot(p[:, 0], p[:, 1], 'ro', lw=1, mec='k')
    plt.plot(n[:, 0], n[:, 1], 'go', lw=1, mec='k')
    plt.scatter(centroids[:, 0],
                centroids[:, 1],
                marker='x',
                s=500,
                linewidth=5,
                color='y',
                zorder=10)
    plt.show()


def P4(data, labels):
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(data)
    pos_data = []
    for x in range(len(labels)):
        if labels[x] == 1:
            pos_data.append(X_pca[x])
    pos_data = np.asarray(pos_data)
    cov_mat_types = ['spherical', 'diag', 'tied', 'full']
    plt.figure(figsize=(16, 16))

    # display predicted scores by the model as a contour plot
    x = np.linspace(-3., 3)
    y = np.linspace(-3., 3)
    X, Y = np.meshgrid(x, y)
    XX = np.array([X.ravel(), Y.ravel()]).T
    for n in range(len(cov_mat_types)):
        cmt = cov_mat_types[n]
        for cn in range(1, 5):
            clf = GMM(n_components=cn, covariance_type=cmt)
            clf.fit(pos_data)
            Z = -(clf.score_samples(XX)[0])
            Z = Z.reshape(X.shape)
            plt.subplot(4, 4, n * 4 + cn)
            CS = plt.contour(X,
                             Y,
                             Z,
                             norm=LogNorm(),
                             levels=np.logspace(0, 2.5, 10))
            CB = plt.colorbar(CS, shrink=0.8, extend='both')
            plt.scatter(pos_data[:, 0], pos_data[:, 1], c='cyan')
            plt.title('Type:' + cmt + '  N:' + str(cn))
