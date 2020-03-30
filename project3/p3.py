from matplotlib.colors import LogNorm
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn import metrics
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
    k_values = [1, 2, 3, 4, 5, 10, 20, 30, 40, 50]

    totals = []

    # iterate through each k value
    for n_comp in range(data.shape[1]):
        # build the PCA model
        pca_mod = PCA(n_components=n_comp)
        pca_mod.fit(data)
        var_ratio = pca_mod.explained_variance_ratio_

        total = sum(var_ratio)
        totals.append(total)

        if n_comp in k_values:
            print(f'k={n_comp}\tFraction of Total Variance: {total:.5}')

    # plot the explained variance vs. the number of components
    plt.plot(totals)
    plt.title('Explained Variance vs. Number of Components')
    plt.xlabel('number of components')
    plt.ylabel('explained variance')
    plt.show()


def plot_poison_nopoison(data, labels):
    """
    Visualize a two dimensional plot of poisonous versus non-poisonous mushrooms
    """

    # build the model with k components
    pca_mod = PCA(n_components=2)
    pca_fit = pca_mod.fit_transform(data)

    # get nparrays for poisonous and nopoisonous mushrooms
    n = pca_fit[labels == 0]
    p = pca_fit[labels != 0]

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

    # get nparrays for poisonous and nonpoisonous mushrooms
    n = pca_fit[labels == 0]
    p = pca_fit[labels != 0]

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


def plot_density(axis, model, npn, pn):
    axis.scatter(npn[:, 0], npn[:, 1], 0.8, color='green')
    axis.scatter(pn[:, 0], pn[:, 1], 0.8, color='red')

    x = np.linspace(-7., 7., num=100)
    y = np.linspace(-13., 13., num=100)
    X, Y = np.meshgrid(x, y)
    XX = np.array([X.ravel(), Y.ravel()]).T
    Z = -model.score_samples(XX)
    Z = Z.reshape(X.shape)
    axis.contour(X, Y, Z, levels=20)


def plot_d(axis, model, npn, pn):
    """
    plot the density of the gaussian mixture model
    """
    axis.scatter(npn[:, 0], npn[:, 1], 0.8, color='green')
    axis.scatter(pn[:, 0], pn[:, 1], 0.8, color='red')

    x = np.linspace(-7., 7., num=100)
    y = np.linspace(-13., 13., num=100)
    X, Y = np.meshgrid(x, y)
    XX = np.array([X.ravel(), Y.ravel()]).T
    Z = -model.score_samples(XX)
    Z = Z.reshape(X.shape)
    axis.contour(X, Y, Z, levels=15)


def plot_gaussian_mixture_models(data, labels):
    """
    Fit Gaussian mixture models for the positive (poisonous) examples in 2d projected
    data. Vary the number of mixture components from 1 to 4 and the covariance matrix
    type 'spherical', 'diag', 'tied', 'full' (that's 16 models). Show square plots of
    the estimated density contours presented in a 4x4 grid - one row each for a number
    of mixture components and one column each for a convariance matrix type.
    """

    # build the model with two components
    pca_mod = PCA(n_components=2)
    pca_fit = pca_mod.fit_transform(data)

    # get nparrays for poisonous and nonpoisonous mushrooms
    n = pca_fit[labels == 0]
    p = pca_fit[labels != 0]

    # create a list of covariance types
    cov_types = ['spherical', 'diag', 'tied', 'full']
    fig, ax = plt.subplots(nrows=4, ncols=4, figsize=(65, 65))

    # display predicted scores by the model as a contour plot
    x = np.linspace(-3., 3)
    y = np.linspace(-3., 3)
    X, Y = np.meshgrid(x, y)
    XX = np.array([X.ravel(), Y.ravel()]).T
    for i, ct in enumerate(cov_types):
        for num_comps in range(1, 5):
            gm = GaussianMixture(n_components=num_comps,
                                 covariance_type=ct,
                                 random_state=12345)
            gm.fit(pca_fit)
            plot_d(ax[num_comps - 1][i], gm, n, p)


def gaussian_predictions(train, train_labels, test, test_labels):
    """
    Fit two Gaussian mixture models, one for the positive examples and one for the
    negative examples in your 2d projected data. Use 4 mixture components and full
    convariance for each model. Predict the test example labels by picking the labels
    corresponding to the larger of the two models' probabilities. Print the accuracy
    of predictions on the test data
    """

    # build the model with two components
    pca_mod = PCA(n_components=2)
    pca_train = pca_mod.fit_transform(train)

    # get nparrays for poisonous and nonpoisonous mushrooms
    n = pca_train[train_labels == 0]
    p = pca_train[train_labels != 0]

    # build model for poisonous mushrooms using 4 components
    gm_p = GaussianMixture(n_components=4,
                           covariance_type='full',
                           random_state=12345)
    gm_p.fit(p)

    # build model for non-poisonous mushrooms using 4 components
    gm_n = GaussianMixture(n_components=4,
                           covariance_type='full',
                           random_state=12345)
    gm_n.fit(n)

    # build model and score for test data
    pca_test = pca_mod.transform(test)
    p_scores = gm_p.score_samples(pca_test)
    n_scores = gm_n.score_samples(pca_test)

    # loop through labels and append best score to array
    predict = []
    for i in range(len(test_labels)):
        # for p_score, n_score in zip(p_scores, n_scores):
        if p_scores[i] > n_scores[i]:
            predict.append(1)
        else:
            predict.append(0)

    print('Prediction Accuracy:', metrics.accuracy_score(test_labels, predict))
