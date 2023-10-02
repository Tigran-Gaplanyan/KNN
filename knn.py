from collections import Counter

import numpy as np
import matplotlib.pyplot as plt

from scipy.spatial.distance import cdist
from sklearn.metrics import accuracy_score
from sklearn.model_selection import KFold

def generate_data(num_samples, num_features):
    """
    Generates a synthetic dataset with a large number of features and a small number of samples using
    a Gaussian distribution.

    Parameters
    ----------
    num_samples : int
        The number of samples to generate.
    num_features : int
        The number of features for each sample.

    Returns
    -------
    X : ndarray of shape (num_samples, num_features)
        The input features for each sample.
    y : ndarray of shape (num_samples,)
        The output labels for each sample.
    """
    # Generate input features with a Gaussian distribution.
    X = np.random.randn(num_samples, num_features)

    # Generate random output labels.
    y = np.random.randint(0, 2, size=num_samples)

    return X, y


class KNNClassifier:
    """
    K-nearest neighbors classifier.

    Parameters
    ----------
    n_neighbors : int, optional (default=5)
        Number of neighbors to use for classification.
    metric : str or callable, optional (default='euclidean')
        Distance metric to use for computing the distances between samples.
        Supported metrics include 'euclidean', 'manhattan', 'chebyshev', and
        any other metric supported by the `scipy.spatial.distance.cdist` function.
    """

    def __init__(self, n_neighbors=5, metric='euclidean'):
        self.n_neighbors = n_neighbors
        self.metric = metric

    def fit(self, X, y):
        """
        Fit the KNN classifier to the training data.

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            The training input samples.
        y : ndarray of shape (n_samples,)
            The target values.
        """
        self.X_train = X
        self.y_train = y


    def predict(self, X):
        """
        Predict the class labels for the input samples.

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            The input samples.

        Returns
        -------
        y_pred : ndarray of shape (n_samples,)
            The predicted class labels.
        """
        dist = cdist(X, self.X_train)
        indices = np.argsort(dist, axis=1)
        k_nearest_indices = indices[:, :self.n_neighbors]
        k_nearest_labels = self.y_train[k_nearest_indices]
        predicted_labels = np.array([Counter(labels).most_common(1)[0][0] for labels in k_nearest_labels])
        return predicted_labels


def evaluate_knn_performance(X, y, estimator, k_values, n_folds=5, metric='euclidean'):
    """Evaluates the performance of the KNN classifier using k-fold cross-validation.

    Parameters:
    -----------
    X : array-like of shape (n_samples, n_features)
        The input samples.
    y : array-like of shape (n_samples,)
        The target values.
    estimator :
        Your or sklearn's model
    k_values : array-like
        The values of k to use for the KNN classifier.
    n_folds : int, optional (default=5)
        The number of folds to use for cross-validation.
    metric : str, optional (default='euclidean')
        The distance metric to use for the KNN classifier.

    Returns:
    --------
    accuracies : dict
        A dictionary containing the accuracy of the KNN classifier for each value of k on validation set.
        You can interpret this as average value.

    NOTE: you can return other values if you need to.
    """
    kf = KFold(n_splits=n_folds, shuffle=True)
    mean_accuracies = []
    for k in k_values:
        accuracies = []
        for train_indices, test_indices in kf.split(X):
            knn = estimator(n_neighbors=k, metric=metric)
            knn.fit(X[train_indices], y[train_indices])
            y_pred = knn.predict(X[test_indices])
            accuracy = np.mean(y_pred == y[test_indices])
            accuracies.append(accuracy)
        mean_accuracy=np.mean(accuracies)
        mean_accuracies.append(mean_accuracy)
    return mean_accuracies



def plot_learning_curve(X, y, k, n_folds = 5,metric='euclidean', *args, **kwargs):
    """
    Plot the learning curve for a KNN classifier using different values of k.

    NOTE: You can modify arguments as you like, except `k`.

    Parameters:
    ----------
    k : int or array like

    Returns:
    -------
    None
    """
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    train_scores = np.zeros((len(k), kf.get_n_splits()))
    test_scores = np.zeros((len(k), kf.get_n_splits()))

    for k_idx, k_value in enumerate(k):
        for fold_idx, (train_index, test_index) in enumerate(kf.split(X)):
            knn = KNNClassifier(n_neighbors=k_value, metric=metric)
            knn.fit(X[train_index], y[train_index])
            train_pred = knn.predict(X[train_index])
            train_accuracy = accuracy_score(y[train_index], train_pred)
            test_pred = knn.predict(X[test_index])
            test_accuracy = accuracy_score(y[test_index], test_pred)
            train_scores[k_idx, fold_idx] = train_accuracy
            test_scores[k_idx, fold_idx] = test_accuracy

    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)

    plt.plot(k, train_scores_mean, label='Training score', color='blue', marker='o')
    plt.fill_between(k, train_scores_mean - train_scores_std, train_scores_mean + train_scores_std, alpha=0.1,
                     color='blue')
    plt.plot(k, test_scores_mean, label='Cross-validation score', color='red', marker='o')
    plt.fill_between(k, test_scores_mean - test_scores_std, test_scores_mean + test_scores_std, alpha=0.1,
                     color='red')
    plt.legend(loc='best')
    plt.xlabel('k')
    plt.ylabel('Accuracy')
    plt.title('KNN learning curve')
    plt.show()


if __name__ == "__main__":
    X, y = generate_data(100, 1000)
    knn = KNNClassifier()
    knn.fit(X, y)


