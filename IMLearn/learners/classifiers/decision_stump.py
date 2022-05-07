from __future__ import annotations
from typing import Tuple, NoReturn
from ...base import BaseEstimator
import numpy as np
from itertools import product


class DecisionStump(BaseEstimator):
    """
    A decision stump classifier for {-1,1} labels according to the CART algorithm

    Attributes
    ----------
    self.threshold_ : float
        The threshold by which the data is split

    self.j_ : int
        The index of the feature by which to split the data

    self.sign_: int
        The label to predict for samples where the value of the j'th feature is about the threshold
    """

    def __init__(self) -> DecisionStump:
        """
        Instantiate a Decision stump classifier
        """
        super().__init__()
        self.threshold_, self.j_, self.sign_ = None, None, None

    def _fit(self, X: np.ndarray, y: np.ndarray) -> NoReturn:
        """
        fits a decision stump to the given data

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Input data to fit an estimator for

        y : ndarray of shape (n_samples, )
            Responses of input data to fit to
        """

        # Set fitted = True to allow us to import the loss function
        self.fitted_ = True
        self.sign_ = 1
        num_of_features = X.shape[1]

        # Calculate the minimum loss for each feature
        # The losses is a matrix who's rows are the features and where column1 is the
        # threshold with the smallest loss and column2 is the loss for this threshold
        losses = np.empty((num_of_features, 2))
        for f in range(num_of_features):
            losses[f, :] = self._find_threshold(X[:, f], y, self.sign_)

        # Select feature and threshold based on the loss matrix above
        self.j = np.argmin(losses[:, 1])
        self.threshold_ = losses[self.j][0]

    def _predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict responses for given samples using fitted estimator

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Input data to predict responses for

        Returns
        -------
        responses : ndarray of shape (n_samples, )
            Predicted responses of given samples

        Notes
        -----
        Feature values strictly below threshold are predicted as `-sign` whereas values which equal
        to or above the threshold are predicted as `sign`
        """
        return np.array([-self.sign_ if X[i][self.j] < self.threshold_ else self.sign_ for i in range(X.shape[0])])

    def _find_threshold(self, values: np.ndarray, labels: np.ndarray, sign: int) -> Tuple[float, float]:
        """
        Given a feature vector and labels, find a threshold by which to perform a split
        The threshold is found according to the value minimizing the misclassification
        error along this feature

        Parameters
        ----------
        values: ndarray of shape (n_samples,)
            A feature vector to find a splitting threshold for

        labels: ndarray of shape (n_samples,)
            The labels to compare against

        sign: int
            Predicted label assigned to values equal to or above threshold

        Returns
        -------
        thr: float
            Threshold by which to perform split

        thr_err: float between 0 and 1
            Misclassificaiton error of returned threshold

        Notes
        -----
        For every tested threshold, values strictly below threshold are predicted as `-sign` whereas values
        which equal to or above the threshold are predicted as `sign`
        """
        number_of_samples = labels.shape[0]
        sorted_index = np.argsort(values)
        sorted_values, sorted_labels = values[sorted_index], labels[sorted_index]
        losses = np.empty(number_of_samples)
        for i in range(number_of_samples):
            threshold_prediction = ([-sign] * i) + ([sign] * (number_of_samples - i))
            losses[i] = self.loss(np.array(threshold_prediction), sorted_labels)
        the_loss_argmin = np.argmin(losses)
        return sorted_values[the_loss_argmin], losses[the_loss_argmin]

    def _loss(self, X: np.ndarray, y: np.ndarray) -> float:
        """
        Evaluate performance under misclassification loss function

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Test samples

        y : ndarray of shape (n_samples, )
            True labels of test samples

        Returns
        -------
        loss : float
            Performance under missclassification loss function
        """
        misclassified_indexes = np.sign(X) - np.sign(y)
        return float(np.sum(np.abs(y[misclassified_indexes != 0])))
