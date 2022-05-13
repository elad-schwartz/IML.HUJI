import numpy as np
from typing import Tuple

import plotly.graph_objects

from IMLearn.metalearners.adaboost import AdaBoost
from IMLearn.learners.classifiers import DecisionStump
from utils import *
import plotly.graph_objects as go
from plotly.subplots import make_subplots


def generate_data(n: int, noise_ratio: float) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generate a dataset in R^2 of specified size

    Parameters
    ----------
    n: int
        Number of samples to generate

    noise_ratio: float
        Ratio of labels to invert

    Returns
    -------
    X: np.ndarray of shape (n_samples,2)
        Design matrix of samples

    y: np.ndarray of shape (n_samples,)
        Labels of samples
    """
    '''
    generate samples X with shape: (num_samples, 2) and labels y with shape (num_samples).
    num_samples: the number of samples to generate
    noise_ratio: invert the label for this ratio of the samples
    '''
    X, y = np.random.rand(n, 2) * 2 - 1, np.ones(n)
    y[np.sum(X ** 2, axis=1) < 0.5 ** 2] = -1
    y[np.random.choice(n, int(noise_ratio * n))] *= -1
    return X, y


def return_title(text: str):
    '''
    Returns title with global config with given text
    '''
    return {
        'text': text,
        'x': 0.5,
        'y': 0.99,
        'font': {'size': 35}
    }


def get_marker_shapes_list(test_y: np.ndarray):
    '''
    Helper function for styling markers in scatter plots
    '''
    return ['circle' if test_y[i] == -1 else 'x' for i in range(test_y.shape[0])]


def Q1_plot(training_losses: [int], testing_losses: [int]):
    '''
    Plots the classification loss as a function on adaboost ensemble size
    '''
    fig = go.Figure()

    # Adding training and testing losses
    fig.add_trace(go.Scatter(x=[i for i in range(1, 251)], y=training_losses, name="train errors"))
    fig.add_trace(go.Scatter(x=[i for i in range(1, 251)], y=testing_losses, name="test errors"))

    fig.update_layout(
        title=return_title("Classification loss as a function of ensemble size"),
        xaxis_title="Adaboost iterations",
        yaxis_title="Classification Error for iteration"
    )
    fig.show()


def Q2_plot(adaboost_classifier: AdaBoost, test_X: np.ndarray, test_y: np.ndarray, lims: [float]):
    """
    Creates subplots of the dataset and decision boundaries for a list of adaboost ensembles of increasing size
    """
    T = [5, 50, 100, 250]
    traces = [(1, 1), (1, 2), (2, 1), (2, 2)] # Used to place the subplots

    fig = make_subplots(rows=2, cols=2, subplot_titles=[f"Decision boundary for {t}" for t in T])

    for i in range(4):
        fig.add_trace(
            go.Scatter(
                x=test_X[:, 0],
                y=test_X[:, 1],
                mode='markers',
                marker=dict(
                    size=15,
                    color=test_y,
                    symbol=get_marker_shapes_list(test_y),
                    line_width=2,
                    line_color='black'
                )
            ), col=traces[i][0], row=traces[i][1])

        partial_predict = lambda test_data: adaboost_classifier.partial_predict(test_data, T[i])

        fig.add_trace(decision_surface(partial_predict, lims[0], lims[1], showscale=False),
                      col=traces[i][0], row=traces[i][1])

    fig.update_layout(title=return_title('Decision Boundaries of Adaboost ensembles'),showlegend=False)
    fig.show()


def Q3_plot(adaboost_classifier: AdaBoost, test_X: np.ndarray, testing_losses: [float], test_y: np.ndarray,
            lims: [int]):
    '''
    PLots the dataset and decision boundary for the best performing ensemble
    '''
    best_performing_ensemble = np.argmin(testing_losses)
    accuracy = 1 - testing_losses[best_performing_ensemble]
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=test_X[:, 0],
        y=test_X[:, 1],
        mode='markers',
        marker=dict(
            size=15,
            color=test_y,
            symbol=get_marker_shapes_list(test_y),
            line_width=2,
            line_color='black'
        )
    ))
    partial_predict = lambda test_data: adaboost_classifier.partial_predict(test_data, 238)
    fig.add_trace(decision_surface(partial_predict, lims[0], lims[1], showscale=False))
    fig.update_layout(
        title=return_title(f'Decision boundary ensemble of size 238 '
                           f'with accuracy {accuracy}')
    )
    fig.show()


def Q4_plot(adaboost_classifier: AdaBoost, train_X: np.ndarray, train_y: np.ndarray, lims: [int]):
    '''
    Plot of weighted training samples by their respective weights in the last iteration of the adaboost
    algorithm
    '''
    max_D_t = np.max(adaboost_classifier.D)

    # I have chosen to multiply by 20 to increase graph visibility whilst still maintaining the marker
    # sizes proportional to their weight in the final iteration
    normalized_list = (adaboost_classifier.D / max_D_t) * 20

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=train_X[:, 0],
            y=train_X[:, 1],
            mode='markers',
            marker=dict(
                size=normalized_list,
                color=train_y,
                symbol=get_marker_shapes_list(train_y),
                line_width=5,
                line_color='black'
            )
        )
    )
    fig.add_trace(decision_surface(adaboost_classifier.predict, lims[0], lims[1], showscale=False))
    fig.update_layout(
        title=return_title('Plot with weighted training samples')
    )
    fig.show()


def fit_and_evaluate_adaboost(noise, n_learners=250, train_size=5000, test_size=500):
    (train_X, train_y), (test_X, test_y) = generate_data(train_size, noise), generate_data(test_size, noise)

    adaboost_classifier = AdaBoost(DecisionStump, n_learners)
    adaboost_classifier.fit(train_X, train_y)

    training_losses = []
    # Training
    for i in range(adaboost_classifier.iterations_):
        training_losses.append(adaboost_classifier.partial_loss(train_X, train_y, i))

    # Testing
    testing_losses = []
    for i in range(adaboost_classifier.iterations_):
        testing_losses.append(adaboost_classifier.partial_loss(test_X, test_y, i))

    lims = np.array([np.r_[train_X, test_X].min(axis=0), np.r_[train_X, test_X].max(axis=0)]).T + np.array(
        [-.1, .1])

    # Question 1: Train- and test errors of AdaBoost in noiseless case
    Q1_plot(training_losses, testing_losses)

    # Question 2: Plotting decision surfaces
    if noise == 0:
        Q2_plot(adaboost_classifier, test_X, test_y, lims)

    # Question 3: Decision surface of best performing ensemble
    if noise == 0:
        Q3_plot(adaboost_classifier, test_X, testing_losses, test_y, lims)

    # Question 4: Decision surface with weighted samples
    Q4_plot(adaboost_classifier, train_X, train_y, lims)


if __name__ == '__main__':
    np.random.seed(0)
    fit_and_evaluate_adaboost(0)
    fit_and_evaluate_adaboost(0.4)
