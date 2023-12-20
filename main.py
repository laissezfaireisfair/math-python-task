from BernoulliExperiment import BernoulliExperiment
from PoissonModel import PoissonModel
from ExponentialModel import ExponentialModel
import numpy as np
import matplotlib.pyplot as plt


def show_and_log_histogram(array: np.ndarray, name: str = 'unnamed') -> None:
    plt.hist(array, bins='sturges')
    plt.title(name)
    plt.show()

    text = '[' + ', '.join([str(e) for e in array]) + ']'
    print(f'Array {name}: {text} \n')


def run_bernoulli_experiments(numerator: int, denominator: int, series_length: int) -> None:
    experiment = BernoulliExperiment(numerator, denominator)

    successes, distances = experiment.run_series(series_length)

    show_and_log_histogram(successes, name='Bernoulli experiment successes count')
    show_and_log_histogram(distances, name='Bernoulli experiment success distances')


def run_poisson_model(parameter: float, series_length: int) -> None:
    model = PoissonModel(parameter)

    distances = model.run_series_and_get_distances(series_length)

    show_and_log_histogram(distances, name='Poisson model success distances')


def run_exponential_model(parameter: float, series_length: int) -> None:
    model = ExponentialModel(parameter)

    successes_count = model.run_series_and_get_successes_count(series_length)

    show_and_log_histogram(successes_count, name='Exponential model successes count')


def main():
    numerator = 50
    denominator = 100
    series_length = 1000

    run_bernoulli_experiments(numerator, denominator, series_length)
    run_poisson_model(numerator, series_length)
    run_exponential_model(numerator, series_length)


main()
