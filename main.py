from BernoulliExperiment import BernoulliExperiment
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

    show_and_log_histogram(successes, name='Bernoulli experiment successes')
    show_and_log_histogram(distances, name='Bernoulli experiment success distances')


def main():
    numerator = 50
    denominator = 100
    series_length = 1000

    run_bernoulli_experiments(numerator, denominator, series_length)


main()
