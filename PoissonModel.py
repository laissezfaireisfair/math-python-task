import random as rand
import numpy as np


class PoissonModel:
    def __init__(self, parameter: float):
        self.parameter: float = parameter

    def gen_x(self) -> int:
        return np.random.poisson(self.parameter)

    def run_and_get_success_points(self) -> np.ndarray:
        x = self.gen_x()
        successes = np.array([rand.random() for _ in range(x)])
        successes = np.sort(successes)
        return successes

    @staticmethod
    def _count_distances(successes: np.ndarray) -> [float]:
        return [successes[i + 1] - successes[i] for i in range(successes.size - 1)]

    def run_series_and_get_distances(self, count: int) -> np.ndarray:
        def get_distances():
            successes = self.run_and_get_success_points()
            return PoissonModel._count_distances(successes)

        distances = (get_distances() for _ in range(count))
        distances_flatten = [distance for xs in distances for distance in xs]

        return np.array(distances_flatten)
