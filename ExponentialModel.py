import numpy as np
from itertools import repeat


class ExponentialModel:
    def __init__(self, parameter: float):
        self.parameter: float = 1 / parameter

    def run_series_and_get_successes_count(self, series_length: int) -> np.ndarray:
        def get_successes_count() -> int:
            def exponential_distribution():
                while True:
                    yield np.random.exponential(self.parameter)

            gen_sum = 0
            for count, number in enumerate(exponential_distribution()):
                gen_sum += number
                count += 1
                if gen_sum > 1:
                    return count

        successes_count = [get_successes_count() for _ in range(series_length)]

        return np.array(successes_count)
