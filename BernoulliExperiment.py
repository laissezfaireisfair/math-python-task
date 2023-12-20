import random as rand
import numpy as np
import enum


class BernoulliExperiment:
    def __init__(self, p_numerator: float, p_denominator: int):
        self.p_numerator: float = p_numerator
        self.p_denominator: int = p_denominator

    def get_success_probability(self) -> float:
        return min(self.p_numerator / self.p_denominator, 1.)

    def run_once(self) -> bool:
        rand_float = rand.random()
        return rand_float < self.get_success_probability()

    def run_segment(self) -> np.ndarray:
        sub_segments_count = self.p_denominator
        results = [self.run_once() for _ in range(sub_segments_count)]
        return np.array(results, dtype=np.bool_)

    def _count_distances_between_successes(self, segment_results: np.ndarray) -> [float]:
        class Status(enum.Enum):
            looking_for_first_success = 0
            looking_for_fail = 1
            enumerating_through_fail = 2

        sub_segment_length = 1. / self.p_denominator
        distances = []

        status = Status.looking_for_first_success
        current_fail_count = 0
        for i in range(segment_results.size):
            if status == Status.looking_for_first_success:
                if segment_results[i]:
                    status = Status.looking_for_fail
                continue

            if status == Status.looking_for_fail:
                if segment_results[i]:
                    distances.append(0)
                    continue
                status = Status.enumerating_through_fail
                current_fail_count = 1
                continue

            if status == Status.enumerating_through_fail:
                if not segment_results[i]:
                    current_fail_count += 1
                    continue
                distances.append(current_fail_count * sub_segment_length)
                status = Status.looking_for_fail

        return distances

    def run_series(self, length: int) -> (np.ndarray, np.ndarray):
        def get_successes_count_and_distances() -> ([int], [float]):
            segment_results = self.run_segment()

            step_successes_count = np.count_nonzero(segment_results)

            step_distances = self._count_distances_between_successes(segment_results)
            return step_successes_count, step_distances

        successes_and_distances = (get_successes_count_and_distances() for _ in range(length))

        successes_count, distances = zip(*successes_and_distances)
        distances_flatten = [distance for xs in distances for distance in xs]

        return np.array(successes_count), np.array(distances_flatten)
