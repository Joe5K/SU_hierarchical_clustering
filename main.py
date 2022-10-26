import matplotlib.pyplot as plt
from copy import deepcopy
from functools import lru_cache


class HierarchicalClustering:
    MAX_CACHE_SIZE = None

    def __init__(self, filename: str, metrics_function: str, distance_function: str):
        self.data = []
        self.history = {}
        self.metrics_function = getattr(self, f"_{metrics_function}_linkage")
        self.distance_function = getattr(self, f"_{distance_function}_distance")
        self._load_initial_clusters(filename)
        self._cluster()
        self.plot()

    def _load_initial_clusters(self, filename: str) -> None:
        data = []
        with open(filename, "r") as reader:
            while line := reader.readline():
                coordinates = line.replace("\n", "").replace(",", ";").split(";")
                data.append([tuple(float(i) for i in coordinates)])
        self.data = data

    @staticmethod
    @lru_cache(maxsize=MAX_CACHE_SIZE)
    def _euclidean_distance(first_vector: tuple[int], second_vector: tuple[int]) -> float:
        distance = 0
        for i, j in zip(first_vector, second_vector):
            distance += (i - j) ** 2
        return distance  # no need for sqrt, we just want the highest/lowest value

    @staticmethod
    @lru_cache(maxsize=MAX_CACHE_SIZE)
    def _manhattan_distance(first_vector: tuple[int], second_vector: tuple[int]) -> float:
        distance = 0
        for i, j in zip(first_vector, second_vector):
            distance += abs(i - j)
        return distance

    @lru_cache(maxsize=MAX_CACHE_SIZE)
    def _single_linkage(self, first_cluster: tuple[int], second_cluster: tuple[int]) -> float:
        min_distance = float("inf")

        for i_data in first_cluster:
            for j_data in second_cluster:
                distance = self.distance_function(i_data, j_data)
                if distance < min_distance:
                    min_distance = distance

        return min_distance

    @lru_cache(maxsize=MAX_CACHE_SIZE)
    def _average_linkage(self, first_cluster: tuple[int], second_cluster: tuple[int]) -> float:
        distance, counter = 0, 0

        for i_data in first_cluster:
            for j_data in second_cluster:
                distance += self.distance_function(i_data, j_data)
                counter += 1

        return distance / counter

    @lru_cache(maxsize=MAX_CACHE_SIZE)
    def _complete_linkage(self, first_cluster: tuple[int], second_cluster: tuple[int]) -> float:
        max_distance = 0

        for i_data in first_cluster:
            for j_data in second_cluster:
                distance = self.distance_function(i_data, j_data)
                if distance > max_distance:
                    max_distance = distance

        return max_distance

    def _find_clusters_to_merge(self) -> tuple[int, 2]:
        first, second = None, None
        min_distance = float("inf")

        for i_idx, i_data in enumerate(self.data):
            for j_idx, j_data in enumerate(self.data[i_idx+1:]):
                distance = self.metrics_function(tuple(i_data), tuple(j_data))

                if distance < min_distance:
                    min_distance = distance
                    first, second = i_idx, j_idx + i_idx + 1

        return first, second

    def _cluster(self) -> None:
        while len(self.data) > 1:
            first, second = self._find_clusters_to_merge()
            self.data[first].extend(self.data.pop(second))
            self.history[len(self.data)] = deepcopy(self.data)

            print(f"{len(self.data)} clusters left")

        self.distance_function.cache_clear()
        self.metrics_function.cache_clear()

    def plot(self) -> None:
        while True:
            number = input("Which number of clusters do you want to plot?\n")
            if number.isnumeric() and (data := self.history.get(int(number))):
                for cluster in data:
                    cluster = [*zip(*cluster)]
                    plt.scatter(cluster[0], cluster[1])
                plt.title(f"{filename_.split('.')[0]}_{number}_{distance_function_}_{metrics_function_}")
                plt.show()
            else:
                print("Wrong input")


if __name__ == "__main__":
    filename_ = "clusters3.csv"
    metrics_function_ = "complete"
    distance_function_ = "euclidean"

    clustering = HierarchicalClustering(filename_, metrics_function_, distance_function_)
