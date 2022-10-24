from math import sqrt
import matplotlib.pyplot as plt

class HierarchicalClustering:
    def __init__(self, filename: str, metrics_function: str, distance_function: str, goal_number_of_clusters: int):
        self.data = []
        self.metrics_function = getattr(self, f"_{metrics_function}")
        self.distance_function = getattr(self, f"_{distance_function}")
        self.goal_number_of_clusters = goal_number_of_clusters
        self._load_initial_clusters(filename)
        self._cluster()
        self.visualize_2d()

    def _load_initial_clusters(self, filename: str) -> None:
        data = []
        with open(filename, "r") as reader:
            while line := reader.readline():
                coordinates = line.replace("\n", "").replace(",", ";").split(";")
                data.append([tuple(float(i) for i in coordinates)])
        self.data = data

    def _euclidean_distance(self, first_vector, second_vector):
        distance = 0
        for i, j in zip(first_vector, second_vector):
            distance += (i - j) ** 2
        return sqrt(distance)

    def _single_linkage(self, first_cluster, second_cluster):
        min_distance = float("inf")

        for i_data in first_cluster:
            for j_data in second_cluster:
                distance = self.distance_function(i_data, j_data)
                if distance < min_distance:
                    min_distance = distance

        return min_distance

    def _average_linkage(self, first_cluster, second_cluster):
        distance, counter = 0, 0

        for i_idx, i_data in enumerate(first_cluster):
            for j_data in second_cluster[i_idx:]:
                distance += self.distance_function(i_data, j_data)
                counter += 1

        return distance / counter

    def _find_clusters_to_merge(self):
        first, second = None, None
        min_distance = float("inf")

        for i_idx, i_data in enumerate(self.data):
            for j_idx, j_data in enumerate(self.data[i_idx+1:]):
                distance = self.metrics_function(i_data, j_data)

                if distance < min_distance:
                    min_distance = distance
                    first, second = i_idx, j_idx + i_idx + 1

        return first, second

    def _merge(self, first, second):
        self.data[first].extend(self.data[second])
        del self.data[second]

    def _cluster(self):
        while len(self.data) > self.goal_number_of_clusters:
            first, second = self._find_clusters_to_merge()
            self._merge(first, second)
            print(f"{len(self.data)} clusters left")

    def visualize_2d(self):
        def get_cmap(n, name='hsv'):
            return plt.cm.get_cmap(name, n + 1)

        cmap = get_cmap(len(self.data))

        for i, c in enumerate(self.data):
            # transpose the data
            transposed = list(map(list, zip(*c)))

            plt.scatter(transposed[0], transposed[1], c=cmap(i))

        plt.show()


if __name__ == "__main__":
    filename_ = "clusters3.csv"
    metrics_function_ = "single_linkage"
    distance_function_ = "euclidean_distance"
    number_of_clusters_ = 3

    clustering = HierarchicalClustering(filename_, metrics_function_, distance_function_, number_of_clusters_)
