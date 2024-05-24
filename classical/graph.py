import numpy as np
import numpy.typing as npt
import matplotlib.pyplot as plt
import math as m

class Graph:
    adj_matrix: npt.NDArray[np.uint8]
    num_vertices: int

    def __init__(self, adjcency_matrix: npt.NDArray[np.uint8]):
        self.adj_matrix = adjcency_matrix
        self.num_vertices = len(self.adj_matrix)

    def __eq__(self, value: object) -> bool:
        return self.adj_matrix == value.adj_matrix

    def to_bitstring(self) -> str:
        mask = np.triu(np.ones(self.adj_matrix.shape), k=1)
        elements = self.adj_matrix[mask == 1]
        result = "".join(map(str, elements))
        return result

    def plot_graph(self) -> None:
        # plt.scatter(np.arange(0, 5), np.arange(5, 10))

        radius = 5
        positions_x = list(map(lambda x: radius * m.cos(x * 2 * m.pi / self.num_vertices), np.arange(0, self.num_vertices)))
        positions_y = list(map(lambda x: radius * m.sin(x * 2 * m.pi / self.num_vertices), np.arange(0, self.num_vertices)))

        for i in range(self.num_vertices):
            for j in range(self.num_vertices):
                if self.adj_matrix[i, j] == 1:
                    plt.plot([positions_x[i], positions_y[i]], [positions_x[j], positions_y[j]], color="b")

        plt.scatter(positions_x, positions_y, c='r')

        plt.show()