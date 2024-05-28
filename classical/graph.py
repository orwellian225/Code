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

        np.fill_diagonal(self.adj_matrix, 0)

    def __eq__(self, value: object) -> bool:
        return self.adj_matrix == value.adj_matrix
        
    def __str__(self):
        return self.to_bitstring()

    def get_edge_sequence(self) -> np.ndarray:
        return np.sum(self.adj_matrix, axis=1)

    def get_edge_count(self) -> int:
        return np.sum(self.adj_matrix) / 2

    def get_connectivity_sequence(self) -> np.ndarray:
        sequence = np.zeros(self.num_vertices)

        for i in range(self.num_vertices):
            for j in range(i + 1, self.num_vertices):
                binary_idx = self.num_vertices - j - 1 
                sequence[i] += self.adj_matrix[i,j] * 2**binary_idx

        return sequence

    def permute(self, vertex_1, vertex_2) -> None:
        self.adj_matrix[[vertex_1, vertex_2]] = self.adj_matrix[[vertex_2, vertex_1]]
        self.adj_matrix = np.transpose(self.adj_matrix)
        self.adj_matrix[[vertex_1, vertex_2]] = self.adj_matrix[[vertex_2, vertex_1]]
        self.adj_matrix = np.transpose(self.adj_matrix)

    def copy_permute(self, vertex_1, vertex_2):
        copy = Graph(np.copy(self.adj_matrix))
        copy.permute(vertex_1, vertex_2)
        return copy


    def to_bitstring(self) -> str:
        mask = np.triu(np.ones(self.adj_matrix.shape), k=1)
        elements = self.adj_matrix[mask == 1]
        result = "".join(map(str, elements))
        return result

    def plot_graph(self, vertex_labels: None | bool) -> None:
        plt.figure(figsize=(2,2))
        plt.axis('off')

        radius = 5
        positions_x = list(map(lambda x: radius * m.cos(x * 2 * m.pi / self.num_vertices), np.arange(0, self.num_vertices)))
        positions_y = list(map(lambda x: radius * m.sin(x * 2 * m.pi / self.num_vertices), np.arange(0, self.num_vertices)))

        if vertex_labels is not None and vertex_labels:
            for i in range(self.num_vertices):
                plt.text(positions_x[i], positions_y[i], str(i))

        for i in range(self.num_vertices):
            for j in range(self.num_vertices):
                if self.adj_matrix[i,j] == 1:
                    plt.plot(
                        (positions_x[i], positions_x[j]),
                        (positions_y[i], positions_y[j]),
                        color='blue'
                    )

        plt.scatter(positions_x, positions_y, c='r')
        plt.show()