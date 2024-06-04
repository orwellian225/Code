import numpy as np
import numpy.typing as npt
import matplotlib.pyplot as plt
import math as m

class Graph:
    adj_matrix: npt.NDArray[np.uint8]
    num_vertices: int

    def bitstring_to_adjacency_matrix(bitstring: str) -> npt.NDArray[np.uint8]:
        """
            UTILITY: Convert a bitstring into an adjacency matrix
            
            matrix_row = floor((i + 1) / num_vertices)
            matrix_col = i % (num_vertices - 1) + 1 + matrix_row
        """
        num_vertices = (1 + m.sqrt( 1 + 8*len(bitstring) )) / 2

        if num_vertices != m.floor(num_vertices):
            raise ValueError(f"Not a valid bitstring n = {num_vertices}")

        num_vertices = int(num_vertices)
        result = np.zeros((num_vertices, num_vertices), dtype=np.uint8)

        for i, c in enumerate(bitstring):
            row = m.floor((i + 1) / num_vertices)
            col = i % (num_vertices - 1) + 1 + row 
            edge = int(c)

            print(i, row, col, c)

            result[row, col] = edge
            result[col, row] = edge

        return result

    def __init__(self, adjcency_matrix: npt.NDArray[np.uint8]):
        self.adj_matrix = adjcency_matrix
        self.num_vertices = len(self.adj_matrix)

        np.fill_diagonal(self.adj_matrix, 0)

    def __eq__(self, value: object) -> bool:
        return (self.adj_matrix == value.adj_matrix).all()
        
    def __str__(self):
        return self.to_bitstring()

    def get_edge_sequence(self) -> np.ndarray:
        """
            The degree of each vertex
        """
        return np.sum(self.adj_matrix, axis=1)

    def get_edge_count(self) -> int:
        """
            The total number of edges in the graph
        """
        return np.sum(self.adj_matrix) / 2

    def get_connectivity_sequence(self) -> np.ndarray:
        """
            The upper triangular matrix converted into a binary string, and then into a decimal number
            eg:

                * 0 1 0 1 => 2**2 * 1 + 2**1 * 0 + 2**0 * 1 = 5
                * 1 0 1 0 => 2**1 * 1 + 2**0 * 0 = 2
                * 0 1 0 0 => 2**0 * 0 = 0
                * 1 0 0 0 => 0

                The above adjacency matrix (and relevant graph) has a connectivity sequence of 5 2 0
        """
        sequence = np.zeros(self.num_vertices)

        for i in range(self.num_vertices):
            for j in range(i + 1, self.num_vertices):
                binary_idx = self.num_vertices - j - 1 
                sequence[i] += self.adj_matrix[i,j] * 2**binary_idx

        return sequence

    def permute(self, vertex_1: int, vertex_2: int) -> None:
        """
            Perform an in-place permutation of a graph's vertices§
        """
        self.adj_matrix[[vertex_1, vertex_2]] = self.adj_matrix[[vertex_2, vertex_1]]
        self.adj_matrix = np.transpose(self.adj_matrix)
        self.adj_matrix[[vertex_1, vertex_2]] = self.adj_matrix[[vertex_2, vertex_1]]
        self.adj_matrix = np.transpose(self.adj_matrix)

    def copy_permute(self, vertex_1: int, vertex_2: int):
        """
            Make a copy of the graph and then apply a permutation to the copy
        """
        copy = Graph(np.copy(self.adj_matrix))
        copy.permute(vertex_1, vertex_2)
        return copy

    def to_bitstring(self) -> str:
        """
            Convert the upper triangle adjacency matrix into a bitstring
        """
        mask = np.triu(np.ones(self.adj_matrix.shape), k=1)
        elements = self.adj_matrix[mask == 1]
        result = "".join(map(str, elements))
        return result

    def to_bitint(self) -> int:
        """
            Convert the bitstring into an integer
        """
        bitstring = self.to_bitstring()
        return int(bitstring, 2)

    def plot_graph(self, vertex_labels: bool = False) -> None:
        plt.figure(figsize=(2,2))
        plt.axis('off')

        radius = 5
        positions_x = list(map(lambda x: radius * m.cos(x * 2 * m.pi / self.num_vertices), np.arange(0, self.num_vertices)))
        positions_y = list(map(lambda x: radius * m.sin(x * 2 * m.pi / self.num_vertices), np.arange(0, self.num_vertices)))

        if vertex_labels:
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