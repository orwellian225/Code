import numpy as np
import matplotlib.pyplot as plt
import math as m

import numpy.typing as npt
from typing import Self

class Properties:
    id: int # The enumerating ID of the graph
    order: int # The number of vertices
    size: int # The number of edges
    encoding_length: int # The length of the encoding string

    def __init__(self, id: int, order: int, size: int):
        self.id = id
        self.order = order
        self.size = size
        self.encoding_length = int(order * (order - 1) / 2)

class Graph:
    matrix: npt.NDArray[np.int8]
    properties: Properties

    def __init__(self, matrix: npt.NDArray[np.int8]) -> Self: 
        n = len(matrix)

        # Diagonal must be zeros
        for i in range(n):
            assert(matrix[i,i] == 0)

        # Must be symmetrical
        for i in range(n):
            for j in range(n):
                assert(matrix[i,j] == matrix[j,i])

        self.matrix = matrix
        self.properties = Properties(int(self.to_bitstring(), 2), len(self.matrix), np.sum(self.matrix) / 2)

    def __getitem__(self, key: int) -> npt.NDArray[np.int8]:
        return self.matrix[key]

    def __eq__(self, other) -> bool:
        try:
            return (self.matrix == other.matrix).all()
        except AttributeError:
            return False

    def __str__(self) -> str:
        return self.to_bitstring()

    def to_bitstring(self) -> str:
        """
            * Handles wildcard edges as if they are empty edges
        """
        mask = np.triu(np.ones(self.matrix.shape), k=1)
        elements = self.matrix[mask == 1]
        elements[elements == -1] = 0 # reset wildcard edges to 0
        result = "".join(map(str, elements))[::-1]
        return result

    def get_edge_sequence(self) -> list:
        return np.sum(self.matrix, axis=1).tolist()

    def plot_graph(self, axes: plt.axes, radius=5, vertex_labels=False):
        axes.tick_params(
            left=False, right=False, bottom=False, top=False,
            labelleft=False, labelright=False, labelbottom=False, labeltop=False
        )
        axes.margins(0.1)

        positions_x = list(map(lambda x: radius * m.cos(x * 2 * m.pi / self.properties.order), np.arange(0, self.properties.order)))
        positions_y = list(map(lambda x: radius * m.sin(x * 2 * m.pi / self.properties.order), np.arange(0, self.properties.order)))

        if vertex_labels:
            for i in range(self.properties.order):
                axes.text(positions_x[i] + 0.3, positions_y[i] + 0.3, str(i + 1))

        for i in range(self.properties.order):
            for j in range(self.properties.order):
                if self.matrix[i,j] == 1:
                    axes.plot(
                        (positions_x[i], positions_x[j]),
                        (positions_y[i], positions_y[j]),
                        color='black'
                    )
                elif self.matrix[i,j] == -1:
                    axes.plot(
                        (positions_x[i], positions_x[j]),
                        (positions_y[i], positions_y[j]),
                        color='grey',
                        alpha=0.4
                    )

        axes.scatter(positions_x, positions_y, c='black', s=2)

    def complete_n(n: int) -> Self:
        new_matrix = np.ones((n, n), dtype=np.int8)
        for i in range(n):
            new_matrix[i,i] = 0

        return Graph(new_matrix)

    def empty_n(n: int) -> Self:
        return Graph(np.zeros((n, n), dtype=np.int8))

    def cycle_n(n: int) -> Self:
        new_matrix = np.zeros((n, n), dtype=np.int8)
        for i in range(n - 1):
            new_matrix[i][i + 1] = 1
            new_matrix[i + 1][i] = 1

        new_matrix[0][n - 1] = 1
        new_matrix[n - 1][0] = 1
        return Graph(new_matrix)

    def subgraph_cycle_n(n: int) -> Self:
        """
            Create a cycle graph, but use the value of all other edges as a wildcard to indicate 
            that the cycle exists as a subgraph
        """
        new_matrix = np.ones((n, n), dtype=np.int8) * -1
        for i in range(n - 1):
            new_matrix[i][i + 1] = 1
            new_matrix[i + 1][i] = 1

            # set diagonal to zero
            new_matrix[i][i] = 0
        new_matrix[n - 1][n - 1] = 0

        new_matrix[0][n - 1] = 1
        new_matrix[n - 1][0] = 1
        return Graph(new_matrix)

    def path_n(n: int) -> Self:
        new_matrix = np.zeros((n, n), dtype=np.int8)
        for i in range(n - 1):
            new_matrix[i][i + 1] = 1
            new_matrix[i + 1][i] = 1

        return Graph(new_matrix)

    def subgraph_path_n(n: int) -> Self:
        """
            Create a path graph, but use the value of all other edges as a wildcard to indicate
            that the path exists as a subgraph
        """
        new_matrix = np.ones((n, n), dtype=np.int8) * -1
        for i in range(n - 1):
            new_matrix[i][i + 1] = 1
            new_matrix[i + 1][i] = 1

            # set diagonal to zero
            new_matrix[i][i] = 0
        new_matrix[n - 1][n - 1] = 0

        return Graph(new_matrix)

    def from_id(id: int, n: int) -> Self:
        """
        Return the graph of order n identified by the integer id as a bitstring
        """
        matrix = np.zeros((n, n), dtype=np.int8)

        bitstring = bin(id)[2::].zfill(int(n*(n-1)/2))[::-1]
        if len(bitstring) > n*(n-1)/2:
            raise ValueError(f"Specified ID {id} is larger than max id of graph order {n} | Max ID = 2^(n(n-1)/2) = {int(2**(n*(n-1)/2))}")

        row_indent = 0
        row_step = 0
        for i, b in enumerate(bitstring):
            row = int(m.floor((i + row_indent) / (n - 1)))
            col = row_indent + row_step + 1
            # print(row, col, b)

            row_step += 1
            if row_step == n - row_indent - 1:
                row_indent += 1
                row_step = 0

            matrix[row][col] = int(b)
            matrix[col][row] = int(b)

        return Graph(matrix)
