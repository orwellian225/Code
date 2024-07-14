import numpy as np
import numpy.typing as npt
import matplotlib.pyplot as plt
import math as m
from copy import deepcopy

class Graph:
    adj_matrix: npt.NDArray[np.uint8]
    num_vertices: int

    def bitstring_to_adjacency_matrix(bitstring: str) -> npt.NDArray[np.uint8]:
        """
            UTILITY: Convert a bitstring into an adjacency matrix
        """
        num_vertices = (1 + m.sqrt( 1 + 8*len(bitstring) )) / 2

        if num_vertices != m.floor(num_vertices):
            raise ValueError(f"Not a valid bitstring n = {num_vertices}")

        num_vertices = int(num_vertices)
        result = np.zeros((num_vertices, num_vertices), dtype=np.uint8)

        row_indent = 0
        row_step = 0
        for i, char in enumerate(bitstring):
            row = m.floor((i + row_indent) / (num_vertices - 1))
            col = row_indent + row_step + 1
            edge = int(char)

            row_step += 1
            if row_step == num_vertices - row_indent - 1:
                row_indent += 1
                row_step = 0

            result[row][col] = edge
            result[col][row] = edge

        return result

    def __init__(self, adjcency_matrix: npt.NDArray[np.uint8]):
        self.adj_matrix = adjcency_matrix
        self.num_vertices = len(self.adj_matrix)

        np.fill_diagonal(self.adj_matrix, 0)

    def __eq__(self, value: object) -> bool:
        return (self.adj_matrix == value.adj_matrix).all()
        
    def __str__(self):
        return self.to_bitstring()

    def __getitem__(self, key):
        return self.adj_matrix[key]

    def has_subgraph(self, subgraph):
        suborder = subgraph.num_vertices

        def build(i: int, d: int, w: int, current: list, result: list):
            if len(current) == d:
                result.append(current)
                return 

            for j in range(i + 1, w):
                c2 = deepcopy(current)
                c2.append(j)
                build(j, d, w, c2, result)

        subgraph_vertices = []
        for i in range(self.num_vertices - suborder + 1):
            build(i, suborder, self.num_vertices, [ i ], subgraph_vertices)            

        for h in subgraph_vertices:
            result = True
            for i in range(len(h) - 1):
                for j in range(i, len(h)):
                    # print(f"Edge pairs: ({h[i]}, {h[j]}) -> ({i}, {j})")
                    # All edges must match
                    result = result and self[h[i]][h[j]] == subgraph[i][j]

            # if all edges match, then return true
            if result:
                return True

        # if none of the subgraphs are found
        return False



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