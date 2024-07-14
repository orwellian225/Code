from manim import Scene, Graph, Text, VGroup, Create, RIGHT, LEFT, UP, DOWN, GREEN, WHITE, BLACK
import math as m
from copy import deepcopy

def construct_all_subgraphs(order: int, sub_order: int):
    subgraph_vertices = []
    for i in range(order - sub_order + 1):
        build(i, sub_order, order, [ i ], subgraph_vertices)

    return subgraph_vertices

def build(i: int, d: int, w: int, current: list, result: list):
    # print(i, d, w, current, result)
    if len(current) == d:
        result.append(current)
        return 

    for j in range(i + 1, w):
        c2 = deepcopy(current)
        c2.append(j)
        build(j, d, w, c2, result)

class SubgraphSearch(Scene):
    order = 8
    sub_order = 4

    def construct(self):
        G_vertices = []        
        G_edges = []

        for i in range(self.order):
            G_vertices.append(i)

        for i in range(0, self.order - 1):
            for j in range(i + 1, self.order):
                G_edges.append((i, j))

        graph_G = Graph(G_vertices, G_edges, layout="circular", labels=True)
        label_G = Text(f"Graph of order {self.order}")
        group_G = VGroup(
            label_G, graph_G
        ).arrange(direction=DOWN)
        self.play(Create(group_G))
        self.play(group_G.animate.shift(4 * LEFT))

        label_H = Text(f"Subgraphs of order {self.sub_order}")
        label_H.shift(3 * RIGHT + 3.5 * UP)
        self.play(Create(label_H))

        dots_G = graph_G.vertices
        edges_G = graph_G.edges

        subgraph_vertices = construct_all_subgraphs(self.order, self.sub_order)

        for i in range(m.comb(self.order, self.sub_order)):
            sub_vertices = []
            sub_edges = []

            for j in range(self.sub_order):
                dots_G[subgraph_vertices[i][j]][0].set_color(GREEN)
                dots_G[subgraph_vertices[i][j]][1].set_color(WHITE)
                sub_vertices.append(subgraph_vertices[i][j])

            for j in range(self.sub_order - 1):
                for k in range(j + 1, self.sub_order):
                    sub_edges.append((subgraph_vertices[i][j], subgraph_vertices[i][k]))
                    edges_G[(subgraph_vertices[i][j], subgraph_vertices[i][k])].set_color(GREEN)

            new_subgraph = Graph(
                sub_vertices,
                sub_edges,
                labels=True,
                layout="circular"
            ).scale(0.15)
            new_subgraph.shift(0.8 * LEFT + 2 * UP + i % 10 * 0.8 * RIGHT + i // 10 * 0.8 * DOWN)
            self.play(Create(new_subgraph, lag_ratio=0))

            for j in range(self.sub_order):
                dots_G[subgraph_vertices[i][j]][0].set_color(WHITE)
                dots_G[subgraph_vertices[i][j]][1].set_color(BLACK)

            for j in range(self.sub_order - 1):
                for k in range(j + 1, self.sub_order):
                    edges_G[(subgraph_vertices[i][j], subgraph_vertices[i][k])].set_color(WHITE)

        self.wait(duration = 5)