from queue import LifoQueue
from typing import Callable
from copy import deepcopy

import math as m

from graph import Graph

def enumerate_linear_G(n: int, work: Callable[[Graph], None]):
    for i in range(0, int(2**(n*(n-1)/2))):
        g = Graph.from_id(i, n)
        work(g)

def enumerate_linear_H(n: int, work: Callable[[Graph], None]):
    """
        Optimised Linear G(n) construction by manually removing some automorphic graphs from the list
            * All graphs with a single edge (there are $nC2$ of them) are removed, and only one is inserted
            * All graphs will only missing a single edge (there are $nC2$) are removed, and only one is inserted
    """
    l = int(n*(n-1)/2)

    # Doing this seperately to avoid running log2 on i = 0
    zero_graph = Graph.from_id(0, n)
    work(zero_graph)

    # Adding the single edge graph manually, then just avoiding adding any graph with a single edge into the list further down
    one_edge_graph = Graph.from_id(1, n)
    work(one_edge_graph)

    # Adding the missing one edge graph manually
    missing_one_edge_graph = Graph.from_id(2**l - 2, n)
    work(missing_one_edge_graph)

    for i in range(1, int(2**l)):
        # If the graph will have a single edge i.e. binary string has a single 1
        # Or if the graph will have a single missing edge i.e. binary string has a single 0
        if i.bit_count() != 1 and i.bit_count() != l - 1:
            g = Graph.from_id(i, n)
            work(g)

def enumerate_tree_G(n: int, work: Callable[[Graph], None]):
    stack = LifoQueue()
    stack.put([])

    while not stack.empty():
        item = stack.get()

        if len(item) < int(n*(n-1)/2):
            item_left_child = deepcopy(item)
            item_left_child.append('0')

            item_right_child = deepcopy(item)
            item_right_child.append('1')

            stack.put(item_right_child)
            stack.put(item_left_child)

        else:
            graph_id = int("".join(item), base=2)
            g = Graph.from_id(graph_id, n)
            work(g)