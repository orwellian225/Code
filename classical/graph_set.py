from queue import LifoQueue
from typing import Callable
from copy import deepcopy

import numpy as np

from graph import Graph

def iterate_linear_Gn(n: int, work: Callable[[Graph], None]):
    for i in range(0, int(2**(n*(n-1)/2))):
        g = Graph.from_id(i, n)
        work(g)

def iterate_tree_Gn(n: int, work: Callable[[Graph], None]):
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