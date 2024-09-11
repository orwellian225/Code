from queue import LifoQueue
from copy import deepcopy
from graph import Graph

import sys

def subgraph_search(G: Graph, H: Graph):
    """
        Decide if H is in G
            * return True if H in G
            * return False if H not in G
    """

    stack = LifoQueue()
    stack.put([])

    while not stack.empty():
        item = stack.get()

        if len(item) == H.properties.order:
            local_result = True
            for vert_a_H, vert_a_G in enumerate(item):
                for vert_b_H, vert_b_G in enumerate(item[vert_a_H + 1:]):
                    vert_b_H += vert_a_H + 1
                    if vert_a_G == vert_b_G or H[vert_a_H][vert_b_H] == -1:
                        continue

                    local_result = local_result and (G[vert_a_G][vert_b_G] == H[vert_a_H][vert_b_H])

            if local_result:
                return True
        else:
            for i in range((item[-1] + 1) if len(item) > 0 else 0, G.properties.order):
                new_item = deepcopy(item)
                new_item.append(i)
                stack.put(new_item)

    return False

def is_upper_ramsey_tree_search(n: int, F: Graph, H: Graph):
    stack = LifoQueue()

    for i in range(n):
        new_item = []
        for _ in range(i):
            new_item.append("1")
        for _ in range(n - 1 - i):
            new_item.append("0")

        stack.put(new_item)

    while not stack.empty():
        item = stack.get()
        # print(item)
        current_G = Graph.from_bitstring("".join(item), n)
        # print("Testing\n", current_G.matrix, "\n", F.matrix, "\n", H.matrix, f"\n H in G ({subgraph_search(current_G, H)}), F in G ({subgraph_search(current_G, F)})")

        if len(item) < int(n*(n-1)/2):
            item_left_child = deepcopy(item)
            item_left_child.append('0')

            item_right_child = deepcopy(item)
            item_right_child.append('1')

            right_G = Graph.from_bitstring("".join(item_right_child), n)
            if not (subgraph_search(right_G, F) or subgraph_search(right_G, H)):
                stack.put(item_right_child)

            left_G = Graph.from_bitstring("".join(item_left_child), n)
            if not (subgraph_search(left_G, F) or subgraph_search(left_G, H)):
                stack.put(item_left_child)
        else:
            graph_id = int("".join(item), base=2)
            G = Graph.from_id(graph_id, n)
            if not(subgraph_search(G, F) or subgraph_search(G, H)):
                return False

    return True

def main():
    if len(sys.argv) == 5:
        min_n = int(sys.argv[1])
        max_n = int(sys.argv[2])
        clique_order = int(sys.argv[3])
        iset_order = int(sys.argv[4])
    else:
        min_n = int(input("Start n: "))
        max_n = int(input("End n: "))
        clique_order = int(input("Clique Order: "))
        iset_order = int("Independent Set Order: ")

    clique = Graph.complete_n(clique_order)
    iset = Graph.empty_n(iset_order)

    print("n, k, l, result")
    for n in range(min_n, max_n + 1, 1):
        result = is_upper_ramsey_tree_search(n, clique, iset)

        print(f"{n}, {clique_order}, {iset_order}, {result}")
        with open("./data/cis_computation_result.csv", "a+") as f:
            f.write(f"{n},{clique_order},{iset_order},{result}\n")

if __name__ == "__main__":
    main()