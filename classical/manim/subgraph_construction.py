import manim
from copy import deepcopy

def build(i: int, d: int, w: int, current: list, result: list):
    # print(i, d, w, current, result)
    if len(current) == d:
        result.append(current)
        return 

    for j in range(i + 1, w):
        c2 = deepcopy(current)
        c2.append(j)
        build(j, d, w, c2, result)

class SubgraphConstruction(manim.Scene):

    def construct(self):
        order = 5 # Tree width
        suborder = 3 # Tree depth

        result = []
        for i in range(order - suborder + 1):
            build(i, suborder, order, [ i ], result)

        self.play(
            manim.Create(manim.Dot(
                [0., 4., 0.],
                0.03
            ))
        )

        depth_1 = []
        for i in range(order):
            print([float(i - 2), 0., 0.])
            depth_1.append(manim.Create(manim.Dot(
                [float(i - 2) * 0.5, 3., 0.],
                0.03
            )))
        self.play(*depth_1)
        
        self.wait(duration=5)