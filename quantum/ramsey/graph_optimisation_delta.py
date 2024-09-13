"""
    Graphing the Trotterization Delta created for testing an AQO problem
    
    Plotting Eigenvalues (Energy) against delta 
"""

import matplotlib.pyplot as plt
import ramsey_util as ru
import seaborn as sns
import polars as pl
import numpy as np
import math as m

from adiabatic_optimisation import AdiabaticOptimisationProblem
import qiskit as qk
from qiskit_aer import AerSimulator, StatevectorSimulator
from qiskit_aer.primitives import EstimatorV2

k = 2
l = 4
max_n = 5 + 1
initial_hams = [ru.generate_initial_hamiltonian(n) for n in range(2, max_n)]
problem_hams = [ru.generate_ramsey_hamiltonian(n, k, l) for n in range(2, max_n)]
min_energies = [np.min(np.linalg.eigvals(h)).astype(np.float32) for h in problem_hams]
max_time_space = [.5, 1., 2., 3., 4., 5.]
steps_space = [1, 2, 5, 10, *[i for i in range(10, 101, 5)]]

data = {
    "max_t": [],
    "steps": [],
    "delta": [],
    "graph_order": [],
    "estimated_energy": [],
}

sim = StatevectorSimulator(device="GPU")
estimator = EstimatorV2()

for n in range(2, max_n):
    for T in max_time_space:
        for steps in steps_space:
            print(f"\rSimulation R({k},{l}) ?= {n} for T = {T} & delta = {T / steps}", end="")

            aqo = AdiabaticOptimisationProblem(
                initial_hamiltonian=initial_hams[n - 2],
                problem_hamiltonian=problem_hams[n - 2],
                time=T,
                steps=steps
            )
            aqo.generate([1 / m.sqrt(2**aqo.num_qubits) for _ in range(2**aqo.num_qubits)])
            aqo.circuit = aqo.circuit.decompose()

            observables = [aqo.problem_hamiltonian]
            pub = (qk.transpile(aqo.circuit, sim), observables)
            job = estimator.run([pub])

            data["max_t"].append(T)
            data["steps"].append(steps)
            data["delta"].append(T / steps)
            data["graph_order"].append(n)
            data["estimated_energy"].append(job.result()[0].data.evs[0])
print("")

df = pl.from_dict(data)
fig, axes = plt.subplots(ncols=len(max_time_space), figsize=(len(max_time_space) * 5, 5))
fig.suptitle(f"Evaluating AQO performance at various $T$ and $M$ for R({k},{l})")

for i, t in enumerate(max_time_space):
    _ = sns.lineplot(data=df.filter(pl.col("max_t") == t), hue="graph_order", x="delta", y="estimated_energy", palette="Set2", ax=axes[i], sort=True)
    for n in range(2, max_n):
        axes[i].axhline(y=min_energies[n - 2], color=sns.color_palette("Set2")[n - 2], linestyle="--")
    _ = axes[i].set_title(f'Estimated Energy vs $\Delta t$ at $T={t}$')

plt.savefig(f"./visualizations/hyperparam_tuning_R({k}_{l}).pdf")
