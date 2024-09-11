import qiskit as qk
import qiskit.circuit as qkc
import qiskit.quantum_info as qki
from qiskit_aer import AerSimulator
from qiskit_aer.primitives import EstimatorV2
import numpy as np
import math as m

class AdiabaticOptimisationProblem:
    def __init__(
            self,
            initial_hamiltonian: qki.operators.Operator,
            problem_hamiltonian: qki.operators.Operator,
            time: float,
            steps: int
        ):

        if initial_hamiltonian.num_qubits != problem_hamiltonian.num_qubits:
            raise ValueError(f"Initial Hamiltonian and Problem Hamiltonian have incorrect dimensions, {initial_hamiltonian.num_qubits} != {problem_hamiltonian.num_qubits}")

        self.initial_hamiltonian = initial_hamiltonian
        self.problem_hamiltonian = problem_hamiltonian
        self.num_qubits = self.initial_hamiltonian.num_qubits
        self.qubit_list = [x for x in range(self.num_qubits)]

        self.time = time
        self.steps = steps

        self.delta = self.time / self.steps
        self.circuit = qk.QuantumCircuit(self.num_qubits)

    def generate(self, initial_state):
        self.circuit.initialize(initial_state)
        for step in range(self.steps):
            interpolated_op = (step * self.delta / self.time) * self.initial_hamiltonian + (1 - step * self.delta / self.time) * self.problem_hamiltonian
            local = qkc.library.PauliEvolutionGate(interpolated_op, time=self.delta, label=f"U({(step + 1) * self.delta},{step * self.delta})")
            self.circuit.append(local, qargs=self.qubit_list)

# num_qubits = 1
# temp = AdiabaticOptimisationProblem(
#     qki.operators.SparsePauliOp(["I", "X"], np.array([0.5, -0.5])),
#     qki.operators.SparsePauliOp(["I", "Z"], np.array([0.5, -0.5])),
#     5.,
#     3
# )

# print(temp.initial_hamiltonian.to_matrix())
# print(temp.problem_hamiltonian.to_matrix())

# equal_prob = 1 / m.sqrt(2**temp.num_qubits)
# temp.generate([equal_prob for _ in range(2**temp.num_qubits)])
# temp.circuit = temp.circuit.decompose()
# print(temp.circuit.draw())
# print("Operations:", temp.circuit.count_ops())
# print("Depth:", temp.circuit.depth())

# simulator = AerSimulator()
# estimator = EstimatorV2()
# observables = [temp.problem_hamiltonian]
# pub = (qk.transpile(temp.circuit, simulator), observables)
# job = estimator.run([pub])

# print(job.result()[0].data.evs)
# # print(job.result()[0].data.stds)
# # print("Energy in state:", job.result())