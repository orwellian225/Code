import ramsey_util as ru
import numpy as np

from qiskit_aer import AerSimulator, StatevectorSimulator 
from qiskit_aer.primitives import EstimatorV2, SamplerV2, Estimator
import qiskit_algorithms as qka
import qiskit as qk
import qiskit.circuit.library as qkl

n = 6
length = n * (n - 1) // 2
problem_hamiltonian = ru.generate_ramsey_hamiltonian(n, 3, 3)
print("Numpy Min Energy:", np.min(np.linalg.eigvals(problem_hamiltonian)))

simulator = StatevectorSimulator(device="CPU")
estimator = Estimator()

ansatz = qkl.EfficientSU2(length)
initial_point = np.random.random(ansatz.num_parameters)
spsa = qka.optimizers.SPSA(maxiter=300)

vqe = qka.VQE(estimator, ansatz, optimizer=spsa)
result = vqe.compute_minimum_eigenvalue(operator=problem_hamiltonian)

print("VQE Min Energy:", result.eigenvalue)
