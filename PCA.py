import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.preprocessing import StandardScaler
from qiskit.quantum_info import Operator, SparsePauliOp
from qiskit_algorithms import VQE
from qiskit_algorithms.optimizers import SPSA
from qiskit.circuit.library import real_amplitudes
from qiskit_aer import AerSimulator
from qiskit.primitives import BackendEstimatorV2
import matplotlib.pyplot as plt
import seaborn as sns

# Προεπεξεργασία
data = load_breast_cancer()
# Επιλογή 4 χαρακτηριστικών για 4x4 matrix (2 qubits)
X = data.data[:, :4] 
X_scaled = StandardScaler().fit_transform(X)

# Υπολογισμός και κανονικοποίηση πίνακα συνδιακύμανσης
cov_matrix = np.cov(X_scaled.T)
cov_matrix = cov_matrix / np.trace(cov_matrix)

# Κλασική Baseline
classical_evs = np.linalg.eigvalsh(cov_matrix)
class_max = max(classical_evs)

# Κβαντική Υλοποίηση (VQPCA)
# Ορισμός του τελεστή
op = Operator(cov_matrix)
observable = SparsePauliOp.from_operator(op)
observable_inv = -1 * observable # Αντιστροφή για εύρεση μεγίστου

# Ρύθμιση VQE
backend = AerSimulator()
estimator = BackendEstimatorV2(backend=backend)

ansatz = real_amplitudes(num_qubits=2, reps=3, entanglement='full')
optimizer = SPSA(maxiter=50)

# Εκτέλεση
vqe = VQE(estimator, ansatz, optimizer)
vqe_result = vqe.compute_minimum_eigenvalue(observable_inv)

# Αποτελέσματα
quantum_max = -vqe_result.eigenvalue.real

print(f"Classical Max Eigenvalue: {class_max:.4f}")
print(f"Quantum Max Eigenvalue:   {quantum_max:.4f}")
print(f"Absolute Error:           {abs(class_max - quantum_max):.6f}")


# Χάρτης Θερμότητας / Πίνακα Συνδιακύμανσης
plt.figure(figsize=(6, 5))
sns.heatmap(cov_matrix, annot=True, cmap='coolwarm', fmt='.4f', 
            xticklabels=[f'Feature {i+1}' for i in range(4)], 
            yticklabels=[f'Feature {i+1}' for i in range(4)])
plt.title("Πίνακας Συνδιακύμανσης (4 Χαρακτηριστικά)")
plt.show()

# Σύγκριση Κύριας Ιδιοτιμής
plt.figure(figsize=(7, 5))
methods = ['Κλασική PCA (NumPy)', 'Κβαντική VQPCA (Qiskit VQE)']
eigenvalues = [class_max, quantum_max]

bars = plt.bar(methods, eigenvalues, color=['#4C72B0', '#55A868'], width=0.5)
plt.ylabel('Τιμή Κύριας Ιδιοτιμής (Max Variance)')
plt.title('Σύγκριση Εξαγωγής Κύριας Ιδιοτιμής')
plt.ylim(0, max(class_max, quantum_max) * 1.2) # Αφήνουμε χώρο πάνω από τις μπάρες

for bar in bars:
    yval = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2, yval + 0.01, 
             f'{yval:.4f}', ha='center', va='bottom', fontweight='bold')

plt.show()