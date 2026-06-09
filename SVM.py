import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.decomposition import PCA
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report
from qiskit.circuit.library import zz_feature_map
from qiskit_machine_learning.kernels import FidelityQuantumKernel
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix

# Προεπεξεργασία
data = load_breast_cancer()
X_standard = StandardScaler().fit_transform(data.data)

# Μείωση σε 2 χαρακτηριστικά
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_standard)

scaler_quantum = MinMaxScaler(feature_range=(-1, 1))
X_quantum = scaler_quantum.fit_transform(X_pca)

# Διαχωρισμός σε σύνολα εκπαίδευσης και δοκιμής
X_train, X_test, y_train, y_test = train_test_split(
    X_quantum, data.target, test_size=0.2, random_state=42
)

# Κλασικό SVM
classical_svc = SVC(kernel='rbf')
classical_svc.fit(X_train, y_train)
y_pred_class = classical_svc.predict(X_test)

# Κβαντικό SVM
# Ορισμός του Quantum Feature Map
feature_map = zz_feature_map(feature_dimension=2, reps=1, entanglement='linear')

# Ορισμός του Κβαντικού Πυρήνα
qkernel = FidelityQuantumKernel(feature_map=feature_map)

# Υπολογισμός των Πινάκων Πυρήνα
matrix_train = qkernel.evaluate(x_vec=X_train)
matrix_test = qkernel.evaluate(x_vec=X_test, y_vec=X_train)

# Εκπαίδευση του Υβριδικού QSVMσ
qsvm = SVC(kernel='precomputed')
qsvm.fit(matrix_train, y_train)
y_pred_quant = qsvm.predict(matrix_test)

# Αποτελέσματα
print("Classical SVM Accuracy:", classical_svc.score(X_test, y_test))
print("QSVM Accuracy:", qsvm.score(matrix_test, y_test))

print("--- Αποτελέσματα Κλασικού SVM (RBF Kernel) ---")
print(f"Accuracy:  {accuracy_score(y_test, y_pred_class):.4f}")
print(f"Precision: {precision_score(y_test, y_pred_class):.4f}")
print(f"Recall:    {recall_score(y_test, y_pred_class):.4f}")
print(f"F1-Score:  {f1_score(y_test, y_pred_class):.4f}")
print("-" * 45)

print("--- Αποτελέσματα Κβαντικού SVM (ZZFeatureMap) ---")
print(f"Accuracy:  {accuracy_score(y_test, y_pred_quant):.4f}")
print(f"Precision: {precision_score(y_test, y_pred_quant):.4f}")
print(f"Recall:    {recall_score(y_test, y_pred_quant):.4f}")
print(f"F1-Score:  {f1_score(y_test, y_pred_quant):.4f}")



# Δισδιάστατη απεικόνιση των δεδομένων
plt.figure(figsize=(8, 6))
# Σχεδιασμός σημείων εκπαίδευσης και δοκιμής
scatter = plt.scatter(X_quantum[:, 0], X_quantum[:, 1], c=data.target, 
                      cmap='bwr', alpha=0.7, edgecolors='k')
plt.title("Κατανομή Δεδομένων Καρκίνου του Μαστού (2 Κύριες Συνιστώσες)")
plt.xlabel("Κύρια Συνιστώσα 1 (Κλιμακωμένη -1 έως 1)")
plt.ylabel("Κύρια Συνιστώσα 2 (Κλιμακωμένη -1 έως 1)")
# Προσθήκη υπομνήματος
plt.legend(handles=scatter.legend_elements()[0], labels=['Κακοήθης', 'Καλοήθης'])
plt.grid(True, linestyle='--', alpha=0.5)
plt.show()

# Πίνακες Σύγχυσης
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

# Κλασικό SVM
cm_class = confusion_matrix(y_test, y_pred_class)
sns.heatmap(cm_class, annot=True, fmt='d', cmap='Blues', ax=axes[0],
            xticklabels=['Κακοήθης', 'Καλοήθης'], yticklabels=['Κακοήθης', 'Καλοήθης'])
axes[0].set_title('Πίνακας Σύγχυσης: Κλασικό SVM')
axes[0].set_xlabel('Προβλεπόμενη Κλάση')
axes[0].set_ylabel('Πραγματική Κλάση')

# Κβαντικό SVM
cm_quant = confusion_matrix(y_test, y_pred_quant)
sns.heatmap(cm_quant, annot=True, fmt='d', cmap='Greens', ax=axes[1],
            xticklabels=['Κακοήθης', 'Καλοήθης'], yticklabels=['Κακοήθης', 'Καλοήθης'])
axes[1].set_title('Πίνακας Σύγχυσης: Κβαντικό SVM (QSVM)')
axes[1].set_xlabel('Προβλεπόμενη Κλάση')
axes[1].set_ylabel('Πραγματική Κλάση')

plt.tight_layout()
plt.show()