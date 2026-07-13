import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
X_xor = np.array([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=float)
y_xor = np.array([0, 1, 1, 0])
def sigmoid(x):
return 1 / (1 + np.exp(-x))
def compute_loss_bce(y_true, y_pred):
y_pred = np.clip(y_pred, 1e-7, 1 - 1e-7)
return -np.mean(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))
# Architecture 2-2-1 : 2 entrées → 2 neurones cachés → 1 sortie
# TODO : initialiser les poids (seed=42, facteur 0.5)
# W1 shape [2, 2], b1 shape [2,]
# W2 shape [2, 1], b2 shape [1,]
np.random.seed(42)
learning_rate = 0.5
n_epochs = 10000
losses = []
for epoch in range(n_epochs):
    # TODO : forward pass couche 1 (cachée)
# z1 = X_xor @ W1 + b1 → a1 = sigmoid(z1) shape attendu : [4, 2]
# TODO : forward pass couche 2 (sortie)
# z2 = a1 @ W2 + b2 → a2 = sigmoid(z2) shape attendu : [4, 1]
Devrait afficher :
La frontière de décision sur le plot doit montrer une séparation non-linéaire : une région rouge pour les points
(0,0) et (1,1), une région bleue pour (0,1) et (1,0). Aucune droite ne pourrait faire ça.
Qualité :
# y_pred = a2.flatten()
loss = compute_loss_bce(y_xor, y_pred)
losses.append(loss)
# Backprop couche 2 — même simplification qu'en phase 2 (BCE+sigmoid) :
# error2 = prédiction - cible → shape [4,]
# ∂L/∂W2 = (1/n) * a1^T · error2 → shape [2, 1]
# ∂L/∂b2 = moyenne(error2) → scalaire
# TODO : calculer error2, dW2, db2 (attention au reshape pour aligner les shapes)
# Backprop couche 1 — chain rule : on remonte à travers la couche 2 PUIS a1
# dérivée de sigmoid : σ'(x) = σ(x) * (1 - σ(x)) = a1 * (1 - a1)
# error1 = (error2 rétropropagé via W2) * σ'(z1) → shape [4, 2]
# ∂L/∂W1 = (1/n) * X^T · error1 → shape [2, 2]
# ∂L/∂b1 = moyenne(error1 sur le batch) → shape [2,]
# TODO : calculer error1, dW1, db1
# TODO : mettre à jour W1, b1, W2, b2
if epoch % 2000 == 0:
acc = np.mean((y_pred > 0.5) == y_xor)
print(f"Epoch {epoch:5d} | Loss: {loss:.4f} | Accuracy: {acc:.2%}")
# Frontière de décision (code fourni)
xx, yy = np.meshgrid(np.linspace(-0.5, 1.5, 200), np.linspace(-0.5, 1.5, 200))
grid = np.c_[xx.ravel(), yy.ravel()]
z1g = sigmoid(np.dot(grid, W1) + b1)
z2g = sigmoid(np.dot(z1g, W2) + b2).reshape(xx.shape)
plt.figure(figsize=(8, 6))
plt.contourf(xx, yy, z2g, alpha=0.4, cmap='RdBu')
plt.scatter(X_xor[:, 0], X_xor[:, 1], c=y_xor, s=100, cmap='RdBu', edgecolors='k')
plt.title("XOR : frontière de décision du réseau 2-2-1")
plt.savefig("phase3_xor_boundary.png", dpi=100, bbox_inches='tight')
print(f"\nLoss finale : {losses[-1]:.4f}")
print(f"Accuracy finale : {np.mean((y_pred > 0.5) == y_xor):.2%}")
