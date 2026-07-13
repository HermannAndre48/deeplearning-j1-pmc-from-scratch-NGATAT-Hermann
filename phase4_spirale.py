import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
def generate_spiral(n_points=200, noise=0.1, seed=42):
"""Génère deux spirales entrelacées : classe 0 et classe 1."""
np.random.seed(seed)
n = n_points // 2
theta0 = np.linspace(0, 4 * np.pi, n) + np.random.randn(n) * noise
theta1 = np.linspace(0, 4 * np.pi, n) + np.random.randn(n) * noise + np.pi
r = np.linspace(0.1, 1.0, n)
X0 = np.c_[r * np.cos(theta0), r * np.sin(theta0)]
X1 = np.c_[r * np.cos(theta1), r * np.sin(theta1)]
X = np.vstack([X0, X1])
y = np.hstack([np.zeros(n), np.ones(n)])
return X, y
def sigmoid(x):
return 1 / (1 + np.exp(-x))
# TODO : implémenter relu(x) → max(0, x) — np.maximum(0, x)
def relu(x):
pass
# TODO : implémenter relu_grad(x) → dérivée de relu : 1 si x > 0, sinon 0
def relu_grad(x):
pass
def bce_loss(y_true, y_pred):
y_pred = np.clip(y_pred, 1e-7, 1 - 1e-7)
return -np.mean(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))
X, y = generate_spiral(n_points=400, noise=0.15)
# Architecture 2-64-64-1 avec initialisation He (std = sqrt(2 / n_entrées de la couche))
# TODO : initialiser W1 [2, 64], b1 [64], W2 [64, 64], b2 [64], W3 [64, 1], b3 [1]
# chaque W = randn(shape) * std_He de la couche ; chaque b initialisé à zéro
np.random.seed(42)
lr = 0.01
n_epochs = 2000
losses = []
for epoch in range(n_epochs):
# TODO : forward — 3 couches
# couche 1 : z1 = X @ W1 + b1 → a1 = relu(z1)
# couche 2 : z2 = a1 @ W2 + b2 → a2 = relu(z2)
# sortie : z3 = a2 @ W3 + b3 → y_pred = sigmoid(z3).flatten()
loss = bce_loss(y, y_pred)
losses.append(loss)
# TODO : backward — remonter les gradients couche par couche (même schéma qu'en phase
3)
# sortie : err3 = prédiction - y → gradients W3, b3
# couche 2 : err3 rétropropagé via W3, puis * relu_grad(z2) → gradients W2, b2
# couche 1 : err2 rétropropagé via W2, puis * relu_grad(z1) → gradients W1, b1
# rappel gradient d'une couche : ∂L/∂W = (1/n) * (entrée de la couche)^T · err
# ∂L/∂b = moyenne(err sur le batch)
# TODO : mettre à jour W1, b1, W2, b2, W3, b3
if epoch % 500 == 0:
acc = np.mean((y_pred > 0.5) == y)
print(f"Epoch {epoch:4d} | Loss: {loss:.4f} | Accuracy: {acc:.2%}")
# Frontière de décision (code fourni)
h = 0.02
xx, yy = np.meshgrid(np.arange(X[:, 0].min() - 0.2, X[:, 0].max() + 0.2, h),
np.arange(X[:, 1].min() - 0.2, X[:, 1].max() + 0.2, h))
grid = np.c_[xx.ravel(), yy.ravel()]
a1g = relu(np.dot(grid, W1) + b1)
a2g = relu(np.dot(a1g, W2) + b2)
zg = sigmoid(np.dot(a2g, W3) + b3).reshape(xx.shape)
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
axes[0].contourf(xx, yy, zg, alpha=0.4, cmap='RdBu')
axes[0].scatter(X[:, 0], X[:, 1], c=y, cmap='RdBu', s=10, edgecolors='none')
axes[0].set_title("Frontière de décision (2-64-64-1)")
axes[1].plot(losses)
axes[1].set_xlabel("Epoch"); axes[1].set_ylabel("Loss BCE")
axes[1].set_title("Courbe de loss spirale")
plt.savefig("phase4_spirale.png", dpi=100, bbox_inches='tight')
print(f"\nLoss finale : {losses[-1]:.4f}")
print(f"Accuracy finale : {np.mean((y_pred > 0.5) == y):.2%}")