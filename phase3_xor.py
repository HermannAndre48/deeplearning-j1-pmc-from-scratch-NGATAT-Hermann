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

np.random.seed(42)
W1 = np.random.randn(2, 2) * 1.0
b1 = np.zeros(2)
W2 = np.random.randn(2, 1) * 1.0
b2 = np.zeros(1)

learning_rate = 5.0
n_epochs = 50000
losses = []

for epoch in range(n_epochs):
    z1 = np.dot(X_xor, W1) + b1
    a1 = sigmoid(z1)
    
    z2 = np.dot(a1, W2) + b2
    a2 = sigmoid(z2)
    y_pred = a2.flatten()
    
    loss = compute_loss_bce(y_xor, y_pred)
    losses.append(loss)
    
    error2 = y_pred - y_xor
    dW2 = (1 / len(y_xor)) * np.dot(a1.T, error2.reshape(-1, 1))
    db2 = np.mean(error2)
    
    error1 = np.dot(error2.reshape(-1, 1), W2.T) * a1 * (1 - a1)
    dW1 = (1 / len(y_xor)) * np.dot(X_xor.T, error1)
    db1 = np.mean(error1, axis=0)
    
    W1 = W1 - learning_rate * dW1
    b1 = b1 - learning_rate * db1
    W2 = W2 - learning_rate * dW2
    b2 = b2 - learning_rate * db2
    
    if epoch % 2000 == 0:
        acc = np.mean((y_pred > 0.5) == y_xor)
        print(f"Epoch {epoch:5d} | Loss: {loss:.4f} | Accuracy: {acc:.2%}")

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
