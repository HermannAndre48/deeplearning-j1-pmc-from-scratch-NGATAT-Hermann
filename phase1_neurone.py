import numpy as np
X = np.array([
    [0.2, 0.1],
    [0.8, 0.9],
    [0.3, 0.7],
    [0.9, 0.2],
])

y = np.array([0, 1, 1, 0])


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def forward(X, w, b):
    z = np.dot(X, w) + b
    return sigmoid(z)


def compute_loss(y_true, y_pred):
    y_pred = np.clip(y_pred, 1e-7, 1 - 1e-7)
    return -np.mean(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))

w = np.array([0.5, -0.3])
b = 0.1

# Test scénario normal
y_pred = forward(X, w, b)
loss = compute_loss(y, y_pred)
print("Prédictions :", np.round(y_pred, 3))
print("Étiquettes :", y)
print("Loss BCE :", f"{loss:.4f}")
print()

# Test avec entrées à 0
X_zeros = np.zeros((4, 2))
y_pred_zeros = forward(X_zeros, w, b)
loss_zeros = compute_loss(y, y_pred_zeros)
print("Prédictions (X=0) :", np.round(y_pred_zeros, 3))
print("Loss (X=0) :", f"{loss_zeros:.4f}")
print()

# Test avec poids à 0
w_zero = np.zeros(2)
b_zero = 0
y_pred_adv = forward(X, w_zero, b_zero)
loss_adv = compute_loss(y, y_pred_adv)
print("Prédictions (w=0, b=0) :", np.round(y_pred_adv, 3))
print("Loss (w=0, b=0) :", f"{loss_adv:.4f}")