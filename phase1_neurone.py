import numpy as np
X = np.array([
    [0.2, 0.1],
    [0.8, 0.9],
    [0.3, 0.7],
    [0.9, 0.2],
])

y = np.array([0, 1, 1, 0])


# TODO : implémenter sigmoid(x)
# formule : 1 / (1 + exp(-x))
# numpy : np.exp(-x)
def sigmoid(x):
    pass


# TODO : implémenter forward(X, w, b)
# étape 1 : somme pondérée z = X @ w + b (numpy : np.dot)
# étape 2 : retourner sigmoid(z)
def forward(X, w, b):
    pass


# TODO : implémenter compute_loss(y_true, y_pred) — Binary Cross-Entropy
# formule : -mean( y*log(ŷ) + (1-y)*log(1-ŷ) )
# clamper y_pred entre 1e-7 et 1-1e-7 avant le log → np.clip
def compute_loss(y_true, y_pred):
    pass

# Poids fixés — pas encore d'entraînement dans cette phase
w = np.array([0.5, -0.3])
b = 0.1
# TODO : appeler forward, puis compute_loss, puis afficher y_pred.round(3) et loss