import numpy as np
import tensorflow as tf
from tensorflow import keras
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import make_classification
import time
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
# ---- Partie 1 : Breast Cancer (petit dataset, 569 exemples) ----
data = load_breast_cancer()
X, y = data.data, data.target
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)
EPOCHS = 200
n_features = X_train.shape[1]
# TODO : implémenter run_numpy(X_train, y_train, X_test, y_test, epochs)
# - architecture [n_features → 16 → 8 → 1]
# - initialisation He (std = sqrt(2 / n_entrées))
# - activation couches cachées : relu, sortie : sigmoid
# - BCE loss, descente de gradient full-batch, lr=0.2 (plus grand que le 1e-3 d'Adam :
en full-batch un lr trop petit n'apprend rien)
# - retourner : (train_time_s, final_loss, test_accuracy)
def run_numpy(X_tr, y_tr, X_te, y_te, epochs):
pass # TODO
# TODO : implémenter run_keras(X_train, y_train, X_test, y_test, epochs)
# - même architecture en Keras : [Dense(16, relu), Dense(8, relu), Dense(1, sigmoid)]
# - binary_crossentropy, Adam lr=0.001, batch=32
# - retourner : (train_time_s, final_loss, test_accuracy)
def run_keras(X_tr, y_tr, X_te, y_te, epochs):
pass # TODO
# ---- Partie 2 : dataset synthétique large (make_classification) ----
# TODO : générer un dataset avec make_classification :
# n_samples=50_000, n_features=30, n_informative=15, random_state=42
# même split 80/20, même normalisation StandardScaler
# mêmes fonctions run_numpy / run_keras, même EPOCHS
# ---- Tableau comparatif agrégé (code fourni) ----
# Remplissez ces variables à partir de vos appels :
results = {
'breast_cancer': {
'numpy': {'time': 0, 'loss': 0, 'acc': 0}, # à remplacer
'keras': {'time': 0, 'loss': 0, 'acc': 0},
},
'large_50k': {
'numpy': {'time': 0, 'loss': 0, 'acc': 0},
'keras': {'time': 0, 'loss': 0, 'acc': 0},
},
}
print("\n=== COMPARAISON NUMPY vs KERAS ===")
for dataset_name, res in results.items():
ratio = res['numpy']['time'] / res['keras']['time'] if res['keras']['time'] > 0 else
float('nan')
print(f"\n [{dataset_name}]")
print(f" Numpy : {res['numpy']['time']:.1f}s | loss {res['numpy']['loss']:.4f} |
acc {res['numpy']['acc']:.4f}")
print(f" Keras : {res['keras']['time']:.1f}s | loss {res['keras']['loss']:.4f} |
acc {res['keras']['acc']:.4f}")
print(f" Keras est {ratio:.1f}x plus rapide sur ce dataset")
# Barplot temps (code fourni)
labels = ['Breast Cancer\n(569 ex)', 'Large dataset\n(50 000 ex)']
t_numpy = [results['breast_cancer']['numpy']['time'], results['large_50k']['numpy']
['time']]
t_keras = [results['breast_cancer']['keras']['time'], results['large_50k']['keras']
['time']]
x = np.arange(len(labels))
width = 0.35
fig, ax = plt.subplots(figsize=(8, 5))
ax.bar(x - width/2, t_numpy, width, label='Numpy from-scratch', color='steelblue')
ax.bar(x + width/2, t_keras, width, label='Keras', color='darkorange')
ax.set_ylabel("Temps (s) pour 200 epochs")
ax.set_xticks(x); ax.set_xticklabels(labels)
ax.set_title("Numpy vs Keras : coût d'entraînement comparé")
ax.legend()
plt.tight_layout()
plt.savefig("phase10_timing.png", dpi=100, bbox_inches='tight')
print("\nBarplot sauvegardé : phase10_timing.png")