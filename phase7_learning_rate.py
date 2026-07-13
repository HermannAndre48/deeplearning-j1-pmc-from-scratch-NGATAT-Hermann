import numpy as np
import tensorflow as tf
from tensorflow import keras
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import time
(X_train, y_train), (X_test, y_test) = keras.datasets.mnist.load_data()
X_train = X_train.reshape(-1, 784).astype('float32') / 255.0
X_test = X_test.reshape(-1, 784).astype('float32') / 255.0
learning_rates = [1e-7, 1e-3, 1.0]
lr_labels = ['trop petit (1e-7)', 'sweet spot (1e-3)', 'trop grand (1.0)']
results = []
histories = {}
# TODO : même structure que la phase 6, mais on itère sur les learning rates
# pour chaque (lr, label) dans zip(learning_rates, lr_labels) :
# 1. tf.random.set_seed(42)
# 2. même modèle [Dense(128, relu), Dense(64, relu), Dense(10, softmax)]
# 3. compiler avec Adam(learning_rate=lr)
# 4. entraîner 10 epochs, batch=64, validation_split=0.1, verbose=0
# 5. stocker val_loss final, test_accuracy, temps dans results
# 6. stocker val_losses dans histories[label]
for lr, label in zip(learning_rates, lr_labels):
pass # TODO
# Tableau récapitulatif (code fourni)
print("\n=== TABLEAU COMPARATIF LEARNING RATE ===")
print(f"{'LR':8s} | {'Label':24s} | {'Val loss final':14s} | {'Test acc':10s} | {'Temps
(s)':10s}")
print("-" * 80)
for r in results:
print(f"{r['lr']:.0e} | {r['label']:24s} | {r['val_loss_final']:.4f} |
{r['test_accuracy']:.4f} | {r['train_time_s']:.0f}")
# Courbe superposée (code fourni)
plt.figure(figsize=(10, 5))
for label, val_losses in histories.items():
plt.plot(range(1, 11), val_losses, label=label, linewidth=2)
plt.xlabel("Epoch"); plt.ylabel("Val Loss")
plt.title("Impact du learning rate sur la convergence (MNIST)")
plt.legend(); plt.yscale('log')
plt.savefig("phase7_lr_curve.png", dpi=100, bbox_inches='tight')
print("\nCourbe sauvegardée : phase7_lr_curve.png")