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
depths = [1, 2, 3]
widths = [8, 64, 256]
results = []
# TODO : pour chaque combinaison (depth, width) :
# 1. tf.random.set_seed(42)
# 2. construire le modèle :
# - `depth` couches Dense(width, relu)
# - couche de sortie Dense(10, softmax)
# - compter le nombre total de paramètres : model.count_params()
# 3. compiler : Adam(lr=1e-3), sparse_categorical_crossentropy, accuracy
# 4. entraîner 15 epochs, batch=64, validation_split=0.1, verbose=0, mesurer le temps
# 5. évaluer sur X_test : test_loss, test_accuracy
# 6. repérer la première epoch où val_accuracy > 0.95 (ou "N/A" si jamais atteint)
# 7. stocker dans results : depth, width, n_params, test_accuracy, val_loss_final,
# epoch_95, train_time_s
for depth in depths:
for width in widths:
pass # TODO
# Tableau récapitulatif (ok je vous fournis ça)
print("\n=== ABLATION STUDY : PROFONDEUR x LARGEUR ===")
print(f"{'Depth':6s} | {'Width':6s} | {'Params':10s} | {'Test acc':10s} | {'Val loss':10s}
| {'Epoch >95%':10s} | {'Temps (s)':10s}")
print("-" * 80)
for r in results:
print(
    f"{r['depth']:6d} | {r['width']:6d} | {r['n_params']:10,d} | "
f"{r['test_accuracy']:.4f} | {r['val_loss_final']:.4f} | "
f"{str(r['epoch_95']):10s} | {r['train_time_s']:.0f}"
)
# Heatmap accuracy (code fourni ici aussi, plus simple de l'avoir)
acc_grid = np.array([
[r['test_accuracy'] for r in results if r['depth'] == d]
for d in depths
])
fig, ax = plt.subplots(figsize=(7, 4))
im = ax.imshow(acc_grid, cmap='RdYlGn', vmin=0.85, vmax=0.99)
ax.set_xticks(range(len(widths))); ax.set_xticklabels(widths)
ax.set_yticks(range(len(depths))); ax.set_yticklabels(depths)
ax.set_xlabel("Largeur (neurones par couche)")
ax.set_ylabel("Profondeur (couches cachées)")
ax.set_title("Test accuracy selon l'architecture (MNIST)")
for i in range(len(depths)):
for j in range(len(widths)):
ax.text(j, i, f"{acc_grid[i, j]:.3f}", ha='center', va='center', fontsize=9)
plt.colorbar(im)
plt.tight_layout()
plt.savefig("phase9_ablation_heatmap.png", dpi=100, bbox_inches='tight')
print("\nHeatmap sauvegardée : phase9_ablation_heatmap.png")
