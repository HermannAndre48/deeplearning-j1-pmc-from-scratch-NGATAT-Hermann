import numpy as np
from sklearn.datasets import load_breast_cancer, load_wine, load_digits
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import tensorflow as tf
from tensorflow import keras
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import time

print("="*80)
print("PHASE 8 : Pipeline complet sur dataset personnel")
print("="*80)

# ============================================================================
# SCÉNARIO 1 : NORMAL - Breast Cancer pipeline complet
# ============================================================================

print("\n" + "="*80)
print("SCÉNARIO 1 : NORMAL - Breast Cancer (569 exemples, 30 features)")
print("="*80)

# ---- 1. Chargement Breast Cancer ----
data = load_breast_cancer()
X, y = data.data, data.target
n_features = X.shape[1]
n_samples = X.shape[0]
print(f"\nDataset chargé : {n_samples} exemples, {n_features} features")
print(f"Classes : {np.unique(y)} (0=malin, 1=bénin)")

# ---- 2. Split et normalisation ----
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

print(f"Train : {X_train_scaled.shape} | Test : {X_test_scaled.shape}")

# ---- 3. Fonctions d'activation et dérivées (NumPy) ----
def relu(x):
    return np.maximum(0, x)

def relu_grad(x):
    return (x > 0).astype(float)

def sigmoid(x):
    x = np.clip(x, -500, 500)
    return 1 / (1 + np.exp(-x))

def binary_crossentropy(y_true, y_pred):
    y_pred = np.clip(y_pred, 1e-7, 1 - 1e-7)
    return -np.mean(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))

def binary_accuracy(y_true, y_pred):
    y_pred_binary = (y_pred > 0.5).astype(int)
    return np.mean(y_true == y_pred_binary)

# ---- 4. Pipeline NumPy from-scratch ----
print("\n--- Pipeline NumPy from-scratch ---")

# Initialisation He
np.random.seed(42)
W1 = np.random.randn(n_features, 16) * np.sqrt(2 / n_features)
b1 = np.zeros((1, 16))
W2 = np.random.randn(16, 8) * np.sqrt(2 / 16)
b2 = np.zeros((1, 8))
W3 = np.random.randn(8, 1) * np.sqrt(2 / 8)
b3 = np.zeros((1, 1))

learning_rate = 0.01
losses_numpy = []

for epoch in range(200):
    # Forward
    z1 = X_train_scaled @ W1 + b1
    a1 = relu(z1)
    z2 = a1 @ W2 + b2
    a2 = relu(z2)
    z3 = a2 @ W3 + b3
    a3 = sigmoid(z3)
    
    # Loss
    loss = binary_crossentropy(y_train.reshape(-1, 1), a3)
    losses_numpy.append(loss)
    
    # Backprop
    dz3 = a3 - y_train.reshape(-1, 1)
    dW3 = (a2.T @ dz3) / len(y_train)
    db3 = np.sum(dz3, axis=0, keepdims=True) / len(y_train)
    
    da2 = dz3 @ W3.T
    dz2 = da2 * relu_grad(z2)
    dW2 = (a1.T @ dz2) / len(y_train)
    db2 = np.sum(dz2, axis=0, keepdims=True) / len(y_train)
    
    da1 = dz2 @ W2.T
    dz1 = da1 * relu_grad(z1)
    dW1 = (X_train_scaled.T @ dz1) / len(y_train)
    db1 = np.sum(dz1, axis=0, keepdims=True) / len(y_train)
    
    # Update
    W1 -= learning_rate * dW1
    b1 -= learning_rate * db1
    W2 -= learning_rate * dW2
    b2 -= learning_rate * db2
    W3 -= learning_rate * dW3
    b3 -= learning_rate * db3
    
    if (epoch + 1) % 40 == 0:
        print(f"  Epoch {epoch+1:3d} : Loss = {loss:.4f}")

# Test NumPy
z1_test = X_test_scaled @ W1 + b1
a1_test = relu(z1_test)
z2_test = a1_test @ W2 + b2
a2_test = relu(z2_test)
z3_test = a2_test @ W3 + b3
y_pred_numpy = sigmoid(z3_test)

loss_final_numpy = binary_crossentropy(y_test.reshape(-1, 1), y_pred_numpy)
acc_final_numpy = binary_accuracy(y_test, y_pred_numpy)

print(f"\nRésultats NumPy from-scratch:")
print(f"  Loss finale : {loss_final_numpy:.4f}")
print(f"  Test accuracy : {acc_final_numpy:.4f}")

# ---- 5. Pipeline Keras ----
print("\n--- Pipeline Keras ---")

tf.random.set_seed(42)
model_keras = keras.Sequential([
    keras.layers.Dense(16, activation='relu', input_shape=(n_features,)),
    keras.layers.Dense(8, activation='relu'),
    keras.layers.Dense(1, activation='sigmoid')
])

model_keras.compile(
    optimizer=keras.optimizers.Adam(learning_rate=0.001),
    loss='binary_crossentropy',
    metrics=['accuracy']
)

start_keras = time.time()
history_keras = model_keras.fit(
    X_train_scaled, y_train,
    epochs=50,
    batch_size=32,
    validation_split=0.1,
    verbose=0
)
time_keras = time.time() - start_keras

loss_keras_final, acc_keras_final = model_keras.evaluate(
    X_test_scaled, y_test, verbose=0
)

print(f"Résultats Keras:")
print(f"  Loss finale : {loss_keras_final:.4f}")
print(f"  Test accuracy : {acc_keras_final:.4f}")
print(f"  Temps entraînement : {time_keras:.1f}s")

# ---- 6. Comparaison ----
print("\n" + "-"*80)
print("COMPARAISON : NumPy vs Keras")
print("-"*80)
print(f"NumPy from-scratch | Loss finale : {loss_final_numpy:.4f} | Test accuracy : {acc_final_numpy:.4f}")
print(f"Keras             | Loss finale : {loss_keras_final:.4f} | Test accuracy : {acc_keras_final:.4f}")
gain_acc = (acc_keras_final - acc_final_numpy) * 100
print(f"Gain Keras vs NumPy : {gain_acc:+.1f} points de %")

# ============================================================================
# SCÉNARIO 2 : CAS LIMITE - Données manquantes (zéros => médiane)
# ============================================================================

print("\n" + "="*80)
print("SCÉNARIO 2 : CAS LIMITE - Gestion des zéros suspects")
print("="*80)

# Simuler des zéros suspects : remplacer quelques valeurs > 90e percentile par 0
X_suspicious = X.copy()
for col in range(X.shape[1]):
    p90 = np.percentile(X[:, col], 90)
    n_zeros = int(0.05 * len(X))
    corrupt_idx = np.random.choice(len(X), n_zeros, replace=False)
    X_suspicious[corrupt_idx, col] = 0

# Traiter les zéros avec la médiane
X_treated = X_suspicious.copy()
for col in range(X_treated.shape[1]):
    non_zero = X_treated[:, col][X_treated[:, col] != 0]
    if len(non_zero) > 0:
        median_val = np.median(non_zero)
        X_treated[:, col] = np.where(
            X_treated[:, col] == 0,
            median_val,
            X_treated[:, col]
        )

# Pipeline Keras sur données brutes vs traitées
X_train_sus, X_test_sus, y_train_sus, y_test_sus = train_test_split(
    X_suspicious, y, test_size=0.2, random_state=42
)
X_train_treated, X_test_treated, y_train_t, y_test_t = train_test_split(
    X_treated, y, test_size=0.2, random_state=42
)

scaler_sus = StandardScaler()
X_train_sus_scaled = scaler_sus.fit_transform(X_train_sus)
X_test_sus_scaled = scaler_sus.transform(X_test_sus)

scaler_treated = StandardScaler()
X_train_treated_scaled = scaler_treated.fit_transform(X_train_treated)
X_test_treated_scaled = scaler_treated.transform(X_test_treated)

# Entraîner sur données brutes
print("\nEntraînement sur données avec zéros suspects (sans traitement)...")
tf.random.set_seed(42)
model_sus = keras.Sequential([
    keras.layers.Dense(16, activation='relu', input_shape=(n_features,)),
    keras.layers.Dense(8, activation='relu'),
    keras.layers.Dense(1, activation='sigmoid')
])
model_sus.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
model_sus.fit(X_train_sus_scaled, y_train_sus, epochs=50, batch_size=32,
              validation_split=0.1, verbose=0)
loss_sus, acc_sus = model_sus.evaluate(X_test_sus_scaled, y_test_sus, verbose=0)

# Entraîner sur données traitées
print("Entraînement sur données avec zéros remplacés par médiane...")
tf.random.set_seed(42)
model_treated = keras.Sequential([
    keras.layers.Dense(16, activation='relu', input_shape=(n_features,)),
    keras.layers.Dense(8, activation='relu'),
    keras.layers.Dense(1, activation='sigmoid')
])
model_treated.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
model_treated.fit(X_train_treated_scaled, y_train_t, epochs=50, batch_size=32,
                  validation_split=0.1, verbose=0)
loss_treated, acc_treated = model_treated.evaluate(X_test_treated_scaled, y_test_t, verbose=0)

print(f"\nRésultats CAS LIMITE :")
print(f"  Avec zéros suspects    : Accuracy = {acc_sus:.4f}")
print(f"  Avec zéros => médiane  : Accuracy = {acc_treated:.4f}")
print(f"  Impact du nettoyage    : {(acc_treated - acc_sus) * 100:+.1f} points")

# ============================================================================
# SCÉNARIO 3 : ADVERSARIAL - Valeurs extrêmes
# ============================================================================

print("\n" + "="*80)
print("SCÉNARIO 3 : ADVERSARIAL - Prédictions sur données hors distribution")
print("="*80)

# Créer des exemples extrêmes
X_extreme = np.array([[99999] * n_features])
X_extreme_scaled = scaler.transform(X_extreme)

# Prédictions NumPy
z1_ext = X_extreme_scaled @ W1 + b1
a1_ext = relu(z1_ext)
z2_ext = a1_ext @ W2 + b2
a2_ext = relu(z2_ext)
z3_ext = a2_ext @ W3 + b3
y_pred_numpy_extreme = sigmoid(z3_ext)[0, 0]

# Prédictions Keras
y_pred_keras_extreme = model_keras.predict(X_extreme_scaled, verbose=0)[0, 0]

print(f"\nPrédiction sur données extrêmes ([99999]*30) :")
print(f"  NumPy : {y_pred_numpy_extreme:.4f} (confiance : {max(y_pred_numpy_extreme, 1-y_pred_numpy_extreme):.2%})")
print(f"  Keras : {y_pred_keras_extreme:.4f} (confiance : {max(y_pred_keras_extreme, 1-y_pred_keras_extreme):.2%})")

print(f"\n SIGNAL D'ALARME : Les modèles sont très confiants sur des données")
print(f"   complètement hors de la distribution d'entraînement.")
print(f"   Cela indique un risque en production : la prédiction n'est pas fiable.")
print(f"   Solution : implémenter une détection d'anomalies ou un seuil de confiance.")

# ============================================================================
# POUR ALLER PLUS LOIN : Second dataset (Wine) + Explorations
# ============================================================================

print("\n" + "="*80)
print("POUR ALLER PLUS LOIN : Dataset Wine (178 exemples, 3 classes)")
print("="*80)

data_wine = load_wine()
X_wine, y_wine = data_wine.data, data_wine.target
print(f"\nWine dataset : {X_wine.shape[0]} exemples, {X_wine.shape[1]} features, {len(np.unique(y_wine))} classes")

X_train_wine, X_test_wine, y_train_wine, y_test_wine = train_test_split(
    X_wine, y_wine, test_size=0.2, random_state=42
)

scaler_wine = StandardScaler()
X_train_wine_scaled = scaler_wine.fit_transform(X_train_wine)
X_test_wine_scaled = scaler_wine.transform(X_test_wine)

# Keras multiclass (categorical)
tf.random.set_seed(42)
model_wine = keras.Sequential([
    keras.layers.Dense(16, activation='relu', input_shape=(X_wine.shape[1],)),
    keras.layers.Dense(8, activation='relu'),
    keras.layers.Dense(len(np.unique(y_wine)), activation='softmax')
])

model_wine.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

model_wine.fit(X_train_wine_scaled, y_train_wine, epochs=50, batch_size=16,
               validation_split=0.1, verbose=0)

loss_wine, acc_wine = model_wine.evaluate(X_test_wine_scaled, y_test_wine, verbose=0)
print(f"Résultats Wine Keras : Accuracy = {acc_wine:.4f}")

# ============================================================================
# VISUALISATIONS
# ============================================================================

print("\n" + "="*80)
print("Génération des visualisations...")
print("="*80)

fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# Plot 1 : Courbes de loss NumPy vs Keras
ax1 = axes[0, 0]
ax1.plot(losses_numpy, label='NumPy (200 epochs)', linewidth=2, alpha=0.7)
ax1.plot(np.arange(0, 50), history_keras.history['loss'], label='Keras (50 epochs)', linewidth=2, alpha=0.7)
ax1.set_xlabel("Epoch")
ax1.set_ylabel("Loss")
ax1.set_title("Convergence : NumPy vs Keras")
ax1.legend()
ax1.grid(alpha=0.3)

# Plot 2 : Barplot comparaison accuracy
ax2 = axes[0, 1]
models = ['NumPy\nfrom-scratch', 'Keras\nBrainstack']
accuracies = [acc_final_numpy, acc_keras_final]
colors = ['#FF6B6B', '#4ECDC4']
bars = ax2.bar(models, accuracies, color=colors, alpha=0.7, edgecolor='black', linewidth=2)
ax2.set_ylim([0, 1.0])
ax2.set_ylabel("Test Accuracy")
ax2.set_title("Comparaison Accuracy : Breast Cancer")
for bar, acc in zip(bars, accuracies):
    height = bar.get_height()
    ax2.text(bar.get_x() + bar.get_width()/2., height,
             f'{acc:.3f}', ha='center', va='bottom', fontweight='bold')
ax2.grid(axis='y', alpha=0.3)

# Plot 3 : Cas limite - impact du nettoyage
ax3 = axes[1, 0]
methods = ['Zéros suspects', 'Zéros => Médiane']
accs_limit = [acc_sus, acc_treated]
colors_limit = ['#FF6B6B', '#95E1D3']
bars3 = ax3.bar(methods, accs_limit, color=colors_limit, alpha=0.7, edgecolor='black', linewidth=2)
ax3.set_ylim([0, 1.0])
ax3.set_ylabel("Test Accuracy")
ax3.set_title("CAS LIMITE : Impact du nettoyage des données")
for bar, acc in zip(bars3, accs_limit):
    height = bar.get_height()
    ax3.text(bar.get_x() + bar.get_width()/2., height,
             f'{acc:.3f}', ha='center', va='bottom', fontweight='bold')
ax3.grid(axis='y', alpha=0.3)

# Plot 4 : Comparaison datasets
ax4 = axes[1, 1]
datasets_comp = ['Breast Cancer\n(binary)', 'Wine\n(multiclass)']
accs_datasets = [acc_keras_final, acc_wine]
colors_datasets = ['#4ECDC4', '#FFE66D']
bars4 = ax4.bar(datasets_comp, accs_datasets, color=colors_datasets, alpha=0.7, edgecolor='black', linewidth=2)
ax4.set_ylim([0, 1.0])
ax4.set_ylabel("Test Accuracy (Keras)")
ax4.set_title("Comparaison : Datasets différents")
for bar, acc in zip(bars4, accs_datasets):
    height = bar.get_height()
    ax4.text(bar.get_x() + bar.get_width()/2., height,
             f'{acc:.3f}', ha='center', va='bottom', fontweight='bold')
ax4.grid(axis='y', alpha=0.3)

plt.tight_layout()
plt.savefig("phase8_pipeline_comparaison.png", dpi=100, bbox_inches='tight')
print("\n✓ Visualisation sauvegardée : phase8_pipeline_comparaison.png")

# ============================================================================
# TABLEAU RÉCAPITULATIF FINAL
# ============================================================================

print("\n" + "="*80)
print("RÉSUMÉ FINAL - PHASE 8")
print("="*80)
print("\nBREAST CANCER - Scénario Normal:")
print(f"  NumPy from-scratch | Loss : {loss_final_numpy:.4f} | Accuracy : {acc_final_numpy:.4f}")
print(f"  Keras             | Loss : {loss_keras_final:.4f} | Accuracy : {acc_keras_final:.4f}")
print(f"  Gain Keras        | {gain_acc:+.2f}%")

print("\nCAS LIMITE - Données manquantes:")
print(f"  Avec zéros suspects   : {acc_sus:.4f}")
print(f"  Après nettoyage       : {acc_treated:.4f}")
print(f"  Amélioration          : {(acc_treated - acc_sus) * 100:+.2f}%")

print("\nADVERSARIAL - Données extrêmes:")
print(f"  NumPy confiance : {max(y_pred_numpy_extreme, 1-y_pred_numpy_extreme):.2%}")
print(f"  Keras confiance : {max(y_pred_keras_extreme, 1-y_pred_keras_extreme):.2%}")
print(f"   Prédictions non fiables sur hors-distribution!")

print("\nEXPLORATION - Second dataset (Wine):")
print(f"  Keras Multiclass : {acc_wine:.4f}")

print("\n" + "="*80)

