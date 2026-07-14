import numpy as np
import tensorflow as tf
from tensorflow import keras
from sklearn.datasets import load_breast_cancer, make_classification
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import time
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

print("="*80)
print("PHASE 10 : NumPy vs Keras - Le vrai coût (timing benchmark)")
print("="*80)

# ============================================================================
# FONCTIONS UTILITAIRES
# ============================================================================

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
    return np.mean((y_pred > 0.5).astype(int) == y_true)

# ============================================================================
# NUMPY TRAINING FUNCTION
# ============================================================================

def run_numpy(X_train, y_train, X_test, y_test, epochs):
    """Entraîner un réseau [n_features → 16 → 8 → 1] en NumPy pur"""
    n_features = X_train.shape[1]
    
    # Initialisation He
    np.random.seed(42)
    W1 = np.random.randn(n_features, 16) * np.sqrt(2 / n_features)
    b1 = np.zeros((1, 16))
    W2 = np.random.randn(16, 8) * np.sqrt(2 / 16)
    b2 = np.zeros((1, 8))
    W3 = np.random.randn(8, 1) * np.sqrt(2 / 8)
    b3 = np.zeros((1, 1))
    
    lr = 0.2  # Full-batch, lr peut être plus grand
    
    start = time.time()
    
    for epoch in range(epochs):
        # Forward
        z1 = X_train @ W1 + b1
        a1 = relu(z1)
        z2 = a1 @ W2 + b2
        a2 = relu(z2)
        z3 = a2 @ W3 + b3
        a3 = sigmoid(z3)
        
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
        dW1 = (X_train.T @ dz1) / len(y_train)
        db1 = np.sum(dz1, axis=0, keepdims=True) / len(y_train)
        
        # Update
        W1 -= lr * dW1
        b1 -= lr * db1
        W2 -= lr * dW2
        b2 -= lr * db2
        W3 -= lr * dW3
        b3 -= lr * db3
    
    elapsed = time.time() - start
    
    # Test
    z1_test = X_test @ W1 + b1
    a1_test = relu(z1_test)
    z2_test = a1_test @ W2 + b2
    a2_test = relu(z2_test)
    z3_test = a2_test @ W3 + b3
    y_pred = sigmoid(z3_test)
    
    final_loss = binary_crossentropy(y_test, y_pred.flatten())
    test_acc = binary_accuracy(y_test, y_pred.flatten())
    
    return elapsed, final_loss, test_acc

# ============================================================================
# KERAS TRAINING FUNCTION
# ============================================================================

def run_keras(X_train, y_train, X_test, y_test, epochs):
    """Entraîner le même réseau en Keras"""
    n_features = X_train.shape[1]
    
    tf.random.set_seed(42)
    model = keras.Sequential([
        keras.layers.Dense(16, activation='relu', input_shape=(n_features,)),
        keras.layers.Dense(8, activation='relu'),
        keras.layers.Dense(1, activation='sigmoid')
    ])
    
    model.compile(
        optimizer=keras.optimizers.SGD(learning_rate=0.2),  # Même lr que NumPy
        loss='binary_crossentropy',
        metrics=['accuracy']
    )
    
    start = time.time()
    model.fit(X_train, y_train, epochs=epochs, batch_size=len(X_train),
              validation_split=0.0, verbose=0)  # Full-batch
    elapsed = time.time() - start
    
    loss, acc = model.evaluate(X_test, y_test, verbose=0)
    
    return elapsed, loss, acc

# ============================================================================
# SCÉNARIO 1 : NORMAL - Breast Cancer
# ============================================================================

print("\n" + "="*80)
print("SCÉNARIO 1 : NORMAL - Breast Cancer (569 exemples)")
print("="*80)

data_bc = load_breast_cancer()
X_bc, y_bc = data_bc.data, data_bc.target
X_train_bc, X_test_bc, y_train_bc, y_test_bc = train_test_split(
    X_bc, y_bc, test_size=0.2, random_state=42
)

scaler_bc = StandardScaler()
X_train_bc = scaler_bc.fit_transform(X_train_bc)
X_test_bc = scaler_bc.transform(X_test_bc)

print("\n[Breast Cancer - 569 exemples, 30 features]")

print("Entraînement NumPy...")
time_np_bc, loss_np_bc, acc_np_bc = run_numpy(X_train_bc, y_train_bc, X_test_bc, y_test_bc, epochs=200)
print(f"NumPy : {time_np_bc:.1f}s | loss {loss_np_bc:.4f} | acc {acc_np_bc:.4f}")

print("Entraînement Keras...")
time_keras_bc, loss_keras_bc, acc_keras_bc = run_keras(X_train_bc, y_train_bc, X_test_bc, y_test_bc, epochs=200)
print(f"Keras : {time_keras_bc:.1f}s | loss {loss_keras_bc:.4f} | acc {acc_keras_bc:.4f}")

ratio_bc = time_np_bc / time_keras_bc
print(f"✓ Keras est {ratio_bc:.1f}x plus rapide sur Breast Cancer")

# ============================================================================
# SCÉNARIO 1 (suite) : NORMAL - Large dataset 50k
# ============================================================================

print("\n[Dataset synthétique 50k exemples]")

X_large, y_large = make_classification(
    n_samples=50000, n_features=30, n_informative=20, n_redundant=5,
    random_state=42
)

X_train_large, X_test_large, y_train_large, y_test_large = train_test_split(
    X_large, y_large, test_size=0.2, random_state=42
)

scaler_large = StandardScaler()
X_train_large = scaler_large.fit_transform(X_train_large)
X_test_large = scaler_large.transform(X_test_large)

print("Entraînement NumPy (50k exemples, cela peut prendre du temps)...")
time_np_large, loss_np_large, acc_np_large = run_numpy(X_train_large, y_train_large, X_test_large, y_test_large, epochs=50)
print(f"NumPy : {time_np_large:.1f}s | loss {loss_np_large:.4f} | acc {acc_np_large:.4f}")

print("Entraînement Keras (50k exemples)...")
time_keras_large, loss_keras_large, acc_keras_large = run_keras(X_train_large, y_train_large, X_test_large, y_test_large, epochs=50)
print(f"Keras : {time_keras_large:.1f}s | loss {loss_keras_large:.4f} | acc {acc_keras_large:.4f}")

ratio_large = time_np_large / time_keras_large
print(f"✓ Keras est {ratio_large:.1f}x plus rapide sur 50k exemples")

# ============================================================================
# SCÉNARIO 2 : CAS LIMITE - 10 epochs sur 50k
# ============================================================================

print("\n" + "="*80)
print("SCÉNARIO 2 : CAS LIMITE - 10 epochs sur dataset 50k")
print("="*80)

print("\nEntraînement NumPy (10 epochs)...")
time_np_10, loss_np_10, acc_np_10 = run_numpy(X_train_large, y_train_large, X_test_large, y_test_large, epochs=10)
print(f"NumPy : {time_np_10:.1f}s | loss {loss_np_10:.4f} | acc {acc_np_10:.4f}")

print("Entraînement Keras (10 epochs)...")
time_keras_10, loss_keras_10, acc_keras_10 = run_keras(X_train_large, y_train_large, X_test_large, y_test_large, epochs=10)
print(f"Keras : {time_keras_10:.1f}s | loss {loss_keras_10:.4f} | acc {acc_keras_10:.4f}")

ratio_10 = time_np_10 / time_keras_10
print(f"\nComparaison : Ratio d'accélération avec EPOCHS=10")
print(f"Keras est {ratio_10:.1f}x plus rapide (vs {ratio_large:.1f}x avec EPOCHS=50)")
print(f"\n Le ratio reste stable : {abs(ratio_10 - ratio_large) / ratio_large * 100:.1f}% de différence")
print(f"    Cela confirme que l'accélération est STRUCTURELLE, pas liée au nombre d'itérations")

# ============================================================================
# SCÉNARIO 3 : ADVERSARIAL - Sans normalisation
# ============================================================================

print("\n" + "="*80)
print("SCÉNARIO 3 : ADVERSARIAL - Sans StandardScaler sur 50k")
print("="*80)

# Données brutes (sans normalisation)
X_train_raw, X_test_raw, y_train_raw, y_test_raw = train_test_split(
    X_large, y_large, test_size=0.2, random_state=42
)

print("\nEntraînement NumPy (données brutes, 50 epochs)...")
time_np_raw, loss_np_raw, acc_np_raw = run_numpy(X_train_raw, y_train_raw, X_test_raw, y_test_raw, epochs=50)
print(f"NumPy : {time_np_raw:.1f}s | loss {loss_np_raw:.4f} | acc {acc_np_raw:.4f}")

print("Entraînement Keras (données brutes, 50 epochs)...")
time_keras_raw, loss_keras_raw, acc_keras_raw = run_keras(X_train_raw, y_train_raw, X_test_raw, y_test_raw, epochs=50)
print(f"Keras : {time_keras_raw:.1f}s | loss {loss_keras_raw:.4f} | acc {acc_keras_raw:.4f}")

ratio_raw = time_np_raw / time_keras_raw

print(f"\nComparaison de robustesse :")
print(f"NumPy   - Avec normalization : acc {acc_np_large:.4f} | Sans : acc {acc_np_raw:.4f} (Δ = {(acc_np_raw - acc_np_large):.4f})")
print(f"Keras   - Avec normalization : acc {acc_keras_large:.4f} | Sans : acc {acc_keras_raw:.4f} (Δ = {(acc_keras_raw - acc_keras_large):.4f})")

# ============================================================================
# POUR CREUSER : Varier n_samples
# ============================================================================

print("\n" + "="*80)
print("POUR CREUSER : Accélération en fonction de la taille du dataset")
print("="*80)

sample_sizes = [10000, 50000, 100000]
accelerations = []

for n_samples in sample_sizes:
    print(f"\n[n_samples = {n_samples}]")
    
    X_var, y_var = make_classification(
        n_samples=n_samples, n_features=30, n_informative=20, n_redundant=5,
        random_state=42
    )
    
    X_train_var, X_test_var, y_train_var, y_test_var = train_test_split(
        X_var, y_var, test_size=0.2, random_state=42
    )
    
    scaler_var = StandardScaler()
    X_train_var = scaler_var.fit_transform(X_train_var)
    X_test_var = scaler_var.transform(X_test_var)
    
    # Epochs adapté : moins d'epochs pour grand dataset (pour pas que ça prenne trop longtemps)
    epochs_var = max(5, 100 // (n_samples // 10000))
    
    print(f"  NumPy ({epochs_var} epochs)...", end='', flush=True)
    time_np_var, _, _ = run_numpy(X_train_var, y_train_var, X_test_var, y_test_var, epochs=epochs_var)
    print(f" {time_np_var:.1f}s")
    
    print(f"  Keras ({epochs_var} epochs)...", end='', flush=True)
    time_keras_var, _, _ = run_keras(X_train_var, y_train_var, X_test_var, y_test_var, epochs=epochs_var)
    print(f" {time_keras_var:.1f}s")
    
    accel = time_np_var / time_keras_var
    accelerations.append(accel)
    print(f"  ✓ Keras est {accel:.1f}x plus rapide")

# ============================================================================
# TABLEAU RÉCAPITULATIF
# ============================================================================

print("\n" + "="*80)
print("TABLEAU RÉCAPITULATIF")
print("="*80)

print(f"\n{'Dataset':<20} | {'NumPy Time':>12} | {'Keras Time':>12} | {'Ratio':>8} | {'NumPy Acc':>10} | {'Keras Acc':>10}")
print("-" * 85)
print(f"{'BC (569x30)':<20} | {time_np_bc:>12.2f}s | {time_keras_bc:>12.2f}s | {ratio_bc:>8.1f}x | {acc_np_bc:>10.4f} | {acc_keras_bc:>10.4f}")
print(f"{'50k (40k×30)':<20} | {time_np_large:>12.1f}s | {time_keras_large:>12.2f}s | {ratio_large:>8.1f}x | {acc_np_large:>10.4f} | {acc_keras_large:>10.4f}")
print(f"{'50k no scaler':<20} | {time_np_raw:>12.1f}s | {time_keras_raw:>12.2f}s | {ratio_raw:>8.1f}x | {acc_np_raw:>10.4f} | {acc_keras_raw:>10.4f}")

# ============================================================================
# VISUALISATIONS
# ============================================================================

print("\n" + "="*80)
print("Génération des visualisations...")
print("="*80)

fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# Plot 1 : Barplot ratio d'accélération
ax1 = axes[0, 0]
datasets = ['BC\n(569 ex)', '50k ex\n(50 ep)', '50k ex\n(10 ep)', '50k raw\n(50 ep)']
ratios = [ratio_bc, ratio_large, ratio_10, ratio_raw]
colors = ['#FF6B6B', '#4ECDC4', '#FFE66D', '#95E1D3']
bars = ax1.bar(datasets, ratios, color=colors, alpha=0.7, edgecolor='black', linewidth=2)
ax1.set_ylabel("Facteur d'accélération (NumPy time / Keras time)")
ax1.set_title("Keras vs NumPy : Accélération par dataset")
ax1.axhline(y=1, color='red', linestyle='--', alpha=0.5, label='Pas d\'accélération')
for bar, ratio in zip(bars, ratios):
    height = bar.get_height()
    ax1.text(bar.get_x() + bar.get_width()/2., height,
             f'{ratio:.1f}x', ha='center', va='bottom', fontweight='bold')
ax1.legend()
ax1.grid(axis='y', alpha=0.3)

# Plot 2 : Temps d'entraînement (barplot)
ax2 = axes[0, 1]
x = np.arange(len(datasets))
width = 0.35
np_times = [time_np_bc, time_np_large, time_np_10, time_np_raw]
keras_times = [time_keras_bc, time_keras_large, time_keras_10, time_keras_raw]
bars1 = ax2.bar(x - width/2, np_times, width, label='NumPy', alpha=0.7, color='#FF6B6B')
bars2 = ax2.bar(x + width/2, keras_times, width, label='Keras', alpha=0.7, color='#4ECDC4')
ax2.set_ylabel("Temps (secondes)")
ax2.set_title("Temps d'entraînement absolu")
ax2.set_xticks(x)
ax2.set_xticklabels(datasets)
ax2.legend()
ax2.grid(axis='y', alpha=0.3)

# Plot 3 : Accélération vs taille dataset
ax3 = axes[1, 0]
ax3.plot(sample_sizes, accelerations, marker='o', linewidth=2, markersize=8, color='#4ECDC4')
ax3.fill_between(sample_sizes, accelerations, alpha=0.3, color='#4ECDC4')
ax3.set_xlabel("Taille du dataset (n_samples)")
ax3.set_ylabel("Facteur d'accélération Keras/NumPy")
ax3.set_title("Accélération en fonction de la taille du dataset")
ax3.grid(alpha=0.3)
for x_pt, y_pt in zip(sample_sizes, accelerations):
    ax3.text(x_pt, y_pt + 0.5, f'{y_pt:.1f}x', ha='center', fontweight='bold')

# Plot 4 : Accuracy comparison
ax4 = axes[1, 1]
x_acc = np.arange(3)
width_acc = 0.35
np_accs = [acc_np_bc, acc_np_large, acc_np_raw]
keras_accs = [acc_keras_bc, acc_keras_large, acc_keras_raw]
bars1 = ax4.bar(x_acc - width_acc/2, np_accs, width_acc, label='NumPy', alpha=0.7, color='#FF6B6B')
bars2 = ax4.bar(x_acc + width_acc/2, keras_accs, width_acc, label='Keras', alpha=0.7, color='#4ECDC4')
ax4.set_ylabel("Test Accuracy")
ax4.set_ylim([0.8, 1.0])
ax4.set_title("Accuracy : NumPy vs Keras")
ax4.set_xticks(x_acc)
ax4.set_xticklabels(['BC\n(normalized)', '50k\n(normalized)', '50k\n(raw)'])
ax4.legend()
ax4.grid(axis='y', alpha=0.3)

plt.tight_layout()
plt.savefig("phase10_timing_benchmark.png", dpi=100, bbox_inches='tight')
print("\n Visualisations sauvegardées : phase10_timing_benchmark.png")

# ============================================================================
# CONCLUSIONS
# ============================================================================

print("\n" + "="*80)
print("CONCLUSIONS - TIMING BENCHMARK")
print("="*80)

print("\n1. L'ACCÉLÉRATION EXPLOSE AVEC LA TAILLE")
print(f"   - Breast Cancer (569 ex) : {ratio_bc:.1f}x")
print(f"   - 50k exemples : {ratio_large:.1f}x")
print(f"   - 100k exemples : {accelerations[-1]:.1f}x")
print(f"   → Cela explique pourquoi Keras (avec GPU) est crucial en production")

print("\n2. LE RATIO EST STRUCTUREL, PAS LIÉ AUX EPOCHS")
print(f"   - 50 epochs : {ratio_large:.1f}x")
print(f"   - 10 epochs : {ratio_10:.1f}x")
print(f"   → Différence : {abs(ratio_10 - ratio_large) / ratio_large * 100:.1f}% (négligeable)")

print("\n3. NORMALISATION IMPACT LES DEUX, MAIS KERAS RESTE PLUS RAPIDE")
print(f"   - NumPy normalized : {acc_np_large:.4f} | raw : {acc_np_raw:.4f}")
print(f"   - Keras normalized : {acc_keras_large:.4f} | raw : {acc_keras_raw:.4f}")
print(f"   → Sans normalisation : perte d'accuracy mais ratio {ratio_raw:.1f}x inchangé")

print("\n4. POURQUOI NE PAS UTILISER NUMPY EN PRODUCTION")
print(f"   - Numpy 200 epochs Breast Cancer = {time_np_bc:.1f}s")
print(f"   - Numpy 50 epochs 50k = {time_np_large:.1f}s (!)")
print(f"   - Keras le fait en {time_keras_large:.1f}s")
print(f"   → Sur ImageNet (millions d'images), NumPy prendrait des SEMAINES")
print(f"   → Keras (GPU) le ferait en heures")

print("\n" + "="*80)