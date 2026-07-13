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

print("="*80)
print("PHASE 6 : Comparaison des fonctions d'activation et de la profondeur")
print("="*80)

# ============ SCÉNARIO NORMAL : 3 activations ============
print("\n" + "="*80)
print("SCÉNARIO NORMAL : sigmoid, tanh, relu (architecture 128-64)")
print("="*80)

activations = ['sigmoid', 'tanh', 'relu']
results = []
histories = {}

for activation in activations:
    print(f"\nEntraînement avec activation={activation}...")
    
    tf.random.set_seed(42)
    model = keras.Sequential([
        keras.layers.Dense(128, activation=activation, input_shape=(784,)),
        keras.layers.Dense(64, activation=activation),
        keras.layers.Dense(10, activation='softmax')
    ])
    
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=0.001),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    
    start = time.time()
    history = model.fit(
        X_train, y_train,
        epochs=10,
        batch_size=64,
        validation_split=0.1,
        verbose=0
    )
    elapsed = time.time() - start
    
    test_loss, test_acc = model.evaluate(X_test, y_test, verbose=0)
    
    val_losses = history.history['val_loss']
    convergence_epoch = "N/A"
    for epoch, val_loss in enumerate(val_losses):
        if val_loss < 0.1:
            convergence_epoch = f"epoch {epoch+1}"
            break
    
    results.append({
        'activation': activation,
        'val_loss_final': val_losses[-1],
        'test_accuracy': test_acc,
        'convergence_epoch_sub01': convergence_epoch,
        'train_time_s': elapsed
    })
    
    histories[activation] = val_losses
    
    print(f"  Val loss finale : {val_losses[-1]:.4f}")
    print(f"  Test accuracy : {test_acc:.4f}")
    print(f"  Convergence : {convergence_epoch}")
    print(f"  Temps : {elapsed:.1f}s")

# ============ CAS LIMITE : Pas d'activation (linear) ============
print("\n" + "="*80)
print("CAS LIMITE : Sans activation (linéaire)")
print("="*80)

tf.random.set_seed(42)
model_linear = keras.Sequential([
    keras.layers.Dense(128, input_shape=(784,)),
    keras.layers.Dense(64),
    keras.layers.Dense(10, activation='softmax')
])

model_linear.compile(
    optimizer=keras.optimizers.Adam(learning_rate=0.001),
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

start = time.time()
history_linear = model_linear.fit(
    X_train, y_train,
    epochs=10,
    batch_size=64,
    validation_split=0.1,
    verbose=0
)
elapsed_linear = time.time() - start

test_loss_linear, test_acc_linear = model_linear.evaluate(X_test, y_test, verbose=0)
val_losses_linear = history_linear.history['val_loss']

convergence_linear = "N/A"
for epoch, val_loss in enumerate(val_losses_linear):
    if val_loss < 0.1:
        convergence_linear = f"epoch {epoch+1}"
        break

results.append({
    'activation': 'linear',
    'val_loss_final': val_losses_linear[-1],
    'test_accuracy': test_acc_linear,
    'convergence_epoch_sub01': convergence_linear,
    'train_time_s': elapsed_linear
})

histories['linear'] = val_losses_linear

print(f"Val loss finale : {val_losses_linear[-1]:.4f}")
print(f"Test accuracy : {test_acc_linear:.4f}")
print(f"Convergence : {convergence_linear}")
print(f"Temps : {elapsed_linear:.1f}s")
print(f"(Observation : sans activation, la val_loss reste très élevée car les couches restent linéaires)")

# ============ SCÉNARIO ADVERSARIAL : softmax en couches cachées ============
print("\n" + "="*80)
print("SCÉNARIO ADVERSARIAL : softmax en couches cachées (mauvaise pratique)")
print("="*80)

tf.random.set_seed(42)
model_softmax = keras.Sequential([
    keras.layers.Dense(128, activation='softmax', input_shape=(784,)),
    keras.layers.Dense(64, activation='softmax'),
    keras.layers.Dense(10, activation='softmax')
])

model_softmax.compile(
    optimizer=keras.optimizers.Adam(learning_rate=0.001),
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

start = time.time()
history_softmax = model_softmax.fit(
    X_train, y_train,
    epochs=10,
    batch_size=64,
    validation_split=0.1,
    verbose=0
)
elapsed_softmax = time.time() - start

test_loss_softmax, test_acc_softmax = model_softmax.evaluate(X_test, y_test, verbose=0)
val_losses_softmax = history_softmax.history['val_loss']

convergence_softmax = "N/A"
for epoch, val_loss in enumerate(val_losses_softmax):
    if val_loss < 0.1:
        convergence_softmax = f"epoch {epoch+1}"
        break

results.append({
    'activation': 'softmax_hidden',
    'val_loss_final': val_losses_softmax[-1],
    'test_accuracy': test_acc_softmax,
    'convergence_epoch_sub01': convergence_softmax,
    'train_time_s': elapsed_softmax
})

histories['softmax_hidden'] = val_losses_softmax

print(f"Val loss finale : {val_losses_softmax[-1]:.4f}")
print(f"Test accuracy : {test_acc_softmax:.4f}")
print(f"Convergence : {convergence_softmax}")
print(f"Temps : {elapsed_softmax:.1f}s")
print(f"(Observation : softmax en couches cachées distribue trop l'info, apprentissage dégradé)")

# ============ POUR ALLER PLUS LOIN : Profondeurs différentes avec ReLU ============
print("\n" + "="*80)
print("POUR ALLER PLUS LOIN : Profondeur avec ReLU (meilleure activation)")
print("="*80)

depth_configs = {
    'Peu profond': [
        keras.layers.Dense(256, activation='relu', input_shape=(784,)),
        keras.layers.Dense(10, activation='softmax')
    ],
    'Moyen (baseline)': [
        keras.layers.Dense(128, activation='relu', input_shape=(784,)),
        keras.layers.Dense(64, activation='relu'),
        keras.layers.Dense(10, activation='softmax')
    ],
    'Plus profond': [
        keras.layers.Dense(128, activation='relu', input_shape=(784,)),
        keras.layers.Dense(64, activation='relu'),
        keras.layers.Dense(32, activation='relu'),
        keras.layers.Dense(10, activation='softmax')
    ]
}

for config_name, layers in depth_configs.items():
    print(f"\n{config_name}...")
    
    tf.random.set_seed(42)
    model_depth = keras.Sequential(layers)
    model_depth.compile(
        optimizer=keras.optimizers.Adam(learning_rate=0.001),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    
    total_params = model_depth.count_params()
    
    start = time.time()
    history_depth = model_depth.fit(
        X_train, y_train,
        epochs=10,
        batch_size=64,
        validation_split=0.1,
        verbose=0
    )
    elapsed_depth = time.time() - start
    
    test_loss_depth, test_acc_depth = model_depth.evaluate(X_test, y_test, verbose=0)
    val_losses_depth = history_depth.history['val_loss']
    
    convergence_depth = "N/A"
    for epoch, val_loss in enumerate(val_losses_depth):
        if val_loss < 0.1:
            convergence_depth = f"epoch {epoch+1}"
            break
    
    results.append({
        'activation': config_name,
        'val_loss_final': val_losses_depth[-1],
        'test_accuracy': test_acc_depth,
        'convergence_epoch_sub01': convergence_depth,
        'train_time_s': elapsed_depth
    })
    
    print(f"  Paramètres : {total_params:,}")
    print(f"  Val loss finale : {val_losses_depth[-1]:.4f}")
    print(f"  Test accuracy : {test_acc_depth:.4f}")
    print(f"  Convergence : {convergence_depth}")
    print(f"  Temps : {elapsed_depth:.1f}s")

# ============ TABLEAU RÉCAPITULATIF ============
print("\n" + "="*80)
print("TABLEAU RÉCAPITULATIF COMPLET")
print("="*80)
print(f"{'Activation/Profondeur':<25} | {'Val loss final':>12} | {'Test Accuracy':>14} | {'Epoch <0.1':>10} | {'Temps':>8}")
print("-" * 85)

for r in results:
    print(f"{r['activation']:<25} | {r['val_loss_final']:>12.4f} | {r['test_accuracy']:>14.4f} | {str(r['convergence_epoch_sub01']):>10s} | {r['train_time_s']:>8.1f}s")

# ============ VISUALISATION ============
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Courbe de convergence pour les 3 activations
for activation in activations:
    axes[0].plot(range(1, 11), histories[activation], label=activation, linewidth=2)

axes[0].axhline(y=0.1, color='red', linestyle='--', label='Seuil 0.1 loss', alpha=0.5)
axes[0].set_xlabel("Epoch")
axes[0].set_ylabel("Val Loss")
axes[0].set_title("Convergence selon l'activation (sigmoid, tanh, relu)")
axes[0].legend()
axes[0].grid(alpha=0.3)

# Courbe adversarial : cas limite + adversarial
axes[1].plot(range(1, 11), histories['relu'], label='ReLU (normal)', linewidth=2)
axes[1].plot(range(1, 11), histories['linear'], label='Linear (cas limite)', alpha=0.7)
axes[1].plot(range(1, 11), histories['softmax_hidden'], label='Softmax hidden (adversarial)', alpha=0.7)
axes[1].axhline(y=0.1, color='red', linestyle='--', label='Seuil 0.1 loss', alpha=0.5)
axes[1].set_xlabel("Epoch")
axes[1].set_ylabel("Val Loss")
axes[1].set_title("Normal vs Cas limite vs Adversarial")
axes[1].legend()
axes[1].grid(alpha=0.3)

plt.tight_layout()
plt.savefig("phase6_activations_curve.png", dpi=100, bbox_inches='tight')
print("\nCourbes sauvegardées : phase6_activations_curve.png")

print("\n" + "="*80)
print("CONCLUSIONS")
print("="*80)
print("1. ReLU converge plus vite que sigmoid et tanh")
print("2. Sans activation (linear), la val_loss reste élevée : besoin d'activations non-linéaires")
print("3. Softmax en couches cachées dégrade les performances : softmax est réservé à la sortie")
print("4. Profondeur vs Largeur : vérifie si plus profond = mieux sur MNIST")
print("="*80)
