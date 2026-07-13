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

print(f"Train : {X_train.shape} | Test : {X_test.shape}")
print(f"Classes uniques : {np.unique(y_train)}\n")

# ============ SCÉNARIO NORMAL : epochs=5, batch_size=64 ============
print("="*60)
print("SCÉNARIO NORMAL : epochs=5, batch_size=64")
print("="*60)

model = keras.Sequential([
    keras.layers.Dense(128, activation='relu', input_shape=(784,)),
    keras.layers.Dense(64, activation='relu'),
    keras.layers.Dense(10, activation='softmax')
])

model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

start = time.time()
history = model.fit(
    X_train, y_train,
    epochs=5,
    batch_size=64,
    validation_split=0.1,
    verbose=1
)
elapsed = time.time() - start

test_loss, test_acc = model.evaluate(X_test, y_test, verbose=0)

print(f"\nTemps d'entraînement : {elapsed:.1f}s")
print(f"Test accuracy : {test_acc:.4f}")
print(f"Test loss : {test_loss:.4f}")

fig, axes = plt.subplots(1, 2, figsize=(12, 4))
axes[0].plot(history.history['loss'], label='train')
axes[0].plot(history.history['val_loss'], label='val')
axes[0].set_title("Loss")
axes[0].set_xlabel("Epoch")
axes[0].legend()
axes[1].plot(history.history['accuracy'], label='train')
axes[1].plot(history.history['val_accuracy'], label='val')
axes[1].set_title("Accuracy")
axes[1].set_xlabel("Epoch")
axes[1].legend()
plt.savefig("phase5_mnist_curves.png", dpi=100, bbox_inches='tight')
print("Courbes sauvegardées : phase5_mnist_curves.png")

# ============ CAS LIMITE : epochs=0 ============
print("\n" + "="*60)
print("CAS LIMITE : epochs=0")
print("="*60)

model2 = keras.Sequential([
    keras.layers.Dense(128, activation='relu', input_shape=(784,)),
    keras.layers.Dense(64, activation='relu'),
    keras.layers.Dense(10, activation='softmax')
])

model2.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

try:
    history_zero = model2.fit(
        X_train, y_train,
        epochs=0,
        batch_size=64,
        validation_split=0.1,
        verbose=1
    )
    print(f"Historique : {history_zero.history}")
except ValueError as e:
    print(f"Erreur levée (ValueError) : {e}")
except Exception as e:
    print(f"Erreur levée ({type(e).__name__}) : {e}")

# ============ SCÉNARIO ADVERSARIAL : batch_size=1 (SGD pur) ============
print("\n" + "="*60)
print("SCÉNARIO ADVERSARIAL : batch_size=1 (SGD pur)")
print("="*60)

model3 = keras.Sequential([
    keras.layers.Dense(128, activation='relu', input_shape=(784,)),
    keras.layers.Dense(64, activation='relu'),
    keras.layers.Dense(10, activation='softmax')
])

model3.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

print("Entraînement avec batch_size=1 (peut être très lent)...")
start = time.time()
history_bs1 = model3.fit(
    X_train, y_train,
    epochs=5,
    batch_size=1,
    validation_split=0.1,
    verbose=0
)
elapsed_bs1 = time.time() - start

test_loss_bs1, test_acc_bs1 = model3.evaluate(X_test, y_test, verbose=0)

print(f"Temps d'entraînement (batch_size=1) : {elapsed_bs1:.1f}s")
print(f"Temps d'entraînement (batch_size=64) : {elapsed:.1f}s")
speedup = elapsed_bs1 / elapsed
print(f"Ratio (speedup) : {speedup:.1f}x plus lent avec batch_size=1 !")
print(f"Test accuracy (batch_size=1) : {test_acc_bs1:.4f}")
print(f"Test accuracy (batch_size=64) : {test_acc:.4f}")
print(f"Instabilité : courbe de loss batch_size=1 est très irrégulière (pics fréquents)")

fig, axes = plt.subplots(1, 2, figsize=(12, 4))
axes[0].plot(history.history['loss'], label='batch_size=64', linewidth=2)
axes[0].plot(history_bs1.history['loss'], label='batch_size=1 (SGD pur)', alpha=0.7)
axes[0].set_title("Loss : stabilité batch_size=64 vs instabilité SGD pur")
axes[0].set_xlabel("Epoch")
axes[0].legend()
axes[1].plot(history.history['val_accuracy'], label='batch_size=64', linewidth=2)
axes[1].plot(history_bs1.history['val_accuracy'], label='batch_size=1', alpha=0.7)
axes[1].set_title("Validation Accuracy")
axes[1].set_xlabel("Epoch")
axes[1].legend()
plt.savefig("phase5_batch_comparison.png", dpi=100, bbox_inches='tight')
print("Comparaison sauvegardée : phase5_batch_comparison.png")

# ============ COMPARAISON ARCHITECTURES ============
print("\n" + "="*60)
print("COMPARAISON ARCHITECTURES (Pour aller plus loin)")
print("="*60)

architectures = {
    'A (128-64)': [
        keras.layers.Dense(128, activation='relu', input_shape=(784,)),
        keras.layers.Dense(64, activation='relu'),
        keras.layers.Dense(10, activation='softmax')
    ],
    'B (256-128-64) - Plus profonde': [
        keras.layers.Dense(256, activation='relu', input_shape=(784,)),
        keras.layers.Dense(128, activation='relu'),
        keras.layers.Dense(64, activation='relu'),
        keras.layers.Dense(10, activation='softmax')
    ],
    'C (512) - Plus large, moins profonde': [
        keras.layers.Dense(512, activation='relu', input_shape=(784,)),
        keras.layers.Dense(10, activation='softmax')
    ]
}

results = []

for arch_name, layers in architectures.items():
    print(f"\n{arch_name}")
    print("-" * 50)
    
    model_arch = keras.Sequential(layers)
    model_arch.compile(
        optimizer='adam',
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    
    total_params = model_arch.count_params()
    print(f"Paramètres totaux : {total_params:,}")
    
    start = time.time()
    history_arch = model_arch.fit(
        X_train, y_train,
        epochs=5,
        batch_size=64,
        validation_split=0.1,
        verbose=0
    )
    elapsed_arch = time.time() - start
    
    val_acc = history_arch.history['val_accuracy'][-1]
    test_loss_arch, test_acc_arch = model_arch.evaluate(X_test, y_test, verbose=0)
    
    results.append({
        'Architecture': arch_name,
        'Params': total_params,
        'Val Accuracy': val_acc,
        'Test Accuracy': test_acc_arch,
        'Time (s)': elapsed_arch
    })
    
    print(f"Val Accuracy (5 epochs) : {val_acc:.4f}")
    print(f"Test Accuracy : {test_acc_arch:.4f}")
    print(f"Temps d'entraînement : {elapsed_arch:.1f}s")

print("\n" + "="*60)
print("TABLEAU RÉCAPITULATIF")
print("="*60)
print(f"{'Architecture':<35} {'Params':>10} {'Val Acc':>10} {'Test Acc':>10} {'Time':>8}")
print("-" * 75)
for r in results:
    print(f"{r['Architecture']:<35} {r['Params']:>10,} {r['Val Accuracy']:>10.4f} {r['Test Accuracy']:>10.4f} {r['Time']:>8.1f}s")

best = max(results, key=lambda x: x['Test Accuracy'])
print(f"\n✓ Meilleure : {best['Architecture']} avec {best['Test Accuracy']:.4f} d'accuracy")
