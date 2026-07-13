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
print("PHASE 7 : Impact du learning rate et des optimizers")
print("="*80)

# ============ SCÉNARIO NORMAL : 3 learning rates ============
print("\n" + "="*80)
print("SCÉNARIO NORMAL + CAS LIMITE + ADVERSARIAL : Learning rates")
print("="*80)
learning_rates = [1e-7, 1e-3, 1.0]
lr_labels = ['trop petit (1e-7)', 'sweet spot (1e-3)', 'trop grand (1.0)']
results = []
histories = {}

for lr, label in zip(learning_rates, lr_labels):
    print(f"\nEntraînement avec lr={lr:.0e} ({label})...")
    
    tf.random.set_seed(42)
    model = keras.Sequential([
        keras.layers.Dense(128, activation='relu', input_shape=(784,)),
        keras.layers.Dense(64, activation='relu'),
        keras.layers.Dense(10, activation='softmax')
    ])
    
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=lr),
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
    delta_loss = val_losses[0] - val_losses[-1]
    
    results.append({
        'lr': lr,
        'label': label,
        'val_loss_final': val_losses[-1],
        'test_accuracy': test_acc,
        'train_time_s': elapsed,
        'delta_loss': delta_loss
    })
    
    histories[label] = val_losses
    
    print(f"  Val loss : {val_losses[0]:.4f} → {val_losses[-1]:.4f} (Δ = {delta_loss:.4f})")
    print(f"  Test accuracy : {test_acc:.4f}")
    print(f"  Temps : {elapsed:.1f}s")

# ============ POUR ALLER PLUS LOIN : Adam vs SGD ============
print("\n" + "="*80)
print("POUR ALLER PLUS LOIN : Adam vs SGD")
print("="*80)

optimizers_to_test = [
    ('Adam lr=1e-3', keras.optimizers.Adam(learning_rate=1e-3)),
    ('SGD lr=1e-3', keras.optimizers.SGD(learning_rate=1e-3)),
    ('SGD lr=1e-2', keras.optimizers.SGD(learning_rate=1e-2))
]

for opt_name, optimizer in optimizers_to_test:
    print(f"\nEntraînement avec {opt_name}...")
    
    tf.random.set_seed(42)
    model_opt = keras.Sequential([
        keras.layers.Dense(128, activation='relu', input_shape=(784,)),
        keras.layers.Dense(64, activation='relu'),
        keras.layers.Dense(10, activation='softmax')
    ])
    
    model_opt.compile(
        optimizer=optimizer,
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    
    start = time.time()
    history_opt = model_opt.fit(
        X_train, y_train,
        epochs=10,
        batch_size=64,
        validation_split=0.1,
        verbose=0
    )
    elapsed_opt = time.time() - start
    
    test_loss_opt, test_acc_opt = model_opt.evaluate(X_test, y_test, verbose=0)
    val_losses_opt = history_opt.history['val_loss']
    
    convergence_epoch = "N/A"
    for epoch, val_loss in enumerate(val_losses_opt):
        if val_loss < 0.1:
            convergence_epoch = epoch + 1
            break
    
    results.append({
        'lr': None,
        'label': opt_name,
        'val_loss_final': val_losses_opt[-1],
        'test_accuracy': test_acc_opt,
        'train_time_s': elapsed_opt,
        'convergence_epoch': convergence_epoch
    })
    
    histories[opt_name] = val_losses_opt
    
    print(f"  Val loss : {val_losses_opt[0]:.4f} → {val_losses_opt[-1]:.4f}")
    print(f"  Test accuracy : {test_acc_opt:.4f}")
    print(f"  Convergence (<0.1) : epoch {convergence_epoch}")
    print(f"  Temps : {elapsed_opt:.1f}s")

# ============ TABLEAU RÉCAPITULATIF ============
print("\n" + "="*80)
print("TABLEAU RÉCAPITULATIF")
print("="*80)
print(f"{'Configuration':<25} | {'Val loss final':>12} | {'Test Accuracy':>14} | {'Temps':>8}")
print("-" * 70)

for r in results:
    config_name = r.get('label', 'N/A')
    print(f"{config_name:<25} | {r['val_loss_final']:>12.4f} | {r['test_accuracy']:>14.4f} | {r['train_time_s']:>8.1f}s")

# ============ VISUALISATIONS ============
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Graphique 1 : Learning rates en échelle log
ax1 = axes[0]
for label in lr_labels:
    ax1.plot(range(1, 11), histories[label], label=label, linewidth=2, marker='o')
ax1.set_xlabel("Epoch")
ax1.set_ylabel("Val Loss")
ax1.set_title("Impact du learning rate (échelle log)")
ax1.set_yscale('log')
ax1.legend()
ax1.grid(alpha=0.3, which='both')

# Graphique 2 : Adam vs SGD
ax2 = axes[1]
for opt_name, _ in optimizers_to_test:
    ax2.plot(range(1, 11), histories[opt_name], label=opt_name, linewidth=2, marker='o')
ax2.set_xlabel("Epoch")
ax2.set_ylabel("Val Loss")
ax2.set_title("Adam vs SGD")
ax2.legend()
ax2.grid(alpha=0.3)

plt.tight_layout()
plt.savefig("phase7_lr_curve.png", dpi=100, bbox_inches='tight')
print("\nCourbes sauvegardées : phase7_lr_curve.png")

# ============ CONCLUSIONS ============
print("\n" + "="*80)
print("CONCLUSIONS")
print("="*80)
print("\n1. Learning rate trop petit (1e-7):")
print("   → Convergence extrêmement lente, loss décroît à peine")
print("   → En pratique : inutilisable (nécessiterait des milliers d'epochs)")

print("\n2. Learning rate sweet spot (1e-3 avec Adam):")
print("   → Courbe lisse et convergence rapide")
print("   → Baseline : c'est ce qu'il faut viser")

print("\n3. Learning rate trop grand (1.0):")
print("   → Loss oscille ou explose autour de 2.3 (entropie croisée uniforme)")
print("   → Le réseau n'apprend rien : saute par-dessus les minima")

print("\n4. Adam vs SGD:")
print("   → Adam converge généralement plus vite que SGD au même lr")
print("   → SGD peut rattraper si on augmente son learning rate (10x parfois)")
print("   → Adam = meilleur par défaut pour la plupart des problèmes")

print("="*80)