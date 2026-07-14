# Deep Learning en partant de zéro (10 phases)

Projet d'une journée pour vraiment comprendre comment les réseaux de neurones marchent, en commençant par du NumPy pur, puis en montrant les limites avant d'utiliser Keras.

## Structure

**Phases 1-4 : NumPy seulement**
- phase1_neurone.py : simple forward pass avec sigmoid, test de différents poids
- phase2_gradient.py : premiers pas avec la backprop manuelle
- phase3_xor.py : montrer qu'une couche c'est pas assez, résoudre XOR
- phase4_spirale.py : dataset plus complexe avec 2 couches cachées

**Phases 5-7 : Keras (Python 3.12 installé en parallèle)**
- phase5_keras_mnist.py : transition vers Keras, benchmark batch size
- phase6_activations.py : pourquoi sigmoid et tanh vs relu, éviter softmax en caché
- phase7_learning_rate.py : learning rate trop bas/trop haut, Adam vs SGD

**Phase 8 : Pipeline complet**
- phase8_pipeline_personnel.py : reprendre tout sur un dataset réel (Breast Cancer)
  - scénario normal : NumPy vs Keras côte à côte
  - cas limite : données manquantes (zéros => médiane)
  - adversarial : prédictions sur valeurs extrêmes (risque en production)

**Phase 9 : Ablation study**
- phase9_ablation.py : quoi enlever pour comprendre ce qui compte vraiment
  - grille 3×3 : depth [1,2,3] × width [8,64,256]
  - heatmap accuracy pour voir les rendements décroissants
  - config extrême (depth=5, width=512) et detection d'overfitting

**Phase 10 : Timing benchmark**
- phase10_timing.py : le vrai coût de NumPy vs Keras
  - Breast Cancer (petit dataset) : 4x d'accélération
  - 50k exemples : 30x d'accélération
  - courbe d'accélération en fonction de la taille

## Démarrage rapide

```bash
# Phases 1-4 (Python 3.14.5 OK)
python phase1_neurone.py
python phase2_gradient.py
python phase3_xor.py
python phase4_spirale.py

# Phase 8 (numpy + sklearn, pas de TensorFlow)
python phase8_pipeline_personnel.py

# Phases 5-7, 9-10 (besoin Python 3.12)
pip install tensorflow matplotlib scikit-learn
python phase5_keras_mnist.py
python phase6_activations.py
python phase7_learning_rate.py
python phase9_ablation.py
python phase10_timing.py
```

## Observations clés

- **Phase 1** : preuve que le sigmoid peut apprendre simple, mais loss trop élevée
- **Phase 2** : 50 epochs à peine, et la loss converge. Comment ?
- **Phase 3** : XOR impossible avec 1 couche. 2 couches = boom, 100% accuracy
- **Phase 4** : ReLU + He init change tout. Spiral classify = non-linear à la puissance
- **Phase 8** : Keras meilleure accuracy de ~3% sur Breast Cancer, mais au prix d'une complexité énorme
- **Phase 9** : Sur MNIST, le sweet spot c'est depth=1, width=64. Au-delà, rendements décroissants
- **Phase 10** : Keras 4x plus rapide sur petit dataset, 30x sur 50k, courbe exponentielle

## Trucs bizarres rencontrés

- Learning rate = 1.0 → explose, loss oscille autour de 2.3 (entropie croisée uniforme)
- Modèles hyper-confiants sur données hors distribution → piège en production
- Zéros suspects dans les colonnes (ex: données Pima) → remplacer par médiane aide
- Plus de paramètres ≠ meilleur résultat : la phase 9 le prouve avec heatmap
- Le ratio d'accélération Keras/NumPy est structurel : 10 epochs ou 50 epochs, même ratio

## Stack

- NumPy 2.4.6 (calcul)
- Matplotlib 3.11.0 (plots)
- TensorFlow / Keras (phases 5+)
- scikit-learn (datasets, preprocessing)
- Python 3.14.5 (phases 1-4, 8) et Python 3.12 (phases 5-7, 9-10)
