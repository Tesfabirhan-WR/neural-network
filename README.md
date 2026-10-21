# Deep learning et NLP : 7 TP en notebooks

Sept notebooks Jupyter qui font passer d'un premier réseau de neurones de régression à la compression d'un modèle, puis au traitement de texte. Chaque TP suit la même structure : des cellules à lire, des cellules **🎯 Étape** à écrire, des cellules **✅ Contrôle** qui vérifient le résultat, et des questions **✍️** avec une réponse écrite.

## Sommaire

| TP | Notebook | Sujet | Données |
|---|---|---|---|
| 1 | `tp1_regression.ipynb` | Régression avec un réseau de neurones : MSE, MAE, RMSE, R², sous- et surapprentissage | Courbe sinus bruitée, puis `load_diabetes` |
| 2 | `tp2_classification_binaire.ipynb` | Classification à 2 classes : sigmoïde, précision, rappel, seuil, courbe ROC | Cancer du sein (569 × 30) |
| 3 | `tp3_multiclasses.ipynb` | Classification à 10 classes : softmax, matrice de confusion, analyse des erreurs | Chiffres 8 × 8 (`load_digits`) |
| 4 | `tp4_cnn.ipynb` | Réseau convolutif (Conv2D, pooling), comparaison avec un réseau dense, robustesse au décalage | Chiffres 8 × 8 |
| 5 | `tp5_regularisation.ipynb` | Surapprentissage, arrêt précoce, dropout, L2, bruit de mesure, augmentation de données | Chiffres 8 × 8 (300 images) |
| 6 | `tp6_hyperparametres_compression.ipynb` | Recherche d'hyperparamètres, validation sur plusieurs graines, élagage, quantification TFLite | Chiffres 8 × 8 |
| 7 | `tp7_texte_nlp.ipynb` | Prétraitement de texte avec NLTK et spaCy, entités nommées, TF-IDF | 2 400 résumés de films (horreur / comédie) |

## Ce que couvre chaque TP

**TP1 : régression.** Réseau à deux couches cachées, sortie sans activation, perte MSE. On lit les courbes d'apprentissage pour savoir quand arrêter, puis on compare des réseaux de 1, 4, 32 et 256 neurones (sous-apprentissage) et un gros réseau sur 15 points (surapprentissage). On termine sur un vrai jeu, `load_diabetes`, face à une régression linéaire.
Résultats obtenus : plateau vers l'epoch 100 ; R² test de 0,59 / 0,86 / 0,93 / 0,94 pour 1 / 4 / 32 / 256 neurones ; sur 15 points, R² de 0,974 à l'apprentissage contre 0,793 au test ; sur le diabète, 0,453 (linéaire) contre 0,463 (réseau).

**TP2 : classification binaire.** Sortie sigmoïde et perte `binary_crossentropy`. On compare à une référence « bête » (toujours la classe majoritaire), on calcule à la main la matrice de confusion, la précision et le rappel, puis on fait varier le seuil de décision (0,5 / 0,3 / 0,1 / 0,05) pour réduire les tumeurs ratées. Le TP se termine par l'AUC et la courbe ROC.

**TP3 : plusieurs classes.** Sortie softmax à 10 neurones et perte `sparse_categorical_crossentropy`. On lit une matrice de confusion 10 × 10 et on regarde les images mal classées.
Résultats obtenus : 0,962 pour la régression logistique, 0,978 pour le réseau dense (10 erreurs sur 450).

**TP4 : CNN.** Remise des images en forme `(8, 8, 1)`, convolution à la main, construction d'un CNN (6 090 paramètres) et comparaison honnête avec le réseau dense (6 570 paramètres). Le test final décale les images d'un pixel pour montrer qu'un modèle peut s'effondrer sur une variation simple.

**TP5 : régularisation.** On provoque le surapprentissage avec 300 images et 85 000 paramètres, puis on compare cinq remèdes. On mesure le bruit d'échantillonnage (`1,96 × √(p(1−p)/n)`) et la variabilité entre graines pour savoir si un gain est réel. La fin du TP utilise l'augmentation de données par décalage.

**TP6 : hyperparamètres et compression.** Exploration du taux d'apprentissage, recherche aléatoire sur 15 configurations (avec Keras Tuner en bonus), validation sur 5 graines, évaluation unique sur le test, puis élagage (0 à 95 %) et quantification TensorFlow Lite.

**TP7 : texte.** Tokenisation, minuscules, mots vides, racines (NLTK) contre lemmes (spaCy), entités nommées et ambiguïtés, pipeline complet, puis TF-IDF et régression logistique. Le TP mesure sur 5 découpages, et avec seulement 200 textes, ce que le prétraitement apporte vraiment.

## Installation

Python 3.10 ou plus récent.

```bash
pip install numpy pandas matplotlib scipy scikit-learn tensorflow
pip install keras-tuner tensorboard            # TP6, étape 4 (bonus, facultatif)
pip install spacy nltk                         # TP7
python -m spacy download en_core_web_sm        # TP7
```

Les jeux de données des TP1 à TP6 sont fournis par scikit-learn : rien d'autre à télécharger. Le TP7 lit le fichier `textes_films_horreur_comedie.csv`, qui doit se trouver **dans le même dossier que le notebook**.

## Comment travailler

1. Ouvrir les notebooks dans l'ordre (1 à 7) : chaque TP réutilise des idées du précédent.
2. Exécuter les cellules de haut en bas, sans sauter d'étape : les contrôles dépendent des variables définies avant.
3. Respecter les noms de variables demandés dans les consignes (`modele`, `resultats`, `acc`…), sinon le contrôle échoue.
4. Après un changement important, redémarrer le noyau et tout ré-exécuter (*Restart & Run All*).
5. Les graines sont fixées à 42, mais les résultats peuvent varier légèrement d'une machine ou d'une version de TensorFlow à l'autre.

## Autheur

Tesfabirhan REDIE 
- Data Engineer Intern
- 21/10/2026
- Bourg-en-Bresse
