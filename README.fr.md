# Détection de la Cataracte — HOG+SVM vs YOLOv5

**Deux approches pour détecter la cataracte à partir d'images d'yeux — descripteurs classiques vs deep learning**
*Mini Projet · Vidéo Biomédicale · Master BIAM 2025–2026 · FSDM*

> 🇬🇧 [Readme in English](README.md)

---

Dépistage non invasif de la cataracte à partir de photographies d'yeux : étant donné une image d'œil, détecter la région de l'œil et la classer **saine (sain)** ou **cataracte**, avec deux pipelines complémentaires entraînés sur le même jeu de données.

## Provenance des données

1. **Source brute** : [Eye Detection Dataset (Kaggle, icebearogo)](https://www.kaggle.com/datasets/icebearogo/eye-detection-dataset) — ~2000 images de la région oculaire annotées au format YOLO (~14 % labellisées comme classe pathologique).
2. **Curation dans [Roboflow](https://universe.roboflow.com/jaouads-workspace/cataract-eye-detection)** : relecture et nettoyage des annotations, **normalisation des IDs de classes** (les données brutes utilisaient `0 = cataracte` ; remappé en `0 = sain`, `1 = cataracte` — vérifié visuellement sur des patches échantillonnés), découpage train/valid/test, et augmentation (**1385 → 4135 images d'entraînement**), exportée en versions v2/v4 du dataset.
3. Les deux méthodes (patches HOG+SVM et images complètes YOLOv5) ont été entraînées sur cette version curatée.

> Note : une version antérieure du notebook HOG+SVM avait `CLASS_NAMES` inversé (`cataracte`↔`sain`). Les métriques ne sont pas affectées ; les libellés d'affichage ont été corrigés après vérification visuelle d'échantillons.

## Dataset

[cataract-eye-detection v2 (Roboflow)](https://universe.roboflow.com/jaouads-workspace/cataract-eye-detection) — format YOLO, 2 classes (`sain` / `cataracte`), splits train/valid/test.

![Annotations du dataset — vert = sain, rouge = cataracte](figures/dataset_annotations_yolo.png)

## Méthode 1 — HOG + SVM (classique)

Extraction de patches 64×128 depuis les annotations YOLO → Histogram of Oriented Gradients (3780-d) → StandardScaler + LinearSVC.

L'intuition est visible dans le descripteur lui-même : un œil sain produit des gradients forts et bien alignés ; le cristallin trouble d'une cataracte les efface.

![Descripteur HOG : patch sain vs cataracte](figures/hog_descriptor_visualization.png)

| Métrique | Valeur |
|---|---|
| Accuracy (test) | **89,90 %** |
| Précision / Rappel (weighted) | 89,90 % / 89,90 % |
| F1 (weighted / macro) | 0,899 / 0,827 |
| Sensibilité / Spécificité | 71,4 % / 93,9 % |
| Temps d'entraînement | **4,9 s** (CPU) |
| Inférence | **0,03 ms/image** |

Matrice de confusion (198 patches test) : VP 153 · FN 10 · FP 10 · VN 25

![Matrice de confusion HOG+SVM](figures/confusion_matrix_hog_svm.png)

## Méthode 2 — YOLOv5 (deep learning)

Fine-tuning de `yolov5s.pt` (transfer learning COCO) sur les images complètes, 50 epochs, batch 16, GPU T4, early stopping à l'epoch 29.

![Courbes d'entraînement YOLOv5](figures/training_curves_yolo.png)

| Métrique | Valeur |
|---|---|
| mAP@0.5 | **96,55 %** |
| mAP@0.5:0.95 | 80,26 % |
| Précision (meilleure epoch) | 94,50 % |
| Rappel (meilleure epoch) | 90,89 % |
| Temps d'entraînement | 52,5 min (GPU T4) |
| Inférence | ~6,5 ms/image |

![Détections YOLOv5 avec scores de confiance](figures/yolo_detections.png)

## Comparaison directe

![Comparaison HOG+SVM vs YOLOv5](figures/comparison_hog_vs_yolo.png)

| | HOG + SVM | YOLOv5 |
|---|---|---|
| Précision | 93,53 % | **94,50 %** |
| Rappel | 93,43 % | 90,89 % |
| Métriques supp. | F1 0,935 | mAP@0.5 96,6 % |
| Entraînement | **52 s** (CPU) | 52 min (GPU) |
| Interprétabilité | ✅ descripteur visualisable | ❌ boîte noire |
| Score de confiance par boîte | ❌ | ✅ natif |

**Conclusion :** pour un système clinique de dépistage en temps réel, YOLOv5 est la méthode recommandée — meilleure précision, inférence temps réel (~150 FPS) et score de confiance par boîte utile au praticien. HOG+SVM reste une excellente baseline interprétable qui s'entraîne en quelques secondes sans GPU.

## Notebooks (exécutés, résultats inclus)

| Notebook | Description |
|---|---|
| [`notebooks/hog-svm-cataract-roboflow.ipynb`](notebooks/hog-svm-cataract-roboflow.ipynb) | Pipeline classique complet : patches → HOG → SVM → évaluation |
| [`notebooks/yolo-cataract-detection.ipynb`](notebooks/yolo-cataract-detection.ipynb) | Fine-tuning YOLOv5 + évaluation + comparaison |

Aussi consultables sur Kaggle :
- [hog_svm_cataract_roboflow](https://www.kaggle.com/code/jaouadelmorabit/hog-svm-cataract-roboflow)
- [yolo_cataract_detection](https://www.kaggle.com/code/jaouadelmorabit/yolo-cataract-detection)

## Reproduire

```bash
pip install roboflow scikit-image scikit-learn opencv-python matplotlib
jupyter notebook notebooks/hog-svm-cataract-roboflow.ipynb   # CPU suffit
```

Le notebook YOLO attend un runtime GPU Kaggle/Colab et clone ultralytics/yolov5 lui-même.
Configurer la clé Roboflow via les Secrets Kaggle (`ROBOFLOW_API_KEY`) — ne jamais la coder en dur.

## Avertissement médical

Projet de recherche/pédagogique — **pas un dispositif médical**, pas pour la prise de décision clinique.

## Auteur

**Jaouad El Morabit** — Master BIAM 2025–2026, Imagerie Biomédicale
Voir aussi : [video-object-tracking-portfolio](https://github.com/jawadelMorabit-smi/video-object-tracking-portfolio) · [radiogenomics-analytics-framework](https://github.com/jawadelMorabit-smi/radiogenomics-analytics-framework)
