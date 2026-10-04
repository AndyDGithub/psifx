# Optimisations RTMO — données mesurées et pistes restantes

Date : 2026-10-07. Branche : `optim/rtmo-fp16-int8` (part de `feature/pose-rtmo`).
Fait suite à `04_modeles_optimisation.md` §2.1/§2.2. Un fichier `05_optimisations_<modèle>.md` existe dans chaque
branche d'optimisation ; le tableau récapitulatif tous modèles sera fait au merge dans `dev`.

## 1. Ce qui a été mesuré

### 1.1 Cadre

- **Outil** : `psifx video pose rtmo` = RTMO-m (preset rtmlib `balanced`), un seul modèle ONNX, **sans détecteur
  séparé** (un étage). rtmlib 0.0.16.
- **Matériel, logiciel, protocole** : identiques à `05_optimisations_rtmpose.md` §1.1 (RTX 4060 Laptop, CPU
  AVX-VNNI, `onnxruntime-gpu` 1.20.2 ; vitesse par `time_psifx_pose.py`, précision par la CLI psifx sur 50 images
  COCO). Résultats bruts : `benchmark/results/psifx_cli/optim-rtmo-fp16-int8.csv`.
- **Optimisations testées** (option `--precision`) : `fp16` (conversion onnxruntime, I/O float32) et `int8`
  (statique QDQ S8S8, poids par canal, `Conv`/`MatMul` seulement, calibrée sur 32 images de
  `tests/integration/data/Multi.mp4`). `torch.compile` sans objet (modèle ONNX).

### 1.2 Résultats (50 images COCO)

| Appareil | Précision | ms/image | FPS | vs fp32 même appareil | OKS | PCK | Détection | Temps CLI (s) |
|---|---|---|---|---|---|---|---|---|
| CPU | fp32 | 149.1 | 6.7 | — | 0.826 | 0.880 | 92 % | 14.0 |
| CPU | int8 | 255.6 | 3.9 | −42 % | 0.584 | 0.680 | 72 % | 33.6 |
| GPU | fp32 | 13.0 | 76.8 | — | 0.826 | 0.880 | 92 % | 7.7 |
| GPU | **fp16** | **9.1** | **109.6** | **+43 %** | 0.826 | 0.880 | 92 % | 8.5 |
| GPU | int8 | 17.7 | 56.6 | −26 % | 0.580 | 0.662 | 70 % | 15.7 |

Valeurs INT8 = deuxième passe, après les corrections de quantification (première passe : 258.2 / 18.1 ms, même
précision). Repère : sur l'extrait de test de 8 images, RTMO fp32 fait OKS 0.943 / PCK 1.000.

### 1.3 Ce qu'on en retient

- **RTMO fp16 sur GPU = le meilleur résultat de toutes les branches d'optimisation** : +43 %, 110 FPS, aucune perte
  de précision. RTMO étant entièrement convolutif et sans détecteur ni prétraitement lourd hors du modèle,
  l'optimisation porte sur presque tout le temps de calcul (contrairement à ViTPose, où le modèle ne pèse que 25 %).
- **RTMO est déjà le plus rapide sans optimisation** (13 ms/image fp32 GPU, 149 ms CPU contre 274 pour RTMPose).
- **INT8 : à écarter**, mêmes causes que RTMPose (Q/DQ autour des SiLU, pas de noyaux INT8 côté provider CUDA) et
  perte encore plus forte : −24 points d'OKS, **−20 points de détection** (le score de présence de RTMO souffre de
  la quantification).
- Rappel limite connue (benchmark précédent) : taux de détection plus bas que RTMPose (92 % vs 98 %).

### 1.4 Problèmes rencontrés (corrigés sur `dev`)

1. `quant_pre_process` échouait (« Incomplete symbolic shape inference ») : l'inférence de formes symbolique vise les
   transformeurs ; désactivée au profit de l'inférence ONNX standard, comme le recommande la doc.
2. Mêmes corrections communes que RTMPose (opset 11 → 13, passage par fichier, `Conv`/`MatMul` seulement,
   erreur explicite si CUDA retombe sur CPU) — cf. `05_optimisations_rtmpose.md` §1.4.

## 2. Optimisations pas encore testées

| Piste | Pourquoi pour RTMO | Coût | Priorité |
|---|---|---|---|
| **TensorRT Execution Provider fp16** | Le modèle est presque tout le temps de calcul : c'est le candidat où une compilation TensorRT (fusions Conv+SiLU, noyaux fp16 optimisés) a le plus de chances de se voir de bout en bout. Cache de moteur pour ne compiler qu'une fois. | Moyen (TensorRT compatible avec onnxruntime-gpu 1.20/CUDA 12) | **Haute** |
| **TensorRT INT8** | Seul chemin où l'INT8 peut accélérer sur GPU (Tensor Cores INT8 d'Ada) ; la perte de précision mesurée (−24 pts) imposera une calibration soignée et probablement d'exclure la tête de détection. | Moyen-élevé | Moyenne |
| **Batch de plusieurs images vidéo** | L'ONNX de RTMO a une dimension `batch` dynamique ; traiter 4-8 images par appel amortit le coût par appel sur GPU. Nécessite de modifier `predict_video` (la boucle actuelle est image par image). | Moyen | Haute pour la vidéo longue |
| **IOBinding + CUDA graphs** | Entrée de taille fixe (640×640) : condition des CUDA graphs ; supprime copies CPU↔GPU et coût de lancement. | Faible-moyen | Moyenne |
| **Preset `lightweight` (RTMO-s)** / `performance` (RTMO-l) | Déjà exposés par `--mode` ; situer le compromis précision/vitesse de chaque taille. | Très faible | Moyenne (mesure rapide) |
| **Prétraitement sur GPU** | Letterbox + normalisation faits en numpy/OpenCV sur CPU par rtmlib ; à 9 ms/image de modèle, ce coût CPU devient visible. | Moyen | Moyenne (après profilage) |
| INT8 mieux calibré (entropie/percentile, couches sensibles exclues) | Réduirait la perte, sans régler la lenteur hors TensorRT. | Faible | Basse |
| Distillation / élagage | Cf. `04_*` §2.3/§2.4 ; RTMO-m est déjà un modèle temps réel. | Élevé | Basse |

## Sources

- [ONNX Runtime — Quantize ONNX Models](https://www.onnxruntime.ai/docs/performance/model-optimizations/quantization.html) (statique pour les CNN, prétraitement : inférence ONNX pour les modèles non transformeurs)
- [ONNX Runtime — Float16 and Mixed Precision Models](https://onnxruntime.ai/docs/performance/model-optimizations/float16.html)
- [ONNX Runtime — TensorRT Execution Provider](https://onnxruntime.ai/docs/execution-providers/TensorRT-ExecutionProvider.html)
- [ONNX Runtime — CUDA Execution Provider](https://www.onnxruntime.ai/docs/execution-providers/CUDA-ExecutionProvider) (CUDA graphs)
- [Optimum — Accelerated inference on NVIDIA GPUs](https://huggingface.co/docs/optimum-onnx/onnxruntime/usage_guides/gpu) (IOBinding)
- Détails de mesure : `benchmark/results/optim-rtmo-fp16-int8.md`
