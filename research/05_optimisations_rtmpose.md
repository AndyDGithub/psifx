# Optimisations RTMPose — données mesurées et pistes restantes

Date : 2026-10-07. Branche : `optim/rtmpose-fp16-int8` (part de `feature/pose-rtmpose`).
Fait suite à `04_modeles_optimisation.md` §2.1/§2.2. Un fichier `05_optimisations_<modèle>.md` existe dans chaque
branche d'optimisation ; le tableau récapitulatif tous modèles sera fait au merge dans `dev`.

## 1. Ce qui a été mesuré

### 1.1 Cadre

- **Outil** : `psifx video pose rtmpose` = détecteur YOLOX-m + RTMPose-m (preset rtmlib `balanced`), deux modèles
  ONNX exécutés par onnxruntime. rtmlib 0.0.16.
- **Matériel** : RTX 4060 Laptop (8 Go, ~0.9 Go occupés par d'autres applications), CPU avec AVX-VNNI.
- **Logiciel** : `psifx-env`, Python 3.11, `onnxruntime-gpu` 1.20.2 (remplace `onnxruntime` 1.30 CPU), `onnx` 1.17
  (compatible protobuf 4.25.9 exigé par MediaPipe), cuDNN 9.10 via pip (`LD_LIBRARY_PATH`, cf. `benchmark/README.md`).
- **Vitesse** : `benchmark/time_psifx_pose.py`, 50 images COCO letterboxées 640×640, 2 passes d'échauffement puis
  3 passes chronométrées (ms/image en régime établi).
- **Précision** : `benchmark/run_psifx_cli_benchmark.py`, CLI psifx réelle sur les mêmes 50 images (OKS, PCK@0.2,
  taux de détection). Résultats bruts : `benchmark/results/psifx_cli/optim-rtmpose-fp16-int8.csv`.
- **Optimisations testées** (option `--precision`) :
  - `fp16` : conversion des deux modèles avec `onnxruntime.transformers.float16`, entrées/sorties gardées en float32 ;
  - `int8` : quantification statique QDQ S8S8 (recommandée par onnxruntime pour les CNN), poids par canal,
    uniquement `Conv`/`MatMul`, calibrée sur 32 images de `tests/integration/data/Multi.mp4` (pas sur les images
    évaluées).
  - `torch.compile` : sans objet (modèles ONNX, déjà optimisés par onnxruntime).

### 1.2 Résultats (50 images COCO)

| Appareil | Précision | ms/image | FPS | vs fp32 même appareil | OKS | PCK | Détection | Temps CLI (s) |
|---|---|---|---|---|---|---|---|---|
| CPU | fp32 | 274.5 | 3.6 | — | 0.873 | 0.935 | 98 % | 20.6 |
| CPU | int8 | 436.9 | 2.3 | −59 % | 0.647 | 0.824 | 98 % | 49.6 |
| GPU | fp32 | 24.6 | 40.7 | — | 0.873 | 0.935 | 98 % | 8.3 |
| GPU | **fp16** | **22.1** | **45.3** | **+11 %** | 0.873 | 0.935 | 98 % | 10.7 |
| GPU | int8 | 32.3 | 31.0 | −31 % | 0.648 | 0.806 | 96 % | 23.5 |

« Temps CLI » = commande psifx complète sur les 50 images (chargement, conversion/quantification, inférence).
Repère : sur l'extrait de test de 8 images, RTMPose fp32 fait OKS 0.935 / PCK 0.988 (MediaPipe : 0.647 / 0.968).

### 1.3 Ce qu'on en retient

- **GPU vs CPU : ×11** (274 → 25 ms/image), bien plus que n'importe quelle optimisation de modèle.
- **fp16 : petit gain gratuit** (+11 %), précision strictement identique → à garder sur GPU.
- **INT8 : à écarter** sur ce matériel, plus lent ET moins précis (−23 points d'OKS) :
  - sur CPU, même avec AVX-VNNI : l'activation SiLU (`Sigmoid` × `Mul`) reste en float entre chaque convolution,
    d'où une paire Quantize/Dequantize à chaque couche qui annule le gain des convolutions INT8 ;
  - sur GPU, le provider CUDA n'a pas de vrais noyaux INT8 pour ce format QDQ (la doc onnxruntime renvoie à
    TensorRT et aux Tensor Cores INT8) ;
  - la perte vient des convolutions, pas de la tête SimCC : ne quantifier que `Conv` donne OKS 0.657 (vs 0.647).

### 1.4 Problèmes rencontrés (corrigés sur `dev`, tests de non-régression ajoutés)

1. **INT8 → aucune détection (OKS 0)** : la sortie `dets` de YOLOX concatène coordonnées (0-640) et score (0-1)
   dans un seul tenseur ; une seule échelle INT8 arrondit tous les scores à 0. Corrigé en ne quantifiant que
   `Conv`/`MatMul` (post-traitement gardé en float32).
2. YOLOX et RTMPose sont en **opset 11** ; la quantification par canal demande l'opset 13 → conversion automatique.
3. onnxruntime 1.20 échoue à relire son fichier intermédiaire si `quant_pre_process` reçoit un modèle en mémoire →
   on lui passe un fichier.
4. Environnement : `onnx` 1.23 tire protobuf 7 et casse MediaPipe → `onnx` 1.17 + protobuf 4.25.9.
5. `onnxruntime-gpu` sans cuDNN sur le `LD_LIBRARY_PATH` retombe sur CPU sans erreur → l'outil psifx lève
   maintenant une erreur explicite.

## 2. Optimisations pas encore testées

Classées par rapport gain attendu / coût, pour ce modèle précis.

| Piste | Pourquoi pour RTMPose | Coût | Priorité |
|---|---|---|---|
| **Batcher les crops de pose** | rtmlib appelle RTMPose **une fois par personne détectée**, en boucle (`rtmpose.py`) : avec N personnes, N inférences séquentielles. Un seul appel en batch amortirait le coût fixe par appel. | Faible-moyen (réécrire l'appel au modèle de pose, l'ONNX a une dimension batch) | **Haute** pour la vidéo multi-personnes |
| **Détecter une image sur N + suivre les boîtes** | Le détecteur YOLOX tourne à chaque image ; entre deux détections, les boîtes peuvent être déduites des keypoints précédents (pratique standard des pipelines top-down vidéo). | Moyen (logique de suivi, risque de perte de personne) | Haute |
| **TensorRT Execution Provider (fp16, puis INT8)** | Seul chemin où l'INT8 peut réellement accélérer sur GPU (Tensor Cores INT8, présents sur Ada/RTX 4060) ; fp16 TensorRT fusionne aussi Conv+SiLU. Cache de moteur pour ne compiler qu'une fois. | Moyen (installer TensorRT compatible avec onnxruntime-gpu 1.20/CUDA 12, gérer le cache) | Haute |
| **IOBinding + CUDA graphs (onnxruntime)** | Évite les copies CPU↔GPU à chaque appel et le coût de lancement des noyaux ; les entrées sont de taille fixe (640×640 et 256×192), condition requise par les CUDA graphs. | Faible-moyen (contourner l'appel `session.run` de rtmlib) | Moyenne |
| **Preset `lightweight`** (YOLOX-tiny + RTMPose-s) | Simple changement d'option déjà exposée (`--mode`) ; à mesurer en précision ET vitesse pour situer le compromis. | Très faible | Moyenne (mesure rapide) |
| **INT8 mieux calibré** (entropie/percentile, calibration sur images proches de COCO, exclure les couches sensibles) | Pourrait réduire la perte de précision, mais ne résout pas la lenteur (SiLU/Q-DQ) sans TensorRT. | Faible | Basse (seulement avec TensorRT) |
| Quantization-aware training | Récupère la précision INT8, mais demande d'entraîner (MMPose). | Élevé | Basse |
| Distillation / élagage | Cf. `04_*` §2.3/§2.4 ; RTMPose est déjà un modèle léger. | Élevé | Basse |

## Sources

- [ONNX Runtime — Quantize ONNX Models](https://www.onnxruntime.ai/docs/performance/model-optimizations/quantization.html) (statique pour les CNN, QDQ S8S8, Tensor Cores INT8)
- [ONNX Runtime — Float16 and Mixed Precision Models](https://onnxruntime.ai/docs/performance/model-optimizations/float16.html)
- [ONNX Runtime — TensorRT Execution Provider](https://onnxruntime.ai/docs/execution-providers/TensorRT-ExecutionProvider.html) (`trt_fp16_enable`, `trt_int8_enable`, cache de moteur)
- [ONNX Runtime — CUDA Execution Provider](https://www.onnxruntime.ai/docs/execution-providers/CUDA-ExecutionProvider) (CUDA graphs, formes statiques)
- [Optimum — Accelerated inference on NVIDIA GPUs](https://huggingface.co/docs/optimum-onnx/onnxruntime/usage_guides/gpu) (IOBinding)
- rtmlib 0.0.16, `rtmlib/tools/pose_estimation/rtmpose.py` (boucle par boîte, lu dans le devcontainer)
- Détails de mesure : `benchmark/results/optim-rtmpose-fp16-int8.md`
