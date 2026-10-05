# Optimisations ViTPose — données mesurées et pistes restantes

Date : 2026-10-07. Branche : `optim/vitpose-compile-fp16` (part de `feature/pose-vitpose`).
Fait suite à `04_modeles_optimisation.md` §2.1. Un fichier `05_optimisations_<modèle>.md` existe dans chaque
branche d'optimisation ; le tableau récapitulatif tous modèles sera fait au merge dans `dev`.

## 1. Ce qui a été mesuré

### 1.1 Cadre

- **Outil** : `psifx video pose vitpose` = détecteur RT-DETR (`PekingU/rtdetr_r50vd_coco_o365`) + ViTPose-base
  (`usyd-community/vitpose-base-simple`), les deux via `transformers` 5.3.0, PyTorch 2.8.0 (cu128).
- **Matériel** : RTX 4060 Laptop (8 Go, ~0.9 Go occupés par d'autres applications).
- **Vitesse** : `benchmark/time_psifx_pose.py`, 50 images COCO letterboxées 640×640, 2 passes d'échauffement puis 3
  chronométrées. **Précision** : CLI psifx sur les mêmes 50 images. Résultats bruts :
  `benchmark/results/psifx_cli/optim-vitpose-compile-fp16.csv`.
- **Optimisations testées** (sur les deux modèles, détecteur compris) :
  - `--fp16` : `torch.autocast` float16 (garde softmax/normalisations en float32, contrairement à `model.half()`) ;
  - `--compile` : `torch.compile(dynamic=True)`, car la taille du batch de pose dépend du nombre de personnes ;
  - Flash Attention : **déjà actif** — le backbone ViTPose de `transformers` 5.3 utilise SDPA
    (`_supports_sdpa = True`), qui inclut les noyaux de type flash de PyTorch. Rien à ajouter.

### 1.2 Résultats (50 images COCO, GPU)

| Config | ms/image | FPS | vs fp32 | Échauffement (s) | Pic mémoire torch (Mo) | OKS | PCK | Détection | Temps CLI (s) |
|---|---|---|---|---|---|---|---|---|---|
| fp32 | 55.1 | 18.1 | — | 6 | 718 | 0.850 | 0.914 | 96 % | 18.9 |
| `--fp16` | 56.9 | 17.6 | −3 % | 6 | 994 | 0.850 | 0.914 | 96 % | 19.5 |
| `--compile` | 64.5 | 15.5 | −17 % | 122 | 656 | — | — | — | — |
| `--fp16 --compile` | **50.0** | **20.0** | **+10 %** | 139 | 652 | 0.851 | 0.914 | 96 % | 134.9 |

Repère : sur l'extrait de test de 8 images, ViTPose fp32 fait OKS 0.943 / PCK 0.988. Le benchmark de recherche
(`03_resultats.md`) donnait 0.890 sur les 50 images avec les boîtes de la vérité terrain ; ici les boîtes viennent
d'un vrai détecteur, d'où 0.850.

### 1.3 Où part le temps (fp32, moyenne sur 100 images)

| Étape | ms | Part |
|---|---|---|
| Prétraitement détecteur (CPU) | 2.7 | 5 % |
| **Détecteur RT-DETR (GPU)** | **26.7** | **45 %** |
| Post-traitement détecteur | 0.5 | 1 % |
| **Prétraitement ViTPose (CPU, processeur HF)** | **12.8** | **22 %** |
| Modèle ViTPose (GPU) | 13.5 | 23 % |
| Post-traitement ViTPose | 3.1 | 5 % |

### 1.4 Ce qu'on en retient

- **Le transformeur ViTPose, cible de la piste `04_*` §2.1, ne pèse que ~23 % du temps** : même un gain parfait
  sur lui ne peut pas dépasser ~1.3× de bout en bout (loi d'Amdahl).
- `--fp16 --compile` : +10 % seulement, au prix de **~2 min de compilation à chaque appel de la CLI** — rentable
  seulement sur des vidéos de plus de ~25 min. **Pas intéressant pour psifx tel quel.**
- fp16 seul et compile seul ralentissent : autocast ajoute des conversions sans les fusionner ; `dynamic=True`
  produit des noyaux moins spécialisés.
- La précision ne bouge pas (OKS 0.850-0.851).

## 2. Optimisations pas encore testées

| Piste | Pourquoi pour ViTPose | Coût | Priorité |
|---|---|---|---|
| **Détecteur plus léger** (RT-DETR R18, YOLOX via rtmlib…) | Le détecteur est **45 % du temps** : c'est le premier levier. Mesurer la perte éventuelle de boîtes (taux de détection 96 % aujourd'hui). | Faible (option `--detector_name` déjà exposée pour les variantes RT-DETR) | **Haute** |
| **Détecter une image sur N + suivre les boîtes** | Supprime la majorité des appels au détecteur en vidéo ; boîtes déduites des keypoints de l'image précédente. | Moyen | **Haute** |
| **Prétraitement ViTPose sur GPU** | 22 % du temps dans le processeur HF (transformation affine + normalisation en numpy/PIL sur CPU) ; à refaire en torch sur GPU (`grid_sample`/`affine_grid`). | Moyen | Haute |
| **`torch.compile` à formes fixes** | Remplir le batch de pose jusqu'à une taille fixe (ex. 4 personnes) pour compiler sans `dynamic=True`, avec `mode="reduce-overhead"` (CUDA graphs) ; mettre en cache la compilation (`torch.compile` cache/AOT) pour ne pas payer 2 min par appel. | Moyen | Moyenne |
| **bf16 au lieu de fp16** | Même largeur que fp16 sur Ada, sans risque de dépassement ; à mesurer avec compile. | Très faible | Basse (mesure rapide) |
| **Export ONNX + TensorRT fp16/INT8** | Fusion des couches et noyaux INT8 réels sur Tensor Cores ; demande d'exporter détecteur et ViTPose. | Élevé | Moyenne |
| **INT8 via `torchao`** (`Int8DynamicActivationInt8WeightConfig`) | `torch.ao.quantization` est déprécié au profit de `torchao`. La doc indique un gain surtout sur de gros modèles : ViTPose-base (~90 M paramètres) est petit, gain incertain. | Moyen (nouvelle dépendance) | Basse |
| **Autre taille de modèle** (ViTPose++ small/base, `usyd-community/vitpose-plus-*`) | Variantes officielles du même modèle ; ViTPose++ demande `dataset_index` (expert COCO = 0). | Faible | Moyenne (mesure rapide) |
| Distillation / élagage | Cf. `04_*` §2.3/§2.4. | Élevé | Basse |

## Sources

- [Transformers — torch.compile](https://huggingface.co/docs/transformers/v5.8.0/perf_torch_compile) (modes `default`/`reduce-overhead`/`max-autotune`, recompilation)
- [Diffusers — Accelerate inference](https://huggingface.co/docs/diffusers/optimization/fp16) (recompilation sur changement de forme, `dynamic=True`)
- [Transformers — ViTPose](https://huggingface.co/docs/transformers/v5.14.0/model_doc/vitpose) (détecteur RT-DETR, ViTPose++ et `dataset_index`, quantification torchao)
- [Transformers — ViTPose v4.48 (SDPA)](https://huggingface.co/docs/transformers/v4.48.0/model_doc/vitpose)
- [torchao — Inference workflows](https://docs.pytorch.org/ao/stable/workflows/inference.html) (`Int8DynamicActivationInt8WeightConfig`, gain sur gros modèles)
- [PyTorch — torch.ao.quantization migration plan](https://dev-discuss.pytorch.org/t/torch-ao-quantization-migration-plan/2810)
- [ONNX Runtime — TensorRT Execution Provider](https://onnxruntime.ai/docs/execution-providers/TensorRT-ExecutionProvider.html)
- Détails de mesure : `benchmark/results/optim-vitpose-compile-fp16.md`
