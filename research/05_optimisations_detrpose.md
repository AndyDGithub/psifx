# Optimisations DETRPose — données mesurées et pistes restantes

Date : 2026-10-07. Branche : `optim/detrpose-compile-fp16` (part de `feature/pose-detrpose`).
Fait suite à `04_modeles_optimisation.md` §2.1 et à `03_resultats.md` §3 (« DETRPose à réévaluer après
`torch.compile`/quantification »). Un fichier `05_optimisations_<modèle>.md` existe dans chaque branche
d'optimisation ; le tableau récapitulatif tous modèles sera fait au merge dans `dev`.

## 1. Ce qui a été mesuré

### 1.1 Cadre

- **Outil** : `psifx video pose detrpose` = DETRPose-M (`detrpose_hgnetv2_m`), exécuté dans son propre
  environnement (`detrpose-env`, PyTorch 2.14.1 cu130, `numpy==1.23.5` imposé par DETRPose) via **un sous-processus
  par vidéo** : psifx écrit chaque image en PNG dans un dossier temporaire, `infer.py` les relit et renvoie un JSON.
- **Matériel** : RTX 4060 Laptop (8 Go, ~0.9 Go occupés par d'autres applications).
- **Optimisations testées** (transmises au sous-processus) :
  - `--fp16` : `torch.autocast` float16 ;
  - `--compile` : `torch.compile` à formes fixes (entrée 640×640, batch 1), avec 3 passes d'échauffement hors
    chronométrage.
- **Mesures** :
  - temps **du modèle seul** par image, mesuré dans `infer.py` (CUDA synchronisé) ;
  - temps **de bout en bout** (`time_psifx_pose.py`, et une mesure 50 vs 200 images pour séparer coût fixe et coût
    par image) ;
  - précision : CLI psifx sur 50 images COCO. Résultats bruts :
    `benchmark/results/psifx_cli/optim-detrpose-compile-fp16.csv`.

### 1.2 Temps du modèle seul et précision (50 images COCO)

| Appareil | Config | Modèle ms/image | vs fp32 GPU | OKS | PCK | Détection | Temps CLI (s) |
|---|---|---|---|---|---|---|---|
| CPU | fp32 | 297.9 | — | 0.840 | 0.922 | 98 % | 26.0 |
| GPU | fp32 | 23.9 | — | 0.840 | 0.922 | 98 % | 12.7 |
| GPU | `--fp16` | 35.8 | −33 % | 0.840 | 0.922 | 98 % | 13.2 |
| GPU | `--compile` | 15.4 | +55 % | 0.839 | 0.922 | 98 % | 17.6 |
| GPU | `--fp16 --compile` | **11.9** | **×2.0** | 0.840 | 0.922 | 98 % | 17.6 |

Repère : sur l'extrait de test de 8 images, DETRPose fp32 fait OKS 0.951 / PCK 1.000 (le meilleur des quatre
modèles sur cet extrait).

### 1.3 Temps de bout en bout : le pont domine

| Config (GPU) | Coût fixe par vidéo | Bout en bout ms/image | dont modèle |
|---|---|---|---|
| fp32 | 2.6 s | 90.1 | 21.7 |
| `--fp16 --compile` | 8.2 s (compilation) | 101.6 | 14.4 |

(Mesure par différence entre un appel de 50 et un appel de 200 images ; « dont modèle » = médiane mesurée dans
`infer.py`.)

### 1.4 Ce qu'on en retient

- `--fp16 --compile` est **l'optimisation de modèle la plus efficace mesurée** (×2 sur le modèle), sans perte de
  précision. fp16 seul ralentit (conversions non fusionnées) ; compile les fusionne.
- **Mais ~70 ms par image partent dans le pont, pas dans le modèle** : écriture PNG côté psifx, relecture,
  redimensionnement PIL et `ToTensor` dans le sous-processus. De bout en bout, l'optimisation ne se voit presque pas
  (~90-100 ms/image dans les deux cas).
- **Correction de mesure** : `infer.py` chronométrait sans `torch.cuda.synchronize()` (seul le lancement des noyaux
  CUDA asynchrones était compté). Corrigé dans cette branche. **Le même défaut existe dans
  `benchmark/detrpose_infer.py`**, d'où viennent les 13.3 FPS de DETRPose dans `03_resultats.md` : chiffre
  probablement surestimé, à remesurer.

## 2. Optimisations pas encore testées

| Piste | Pourquoi pour DETRPose | Coût | Priorité |
|---|---|---|---|
| **Pont plus efficace** : images brutes (numpy/mémoire partagée) au lieu de PNG | Le pont coûte ~70 ms/image, 3× le modèle. Supprimer l'encodage/décodage PNG est le premier levier, avant toute optimisation de modèle. | Faible-moyen | **Haute** |
| **Sous-processus persistant** alimenté par un pipe | Évite de recharger (et de recompiler : 8 s) le modèle à chaque vidéo, et permet de traiter les images au fil de l'eau au lieu de tout écrire sur disque d'abord (aujourd'hui l'espace disque croît avec la durée de la vidéo). | Moyen | Haute |
| **Prétraitement sur GPU** | Redimensionnement 640×640 et `ToTensor` faits en PIL sur CPU dans `infer.py` ; à faire en torch sur GPU. | Faible | Haute |
| **Lever l'environnement séparé** | Si DETRPose fonctionne avec le numpy 2 de psifx (pin `numpy==1.23.5` peut-être trop strict), le pont disparaît entièrement. À tester dans un environnement jetable. | Faible à tester, risqué à adopter | Moyenne |
| **Export ONNX + TensorRT fp16** | Le dépôt fournit déjà `tools/deployment/export_onnx.py`, `export_tensorrt.py` (`trtexec --fp16`) et `tools/inference/trt_inf.py` : c'est le chemin qui a produit les latences publiées. | Moyen (installer TensorRT) | Moyenne |
| **Autres tailles** (N, S, L, X : configs présentes dans `configs/detrpose/`) | DETRPose-S/N pourraient suffire en précision sur des vidéos peu peuplées. | Très faible (télécharger le checkpoint, `--config`) | Moyenne (mesure rapide) |
| **Batch de plusieurs images** | Le modèle est appelé image par image ; un batch amortit le coût fixe GPU. | Moyen | Moyenne |
| **INT8 via `torchao`** | `torch.ao.quantization` est déprécié ; gain annoncé surtout sur de gros modèles ; l'attention déformable est sensible à la quantification. | Moyen | Basse |
| Limite connue à garder en tête : sous-comptage sur scènes très peuplées (1 personne détectée sur 26 au seuil 0.5, cf. `benchmark/README.md`) | Pas une optimisation, mais à vérifier si on baisse `--score_threshold` ou change de taille. | — | — |

## Sources

- [Transformers — torch.compile](https://huggingface.co/docs/transformers/v5.8.0/perf_torch_compile)
- [PyTorch — CUDA semantics, asynchronous execution](https://docs.pytorch.org/docs/stable/notes/cuda.html#asynchronous-execution) (pourquoi synchroniser avant de chronométrer)
- [torchao — Inference workflows](https://docs.pytorch.org/ao/stable/workflows/inference.html)
- [PyTorch — torch.ao.quantization migration plan](https://dev-discuss.pytorch.org/t/torch-ao-quantization-migration-plan/2810)
- [DETRPose (Hugging Face Space de l'auteur)](https://huggingface.co/spaces/SebasJanampa/DETRPose/tree/main) ; dépôt cloné `SebastianJanampa/DETRPose`, `tools/deployment/` et `configs/detrpose/` (lus dans le devcontainer)
- Détails de mesure : `benchmark/results/optim-detrpose-compile-fp16.md`
