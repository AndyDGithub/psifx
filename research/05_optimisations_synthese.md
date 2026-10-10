# Synthèse des modèles et optimisations — décisions pour `main`

Date : 2026-10-09. Récapitule les quatre notes `05_optimisations_<modèle>.md` (mesures détaillées, problèmes
rencontrés, pistes restantes) et fixe ce qui part dans la bibliothèque. Matériel : RTX 4060 Laptop (8 Go,
partagé), CPU AVX-VNNI. Précision : commande psifx réelle sur 50 images COCO ; vitesse : `time_psifx_pose.py`
(régime établi, synchronisation CUDA).

## 1. Tableau récapitulatif

| Modèle | Config. | Appareil | ms/image | vs fp32 | OKS | PCK@0,2 | Détection | Décision |
|---|---|---|---|---|---|---|---|---|
| MediaPipe (existant) | — | CPU | — | — | 0,283 | 0,432 | 84 % | Conservé (visage + mains) |
| **RTMO** | fp32 | CPU | 149,1 | — | 0,826 | 0,880 | 92 % | **main** |
| | fp32 | GPU | 13,0 | — | 0,826 | 0,880 | 92 % | **main** |
| | **fp16** | GPU | **9,1** | **+43 %** | 0,826 | 0,880 | 92 % | **main** (`--precision fp16`) |
| | int8 | CPU | 255,6 | −42 % | 0,584 | 0,680 | 72 % | Retiré |
| | int8 | GPU | 17,7 | −26 % | 0,580 | 0,662 | 70 % | Retiré |
| **ViTPose** | fp32 | GPU | 55,1 | — | 0,850 | 0,914 | 96 % | **main** |
| | fp16 | GPU | 56,9 | −3 % | 0,850 | 0,914 | 96 % | Retiré |
| | compile | GPU | 64,5 | −17 % | — | — | — | Retiré |
| | fp16 + compile | GPU | 50,0 | +10 % | 0,851 | 0,914 | 96 % | Retiré (~2 min de compilation par appel) |
| RTMPose | fp32 | CPU | 274,5 | — | 0,873 | 0,935 | 98 % | Branche seulement |
| | fp32 | GPU | 24,6 | — | 0,873 | 0,935 | 98 % | Branche seulement |
| | fp16 | GPU | 22,1 | +11 % | 0,873 | 0,935 | 98 % | Branche seulement |
| | int8 | CPU / GPU | 436,9 / 32,3 | −59 % / −31 % | 0,647 | 0,82 / 0,81 | 98 / 96 % | Retiré |
| DETRPose | fp32 | CPU | 297,9 \* | — | 0,840 | 0,922 | 98 % | Branche seulement |
| | fp32 | GPU | 23,9 \* (90,1 de bout en bout) | — | 0,840 | 0,922 | 98 % | Branche seulement |
| | fp16 + compile | GPU | 11,9 \* (101,6 de bout en bout) | ×2 modèle | 0,840 | 0,922 | 98 % | Branche seulement |

\* DETRPose : temps du modèle seul dans son sous-processus ; « de bout en bout » inclut le pont psifx ↔
environnement DETRPose (~70 ms/image).

## 2. Décisions (2026-10-09)

**Sur `main`** :
- `psifx video pose rtmo` (avec `--precision fp32|fp16`) — meilleur compromis précision/vitesse, 1er du
  classement pondéré du banc de recherche, le seul à profiter pleinement du fp16 ;
- `psifx video pose vitpose` — le plus précis des modèles disponibles dans psifx, sans nouvelle dépendance ;
- la base commune COCO-17 (`psifx/video/pose/coco17/`), les tests unitaires, le test de précision COCO et le code
  d'évaluation (`tests/pose_eval/`).

**Pas sur `main`** (le code reste dans sa branche, les mesures dans `dev`) :
- RTMPose — le plus précis via psifx (0,873) mais en grande partie redondant avec RTMO (même dépendance) et
  ViTPose ;
- DETRPose — environnement Python séparé, installation manuelle, pont lent ;
- INT8 (tous modèles) — plus lent et −23 à −24 points d'OKS ;
- `--fp16` / `--compile` de ViTPose — gain marginal, compilation de ~2 min à chaque appel.

**Dépendances** : `rtmlib` passe en extras pip (`psifx[rtm]`, `psifx[rtm-gpu]`) ; l'installation de base de
psifx est inchangée. `onnxruntime-gpu` entre en conflit avec le paquet CPU `onnxruntime` (même dossier
d'installation, le dernier installé gagne) : la documentation donne la commande pour ne garder que la version GPU,
et l'outil lève une erreur explicite si CUDA n'est pas utilisable. Les bibliothèques cuDNN/cuBLAS installées avec
PyTorch sont chargées automatiquement (plus besoin de `LD_LIBRARY_PATH`).

**`research/`, rapport, banc de mesure** : restent sur `dev`, hors de `main`.

## 3. Limites à garder en tête

- RTMO et ViTPose ne donnent que les 17 points du corps : MediaPipe Holistic reste nécessaire pour le visage et les
  mains (d'où l'axe corps entier ci-dessous).
- Un seul suivi de personne par vidéo, comme MediaPipe ; le multi-personnes passe par les masques SAM3.

## 4. Axes d'amélioration retenus pour la suite

1. **Corps entier (RTMW)** — retrouver visage et mains avec un modèle récent (même famille et même dépendance que
   RTMO, `rtmlib`), pour couvrir ce que seul MediaPipe Holistic fournit aujourd'hui.
2. **Multi-personnes natif** — produire une archive par personne détectée sans exiger les masques SAM3, avec un
   suivi d'identité entre images.
3. **Occlusion + SAM 3.1** — évaluer la robustesse à l'occlusion mutuelle (CrowdPose, ou occlusion volontaire sur
   la vidéo d'exemple) ; passer à SAM 3.1 dès son support dans `transformers`, et tester d'ici là un re-prompting
   autour de SAM3.

Non retenus pour l'instant : vitesse vidéo avancée (détection une image sur N, TensorRT, lots) — voir les pistes
détaillées dans chaque `05_optimisations_<modèle>.md`.
