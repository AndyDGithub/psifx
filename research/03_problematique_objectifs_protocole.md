# Problématique, objectifs et protocole du benchmark

Date : 2026-09-28
Contexte : suite à `01_exploration_modeles.md` (candidats identifiés) et `02_methodologie_metriques.md` (méthodologie de mesure et littérature domaine). Ce document formule la problématique du travail, les objectifs concrets qui en découlent, et le protocole du benchmark exécuté dans ce même dossier (`research/benchmark/`). Les résultats et leur interprétation sont dans `03_resultats.md`.

## 1. Problématique

Le backend de pose estimation actuellement intégré à psifx (**MediaPipe**) est systématiquement dominé en précision par plusieurs alternatives identifiées dans `01_*`/`02_*` (RTMPose/ViTPose, potentiellement Sapiens2), d'après deux comparatifs empiriques indépendants trouvés dans la littérature (§6.2 de `02_*`). Mais la précision n'est pas le seul critère qui compte pour psifx : la stabilité temporelle sur vidéo, le coût de calcul (GPU/CPU, mémoire), la licence, et la facilité d'intégration dans l'architecture existante (`tool.py`/`command.py` par backend) sont tout aussi déterminants pour une bibliothèque qui vise à être "prête à l'emploi" pour des chercheurs en sciences humaines, pas seulement performante sur un benchmark académique.

**Question centrale :** *Comment choisir, de façon objective et reproductible, quel(s) backend(s) de pose estimation intégrer à psifx pour améliorer la précision d'extraction sans compromettre les contraintes propres au projet (stabilité temporelle, coût de calcul praticable, licence compatible, cohérence architecturale) ?*

Sous-questions :

- **Q1** — Quelle est l'ampleur réelle du gain de précision des candidats identifiés par rapport à MediaPipe, mesurée dans des conditions comparables à l'usage réel de psifx (vidéo d'interaction humaine à deux personnes, pas uniquement des benchmarks d'images statiques) ?
- **Q2** — Ce gain de précision se traduit-il aussi par une meilleure stabilité temporelle sur vidéo (moins de jitter), ou ces deux dimensions sont-elles indépendantes, comme le suggère le cadre MBRCS discuté en `02_*` (§9.2) ?
- **Q3** — Quel est le compromis vitesse/mémoire associé à chaque candidat, et reste-t-il praticable dans l'environnement cible de psifx (GPU grand public type RTX 4060, mais aussi CPU pour les utilisateurs sans GPU dédié — cas déjà rencontré, cf. `psifx video pose mediapipe` qui tourne en CPU-only dans le devcontainer) ?
- **Q4** — Existe-t-il un backend qui domine simultanément sur toutes ces dimensions, ou faut-il arbitrer via un front de Pareto (aucune solution ne dominant les autres sur tous les critères à la fois) ?

## 2. Objectifs concrets

| # | Objectif | Livrable |
|---|---|---|
| O1 | Définir un protocole de benchmark reproductible, avec des métriques explicites tirées de `02_*` (précision, stabilité temporelle, vitesse, mémoire) | Ce document, §3 |
| O2 | Implémenter un script/pipeline exécutable dans le repo, rejouable par n'importe qui avec le devcontainer déjà en place | `research/benchmark/` (code) |
| O3 | Exécuter effectivement le benchmark sur les candidats retenus | Résultats bruts (CSV/JSON) + `03_resultats.md` |
| O4 | Synthétiser en front de Pareto ou classement documenté, interprétable pour la discussion avec le responsable et Matthew Vowels | `03_resultats.md`, section synthèse |

## 3. Protocole

### 3.1 Backends comparés

| Backend | Rôle | Statut d'intégration |
|---|---|---|
| **MediaPipe Pose** | Référence — backend actuellement intégré à psifx | Déjà dans `psifx/video/pose/mediapipe/` |
| **RTMPose** (via `rtmlib`, poids COCO pré-entraînés) | Candidat "haute priorité" de `01_*`/`02_*` | Nouveau, installation légère (ONNXRuntime, pas besoin de la stack complète `mmcv`/`mmengine`) |
| **ViTPose** (via `transformers`, si le support est disponible dans la version installée) | Deuxième candidat de `01_*`/`02_*`, référence de précision dans plusieurs comparatifs | Nouveau — à vérifier techniquement avant de s'engager (voir §3.4) |

**Explicitement hors scope de cette passe**, pour rester dans un temps raisonnable (consigne du responsable, rappelée en `01_*`) :
- **Sapiens2** — licence à faire valider avant d'y consacrer du temps d'implémentation (cf. `01_*` §4.1).
- **Méthodes anti-occlusion dédiées (DEKR/CrowdPose)** — changeraient le backend de détection multi-personnes, pas un simple ajout de candidat ; à réévaluer selon les résultats de ce premier passage.
- **SAM3/SAM3.1 (tracking/segmentation)** — pilier séparé, déjà couvert par les tests d'intégration existants du repo ; ce benchmark se concentre sur la **pose estimation**, le manque identifié comme prioritaire en `01_*`.

### 3.2 Données

| Jeu de données | Usage | Justification |
|---|---|---|
| Sous-ensemble annoté **COCO val2017 keypoints** (~50 images choisies aléatoirement, contenant au moins une personne annotée) | Mesure de précision absolue (OKS, PCK) | Standard de la littérature (§3 et §6.1 de `02_*`) — permet de comparer nos chiffres à ceux publiés (ViTPose, RTMPose annoncent leurs scores sur COCO) |
| `example/data/Video.mp4` (déjà dans le repo) | Mesure de stabilité temporelle + vitesse + comparaison qualitative | Cas d'usage réel de psifx (interaction à deux personnes, un seul point de vue) — pas de vérité terrain disponible dessus, donc **pas** de précision absolue mesurée sur cette source |

### 3.3 Métriques

| Dimension | Métrique | Calculée sur |
|---|---|---|
| Précision | OKS moyen (Object Keypoint Similarity) et PCK@0.2 | Sous-ensemble COCO annoté |
| Stabilité temporelle | Jerk normalisé moyen des trajectoires de points-clés (dérivée seconde de la position, normalisée par la taille de la personne) — proxy sans vérité terrain, dans l'esprit de la dimension "temporal stability" du cadre MBRCS (`02_*` §9.2) | `Video.mp4` |
| Vitesse | Images/seconde, sur GPU (RTX 4060, devcontainer) et sur CPU | `Video.mp4` |
| Mémoire | Pic de mémoire GPU pendant l'inférence (`torch.cuda.max_memory_allocated` ou équivalent ONNXRuntime) | `Video.mp4` |
| Qualitatif (repris de `01_*`/`02_*`, pas remesuré ici) | Licence, effort d'intégration | — |

### 3.4 Étapes d'exécution

1. Vérifier la faisabilité technique de chaque backend dans l'environnement du devcontainer (versions `transformers`/`rtmlib` disponibles) avant de committer le protocole final.
2. Télécharger et préparer le sous-ensemble COCO annoté (annotations officielles + images correspondantes uniquement, pas le jeu complet).
3. Implémenter un script commun par backend, avec une interface de sortie identique (liste de points-clés normalisés + score de confiance par frame/image), pour que le calcul des métriques soit backend-agnostique.
4. Exécuter chaque backend sur le sous-ensemble COCO → calculer précision.
5. Exécuter chaque backend sur `Video.mp4` → calculer stabilité temporelle, vitesse, mémoire.
6. Agréger les résultats dans un tableau unique, produire un front de Pareto (ex. précision vs vitesse) et/ou un classement pondéré si plus de deux critères sont retenus simultanément.

### 3.5 Limites assumées

- Échantillon COCO réduit (~50 images) plutôt que l'ensemble val2017 complet (5000 images) — pour rester dans un temps d'exécution raisonnable ; les scores obtenus seront donc indicatifs, pas une reproduction exacte des chiffres publiés dans la littérature (qui utilisent l'ensemble complet), mais comparables *entre eux* puisque calculés sur le même sous-ensemble pour chaque backend.
- Pas de vérité terrain sur `Video.mp4` : la précision n'y est pas mesurée directement, seulement la stabilité/vitesse/mémoire — cohérent avec la limite déjà notée en `02_*` (aucune des sources trouvées n'a de vérité terrain publique sur cette vidéo spécifique).
- Un seul type de scène testé pour la stabilité temporelle (`Video.mp4`) — pas de généralisation à d'autres types de vidéos dans cette première passe.
