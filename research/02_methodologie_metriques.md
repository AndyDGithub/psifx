# Méthodologie de mesure de la précision — revue de papiers

Date : 2026-09-23
Contexte : suite à `01_exploration_modeles.md`. Ce document lit une série de papiers rassemblés côté utilisateur (segmentation/pose/tracking, sur ~28 ans de littérature) en se concentrant spécifiquement sur **comment chacun mesure la précision** et **quelle méthode il utilise pour l'améliorer** — pas un résumé général. Objectif : dégager les métriques et les fils méthodologiques qui serviront de base au benchmark complet prévu en `03_*`.

## 1. Papiers couverts

| # | Papier | Auteurs | Année / Venue |
|---|---|---|---|
| 1 | Statistical shape model based segmentation of medical images | Neumann, Lorenz | 1998, *Computerized Medical Imaging and Graphics* |
| 2 | Elastic model-based segmentation of 3-D neuroradiological data sets | Kelemen, Székely, Gerig | 1999, *IEEE Trans. Medical Imaging* |
| 3 | Stable Real-Time 3D Tracking using Online and Offline Information | Vacchetti, Lepetit, Fua | 2004, *IEEE TPAMI* |
| 4 | Model Based Training, Detection and Pose Estimation of Texture-Less 3D Objects in Heavily Cluttered Scenes (LINEMOD++) | Hinterstoisser, Lepetit, Ilic, Holzer, Bradski, Konolige, Navab | 2012, ACCV |
| 5 | Articulated Pose Estimation by a Graphical Model with Image Dependent Pairwise Relations | Chen, Yuille | 2014, NeurIPS |
| 6 | FoundPose: Unseen Object Pose Estimation with Foundation Features | Örnek, Labbé, Tekin, Ma, Keskin, Forster, Hodan | 2023/2024, arXiv 2311.18809 (Meta RL / TU Munich) |
| 7 | FoundationPose: Unified 6D Pose Estimation and Tracking of Novel Objects | Wen, Yang, Kautz, Birchfield | 2024, CVPR (NVIDIA) |
| 8 | GCE-Pose: Global Context Enhancement for Category-level Object Pose Estimation | Li, Xu, Huang, Jung, Yu, Navab, Busam | 2025, CVPR |

Remarque : les deux premiers papiers ne sont pas de la vision par ordinateur "grand public" mais de l'imagerie médicale — ils sont néanmoins directement pertinents ici car ce sont les papiers fondateurs de l'usage d'un **modèle statistique de forme (shape prior)** pour la segmentation, une idée qui réapparaît explicitement dans GCE-Pose (2025) sous une forme moderne apprise. Les avoir dans la liste permet de tracer la filiation méthodologique plutôt que de traiter GCE-Pose comme une idée surgie de nulle part.

## 2. Ce que chaque papier apporte à la question "précision"

### 2.1 Neumann & Lorenz (1998) — modèle statistique de forme

Segmentation d'images médicales par ajustement d'un modèle de forme statistique appris sur un ensemble de formes segmentées manuellement (dans la filiation directe des *Active Shape Models* de Cootes & Taylor). La précision vient du fait que la forme cherchée est **contrainte à rester plausible** (proche du sous-espace de formes déjà observées), ce qui limite les erreurs grossières typiques d'une segmentation pixel-par-pixel non contrainte, surtout en présence de bruit ou de données partielles.

### 2.2 Kelemen, Székely & Gerig (1999) — forme paramétrique par harmoniques sphériques

Prolonge l'idée avec une représentation de surface par séries d'harmoniques sphériques dans un repère centré-objet normalisé, puis une déformation élastique restreinte du modèle statistique pour segmenter une nouvelle image. Contribution méthodologique clé pour la précision : normaliser la représentation (repère invariant) **avant** d'apprendre la statistique de forme, pour que le modèle ne confonde pas variation de pose et variation de forme — distinction encore centrale aujourd'hui (voir §2.8, GCE-Pose sépare explicitement reconstruction de forme et estimation de pose).

### 2.3 Vacchetti, Lepetit & Fua (2004) — stabilité du suivi (drift/jitter)

Ne porte pas sur la précision d'une estimation isolée mais sur la **stabilité de la précision dans le temps** lors d'un suivi 3D vidéo. Constat méthodologique important : les approches purement "en ligne" (chaînage image-à-image) dérivent (*drift*) sur les séquences longues, tandis que les approches purement "hors-ligne" (registration absolue sur images de référence) ne dérivent pas mais tremblent (*jitter*) faute d'information pour chaque frame individuelle. Leur solution : fusionner les deux (formulation en ajustement de faisceaux local combinant frames précédentes + un petit ensemble de *keyframes*). **Directement transposable à l'évaluation de SAM3/SAM3.1 dans psifx** : la précision d'un tracker vidéo ne se résume pas à l'IoU frame-par-frame, il faut aussi mesurer la dérive cumulée sur la longueur de la séquence.

### 2.4 Hinterstoisser et al. (2012) — LINEMOD++, détection d'objets texture-less

Base de référence historique pour la détection et l'estimation de pose 6D d'objets sans texture par appariement de gabarits (*templates*) multimodaux (couleur + profondeur), couvrant une hémisphère de vues. Contribution à la précision : (a) génération automatique des gabarits depuis un modèle 3D plutôt qu'un apprentissage en ligne bruité, (b) un raffinement de pose fin à partir de l'estimation grossière donnée par le gabarit le plus proche, (c) un filtrage des faux positifs par vérification couleur — gain de +13% de détections correctes rapporté par rapport au LINEMOD original. Introduit aussi un jeu de données de référence (15 objets, >1100 frames) qui a servi de standard de comparaison pendant plus d'une décennie — illustration qu'une méthodologie d'évaluation reproductible (dataset + protocole) est elle-même une contribution scientifique durable, pas un simple détail.

### 2.5 Chen & Yuille (2014) — modèle graphique + CNN pour la pose humaine articulée

Pertinent pour le pilier "pose estimation" de psifx (2D, humain). Contribution méthodologique : au lieu de relations spatiales fixes entre articulations (hypothèse classique des modèles graphiques), les relations spatiales entre paires de parties du corps sont **elles-mêmes prédites depuis l'image** par un CNN, en fonction du contexte local — la structure du modèle reste interprétable (graphe) mais les probabilités conditionnelles deviennent apprises et dépendantes de l'image. Résultats supérieurs sur LSP et FLIC (benchmarks standards de pose 2D), avec bonne généralisation zero-shot sur Buffy — un précurseur direct de l'idée de généralisation qu'on retrouve dans FoundPose/FoundationPose dix ans plus tard.

### 2.6 FoundPose — Örnek et al. (2023/2024)

Estimation de pose 6D d'objets **jamais vus** à partir d'un modèle 3D, sans ré-entraînement, en exploitant des descripteurs de patch issus de DINOv2 (modèle de fondation auto-supervisé) pour établir des correspondances 2D-3D entre l'image et des vues synthétiques pré-rendues de l'objet. Contribution à la précision : les descripteurs d'une couche intermédiaire de DINOv2 portent plus d'information positionnelle que ceux de la dernière couche — un choix architectural précis qui améliore directement la qualité des correspondances, donc la précision de pose, sans entraînement supplémentaire. Résultat : état de l'art en estimation de pose "refinement-free" sur le benchmark BOP (7 datasets), et peut se combiner à une méthode de raffinement classique pour encore progresser.

### 2.7 FoundationPose — Wen et al. (2024, NVIDIA)

Unifie estimation de pose ET suivi 6D pour objets nouveaux, avec ou sans modèle CAD (mode *model-based* ou *model-free* à partir de simples images de référence). Contribution à la précision : génération de données synthétiques à grande échelle (LLM + diffusion) pour combler l'écart de généralisation, représentation neuronale implicite centrée-objet pour le rendu de nouvelles vues, réseau de raffinement par transformeur, et un module de sélection de pose qui score plusieurs hypothèses. Chiffres concrets sur YCB-Video : AUC ADD 97.4%, AUC ADD-S 91.5%, contre 88.4%/42.1% pour le meilleur concurrent — l'écart énorme sur ADD-S (métrique tolérante à la symétrie) montre que gérer explicitement l'ambiguïté de symétrie des objets est ce qui fait la différence ici, pas juste "plus de données". 1ère place au classement BOP pour l'estimation de pose model-based d'objets nouveaux au moment de sa sortie.

### 2.8 GCE-Pose — Li et al. (2025)

Retour direct au fil "shape prior" ouvert en 1998 (§2.1-2.2), mais appris et conditionné par l'image. Stratégie "reconstruire d'abord, agréger ensuite" : un module *Semantic Shape Reconstruction* déforme un prototype 3D appris par catégorie (via un modèle de forme linéaire profond) pour compléter la géométrie de l'objet même en cas de vue très partielle, puis un module *Global Context Enhanced fusion* combine les features RGB-D partielles avec ce contexte global reconstruit pour prédire la pose. Métriques utilisées (catégorie d'objet, pas juste instance) : IoU50/IoU75 (chevauchement de bounding box 3D) et seuils angle+distance combinés (5°2cm, 5°5cm, 10°2cm, 10°5cm — pourcentage de prédictions dans ces tolérances). Résultats rapportés sur HouseCat6D : IoU50 79.2%, IoU75 60.6%, 5°2cm 24.8%, 5°5cm 25.7%, 10°2cm 55.4%, 10°5cm 58.4% — les chiffres bien plus bas en tolérance stricte (5°2cm) qu'en tolérance large (10°5cm) illustrent bien pourquoi le choix du seuil de tolérance change radicalement la lecture d'un "score de précision".

## 3. Panorama des métriques de précision rencontrées

| Métrique | Ce qu'elle mesure | Utilisée par | À retenir pour le benchmark `03_*` |
|---|---|---|---|
| PCK (Percentage of Correct Keypoints) | % de points-clés 2D prédits à moins d'une distance seuil (souvent relative à la taille du sujet) du point-clé réel | Chen & Yuille (pose 2D) | Standard pour comparer des modèles de pose humaine 2D (MediaPipe, RTMPose, ViTPose, Sapiens2) sur un même jeu d'images annotées |
| Erreur de reprojection | Distance en pixels entre points 3D reprojetés avec la pose estimée vs la pose réelle | Suivi/pose 3D classique (contexte Vacchetti et alii) | Utile pour le suivi caméra/objet, moins pertinent pour de la pose humaine 2D |
| Dérive cumulée / jitter | Stabilité de l'estimation le long d'une séquence vidéo, pas juste image par image | Vacchetti, Lepetit & Fua | À intégrer explicitement dans `03_*` pour SAM3/SAM3.1 : une bonne précision frame-par-frame peut coexister avec une dérive sur vidéo longue |
| ADD / ADD-S (+ AUC) | Distance moyenne entre les sommets du modèle 3D transformés par la pose prédite vs la pose réelle (ADD-S : variante tolérante aux symétries, prend la distance minimale) | LINEMOD, FoundPose, FoundationPose | Métrique de référence pour la pose 6D rigide ; toujours rapporter ADD **et** ADD-S côte à côte (l'écart entre les deux révèle si l'erreur vient d'une confusion de symétrie) |
| BOP AR (moyenne de VSD, MSSD, MSPD) | Score composite standardisé du challenge BOP, combinant plusieurs façons de comparer surfaces/points/projections | FoundPose, FoundationPose | Permet une comparaison directe avec la littérature récente en pose 6D sans redéfinir ses propres métriques |
| IoU à seuil (25/50/75) | Chevauchement de bounding box 3D (ou masque) entre prédiction et vérité terrain | GCE-Pose, et pertinent pour SAM3 (masques de segmentation) | Toujours préciser le seuil utilisé — un score "IoU" seul sans seuil n'est pas comparable |
| Seuils combinés angle+distance (n°/m cm) | % de prédictions dans une tolérance conjointe de rotation et de translation | GCE-Pose (catégorie) | Choisir un seuil trop large masque les vraies différences entre modèles (cf. écart 5°2cm vs 10°5cm dans GCE-Pose) |

## 4. Fils méthodologiques transversaux, pertinents pour psifx

1. **Un a priori de forme améliore la précision en cas de vue partielle/occlusion**, du modèle statistique de forme classique (1998-1999) jusqu'à la reconstruction de forme apprise par catégorie (GCE-Pose, 2025). Pertinent si psifx doit un jour gérer des scènes avec occlusion partielle (ex. personnes se cachant mutuellement dans une interaction filmée d'un seul point de vue, cas déjà mentionné dans `example/README.md`).
2. **La stabilité temporelle (drift/jitter) est une dimension de précision à part entière**, distincte de la précision par frame — un point que la comparaison SAM3 vs SAM3.1 (`01_*`, gain ~7× en tracking multi-objets) devra vérifier explicitement, pas seulement sur des métriques image-par-image.
3. **Le passage des gabarits/templates (LINEMOD, 2012) aux features de modèles de fondation (DINOv2 dans FoundPose, 2023) est le même saut méthodologique** que celui qu'on observe pour la pose humaine entre MediaPipe/RTMPose (features apprises spécifiquement pour la pose) et Sapiens2 (features de fondation multi-tâches) identifié dans `01_*` — la littérature en pose d'objets 6D a fait cette transition un peu plus tôt, ce qui donne des repères sur les gains et les coûts à attendre.
4. **Le choix du seuil de tolérance change radicalement le score rapporté** (IoU25 vs IoU75, 5°2cm vs 10°5cm) : `03_*` devra fixer et documenter explicitement les seuils utilisés pour chaque comparaison, plutôt que de citer un score unique sans contexte.
5. **Toujours rapporter au moins deux métriques complémentaires** (ex. ADD et ADD-S, ou IoU et un seuil angulaire) : plusieurs papiers ci-dessus montrent qu'un seul chiffre peut masquer un problème spécifique (confusion de symétrie, biais de position vs orientation).

## 5. Implications concrètes pour `03_*` (benchmark)

- Reprendre les métriques déjà standardisées par la communauté (PCK pour la pose 2D, ADD/ADD-S + BOP AR pour la pose 6D, IoU à seuils fixes pour la segmentation) plutôt que d'en inventer de nouvelles — permet de comparer directement les chiffres obtenus avec ceux publiés dans les papiers ci-dessus.
- Prévoir une évaluation de la stabilité temporelle pour tout ce qui touche au tracking vidéo (SAM3/SAM3.1), pas seulement une précision instantanée.
- Documenter précisément les seuils de tolérance utilisés à chaque fois qu'un score de type IoU ou angle/distance est rapporté.
- Réutiliser si possible `example/data/Video.mp4` (déjà dans le repo, déjà utilisé pour valider le devcontainer) comme premier cas d'usage concret pour le benchmark, en complément des datasets publics standards (LSP/FLIC/COCO pour la pose, BOP/YCB-Video/NOCS-REAL275/HouseCat6D pour la pose 6D et catégorie).

---

# Partie 2 — Exploration élargie (consensus, littérature domaine, impact psifx)

Suite demandée après relecture de la partie 1 : explorer plus largement (au-delà des 8 papiers initiaux) pour dégager un consensus de terrain, et relier explicitement chaque piste à un impact concret dans psifx. Recherche menée par WebSearch/WebFetch sur ~20 requêtes supplémentaires ; les sources non concluantes ou redondantes ne sont pas toutes citées individuellement, seules celles ayant apporté une information utilisée dans le texte le sont en bas de document.

## 6. Consensus de terrain (revues 2025-2026)

### 6.1 Pose 2D/3D — revue JETA (août 2026)

Shrestha, Thakur & Wang, *Human Pose Estimation in 2D and 3D: A Survey of Analytical Methods, Benchmarking Frameworks, and Engineering Applications* (DOI [10.3390/jeta4030028](https://doi.org/10.3390/jeta4030028)). Une taxonomie unifiée (représentation du corps, flux top-down/bottom-up, régression vs heatmaps, architectures CNN/Transformer/GCN) et une comparaison chiffrée sur COCO/MPII/Human3.6M/3DPW avec les métriques standard (AP/OKS, PCK/AUC, MPJPE/PA-MPJPE, PVE — confirme le panorama du §3). Consensus dégagé par les auteurs : les meilleures méthodes actuelles partagent deux choix de conception — une sortie structurée qui modélise l'incertitude spatiale (plutôt qu'une simple coordonnée), et des architectures multi-échelle haute résolution qui préservent la précision de localisation sous variation d'échelle et occlusion. Défis identifiés comme non résolus : coût/biais d'annotation, changement de domaine, occlusion.

### 6.2 Comparatifs empiriques directs — avec MediaPipe explicitement testé

Deux études indépendantes comparent directement des outils déjà identifiés en `01_*`/`02_*`, dont **MediaPipe, le backend actuellement utilisé par psifx** :

- Analyse de mouvements de cirque (*Multimedia Tools and Applications*, Springer, 2026) : ViTPose bat nettement YOLOv8 et MediaPipe, avec la MPJPE la plus basse (0.192 m) et le PCK le plus élevé (68.1%).
- Comparatif sur cinématique de marche (gait kinematics) : OpenPose et TensorFlow MoveNet Thunder les plus précis pour la hanche/le genou (OpenPose ~5.1° ± 2.5° d'erreur sur le cycle de marche) ; DeepLabCut et AlphaPose les plus rapides (~30 fps) ; MediaPipe et OpenPose plus proches de 15 fps.

Conclusion convergente : dans deux contextes indépendants et deux méthodologies différentes, MediaPipe est systématiquement dominé en précision par au moins une alternative déjà identifiée dans `01_*` (ViTPose, ou par extension RTMPose de la même famille MMPose). Ça renforce la piste "haute priorité" déjà notée en `01_*`.

## 7. Occlusion et interactions rapprochées — le cas d'usage direct de psifx

La vidéo d'exemple de psifx (`example/README.md`) est justement une interaction rapprochée entre deux personnes filmées d'un seul point de vue — l'occlusion mutuelle partielle est donc un cas réel du projet, pas un scénario académique abstrait.

- **DEKR** (*Bottom-Up Human Pose Estimation via Disentangled Keypoint Regression*, CVPR 2021, toujours une référence) : régresse chaque point-clé sur un canal de features dédié avec des convolutions adaptatives spatialement, ce qui le rend robuste au chevauchement de personnes. Sur **CrowdPose** (jeu de données standard pour les scènes à forte occlusion), DEKR+HRNet-W48 atteint 67.3 AP.
- Travaux 2025-2026 poursuivant cette direction : renforcement par modèles vision-langage (CLIP) pour mieux distinguer les personnes proches, mécanismes de sélection/affûtage de features — gains rapportés de +2.1 à +2.5 AP sur COCO/CrowdPose par rapport aux méthodes précédentes.

Implication pour `03_*` : prévoir un sous-test spécifique occlusion/contact rapproché (CrowdPose comme benchmark secondaire, ou test direct sur `Video.mp4` avec occlusion volontaire), pas seulement les benchmarks standards où les personnes sont bien séparées.

## 8. Segmentation vidéo au-delà de SAM3 — mise à jour depuis `01_*`

- **MOSEv2** (2026) : nouveau benchmark délibérément plus dur que DAVIS/YouTube-VOS, où l'état de l'art dépasse déjà 90% J&F (donc ces benchmarks historiques sont proches de saturés). MOSEv2 ajoute disparition/réapparition fréquente d'objets, occlusions sévères, conditions dégradées (pluie, neige, faible luminosité), objets camouflés.
- Résultats SAM3 sur les challenges 2026 associés (PVUW MOSE, LSVOS MOSEv2) : SAM3 reste compétitif mais les meilleures soumissions l'utilisent avec des techniques d'accompagnement plutôt que "tel quel" — re-prompting automatique piloté par un module de récupération d'objet (3e place MOSEv2, J&F 51.17%), ou prompts enrichis par un tracker externe (*Tracking-Enhanced Prompt*, 1er du challenge PVUW MOSE 2026).

Implication : confirme que SAM3(.1) reste le bon socle (cohérent avec `01_*`), mais nuance — sur des scènes complexes (disparition/réapparition, occlusion sévère, ce qui recoupe directement §7), l'utiliser "out of the box" n'est probablement pas optimal. Avant d'envisager un changement de modèle, `03_*` pourrait tester si un simple mécanisme de re-prompting autour du SAM3 déjà intégré comble une partie du besoin — moins coûteux qu'un changement d'architecture.

## 9. Littérature spécifique aux sciences du comportement — la partie la plus directement actionnable

Recherche volontairement recentrée sur des papiers qui utilisent déjà la pose estimation **dans le même contexte que psifx** (recherche en sciences humaines/codage comportemental), plutôt que des benchmarks génériques de vision par ordinateur.

### 9.1 Feature-Based Behavior Coding (FBBC) / "Behavior Senpai"

Papier trouvé : *Feature-based behavior coding for efficient exploratory analysis using pose estimation* (*Behavior Research Methods*). Directement dans le même espace applicatif que psifx — vaut la peine d'être lu en entier par la suite, pas seulement résumé ici.

- Outils de pose supportés par cet outil, exactement les mêmes familles que celles identifiées en `01_*`/`02_*` : **MediaPipe Holistic**, **YOLOv8/v11**, **RTMPose (Halpe26, whole-body)** — et accepte aussi des sorties **DeepLabCut** pour des modèles personnalisés par l'utilisateur.
- Méthodologie en aval de l'extraction de points-clés : features dérivées (distances, angles, produits vectoriels) → réduction de dimension (UMAP) → clustering visuel supervisé par le chercheur. Le jugement humain reste central — approche "collaborative" assumée, pas une boîte noire.
- Limites reconnues par les auteurs : pas de norme universelle pour concevoir les features (nécessite expérimentation), classification de postures instantanées seulement (pas d'action dynamique), identifiants de personnes pouvant se mélanger en cas d'occlusion/échange de position en multi-personnes — recoupe directement le point occlusion du §7.

**Pertinence directe pour psifx** : c'est un exemple concret de ce qu'un·e chercheur·se en sciences humaines fait *après* l'extraction de pose. Psifx s'arrête aujourd'hui à l'extraction de features brutes (keypoints/masques) ; FBBC montre qu'il y a une demande réelle pour une couche d'analyse en aval. Ça dépasse le cadre actuel du TM, mais à noter comme direction future potentielle — et surtout, c'est une preuve indépendante que MediaPipe/RTMPose/DeepLabCut sont bien les outils que la communauté cible utilise concrètement aujourd'hui, ce qui valide leur choix comme candidats à comparer en `03_*`.

### 9.2 Reproductibilité et fiabilité — pas seulement la précision brute

Plusieurs sources convergent sur un point qui dépasse la simple comparaison de scores de précision :

- La communauté psychologie/neurosciences traverse une "crise de reproductibilité" documentée, et le choix de l'outil de mesure (y compris pour la pose estimation) est identifié comme un facteur pouvant contribuer à la variance entre études.
- La revue systématique sur le suivi markerless chez le rongeur (rats/souris, pas l'humain — mais l'argument structurel s'applique tout aussi bien) trouve que **DeepLabCut domine environ 81%** des publications utilisant ce type d'outil. Un tel niveau de concentration sur un seul outil est explicitement signalé par les auteurs comme un risque : si cet outil a un biais ou une limite, beaucoup de résultats pourraient en être affectés de la même façon. Argument transposable à psifx : justifie de garder plusieurs backends interchangeables (comme le patron actuel `tool.py`/`command.py` par backend le permet déjà) plutôt que de tout miser sur un seul modèle.
- Nouveau cadre d'évaluation 2026 (Fernandez, Bonnin, Fernandez-Maloigne, *Frontiers in Neuroinformatics*) : *From Pose Estimation Accuracy to Motor Biomarker Reliability*. Propose un score composite, le **Motor Biomarker Reliability Composite Score (MBRCS)**, combinant explicitement précision spatiale, **stabilité temporelle**, **robustesse à l'occlusion**, et fidélité cinématique — plutôt qu'une métrique de précision par frame isolée (PCK/MPJPE). Argument central des auteurs : deux architectures avec une précision par frame similaire peuvent diverger fortement sur ces autres dimensions, ce qui change directement les conclusions scientifiques tirées des données.

**Implication directe pour `03_*`** : ce dernier point formalise et confirme ce qui était déjà noté en Partie 1 (§4, point 2 : la stabilité temporelle est une dimension de précision à part entière, cf. Vacchetti et al. 2004) — un cadre 2026 dédié à l'évaluation en sciences du mouvement le confirme et propose une manière concrète de le mesurer. `03_*` devrait envisager d'adopter une variante de ce type de score composite plutôt qu'un chiffre de précision unique, ce qui est cohérent avec la philosophie de psifx (viser des chercheurs en sciences humaines, pas seulement optimiser un score de benchmark vision par ordinateur).

## 10. Segmentation amodale — piste nouvelle, à garder en réserve

Pour compléter le pilier "segmentation" : la littérature récente (A2VIS, segmentation amodale vidéo par modèles de diffusion, VATrack — 2024-2026) travaille sur la **segmentation amodale** : estimer la forme complète d'un objet/personne même quand une partie est cachée par un autre objet ou une autre personne, plutôt que de segmenter seulement la partie visible. C'est exactement le scénario "deux personnes en interaction rapprochée" du cas d'usage psifx (§7). Ce n'est pas un besoin exprimé dans le cahier des charges actuel du TM, et aucun outil aussi mature/packagé que SAM3 ou MMPose n'a été trouvé pour cette tâche précise — à garder en réserve plutôt qu'à intégrer maintenant, cohérent avec la consigne du responsable de ne pas s'éparpiller.

## 11. Tableau d'impact — candidats vs effort d'intégration dans psifx

| Piste | Gain de précision attendu (littérature) | Effort d'intégration dans psifx | Risque principal | Priorité proposée |
|---|---|---|---|---|
| RTMPose/RTMW en remplacement ou complément de MediaPipe | Confirmé empiriquement par deux comparatifs indépendants (§6.2) : ViTPose/famille MMPose > MediaPipe | Faible-moyen : Apache 2.0, API Python stable (MMPose), suit le patron `tool.py`/`command.py` déjà en place | Aucun majeur identifié | **Haute** — piste la mieux étayée par la littérature indépendante |
| Sapiens2 | Potentiellement le plus précis (foundation model, +4 mAP pose / +24.3 mIoU seg vs génération précédente) mais pas comparé directement dans les comparatifs empiriques trouvés ici | Moyen-élevé : licence restrictive à faire valider (`01_*`), poids lourds | Licence, budget GPU | Moyenne — attendre validation licence avant d'investir du temps d'implémentation |
| SAM3.1 (Object Multiplex) | Confirmé : ~7× plus rapide en multi-objets, meilleur sur 6/7 benchmarks VOS vs SAM3 | Moyen : pas encore dans `transformers` (`01_*`) ; des travaux 2026 montrent que le re-prompting autour de SAM3 seul apporte déjà des gains sans changer de modèle (§8) | Dépendance à l'intégration `transformers` à venir | Moyenne — surveiller, et explorer le re-prompting autour de SAM3 actuel en attendant |
| Méthodes robustes à l'occlusion (type DEKR/CrowdPose) | Gains ciblés sur scènes à occlusion/contact rapproché (+2 à +2.5 AP sur CrowdPose) — directement pertinent pour le cas d'usage psifx | Moyen-élevé : changerait le backend de détection multi-personnes, pas un simple remplacement de poids | Complexité d'intégration plus grande que RTMPose | Basse pour l'instant, à réévaluer selon les résultats `03_*` sur occlusion |
| DeepLabCut (comme backend optionnel) | Rapide, très utilisé dans le contexte comportemental (81% des études rongeurs, présent dans FBBC/Behavior Senpai) mais précision/robustesse humaine moins documentée que ViTPose/RTMPose dans les comparatifs trouvés | Faible-moyen techniquement, mais rupture de philosophie : nécessite un ré-entraînement par utilisateur, contrairement aux modèles zero-shot déjà intégrés | Moins "prêt à l'emploi" que le reste de psifx | Basse — à mentionner comme option, pas à intégrer par défaut |
| Segmentation amodale | Pertinent en théorie pour occlusion mutuelle forte (§10), mais aucun outil mature/packagé identifié | Élevé : pas de bibliothèque stable équivalente à MMPose/`transformers` trouvée | Immaturité de la piste | Basse — à garder en réserve |

## 12. Mise à jour des implications pour `03_*`

- Adopter, en plus des métriques standard déjà listées en §3, une évaluation composite type MBRCS (précision spatiale + stabilité temporelle + robustesse à l'occlusion + fidélité) pour la dimension vidéo/tracking du benchmark, plutôt qu'un score de précision isolé.
- Inclure un sous-test occlusion/contact rapproché (CrowdPose en benchmark secondaire, et/ou test direct sur `example/data/Video.mp4` avec occlusion volontaire), en plus des jeux de données standards où les personnes sont bien séparées.
- Comparer explicitement RTMPose/RTMW contre MediaPipe sur les mêmes séquences en priorité — c'est la piste la mieux étayée empiriquement par la littérature indépendante (pas seulement les chiffres publiés par les auteurs de chaque modèle).
- Documenter les questions de licence à trancher (Sapiens2) avant d'y consacrer du temps d'implémentation.
- Ne pas se disperser sur la segmentation amodale ou le ré-entraînement type DeepLabCut pour l'instant — à noter comme pistes de réserve, pas à benchmarker en premier lieu, conformément à la consigne de temps raisonnable donnée en réunion de lancement.

## Sources

- [Statistical shape model based segmentation of medical images (Crossref)](https://doi.org/10.1016/S0895-6111(98)00015-9)
- [Elastic model-based segmentation of 3-D neuroradiological data sets — PubMed](https://pubmed.ncbi.nlm.nih.gov/10628943/)
- [Elastic Model-Based Segmentation of 3-D (PDF, sci.utah.edu)](https://www.sci.utah.edu/~gerig/publications/TMI99-Gerig-Kelemen.pdf)
- [Stable Real-Time 3D Tracking using Online and Offline Information (PDF)](https://vincentlepetit.github.io/files/papers/comp_vacchetti_pami04.pdf)
- [Model Based Training, Detection and Pose Estimation of Texture-Less 3D Objects in Heavily Cluttered Scenes (PDF)](https://vincentlepetit.github.io/files/papers/comp_hinterstoisser_accv12.pdf)
- [Articulated Pose Estimation by a Graphical Model with Image Dependent Pairwise Relations (NeurIPS 2014)](https://proceedings.neurips.cc/paper_files/paper/2014/file/6ac19afba0591c506d495d9d6bed5802-Paper.pdf)
- [FoundPose: Unseen Object Pose Estimation with Foundation Features (arXiv 2311.18809)](https://arxiv.org/pdf/2311.18809)
- [FoundationPose: Unified 6D Pose Estimation and Tracking of Novel Objects (project page, NVIDIA)](https://nvlabs.github.io/FoundationPose/)
- [FoundationPose (arXiv 2312.08344, résultats YCB-Video)](https://arxiv.org/html/2312.08344v2)
- [GCE-Pose: Global Context Enhancement for Category-level Object Pose Estimation (CVPR 2025)](https://colin-de.github.io/GCE-Pose/)
- [GCE-Pose (arXiv 2502.04293, tableaux de résultats)](https://arxiv.org/pdf/2502.04293)
- [Human Pose Estimation in 2D and 3D: A Survey of Analytical Methods, Benchmarking Frameworks, and Engineering Applications (JETA, août 2026)](https://doi.org/10.3390/jeta4030028)
- [Evaluating MediaPipe, YOLOv8, and VitPose for dynamic circus motion analysis (Multimedia Tools and Applications, 2026)](https://link.springer.com/article/10.1007/s11042-026-21316-4)
- [Bottom-Up Human Pose Estimation Via Disentangled Keypoint Regression — DEKR (CVPR 2021)](https://arxiv.org/pdf/2104.02300)
- [Towards Real-Time Multi-Person Pose Estimation via Feature Selection and Sharpening Mechanisms (ScienceDirect)](https://www.sciencedirect.com/org/science/article/pii/S1526149226001062)
- [MOSEv2: A More Challenging Dataset for Video Object Segmentation in Complex Scenes](https://arxiv.org/pdf/2508.05630)
- [Advancing Complex Video Object Segmentation via Tracking-Enhanced Prompt: 1st Winner, 5th PVUW MOSE Challenge](https://arxiv.org/html/2604.00395v1)
- [Re-Prompting SAM 3 via Object Retrieval: 3rd of the 5th PVUW MOSE Track](https://arxiv.org/pdf/2603.23788)
- [Feature-based behavior coding for efficient exploratory analysis using pose estimation (Behavior Research Methods, PMC)](https://pmc.ncbi.nlm.nih.gov/articles/PMC12064611/)
- [Does advancement in marker-less pose-estimation mean more quality research? A systematic review (PMC)](https://pmc.ncbi.nlm.nih.gov/articles/PMC12411509/)
- [From Pose Estimation Accuracy to Motor Biomarker Reliability: A Reliability-Centred Evaluation Framework for Markerless Movement Analysis (Frontiers in Neuroinformatics, 2026)](https://www.frontiersin.org/journals/neuroinformatics/articles/10.3389/fninf.2026.1790241/abstract)
- [Comparing the accuracy of open-source pose estimation methods for measuring gait kinematics (ScienceDirect)](https://www.sciencedirect.com/science/article/abs/pii/S0966636222004738)
- [A2VIS: Amodal-Aware Approach to Video Instance Segmentation (ScienceDirect)](https://www.sciencedirect.com/science/article/abs/pii/S0262885625001313)
- [Using Diffusion Priors for Video Amodal Segmentation](https://arxiv.org/pdf/2412.04623)
