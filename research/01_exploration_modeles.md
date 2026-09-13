# Exploration des modèles — segmentation, pose estimation, tracking

Date : 2026-09-17
Contexte : première partie exploratoire du travail de master, suite à la réunion de lancement avec le responsable. Objectif du document : lister ce qui existe aujourd'hui en segmentation/tracking et pose estimation, évaluer l'intégrabilité dans psifx (licence, maturité, effort), sans y passer un temps disproportionné.

## 1. Rappel du cadrage (réunion de lancement)

- L'objectif général du TM ne change pas : étendre psifx avec les bibliothèques de vision par ordinateur les plus récentes en segmentation, tracking et pose estimation.
- Les modèles cités dans la description initiale (SAM + Sapiens) datent de plusieurs mois ; le responsable a explicitement demandé de vérifier s'ils restent pertinents ou si des modèles plus récents doivent être considérés à leur place — d'où cette phase exploratoire.
- Consigne explicite : rester raisonnable en temps passé sur l'exploration, il y a beaucoup de modèles disponibles et on peut facilement y perdre un temps disproportionné.
- Priorité : pose estimation et segmentation/tracking. Les approches génératives/VLM sont une possibilité évoquée mais **pas la priorité actuelle** — à ne creuser que si le temps le permet.
- Pas de contrainte de temps réel obligatoire : certains pipelines de recherche tournent hors-ligne pendant plusieurs heures/jours (fine-tuning inclus), donc le compromis vitesse/précision est ouvert selon le modèle.
- Le développeur principal de psifx, Matthew Vowels (UNIL), est une ressource externe à solliciter pour les discussions d'implémentation détaillées — à contacter rapidement.
- Rythme de suivi : point toutes les deux semaines, lundi 16h30 (flexible, ajustable si besoin ponctuel).

## 2. Ce qui est déjà en place dans psifx

Vérifié directement dans le code du repo (`psifx/video/`), pas supposé :

| Tâche | Backend actuel | Statut |
|---|---|---|
| Segmentation / tracking vidéo | **SAM3** (Meta, via `transformers.Sam3VideoModel`) | Déjà intégré (`psifx/video/tracking/sam3/`), remplace un ancien backend SAM2/SAMURAI (`absolutely-not-samurai`, visible dans l'historique git) |
| Pose estimation | **MediaPipe** (Google) | Déjà intégré (`psifx/video/pose/mediapipe/`), léger et temps réel, mais pas le plus précis pour un usage recherche |
| Visage | **OpenFace** (fork Baltrusaitis) | Déjà intégré, licence non-commerciale/recherche uniquement |

Conclusion : la partie "SAM" de la description initiale du TM est déjà largement traitée par un contributeur précédent. Le travail à venir porte donc surtout sur (a) vérifier si SAM3 doit être mis à jour, et (b) combler le manque sur la pose estimation "recherche" (au-delà de MediaPipe).

Architecture à respecter (déjà en place, cohérente sur tout le repo) : un module par backend sous `psifx/video/<tâche>/<backend>/`, avec `tool.py` (logique) + `command.py` (CLI argparse), une classe de base commune par tâche (ex. `PoseEstimationTool`, `TrackingTool`), et les dépendances déclarées dans `requirements.txt` (lues dynamiquement par `pyproject.toml`). Toute intégration nouvelle doit suivre ce même patron pour rester cohérente avec la philosophie du projet.

## 3. Segmentation & tracking

### SAM3 (déjà intégré) vs SAM 3.1

- **SAM 3** : modèle actuel dans psifx, segmentation promptable par texte ou points/boîtes, avec suivi vidéo. Intégré via `transformers` (support officiel HF depuis le 19/11/2025).
- **SAM 3.1** (Meta, sorti le 27/03/2026) : introduit *Object Multiplex*, un mécanisme de mémoire partagée qui traite plusieurs objets suivis conjointement au lieu de manière indépendante — Meta rapporte un gain de vitesse d'environ 7× à 128 objets sur un seul GPU H100, avec une amélioration de la précision de suivi vidéo sur 6 des 7 benchmarks testés.
- **Point d'attention pour l'intégration** : à la date de cette recherche, les checkpoints SAM 3.1 ne sont **pas encore supportés par `transformers`** (seuls SAM 3 et SAM 3 Video le sont) — le dépôt HF de SAM 3.1 ne contient que les poids, pas d'intégration `transformers`. Utiliser SAM 3.1 aujourd'hui demanderait de passer par le dépôt officiel `facebookresearch/sam3` directement plutôt que par `transformers`, ce qui est un changement d'architecture d'intégration non négligeable (nouvelle dépendance, API différente) plutôt qu'un simple changement de checkpoint.
- **Recommandation** : rester sur SAM3 (`transformers`) pour l'instant, mais surveiller le support `transformers` de SAM 3.1 — le gain de performance (7× sur du multi-personnes) est directement pertinent pour le cas d'usage psifx (plusieurs personnes filmées en interaction), donc à migrer dès que l'intégration officielle existe. Question à poser à Matthew Vowels : a-t-il déjà un avis/une expérience sur SAM 3.1 ou une préférence d'intégration (via `transformers` uniquement, ou aussi via le repo Meta natif) ?

## 4. Pose estimation

C'est le principal manque actuel (MediaPipe seul, pensé pour le temps réel/mobile plutôt que pour la richesse d'un usage recherche). Deux pistes sérieuses identifiées, avec un compromis licence/maturité très différent :

### 4.1 Sapiens2 (Meta, avril 2026)

- Deuxième génération de la famille de modèles fondation "Sapiens" (la première version, d'août 2024, est celle citée dans la description initiale du TM).
- Couvre 2D pose estimation, segmentation par partie du corps, depth, normales de surface et albedo, en une seule famille de modèles (0.4B à 5B paramètres), résolution native jusqu'à 4K.
- Gain rapporté vs génération précédente : +4 mAP en pose, +24.3 mIoU en segmentation.
- **Licence** : licence Meta personnalisée ("Sapiens2 License"), pas Apache/MIT. Usage et travaux dérivés autorisés sous conditions, avec :
  - obligation de citer l'usage de "Sapiens Materials" dans toute publication ;
  - interdictions explicites d'usage : surveillance et traitement biométrique, deepfakes/contenu trompeur, ré-identification de personnes, pratique professionnelle non autorisée (médical/légal/financier), usage militaire/armement/espionnage/infrastructures critiques.
  - Ces restrictions sont à lire attentivement : psifx traite de la vidéo de vraies personnes (contexte recherche en sciences humaines) — l'usage visé (codage comportemental en recherche académique) semble clairement hors du champ des interdictions, mais ça vaut la peine de le confirmer explicitement avec Matthew Vowels avant d'intégrer, par analogie avec la clause déjà existante pour OpenFace (licence non-commerciale, académique/recherche uniquement) dans la doc psifx actuelle.
- Comparable en esprit à une licence type CC-BY-NC mais avec des clauses d'usage plus prescriptives (façon licence Llama).

### 4.2 ViTPose / RTMPose / RTMW (OpenMMLab, MMPose)

- **ViTPose** : transformer pur (ViT) + tête de décodage légère, jusqu'à 1B de paramètres (ViTPose-G), état de l'art sur COCO/AI Challenger/MPII/OCHuman et benchmarks animaux (AP-10K, APT-36K). Licence **Apache 2.0**.
- **RTMPose** : orienté temps réel, ex. RTMPose-m à 75.8% AP COCO tout en tournant à 90+ FPS sur CPU (Intel i7) ou 430+ FPS sur GPU grand public (GTX 1660 Ti). Licence **Apache 2.0** (framework MMPose).
- **RTMW** (whole-body) : variante corps entier (mains, visage, corps), RTMW-l atteint 70.2 mAP sur COCO-Wholebody — premier modèle open source à dépasser 70 mAP sur ce benchmark. Même licence Apache 2.0.
- Écosystème mature (OpenMMLab), maintenu activement, API Python stable.

### 4.3 Comparaison synthétique

| Modèle | Tâches couvertes | Licence | Maturité intégration | Points forts | Points d'attention |
|---|---|---|---|---|---|
| Sapiens2 | pose + seg + depth + normals (foundation model unique) | Restrictive (Meta, façon Llama) | Récent (04/2026), poids HF disponibles | SOTA, richesse (une seule famille pour plusieurs tâches), haute résolution | Licence à faire valider par le responsable/l'équipe légale UNIL avant usage large ; modèle lourd (jusqu'à 5B) |
| ViTPose(-G) | pose 2D uniquement | Apache 2.0 | Mature, stable | Précision SOTA pose seule, licence sans friction | Ne couvre que la pose (pas seg/depth) |
| RTMPose / RTMW | pose 2D (+ corps entier pour RTMW) | Apache 2.0 | Très mature (MMPose), gros écosystème | Temps réel sur CPU, whole-body dispo, licence sans friction | Moins "foundation model" que Sapiens2, précision légèrement en retrait sur les cas complexes |
| SAM3 (actuel) | segmentation + tracking vidéo | Apache-compatible via `transformers` | Déjà intégré | — | — |
| SAM 3.1 | segmentation + tracking multi-objets accéléré | idem SAM3 (Meta) | Poids dispo, **pas encore dans `transformers`** | ~7× plus rapide en multi-personnes | Nécessiterait une intégration via le repo natif Meta, pas juste un changement de checkpoint |

## 5. Approches génératives / VLM (déprioritisé pour l'instant)

Évoqué en réunion comme piste possible mais pas prioritaire. Recherche rapide (volontairement non approfondie, conformément à la consigne de temps) :

- Des travaux existent (LLaVA-Pose, HumanVLM, benchmarks comme NextMotionQA/MMHU) qui visent à faire comprendre pose/action humaine à des modèles vision-langage plutôt que par des keypoints classiques.
- Constat : ce sont pour l'instant des travaux de recherche/benchmarks, pas des bibliothèques packagées prêtes à l'emploi comparables à MMPose ou aux poids Sapiens/SAM — intégration nettement plus coûteuse et moins stable à ce stade.
- **Recommandation** : ne pas investiguer plus avant pour l'instant ; à reconsidérer si un besoin précis (au-delà de la simple extraction de keypoints/masques) émerge plus tard dans le projet.

## 6. Prochaines étapes proposées

1. Valider ce cadrage avec Matthew Vowels (contact à prendre rapidement, comme suggéré en réunion) : a-t-il un avis sur Sapiens2 vs ViTPose/RTMPose pour la pose estimation, et sur SAM3 vs 3.1 pour le tracking ?
2. Clarifier la question de licence Sapiens2 avec le responsable avant de s'engager dans cette voie (usage académique clairement hors des interdictions listées, mais à faire confirmer formellement étant donné que psifx distribue publiquement via PyPI/Docker).
3. Si Sapiens2 est écarté pour raison de licence, RTMW (whole-body, Apache 2.0) est la piste la plus solide pour combler le manque actuel de psifx sur la pose estimation "recherche" au-delà de MediaPipe.
4. Poser la question du calendrier/effort attendu pour un éventuel changement d'architecture d'intégration si SAM 3.1 est retenu plus tard (dépendance au support `transformers`, à re-vérifier périodiquement).

## Sources

- [Sapiens: Foundation for Human Vision Models (arXiv)](https://arxiv.org/abs/2408.12569)
- [Meta AI Releases Sapiens2 (MarkTechPost)](https://www.marktechpost.com/2026/04/27/meta-ai-releases-sapiens2-a-high-resolution-human-centric-vision-model-for-pose-segmentation-normals-pointmap-and-albedo/)
- [facebook/sapiens2 · Hugging Face](https://huggingface.co/facebook/sapiens2)
- [sapiens2/LICENSE.md · facebookresearch/sapiens2 (GitHub)](https://github.com/facebookresearch/sapiens2/blob/main/LICENSE.md)
- [OpenPose, MediaPipe Pose, RTMPose, ViTPose — The Pose-Tracking Stack For Video In 2026](https://www.forasoft.com/learn/ai-for-video-engineering/articles-ai/openpose-mediapipe-rtmpose-pose-tracking)
- [ViTPose License (Apache 2.0) — GitHub](https://github.com/ViTAE-Transformer/ViTPose/blob/main/LICENSE)
- [RTMW: Real-Time Multi-Person 2D and 3D Whole-body Pose Estimation (arXiv)](https://arxiv.org/html/2407.08634v1)
- [RTMPose: Real-Time Multi-Person Pose Estimation based on MMPose (arXiv)](https://arxiv.org/html/2303.07399v2)
- [SAM 3.1: Object Multiplex and Video Tracking (LearnOpenCV)](https://learnopencv.com/sam-3-whats-new/)
- [SAM 3.1: Faster and More Accessible Real-Time Video Detection and Tracking (Meta AI Blog)](https://ai.meta.com/blog/segment-anything-model-3/)
- [SAM3 · Transformers documentation (Hugging Face)](https://huggingface.co/docs/transformers/model_doc/sam3)
- [HumanVLM: Foundation for Human-Scene Vision-Language Model (ScienceDirect)](https://www.sciencedirect.com/science/article/abs/pii/S1566253525003446)
- [LLaVA-Pose: Enhancing Human Pose and Action Understanding via Keypoint-Integrated Instruction Tuning (arXiv)](https://arxiv.org/pdf/2506.21317)
