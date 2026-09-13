# Modèles et méthodes d'optimisation — pistes pour de futures implémentations

Date : 2026-09-28
Contexte : suite au benchmark (`03_*`), qui a confirmé ViTPose comme meilleur candidat pour la précision mais a aussi mis en évidence deux problèmes concrets — (a) ViTPose n'a pas de détecteur propre et dépend de celui de RTMPose, (b) aucune taille de Sapiens2 n'est assez rapide pour de la vidéo. Ce document rassemble, sans encore implémenter, (1) des modèles supplémentaires qui répondent directement à ces deux problèmes ou élargissent le choix, et (2) des méthodes d'optimisation d'inférence qui pourraient rendre les candidats déjà identifiés (ViTPose, Sapiens2) praticables. Objectif : préparer la discussion avec le responsable/Matthew Vowels, pas encore trancher.

## 1. Modèles supplémentaires identifiés

### 1.1 RTMO — répond directement à deux limites déjà documentées

*Towards High-Performance One-Stage Real-Time Multi-Person Pose Estimation*, même famille OpenMMLab que RTMPose (déjà intégré dans `research/benchmark/`, `rtmlib` le supporte aussi).

- **Un seul étage** : pas de détecteur de personnes séparé — élimine directement la limite documentée en `03_resultats.md` §4 ("ViTPose dépend des boîtes du détecteur RTMPose").
- **Latence quasi indépendante du nombre de personnes** dans l'image (rapporté : +0.5% de latence entre 1 et 10+ personnes) — résout aussi, par construction, le problème de mur mémoire/temps rencontré avec Sapiens2 sur des images à forte densité (`03_resultats.md` §2.2).
- Précision comparable à RTMPose à taille égale (RTMO-m/l ≈ RTMPose-m/l), et **plus rapide que RTMPose dès qu'il y a plus de ~4 personnes** dans l'image.
- Licence : même famille Apache 2.0 que RTMPose.

**Recommandation** : candidat naturel pour un prochain passage de benchmark, en remplacement ou en complément de RTMPose — coûte peu à tester puisque `rtmlib` le supporte déjà (`rtmlib.Body` avec un preset RTMO, cf. code source consulté en `03_*`).

### 1.2 Modèles unifiés multi-tâches (Ultralytics YOLO26) — piste puissante, bloquée par la licence

*Ultralytics YOLO26: Unified Real-Time End-to-End Vision Models* (janvier 2026). Un seul pipeline couvre **détection, segmentation d'instance, pose, classification, détection orientée**, avec un mode de suivi (tracking) compatible avec tous ces modes. Inférence de bout en bout sans NMS. Performance rapportée : 40.9-57.5 mAP COCO à 1.7-11.8 ms de latence (TensorRT, T4) selon la taille de modèle (n/s/m/l/x).

**Pourquoi c'est notable pour ce TM précisément** : le titre du travail couvre segmentation + pose + tracking — YOLO26 couvre les trois dans une seule bibliothèque, avec une API cohérente, potentiellement plus simple à intégrer dans psifx qu'une combinaison SAM3 + ViTPose/RTMO + logique de suivi séparée. `ultralytics` est déjà une dépendance **présente mais inutilisée** dans `requirements.txt` de psifx (héritage d'une intégration SAMURAI/SAM2 antérieure, retirée depuis — vérifié directement dans le code).

⚠️ **Licence bloquante à clarifier avant tout investissement de temps** : Ultralytics distribue sous **AGPL-3.0**, une licence copyleft forte — contrairement à Apache 2.0 (psifx), l'AGPL exige que le code source complet de toute œuvre dérivée soit publié, **y compris pour un usage réseau/SaaS**, pas seulement en cas de redistribution binaire. Une licence commerciale Ultralytics Enterprise existe pour contourner cette contrainte, mais implique un coût et un contact commercial. C'est structurellement le même type de blocage que la licence Sapiens2 déjà signalée en `01_*` §4.1 — à traiter avec la même prudence : **ne pas intégrer sans validation explicite du responsable**, malgré l'attrait technique.

### 1.3 DETRPose — sérieux candidat, meilleur profil licence que RTMO/YOLO26

*Real-Time End-to-End Multi-Person Pose Estimation via Modified Transformer Decoder and Novel Denoising Keypoints* (juin 2025, code+poids publiés). Un-étage/bout-en-bout comme RTMO (pas de détecteur séparé — même bénéfice qu'en §1.1), mais avec un décodeur transformeur (variante de GroupPose/D-FINE) plutôt que l'architecture YOLO de RTMO.

**Chiffres concrets** (COCO val2017, Tesla V100, fp16, TensorRT 8.6.3) :

| Taille | AP | Latence |
|---|---|---|
| DETRPose-S | 67.0 | 4.99 ms |
| DETRPose-M | 69.4 | 7.01 ms |
| DETRPose-L | 72.5 | 9.50 ms |

Comparé à RTMO (état de l'art un-étage jusqu'ici) : DETRPose-L a moins de paramètres, une latence plus faible, une convergence à l'entraînement plus rapide, un AP quasi identique (-0.4 point) et un AR supérieur (+3.2 points) — DETRPose-S égale même les plus grandes variantes YOLOv8-Pose/YOLO11-Pose sur COCO val.

**Licence : Apache 2.0** (confirmé directement dans le dépôt) — contrairement à YOLO26 (AGPL-3.0, §1.2) et SAM-pose2seg (GPL-3.0, §1.4), **aucune friction de licence** pour psifx. Python 3.11 (compatible avec `psifx-env` tel quel, pas besoin d'un environnement séparé comme pour Sapiens2), PyTorch standard.

**C'est probablement le meilleur candidat "un-étage" à tester en premier** parmi ceux identifiés dans ce document : license compatible d'emblée, chiffres publiés meilleurs que RTMO sur le papier, pas de détecteur externe à gérer. À vérifier empiriquement dans un prochain passage de benchmark plutôt qu'à prendre pour argent comptant (même réserve méthodologique que pour tous les chiffres "publiés vs mesurés" déjà rencontrée dans ce TM).

### 1.4 SAM-pose2seg / BBoxMaskPose — idée intéressante, deux vrais obstacles

Adapte SAM 2.1 pour la segmentation guidée par la pose : les points-clés (déjà produits par un modèle de pose comme ViTPose) servent de points d'invite pour affiner le masque de segmentation, particulièrement en scène de foule/occlusion. Fait partie d'un projet plus large, **BBoxMaskPose (BMP)**, qui enchaîne boîte englobante → masque → pose de façon itérative ("Closing the Virtuous Circle", ICCV 2025). SAM-pose2seg en est le composant de segmentation ; a reçu un Best Paper Award à CVWW 2026.

**Résultats rapportés** : 44.6 AP sur COCO val2017, 60.3 AP sur COCOPersons, 34.7 AP sur OCHuman test, 72.7 AP sur CIHP val.

**Deux obstacles concrets, pas juste "encore de la recherche"** :
1. **Licence GPL-3.0** (confirmée dans le dépôt `MiraPurkrabek/BBoxMaskPose`) — copyleft fort, même famille de problème que Sapiens2/YOLO26 pour un projet Apache 2.0 comme psifx (moins strict que l'AGPL de YOLO26 puisqu'il ne se déclenche qu'à la distribution, pas à l'usage réseau/SaaS, mais toujours incompatible pour une intégration directe sans clarification).
2. **Pile de dépendances lourde** : combine MMDetection, MMPose 2.0, ViTPose, SAM 2.1, `mmcv==2.1.0`, `mmdet==3.3.0`, `mmpretrain==1.2.0`, PyTorch 2.1.2/CUDA 12.1 — un empilement bien plus lourd qu'un simple ajout de backend, plus proche d'un environnement de recherche complet que d'une bibliothèque à intégrer.

Une API d'inférence simple existe bien (classes `PMPose`/`BBoxMaskPose`, poids auto-téléchargés depuis HuggingFace, démo Jupyter/HuggingFace Spaces) — donc pas irréalisable à tester ponctuellement. Mais entre la licence et le poids de la pile, ce n'est **pas un candidat d'intégration à court terme** pour psifx — à mentionner au responsable comme piste de recherche intéressante (synergie pose↔segmentation), pas comme prochaine étape d'implémentation.

### 1.5 MobileSAM / EfficientSAM / FastSAM — deux sur trois sans problème de licence

Variantes allégées/distillées de SAM pour le pilier segmentation/tracking (pas un besoin actuel — SAM3 fonctionne déjà dans psifx — mais utile si la vitesse devient un jour un problème pour ce pilier aussi, comme c'est déjà le cas pour Sapiens2 côté pose).

| Modèle | Vitesse | Précision (AP) | Licence |
|---|---|---|---|
| MobileSAM | ~10 ms/image (8 ms encodeur + 4 ms décodeur), encodeur ~100× plus petit que SAM original | ~41 | **Apache 2.0** (vérifié) |
| EfficientSAM | 10-20 images/s | ~45 (meilleur des trois) | **Apache 2.0** (vérifié) |
| FastSAM | ~40 ms/image, ~100 images/s | ~37 (le plus faible) | **AGPL-3.0** (construit sur l'architecture YOLOv8 d'Ultralytics — même famille de problème que YOLO26, §1.2) |

MobileSAM et EfficientSAM sont donc tous deux directement compatibles avec la licence de psifx, contrairement à FastSAM. Si le pilier segmentation/tracking a un jour besoin d'une option plus légère que SAM3, **EfficientSAM (meilleure précision) ou MobileSAM (plus rapide)** sont les choix à tester en priorité — pas FastSAM malgré sa popularité, pour la même raison de licence que YOLO26.

## 2. Méthodes d'optimisation d'inférence

Objectif différent des modèles ci-dessus : rendre les candidats **déjà identifiés** (ViTPose pour la précision, Sapiens2 si la licence est validée) réellement utilisables, sans changer de modèle. Classées par coût d'implémentation croissant.

### 2.1 `torch.compile` + Flash Attention — gain "gratuit", sans ré-entraînement

Pour les modèles PyTorch (ViTPose via `transformers`, Sapiens2) : `torch.compile(mode="reduce-overhead")` combiné à Flash Attention donne un gain rapporté de 20-40% de latence sur des transformeurs, sans toucher aux poids ni ré-entraîner. Flash Attention seule apporte 2-4× de vitesse et une réduction de mémoire de 10-20× sur le calcul d'attention — potentiellement significatif pour Sapiens2 à haute résolution (1024×768), où l'attention sur de longues séquences de patches est probablement un poste de coût important.

**Coût d'implémentation** : très faible — quelques lignes autour du chargement du modèle dans `sapiens2_infer.py`/`ViTPoseBackend`. **Premier candidat à tester** avant d'envisager les options plus lourdes ci-dessous, en particulier pour essayer de rendre Sapiens2 ne serait-ce qu'un peu plus praticable.

### 2.2 Quantification (INT8/FP8) via ONNX Runtime + TensorRT

RTMPose est déjà déployé en ONNX dans `research/benchmark/` (via `rtmlib`/`onnxruntime`) — un chemin de quantification INT8 existe donc déjà en pratique pour ce backend. Pour ViTPose/Sapiens2 (actuellement PyTorch pur), un export ONNX puis une quantification via NVIDIA ModelOpt/TensorRT est documenté comme apportant un gain de 2-10× selon l'architecture (ex. 10× rapporté sur MobileNetV2/T4 ; gains plus modestes mais réels sur des encodeurs transformer, ~1.4× FP8 vs FP16 rapporté sur un cas comparable).

**Coût d'implémentation** : moyen — export ONNX d'un modèle `transformers` est généralement direct (`torch.onnx.export` ou l'export intégré de `transformers`), mais la calibration pour la quantification (surtout INT8) demande un jeu de données représentatif et une étape de validation de précision post-quantification (le sous-échantillon COCO de `research/benchmark/` pourrait servir directement à ça).

### 2.3 Distillation de connaissances — le seul chemin réaliste vers "précision Sapiens2, vitesse RTMPose"

Si l'objectif est un jour d'obtenir une précision proche de Sapiens2 à une vitesse compatible avec de la vidéo, la quantification/compilation seules n'y suffiront probablement pas (Sapiens2-0.4B reste ~20× plus lent que RTMPose même en bf16 optimisé) — la distillation consiste à entraîner un **petit modèle étudiant** à reproduire les sorties d'un grand modèle enseignant. Approches trouvées, applicables au cas de psifx :
- **OKDHP** (*Online Knowledge Distillation for Efficient Pose Estimation*) — distillation en une seule passe d'entraînement, sans modèle enseignant pré-entraîné séparé à maintenir.
- **MobileSAM** a utilisé une distillation "découplée" (juste l'encodeur d'image, pas tout le pipeline) reproductible sur un seul GPU en moins d'une journée — preuve que ce n'est pas forcément un projet démesuré.
- Travaux spécifiques à la pose 6D avec distillation sensible à l'incertitude (2026) — pertinent si psifx élargit un jour vers la pose 3D/6D.

**Coût d'implémentation** : élevé — nécessite un jeu de données d'entraînement, une boucle d'entraînement complète, et une validation soigneuse. **Pas une piste à court terme** pour ce TM, mais la bonne réponse de fond si "précision Sapiens2 en temps réel" devient un objectif explicite plus tard — à noter pour la discussion avec le responsable plutôt qu'à entreprendre maintenant.

### 2.4 Élagage structuré (pruning) — mentionné pour complétude, pas prioritaire

Un survey 2025 (*Vision Transformers on the Edge*) couvre la question en détail. Point méthodologique important trouvé : **l'élagage non structuré n'apporte quasiment aucun gain de vitesse réel** malgré une sparsité élevée sur le papier — seul l'élagage structuré (par bloc/canal, aligné sur la façon dont le matériel exécute les multiplications matricielles) donne un gain mesurable. Plus complexe à mettre en œuvre correctement que la quantification ou `torch.compile`, avec un risque de perte de précision mal maîtrisé sans une méthode récente et soigneuse (ex. CORP, 2026, orienté préservation de la représentation).

**Coût d'implémentation** : élevé, gain incertain sans expertise dédiée — à ne considérer qu'après avoir épuisé les options plus simples (§2.1, §2.2).

## 3. Synthèse — comment ces pistes s'articulent

| Piste | Répond à quel problème identifié en `03_*` | Coût | Priorité proposée |
|---|---|---|---|
| RTMO | ViTPose sans détecteur propre ; instabilité/OOM Sapiens2 sur scènes peuplées | Faible (déjà supporté par `rtmlib`) | **Haute** — à tester dans un prochain passage de benchmark |
| DETRPose | Idem RTMO, avec licence Apache 2.0 native et chiffres publiés légèrement meilleurs | Faible-moyen (nouveau dépôt à intégrer, pas encore dans `rtmlib`) | **Haute** — à tester en parallèle de RTMO |
| `torch.compile` + Flash Attention | Lenteur de Sapiens2/ViTPose | Très faible | **Haute** — gain rapide à essayer sur les backends déjà en place |
| Quantification ONNX/TensorRT | Lenteur de Sapiens2/ViTPose pour un déploiement plus large | Moyen | Moyenne — après §2.1, si le gain ne suffit pas |
| YOLO26 (Ultralytics) | Simplification architecturale (un seul modèle pour pose+seg+tracking) | Faible techniquement, **bloqué par la licence AGPL-3.0** | À soumettre au responsable avant tout investissement |
| MobileSAM / EfficientSAM | Alternative légère à SAM3 pour le pilier segmentation/tracking, si la vitesse y devient un jour un problème | Faible, licence Apache 2.0 clean | Basse — pas de besoin actuel identifié (SAM3 fonctionne), à garder en réserve |
| SAM-pose2seg / BBoxMaskPose | Synergie pose↔segmentation (idée de recherche, pas un besoin exprimé) | Élevé (pile MMDetection/MMPose/SAM2.1 lourde), **bloqué par la licence GPL-3.0** | Basse — à mentionner au responsable comme piste, pas à intégrer |
| Distillation | "Précision Sapiens2, vitesse temps réel" si ça devient un objectif explicite | Élevé | Basse pour l'instant — à garder en réserve |
| Élagage structuré | Idem distillation, alternative plus incertaine | Élevé, gain incertain | Basse |
| FastSAM | Alternative à SAM3 | — | **Écarté** — AGPL-3.0, même blocage que YOLO26 |

## Sources

- [RTMO: Towards High-Performance One-Stage Real-Time Multi-Person Pose Estimation (CVPR 2024)](https://arxiv.org/html/2312.07526v1)
- [RTMPose vs RTMO vs DWPose vs RTMW — discussion officielle mmpose](https://github.com/open-mmlab/mmpose/discussions/3135)
- [Ultralytics YOLO26: Unified Real-Time End-to-End Vision Models (arXiv)](https://arxiv.org/abs/2606.03748)
- [Ultralytics YOLO26 — documentation officielle](https://docs.ultralytics.com/models/yolo26)
- [Ultralytics License (AGPL-3.0 / Enterprise)](https://www.ultralytics.com/license)
- [DETRPose: Real-Time End-to-End Multi-Person Pose Estimation](https://arxiv.org/pdf/2506.13027)
- [DETRPose — dépôt officiel (licence Apache 2.0, chiffres S/M/L)](https://github.com/SebastianJanampa/DETRPose)
- [SAM-pose2seg: Pose-Guided Human Instance Segmentation in Crowds](https://www.alphaxiv.org/abs/2601.08982)
- [BBoxMaskPose — dépôt officiel (licence GPL-3.0, API PMPose/BBoxMaskPose)](https://github.com/MiraPurkrabek/BBoxMaskPose)
- [Faster Segment Anything: Towards Lightweight SAM for Mobile Applications (MobileSAM)](https://arxiv.org/html/2306.14289)
- [MobileSAM — dépôt officiel (licence Apache 2.0)](https://github.com/ChaoningZhang/MobileSAM)
- [EfficientSAM — dépôt officiel (licence Apache 2.0)](https://github.com/yformer/EfficientSAM)
- [Awesome-Efficient-Segment-Anything — comparatif MobileSAM/FastSAM/EfficientSAM](https://github.com/czg1225/Awesome-Efficient-Segment-Anything)
- [Model Quantization and Hardware Acceleration for Vision Transformers: A Comprehensive Survey](https://arxiv.org/pdf/2405.00314)
- [NVIDIA Model-Optimizer (quantization/distillation/pruning toolkit)](https://github.com/NVIDIA/Model-Optimizer)
- [Model Quantization: FP8 Checkpoints with NVIDIA TensorRT](https://developer.nvidia.com/blog/model-quantization-turn-fp8-checkpoints-into-high-performance-inference-engines-with-nvidia-tensorrt/)
- [How does torch.compile speed up a transformer?](https://www.adamcasson.com/posts/torch-compile-vit)
- [FlexAttention: The Flexibility of PyTorch with the Performance of FlashAttention (PyTorch Blog)](https://pytorch.org/blog/flexattention/)
- [Vision Transformers on the Edge: A Comprehensive Survey of Model Compression and Acceleration Strategies](https://arxiv.org/pdf/2503.02891)
- [Online Knowledge Distillation for Efficient Pose Estimation (OKDHP)](https://arxiv.org/pdf/2108.02092)
- [Uncertainty-Aware Knowledge Distillation for Compact and Efficient 6DoF Pose Estimation](https://arxiv.org/pdf/2503.13053)
