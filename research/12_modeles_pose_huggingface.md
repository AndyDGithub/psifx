# Modèles de pose disponibles sur Hugging Face, mesurés sur le talk-show annoté

Date : 2026-11-03. Branche : `eval/pose-models-hf`. Scripts : `research/pose_annotation/models.py` (un
même appel pour chaque modèle) et `compare.py` (précision et vitesse). Résultats :
`.e2e_outputs/extrait_talkshow/pose_annotation/pose_models.json`, avec les prédictions brutes dans
`predictions/`. Matériel : RTX 4060 Laptop (8 Go). Environnement : `psifx-poseeval`, un clone de
`psifx-dev` avec transformers 5.22.1 (nécessaire pour Sapiens2), rfdetr 1.11.2 et ultralytics 8.4.

## 1. Inventaire

J'ai interrogé l'API Hugging Face (tâche `keypoint-detection`, recherches « pose », « vitpose », « rtmw »,
« sapiens », etc.), soit 565 dépôts. Une fois retirés les modèles qui ne font pas de pose humaine (appariement
d'images, LoRA, ControlNet, animaux), il reste sept familles :

| Famille | Dépôts | Licence | Performance annoncée | Testé |
|---|---|---|---|---|
| ViTPose / ViTPose++ | `usyd-community/vitpose-*` (base-simple, base, coco-aic-mpii, plus-small, base, large, huge) | Apache 2.0 | 81,1 AP COCO test-dev (plus grand modèle) | oui |
| SynthPose | `stanfordmimi/synthpose-vitpose-{base,huge}-hf` (ViTPose réentraîné, 52 points dont les 17 COCO) | Apache 2.0 | OpenCapBench (biomécanique), pas d'AP COCO | oui |
| Sapiens2 pose | `facebook/sapiens2-pose-{0.4b,0.8b,1b,5b}` (308 points, entrée 1024×768) | Sapiens2 (Meta, usages restreints) | 76,9 à 82,3 mAP sur leur jeu de test interne, pas d'AP COCO | 0.4b, 0.8b, 1b |
| RTMPose / RTMO / RTMW | poids ONNX d'OpenMMLab via rtmlib ; RTMW aussi sur `akore/rtmw-*` (code distant) | Apache 2.0 | RTMW-x 384 : corps 76,3 AP, corps entier 70,2 AP | oui (rtmlib) |
| RF-DETR Keypoint (preview) | paquet `rfdetr`, exports sur `software-mansion/rf-detr-keypoint-*` | Apache 2.0 | 71,8 AP COCO, 9,7 ms (T4, TensorRT fp16) | oui |
| YOLO26-pose | Ultralytics, `onnx-community/yolo26n-pose-ONNX` | AGPL-3.0 | 57 à 71 AP selon la taille | oui, pour comparaison seulement |
| DETRPose | `SebasJanampa/DETRPose_X_COCO` | Apache 2.0 | 73,3 AP (X) | non : déjà mesuré (`05_optimisations_detrpose.md`), environnement séparé |
| BBoxMaskPose | `vrg-prague/BBoxMaskPose` | GPL-3.0 | 44,6 AP COCO (segmentation) | non : licence, pile MMCV |

Le corps 3D (SAM 3D Body) a sa propre branche (`10_sam3d_body.md`).

## 2. Protocole

- 200 images du talk-show, corrigées à la main (293 personnes, `ground_truth.json`).
- Chaque modèle traite chaque image entière.
  - Les modèles « top-down » (ViTPose, SynthPose, Sapiens2) reçoivent les boîtes du détecteur RT-DETR de
    psifx.
  - rtmlib utilise son détecteur YOLOX.
  - RTMO, RF-DETR et YOLO trouvent les personnes eux-mêmes, en une seule étape.
- Le temps est mesuré image par image, du cadre RGB jusqu'aux personnes, détecteur compris, avec
  synchronisation CUDA. Les 10 premières images servent d'échauffement et ne sont pas comptées.
- **OKS** : moyenne sur les personnes annotées ; une personne non trouvée compte 0. **trouvées** : part des
  personnes appariées avec un OKS ≥ 0,5. **PCK@0,1** : point placé à moins de 10 % de la diagonale de la
  personne, avec un score ≥ 0,3.
- **Points absents.** Les scores de RTMPose et RTMW (SimCC) ne sont pas compris entre 0 et 1. Un seuil fixe
  n'est donc pas comparable d'un modèle à l'autre : avec un seuil de 0,3, RTMW-x semblait inventer 100 % des
  points absents. Deux mesures indépendantes de l'échelle des scores sont donc utilisées :
  - **AUC** : probabilité qu'un point annoté ait un score plus élevé qu'un point annoté absent ;
  - **fantômes@95** : part des points absents conservés au seuil qui garde 95 % des points annotés.

## 3. Résultats

Une seule ligne par modèle, dans sa meilleure configuration de vitesse à précision égale. Les autres
configurations sont dans le JSON.

| Modèle | OKS | Trouvées | PCK@0,1 | Cachés PCK@0,1 | AUC absents | Fantômes@95 | ms/image | Mo GPU |
|---|---|---|---|---|---|---|---|---|
| ViTPose++ huge (pré-annotateur, biaisé) | 0,980 | 98,6 % | 0,995 | 0,990 | 0,961 | 13 % | 258 | 4148 |
| ViTPose++ large fp16 | **0,955** | 96,2 % | **0,986** | 0,962 | 0,960 | 14 % | 74 | 1171 |
| ViTPose++ base fp16 | 0,949 | 95,9 % | 0,982 | 0,962 | 0,954 | 16 % | 52 | 549 |
| ViTPose base coco-aic-mpii | 0,945 | 95,9 % | 0,974 | 0,919 | 0,953 | 16 % | 61 | 691 |
| ViTPose++ small | 0,943 | 96,2 % | 0,973 | 0,909 | 0,948 | 18 % | 54 | 426 |
| SynthPose huge | 0,944 | 96,2 % | 0,971 | 0,907 | 0,957 | 14 % | 216 | 3179 |
| Sapiens2 0.8b fp16 | 0,939 | 95,9 % | 0,977 | 0,943 | 0,950 | 20 % | 1156 | 2081 |
| **ViTPose base-simple (psifx)** fp16 | 0,938 | 95,6 % | 0,972 | 0,911 | 0,950 | 17 % | 48 | 471 |
| Sapiens2 0.4b fp16 | 0,937 | 95,9 % | 0,977 | 0,945 | 0,946 | 21 % | 710 | 1272 |
| RTMW-x 384 (wholebody-performance) fp16 | 0,923 | 94,5 % | 0,975 | **0,981** | 0,875 | 37 % | 30 | ONNX |
| RF-DETR Keypoint fp16 | 0,920 | 93,9 % | 0,938 | 0,809 | **0,988** | **5 %** | 18 | 785 |
| RTMPose-m (body-balanced) fp16 | 0,910 | 94,2 % | 0,967 | 0,957 | 0,929 | 28 % | 21 | ONNX |
| YOLO26-l (AGPL) | 0,907 | 93,2 % | 0,943 | 0,892 | 0,988 | 5 % | 20 | 190 |
| RTMO-s (rtmo-lightweight) | 0,894 | 92,2 % | 0,909 | 0,754 | 0,986 | 5 % | **8** | ONNX |
| **RTMO-m (psifx `body`, défaut)** fp16 | 0,841 | 86,0 % | 0,843 | 0,718 | 0,969 | 6 % | 10 | ONNX |
| RTMO-l | 0,815 | 84,0 % | 0,709 | 0,567 | 0,917 | — \* | 22 | ONNX |

\* Plus de 5 % des points annotés ont un score de 0 : le seuil qui garde 95 % des points vaut 0 et la mesure
n'a pas de sens.

### Ce que montrent les mesures

1. **Les tailles intermédiaires de ViTPose++ sont les meilleures**, avec ViTPose++ large fp16 en tête :
   OKS 0,955 et PCK 0,986, contre 0,938 et 0,972 pour le ViTPose actuel de psifx.
   - Il n'est que 1,5 fois plus lent que ViTPose base-simple fp16 (74 ms contre 48 ms).
   - Sur les points cachés, il passe de 0,911 à 0,962.
   - ViTPose++ base fp16 fait presque aussi bien, pour un temps proche de celui de psifx (52 ms).
   - Les deux sont dans transformers sous licence Apache 2.0 : il suffit de passer le `dataset_index` de
     l'expert COCO.
2. **Le score de ViTPose++ huge est gonflé par le biais d'ancrage.** Il a servi à pré-annoter, et son OKS
   (0,980) dépasse les autres de 2,5 points.
   - Le reste de la famille ViTPose++ profite sans doute d'une partie de ce biais.
   - L'écart entre ViTPose++ et les modèles d'autres familles est donc à considérer comme un maximum.
3. **Les modèles en une seule étape** (RF-DETR, YOLO26, RTMO) **sont les seuls qui reconnaissent bien
   l'absence d'un point.**
   - Leur AUC est d'environ 0,99 et ils ne gardent que 5 % de fantômes.
   - Les modèles top-down en gardent 13 à 37 %.
   - RF-DETR Keypoint est le meilleur compromis de ce groupe : OKS 0,920, 18 ms en fp16, licence Apache 2.0.
     Il reste en version préliminaire.
4. **Le modèle par défaut de psifx (`body`, RTMO-m) est l'un des moins bons sur ce talk-show.**
   - RTMO-m et RTMO-l ratent les personnes en plan rapproché, coupées à la taille.
   - RTMO-s fait mieux : OKS 0,894 contre 0,841.
   - Ce résultat contredit la mesure COCO (`05_optimisations_synthese.md`), où RTMO-m était le meilleur
     compromis. Le cadrage du talk-show (plans rapprochés) est absent de COCO.
5. **RTMW** (RTMPose corps entier, qui donnerait aussi le visage et les mains)
   - Il place le mieux les points cachés : 0,981.
   - Il sépare mal les points absents : AUC 0,875, 37 % de fantômes.
6. **Sapiens2 n'apporte rien sur le corps** pour ce cas, pour un coût 10 à 30 fois plus élevé (0,7 à 1,6 s par
   image en fp16). Son intérêt reste les 308 points (visage détaillé, mains, pieds).
7. **Optimisations.**
   - Le fp16 ne change jamais la précision (à 0,002 près) :
     - ViTPose++ large : −51 % de temps et −46 % de mémoire ;
     - Sapiens2 0.4b : −69 % de temps ;
     - RF-DETR : −59 % de temps ;
     - ONNX (RTMO, RTMW) : −15 à −25 % de temps.
   - `torch.compile` n'ajoute presque rien au fp16 (ViTPose : 48,2 contre 47,8 ms).
   - Le tracé `torchscript` de RF-DETR seul gagne 14 % mais triple la mémoire. Combiné au fp16, il apporte le
     gain principal.
   - Le fp16 de ViTPose ne gagnait que 3 % sur COCO (`05_optimisations_vitpose.md`). Il est rentable ici parce
     que les images contiennent plusieurs personnes et que le modèle est plus gros.

## 4. Pistes

- **Court terme, faible coût :** proposer ViTPose++ base ou large (avec l'expert COCO) dans `psifx video pose
  vitpose`, et activer le fp16 sur GPU.
- **Points inventés :** combiner un modèle top-down précis avec la décision « présent ou absent » d'un modèle
  en une seule étape (RF-DETR), ou calibrer un seuil par modèle sur ces annotations.
- **RF-DETR Keypoint :** à suivre jusqu'à sa version stable (Apache 2.0, une seule étape, rapide).
- **Avant de conclure :** mesurer la chaîne réelle (masques SAM3, puis pose par personne, avec MediaPipe et le
  lissage) avec `evaluate.py --archives`, après la chaîne de nuit. Annoter aussi une autre vidéo, pré-annotée
  par un modèle d'une autre famille, pour mesurer le biais d'ancrage.

## Sources

- API Hugging Face, inventaire du 2026-11-03 : https://huggingface.co/api/models?pipeline_tag=keypoint-detection
- ViTPose et ViTPose++ dans transformers (experts MoE, `dataset_index`) : https://huggingface.co/docs/transformers/model_doc/vitpose
- Fiche `usyd-community/vitpose-plus-base` : https://huggingface.co/usyd-community/vitpose-plus-base
- Sapiens2, article (Table 3) : https://huggingface.co/papers/2604.21681 ; fiche : https://huggingface.co/facebook/sapiens2-pose-1b ;
  points : https://github.com/facebookresearch/sapiens2/blob/main/sapiens/pose/configs/_base_/keypoints308.py
- Sapiens2 dans transformers : https://huggingface.co/docs/transformers/model_doc/sapiens2
- RTMW (table COCO-WholeBody) : https://huggingface.co/akore/rtmw-l-256x192
- SynthPose : https://huggingface.co/stanfordmimi/synthpose-vitpose-huge-hf
- RF-DETR Keypoint : https://rfdetr.roboflow.com/latest/learn/run/keypoints/ ; https://blog.roboflow.com/launch-rf-detr-keypoint-in-roboflow/
- DETRPose-X : https://huggingface.co/SebasJanampa/DETRPose_X_COCO
- BBoxMaskPose : https://huggingface.co/vrg-prague/BBoxMaskPose
- YOLO26-pose : https://learnopencv.com/yolo26-pose-estimation-tutorial/
