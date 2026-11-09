# Plan : reconstruire et reproduire une vidéo en 3D

Date : 2026-11-09. Statut : plan à valider, rien n'est implémenté. Point de départ : `dev` au tag `tm-v1-optimisations`.

## 1. L'idée

Jusqu'ici, psifx extrait des signaux séparés : des masques, des poses 2D, des voix, du texte. La nouvelle direction
les réunit en un seul objet, **une scène 3D dans le temps** : qui est où, dans quelle pose, filmé par quelle caméra, et
qui parle. La vidéo reproduite montre ce que contient cette scène : ce qui manque aux données se voit à l'écran.

Ce qui distinguerait ce travail de la bibliothèque d'origine :

| | psifx d'origine | Cette direction |
|---|---|---|
| Sortie | Fichiers par signal (poses 2D, visage, audio, texte) | Une scène unique, personnes et caméras dans un même repère |
| Corps | Points 2D | Maillage complet avec mains et expressions, paramètres réutilisables |
| Vidéos montées | Non traitées | Plans séparés, puis réunis dans un même repère |
| Validation | Par signal | Aller-retour : la scène reprojetée est comparée à la vidéo et aux annotations |
| Usages nouveaux | – | Nouveau point de vue, mesures d'interaction en 3D, vidéo sans l'apparence des personnes |

**Usage à examiner : partager des interactions sans montrer les visages.** Remplacer les personnes par des avatars
neutres pourrait permettre de partager un enregistrement de recherche en conservant le comportement. Je n'ai trouvé
aucune étude qui compare directement, sur le comportement non verbal mesuré, une reconstruction 3D à la vidéo. C'est
une contribution possible, mais aussi une hypothèse : la silhouette et la voix peuvent rester identifiantes, et le
dire « anonymisé » exigerait une mesure.

## 2. Où on en est : mesures sur la nuit du 6 novembre

Maillages SAM 3D Body image par image sur `ExtraitTalkShow.mp4` (5 min 12, 60 images/s, une image sur 2), masques SAM3
de la chaîne. Le coût mesuré est de **0,69 s par image, soit 1 h 48** pour cet extrait.

**Précision 2D contre les 200 images annotées** (points 2D du maillage réduits à COCO-17, image voisine quand
l'annotée est impaire) :

| Modèle dans la chaîne | OKS | PCK@0,1 |
|---|---|---|
| ViTPose | 0,893 | 0,937 |
| **SAM 3D Body** | **0,849** | **0,904** |
| RTMO | 0,742 | 0,722 |
| MediaPipe | 0,690 | 0,794 |

Le maillage est donc déjà une bonne source de pose 2D. Il ne donne pas de score par point : les mesures de points
fantômes n'ont pas de sens pour lui.

**Stabilité et profondeur :**

| Piste | Images | Largeur d'épaules, cm (p5 / p50 / p95) | Vitesse relative au bassin, mm (p50 / p95) | Profondeur, m (p1 / p50 / p99) |
|---|---|---|---|---|
| 0 | 6 120 | 33,4 / 34,9 / 36,9 | 6,0 / 79,0 | 1,03 / 2,22 / **10,15** |
| 1 | 3 840 | 33,6 / 36,8 / 37,9 | 6,2 / 72,1 | 0,82 / 1,59 / 2,53 |
| 2 | 4 346 | **16,6** / 33,7 / 34,9 | 7,8 / 133,2 | 1,58 / 2,50 / **7,64** |

Les vitesses incluent les changements de plan, donc leur p95 surestime le tremblement.

**Ce que cela montre :**

1. **La profondeur n'est pas une donnée de scène.** Des personnes assises ne se trouvent pas à 10 m. La focale est
   supposée fixe alors que les plans changent : chaque plan a sa propre échelle.
2. **La morphologie varie d'une image à l'autre**, alors qu'une personne ne change pas de corps.
3. **Une personne porte plusieurs numéros.** Sur la planche de contrôle, la même femme est P1 en gros plan et P2 en plan
   large. Il faut fusionner ces pistes avant toute scène. C'est à confirmer sur toute la vidéo.
4. **Les paramètres sont perdus.** `run_tracks.py` garde les sommets et les points (110 Ko par image, 1,3 Go ici), mais
   jette ce que le modèle produit par ailleurs : `shape_params`, `scale_params`, `body_pose_params`, `hand_pose_params`,
   `expr_params`, `global_rot`. Sans eux, ni verrouillage de la morphologie, ni lissage propre, ni export vers un autre
   logiciel.
5. **Le rendu actuel est une superposition**, sans repère commun : il ne permet pas de changer de point de vue.

## 3. Veille : ce qui existe en novembre 2026

| Brique | Candidat | Ce qu'on sait | Décision |
|---|---|---|---|
| Modèle du corps | **MHR** | Apache-2.0 ; 45 paramètres d'identité, 204 de pose, 72 d'expression (FACS), 7 niveaux de détail, rig FBX, conversion SMPL/SMPL-X fournie | **Retenu** : c'est déjà la sortie de SAM 3D Body |
| Maillage par image | **SAM 3D Body** | Déjà intégré en recherche. Licence SAM : redistribution sous les mêmes termes, citation obligatoire | **Retenu** |
| Cohérence temporelle | SAM-Body4D | MIT. Morphologie fixée sur la première image visible, lissage de Kalman de la pose et des mains. Aucune mesure chiffrée dans l'article. Python 3.12, torch 2.7.1 ; 14,5 Go de pic pour une personne par lots de 64 ; complétion des occlusions : 26 min pour 90 images à 5 personnes | **Reprendre la méthode**, sans la complétion ni l'environnement |
| Cohérence temporelle | Fenêtre glissante dans l'espace MHR (arXiv 2512.21573), `SAM3DBody_Smooth` de ComfyUI (fenêtre, seuil de rotation rapide) | Mêmes principes : morphologie verrouillée, lissage de la pose, prudence sur les gestes rapides | Alternatives à comparer au Kalman |
| Vidéos montées | **HumanMM** (CVPR 2025) | Détection des plans, trajectoire caméra, alignement du mouvement entre plans, jeu de test ms-Motion. Code non confirmé | Référence de méthode. À lire avant le jalon 3 |
| Repère monde, caméra mobile | PromptHMR-Vid, GVHMR, Human3R, OnlineHMR, WATCH | Basés sur SMPL/SMPL-X. Une étude d'octobre 2026 décrit une dérive liée à un écart d'échelle entre caméra et personne quand la caméra bouge beaucoup. OnlineHMR signale des tremblements chez Human3R. Pas de comparatif unique | Hors du cas de base. Essai court seulement si une vidéo à caméra mobile l'exige |
| Décor, caméra | MoGe (estimateur prévu par SAM 3D Body), VGGT, Depth Anything 3, Ground4D (VGGT avec masquage du mouvement) | Le masquage des personnes est nécessaire : elles violent l'hypothèse de scène statique | MoGe d'abord, pour la focale |
| Occlusion | Diffusion-VAS | Très coûteux (voir SAM-Body4D) | Écarté pour l'instant |
| Avatar réaliste | ExAvatar (vidéo), LHM, PERSONA (une image), AHOY (2026) | Liés à SMPL-X. Licences lues seulement sur des listes tierces (MIT pour ExAvatar, Apache pour LHM). Reproduisent l'apparence, donc l'identité | Hors périmètre initial, voir la décision 1 |
| Lèvres | SMIRK, EMOCA, EMOTE | La précision des lèvres reste un point faible connu en monoculaire | Mesurer d'abord les `expr_params` de MHR. Pas de nouveau modèle de visage pour l'instant |
| Rendu | Superposition OpenCV actuelle, Blender (`bpy`), nvdiffrast, PyTorch3D | Blender : qualité et export, usage sans écran moins commode. nvdiffrast : très rapide, mais tout est à écrire. Aucun comparatif sur ce cas | Essai au jalon 4 |

## 4. Architecture proposée

```
vidéo ─► plans ─► caméras par plan ───────────────┐
  │                                               ▼
  ├─► SAM3 ─► pistes ─► personnes fusionnées ─► SCÈNE ─► rendus : caméra d'origine, point de vue libre
  │                         │                     ▲   ├► exports : glTF/FBX
  │                         ▼                     │   └► mesures : distance, orientation, synchronie
  │              SAM 3D Body ─► paramètres MHR ─► lissage
  └─► audio ─► diarisation, transcription, voix ↔ personne ──► (dans la scène)
```

Format de la scène, un dossier relu par tous les outils en aval :

```
scene/
  scene.json        # vidéo, plans [début, fin, caméra], caméras (focale, position), personnes (pistes, morphologie)
  people/<id>.npz   # images, pose, mains, expression, rotation et translation dans le monde, statut par image
  speech.vtt        # transcription, chaque segment porte la personne
```

Trois règles :

- **On garde les paramètres, pas les sommets.** MHR recalcule le maillage à la demande. L'estimation est de 1 à 2 Ko
  par image et par personne, contre 110 Ko aujourd'hui.
- **Chaque image dit son statut : `observé`, `interpolé` ou `non observé`.** En gros plan, l'autre personne n'est pas
  filmée. Le rendu ne doit pas faire passer un comportement deviné pour une mesure, et les statistiques ne comptent
  que les images observées.
- **Le lissage ne traverse jamais une coupe.** Sinon, il inventerait un mouvement entre deux plans.

## 5. Jalons

Chaque jalon a une branche, un critère de sortie et une note. Les seuils sont des propositions, à ajuster sur les
premières mesures. Le talk-show est la vidéo principale, avec ses 200 images annotées.

### Jalon 0 : préparer le terrain

- Enregistrer tous les paramètres MHR dans `run_tracks.py`, sans changer le modèle.
- Relancer le talk-show (1 h 48), puis vérifier que les paramètres seuls redonnent les mêmes sommets.
- Écrire `mesh_eval.py`, qui automatise les mesures de la section 2.

**Sortie :** sommets recalculés à moins de 1 mm des sommets du modèle ; OKS inchangé (0,849).

### Jalon 1 : plans et personnes

- Réutiliser la détection de changement de plan de `psifx/video/tracking/sam3/reid.py`.
- Regrouper les plans par caméra d'après le décor (masques des personnes exclus).
- Fusionner les pistes d'une même personne : apparence, morphologie, compatibilité (deux pistes visibles dans la même
  image ne peuvent pas être la même personne).

**Sortie :** sur un échantillon annoté à la main, le bon nombre de personnes sur le talk-show et plus de 95 % des
images attribuées à la bonne personne. La vérité terrain des plans reste à créer (quelques minutes).

### Jalon 2 : cohérence temporelle

- Une morphologie et une échelle par personne, estimées sur les images les plus fiables (médiane plutôt que première
  image, à comparer).
- Lissage plan par plan de la pose et des mains : Kalman contre fenêtre glissante, dans un espace de rotations
  adapté.
- Mesurer si une image sur 2 suffit, ou s'il faut toutes les images pour les gestes rapides.

**Sortie :** accélération médiane divisée par au moins 2, OKS en baisse de moins de 0,01. La borne sur l'OKS protège
contre un lissage qui effacerait les gestes.

### Jalon 3 : caméras et repère commun (le principal risque)

- Une focale par caméra (essai avec MoGe, médiane sur le plan).
- Un sol ou un plan de table comme référence.
- L'échelle vient de la morphologie verrouillée : la taille du corps est le seul mètre disponible.
- Le plan large fixe les positions relatives. Les gros plans apportent la pose fine de la personne cadrée.

**Sortie, par des invariants vérifiables sans vérité terrain 3D :**

- la position du bassin d'une personne assise varie de moins de 10 cm entre plans ;
- la distance entre les deux personnes varie de moins de 20 cm entre plans larges ;
- le maillage reprojeté garde son OKS à 0,02 près.

**Si ça échoue :** une scène par caméra, sans repère global. La reproduction à caméra d'origine reste possible, le
point de vue libre se limite aux plans larges. Ce repli est acceptable, et il faudra l'écrire tel quel dans le rapport.

### Jalon 4 : reproduction

- **Caméra d'origine :** avatars neutres d'une couleur par personne, son et sous-titres. Comparaison côte à côte avec
  l'original.
- **Point de vue libre :** caméra fixe de côté ou orbite ; personnes non observées affichées autrement.
- **Décor minimal :** sol et table comme simples volumes.
- **Essai de rendu :** Blender contre rasteriseur GPU sur le temps par image et la mémoire, avec le GPU de 8 Go.
- **Export :** glTF ou FBX ouvrable dans Blender, à partir du rig MHR.

**Sortie :** vidéo complète du talk-show, avec son, vérifiée à l'œil sur chaque type de plan.

### Jalon 5 : validation

| Question | Mesure | Référence |
|---|---|---|
| Le corps est-il juste en 3D ? | PA-MPJPE, MPJPE sur un sous-ensemble | EMDB ou 3DPW ; articles : 33,2 / 54,8 mm sur 3DPW. Vérité SMPL : conversion à prévoir |
| L'interaction est-elle juste ? | Erreur sur la distance entre personnes | Hi4D ou CHI3D, licences à vérifier |
| La vidéo reproduite est-elle fidèle ? | ViTPose lancé sur le rendu, puis OKS | Les 200 images annotées |
| Les lèvres suivent-elles la parole ? | Corrélation de l'ouverture avec MediaPipe ; activité pendant les tours de parole | Sous-titres de référence |
| L'apparence a-t-elle disparu ? | Réidentification faciale entre original et rendu | À faire seulement si la décision 3 le retient |
| Ça tient hors du talk-show ? | Mêmes mesures | V1 (caméra fixe), V2 (film), V4 (danse) |

### Jalon 6 : usages et bibliothèque

- Mesures d'interaction : distance, orientation mutuelle, gestes pendant la parole, synchronie.
- Commandes psifx, seulement une fois le format stable, par exemple `psifx video mesh sam3d_body inference`,
  `psifx scene build` et `psifx scene render`.
- Version du rapport et note de synthèse.

## 6. Risques

| Risque | Conséquence | Réponse |
|---|---|---|
| Repère commun impossible sur vidéo montée | Pas de point de vue libre global | Repli du jalon 3, décidé tôt |
| Lissage qui efface des gestes rapides | Données fausses, surtout pour la synchronie | Garde-fou OKS, comparaison sur des gestes marqués à la main |
| Mains et lèvres imprécises sur plans larges | Rendu trompeur | Mesurer par type de plan, marquer la confiance |
| Coût : 1 h 48 pour 5 min, avant caméras et rendu | Vidéos longues difficiles | Travailler sur un extrait de 30 s pendant le développement |
| Dépendances : modèle à accès restreint, code non packagé, environnement dédié | Installation fragile | Environnement `sam3d-body` figé, commit du clone noté |
| Validation 3D sur SMPL alors que le modèle sort MHR | Mesures biaisées par la conversion | Comparer sur des points communs, documenter l'erreur de conversion |
| Promettre l'anonymat | Risque éthique pour de vraies données | Ne rien affirmer sans la mesure, parler d'« apparence retirée » |

## 7. Hors périmètre pour l'instant

- Avatars photoréalistes (sauf décision contraire).
- Génération de comportement pour les personnes hors champ : on affiche l'absence de données.
- Contacts main-objet, et reconstruction fine du décor.
- Caméras très mobiles : essai au jalon 5 seulement.

## 8. Décisions à prendre

1. **Niveau de reproduction.** Recommandation : avatars neutres et décor minimal. C'est mesurable, faisable sur 8 Go,
   et ça répond à la question des données. Le photoréalisme demanderait SMPL-X et reproduirait l'identité.
2. **Le rendu à point de vue libre est-il indispensable, ou le repli « une scène par caméra » suffit-il ?**
   Recommandation : viser le repère commun, accepter le repli s'il échoue.
3. **Faut-il faire de « l'apparence retirée » un objectif de la thèse ?** Recommandation : à discuter avec les
   encadrants, c'est la contribution la plus originale, mais elle ajoute une mesure et une question éthique.
4. **Calendrier.** La date de rendu fixe jusqu'où aller après le jalon 4.

## Sources

- SAM 3D Body : https://github.com/facebookresearch/sam-3d-body, https://arxiv.org/abs/2602.15989
- MHR : https://github.com/facebookresearch/MHR, https://arxiv.org/pdf/2511.15586v3
- SAM-Body4D : https://arxiv.org/html/2512.08406v1, https://github.com/gaomingqi/sam-body4d,
  https://raw.githubusercontent.com/gaomingqi/sam-body4d/master/assets/doc/resources.md
- Retargeting par fenêtre glissante dans l'espace MHR : https://www.emergentmind.com/papers/2512.21573
- Nœud de lissage ComfyUI : https://docs.comfy.org/built-in-nodes/SAM3DBody_Smooth.md
- HumanMM : https://arxiv.org/abs/2503.07597,
  https://openaccess.thecvf.com/content/CVPR2025/html/Zhang_HumanMM_Global_Human_Motion_Recovery_from_Multi-shot_Videos_CVPR_2025_paper.html
- PromptHMR : https://openaccess.thecvf.com/content/CVPR2025/papers/Wang_PromptHMR_Promptable_Human_Mesh_Recovery_CVPR_2025_paper.pdf
- Étude de la dérive à caméra rapide : https://arxiv.org/pdf/2610.19729
- OnlineHMR : https://openaccess.thecvf.com/content/CVPR2026/papers/Zhao_OnlineHMR_Video-based_Online_World-Grounded_Human_Mesh_Recovery_CVPR_2026_paper.pdf
- WATCH : https://arxiv.org/pdf/2509.04600
- Ground4D : https://arxiv.org/pdf/2606.28828
- Avatars : https://lingtengqiu.github.io/LHM, https://mks0601.github.io/PERSONA,
  https://awesome.ecosyste.ms/projects/github.com%2Fmks0601%2FExAvatar_RELEASE, https://arxiv.org/html/2603.17975
- Lèvres en monoculaire : https://arxiv.org/pdf/2204.11312, https://arxiv.org/pdf/2306.08990.pdf
- Rendu : https://pkhw2023-ship-it.github.io/medium-posts/posts/03-renderer-comparison-five-libraries/post.html,
  https://pytorch3d.org/docs/renderer
- Anonymisation et comportement : https://arxiv.org/pdf/2410.17098,
  https://www.sciencedirect.com/science/article/pii/S107158192500045X,
  https://www.frontiersin.org/journals/robotics-and-ai/articles/10.3389/frobt.2016.00012/full
- Jeux de données : https://ait.ethz.ch/emdb, https://virtualhumans.mpi-inf.mpg.de/3DPW/license.html,
  https://rich.is.tue.mpg.de/index.html
