# Visage statique et mouvement saccadé : diagnostic et pistes (2026-11-19)

Objectif rappelé : retirer l'apparence en gardant **toute** l'information du visage (expressions) et du corps
(fluidité du geste). Sur les vidéos finales du talk-show, le visage ne bouge pas et le mouvement n'est pas fluide.
Ce document pose le diagnostic et compare les pistes **avant** toute implémentation.

## 1. Diagnostic (mesuré sur nos sorties)

### Visage
- Les 72 paramètres d'expression valent **exactement 0** sur toutes les images (`sam3d_params`, `sam3d_people`,
  `sam3d_smooth`).
- Cause dans le code de SAM 3D Body (`sam_3d_body/models/heads/mhr_head.py`) : `pred_face = ... * 0` et
  `pred_pose_euler[:, -3:] = 0` (mâchoire). **Le modèle n'estime ni l'expression ni la mâchoire, quel que soit le
  mode** : ce n'est pas un réglage, il faut une autre source pour le visage.
- MHR n'a pas de globes oculaires (article MHR, section 3.2) : **le regard ne peut pas être rendu** avec le
  maillage actuel.

### Fluidité
1. **Une image sur deux.** La vidéo est à 60 images/s (18 708 images) ; les paramètres ont été calculés une image
   sur deux (`frames` = 0, 2, 4…). Le rendu **garde** chaque pose jusqu'à la suivante (`render_avatars.py`,
   l. 134-139) : l'avatar bouge par paliers à 30 images/s, sans interpolation.
2. **Tremblement restant après lissage** (accélération entre images calculées consécutives, même plan) :

   | Paramètre | Brut, médiane / p95 | Lissé (lock+sg7), médiane / p95 |
   |---|---|---|
   | Profondeur caméra (`cam_t` z) | 6,6 / 109 mm | 2,7 / **39 mm** |
   | Position caméra (x, y) | 1,7 / 20 mm | 0,6 / 7 mm |
   | Rotation du corps | 0,49 / 7,5° | 0,17 / 2,9° |
   | Angles des articulations | 0,38 / 8,9° | 0,15 / 2,4° |

   La médiane est faible, mais la queue (p95) reste visible : la **profondeur** (l'avatar grossit et rétrécit) et
   des à-coups de rotation. Le lissage est fait sur des angles d'Euler, avec une fenêtre fixe de 7 images.
3. Les mains : à vérifier si `sam3d_params` a été produit en mode `body` (sans le décodeur des mains) ou `full`.

## 2. Pistes pour le visage

| Piste | Principe | Pour | Contre |
|---|---|---|---|
| **V1. MediaPipe → MHR** | Les 52 blendshapes ARKit de MediaPipe Face Landmarker (déjà dans psifx, `poses_mp`) convertis vers les 72 expressions FACS de MHR ; la mâchoire depuis `jawOpen` | Données déjà calculées, rapide, un nœud ComfyUI fait déjà cette conversion (preuve que c'est faisable) | Sortie brute très instable (30 à 70 % d'une image à l'autre selon la doc du nœud), visages petits en plan large, table de conversion à construire (52 → 72, noms FACS) |
| V2. Modèle de visage FLAME (SMIRK, Pixel3DMM) → MHR | Ajuster les expressions MHR sur le maillage FLAME de la zone du visage (deux modèles linéaires : moindres carrés) | Expressions plus riches et asymétriques ; Pixel3DMM a un suivi vidéo | Un modèle de plus, conversion géométrique à écrire, coût de calcul |
| V3. Tête séparée (FLAME) collée sur le corps | Rendre une tête FLAME à la place de la tête MHR | Expressions et regard (FLAME a des yeux) | Raccord cou visible, deux modèles à aligner, plus d'apparence possible |

**Recommandation : V1 d'abord**, avec un lissage temporel des blendshapes et une mesure de relecture (ci-dessous).
V2 seulement si V1 ne transmet pas assez d'expressions. Le regard reste une limite à dire (ou V3 plus tard).

## 3. Pistes pour la fluidité

| Piste | Principe | Effort | Effet attendu |
|---|---|---|---|
| **F0. Interpoler au lieu de garder** | Interpolation entre images calculées (SLERP des rotations, linéaire sur le reste), ou calcul à chaque image | Faible (heures) ; ×2 calcul si toutes les images | Supprime les paliers à 30 images/s |
| **F1. Profondeur stabilisée** | Les personnes du talk-show sont assises : `cam_t` z très lissé ou fixé par plan | Faible | Supprime le « pompage » de taille (p95 39 mm) |
| **F2. Meilleur filtre** | Lisser les rotations en 6D/quaternions (pas en Euler) ; filtre adaptatif (One-Euro) ou lisseur de Kalman avant-arrière, comme SAM-Body4D | Moyen (1 jour) | Moins d'à-coups sans aplatir les gestes rapides |
| F3. Réajustement sur une fenêtre | Optimiser les paramètres MHR sur plusieurs images contre les points 2D ViTPose corps entier (mains, visage) avec un terme de régularité, SAM 3D Body comme départ (ce que fait l'article SAM 3D Body pour la vidéo) | Élevé (plusieurs jours) | Le meilleur contrôle : fidèle au 2D et lisse |
| F4. Modèle vidéo (PromptHMR-Vid, GVHMR) | Remplacer SAM 3D Body par un modèle temporel | Élevé | SMPL-X (autre modèle du corps, conversion vers MHR, licence), mains faibles : non prioritaire |
| F5. SmoothNet | Réseau de raffinement temporel « plug-and-play » | Moyen | Travaille sur des points, pas sur des paramètres MHR : il faudrait revenir aux paramètres (cinématique inverse) |

**Recommandation : F0 + F1 + F2 d'abord** (peu coûteux, causes mesurées), puis mesurer ; F3 seulement si le
résultat ne suffit pas.

## 4. Comment mesurer (avant / après)

- **Fluidité** : accélération par partie (tableau ci-dessus) à 60 images/s, et changement des gestes rapides
  (p95, comme `smooth_params.py`) pour vérifier qu'on n'aplatit pas le mouvement.
- **Corps** : relecture ViTPose sur la vidéo d'avatars (`utility_pose.py`), déjà en place.
- **Visage** : même principe, MediaPipe relu sur la vidéo d'avatars : corrélation, par blendshape, entre le
  signal de l'original et celui relu sur l'avatar (bouche, sourcils, clignements). C'est la mesure d'utilité du
  visage qui manque aujourd'hui.
- **À l'œil** : un extrait côte à côte original / avatar, image par image sur un geste rapide et une phrase.

## 5. Questions ouvertes

- Le regard compte-t-il pour la formation des psychologues ? Si oui, il faut un modèle avec des yeux (V3).
- Quelle taille minimale de visage viser ? Sur les plans larges, aucune méthode ne lira l'expression.
- Les expressions portent un peu d'identité (dynamique propre à chacun) : à mesurer comme la démarche.

## 6. Sous-titres : la voix de l'animateur marquée « off »

- **Cause** : la diarisation a été lancée sans nombre de locuteurs. pyannote a trouvé 3 voix pour 2 personnes :
  l'animateur coupé en SPEAKER_02 (131 s) et SPEAKER_00 (17 s). L'association par les lèvres donne une personne par
  voix : le petit morceau (score 1,52) a pris l'animateur, et le gros morceau, au score négatif (-0,27, sa bouche
  n'est visible que 39 % du temps où il parle, la caméra montre souvent l'invitée), est parti en « hors champ ».
- **Correctif vérifié** (en scratch) : diarisation avec `--num_speakers 2` → 2 voix (136 s et 145 s), l'ancien
  SPEAKER_00 rejoint l'animateur ; l'association donne animateur → personne 0 (0,77), invitée → personne 1 (0,59),
  plus aucune réplique « off ».
- **À faire** : donner à la diarisation le nombre de personnes trouvé par le regroupement (`people.json`), relier les
  voix aux **personnes regroupées** (et non aux pistes SAM3 brutes) pour que la couleur du sous-titre soit celle de
  l'avatar, puis régénérer `subtitles.ass` et les vidéos.

## Sources

- SAM 3D Body, code (`mhr_head.py`) : https://github.com/facebookresearch/sam-3d-body
- MHR: Momentum Human Rig, arXiv 2511.15586 : https://arxiv.org/html/2511.15586v3
- Nœud ComfyUI SAM3DBody FaceExpression (ARKit 52 → MHR 72) : https://docs.comfy.org/built-in-nodes/SAM3DBody_FaceExpression
- Nœud ComfyUI SAM3DBody Smooth : https://docs.comfy.org/built-in-nodes/SAM3DBody_Smooth
- SAM-Body4D (lissage de Kalman des paramètres MHR) : https://www.themoonlight.io/review/sam-body4d-training-free-4d-human-body-mesh-recovery-from-videos
- World-Coordinate Human Motion Retargeting via SAM 3D Body, arXiv 2512.21573 : https://arxiv.org/html/2512.21573v1
- SMIRK (CVPR 2024) : https://georgeretsi.github.io/smirk/
- PromptHMR (CVPR 2025) : https://openaccess.thecvf.com/content/CVPR2025/papers/Wang_PromptHMR_Promptable_Human_Mesh_Recovery_CVPR_2025_paper.pdf
- SmoothNet (ECCV 2022) : https://arxiv.org/abs/2112.13715
- Carte du modèle MediaPipe Blendshape V2 : https://dcmpx.remotevs.com/com/googleapis/storage/SL/mediapipe-assets/Model%20Card%20Blendshape%20V2.pdf
