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

## 7. Sous-titres et nombre de personnes : fait le 2026-11-19 (branche `fix/speaker-link`)

Vérité terrain : `example/data/ExtraitTalkShow_Sous-titre.txt` (DCL, MC, EAC, DLC = 4 voix), jugée telle qu'écrite
(`subtitle_eval.py --expected person_0=DCL person_1=MC "off_*=DLC,EAC"`).

| Chaîne | Mots au bon locuteur | David | Voix comptées |
|---|---|---|---|
| Avant (lèvres MediaPipe, diarisation automatique) | 48,1 % | 1,6 % | 3 |
| Locuteur actif Light-ASD sur les personnes regroupées | 88,7 % | 94,9 % | 3 |
| **Diarisation volontairement trop découpée (6) + visages + empreintes de voix** | **92,8 %** | 93,3 % | **4 (juste)** |

- Le nombre de personnes n'est plus donné : 6 n'est qu'une borne haute ; les morceaux d'une même voix se
  ressemblent à 0,91-0,93, les voix différentes à 0,03-0,33, et le résultat ne bouge pas entre les seuils 0,4 et 0,6.
- Reste : les 6 mots de la voix derrière la caméra sont collés par Whisper à une ligne de l'explicateur ;
  à valider sur une deuxième vidéo.

## 8. Fidélité du mouvement : fait le 2026-11-19/20 (branche `optim/motion-fidelity`)

**Diagnostic par partie du corps** (`motion_diag.py` : articulations MHR en repère caméra, accélération et « pics »
= une articulation à plus de 40 mm de la moyenne de ses deux voisines, donc une téléportation d'une image) :

- les plus gros pics sont des erreurs de **profondeur** (jusqu'à 1,7 m sur une image : l'avatar rétrécit puis revient) ;
- le lissage Savitzky-Golay **étale** une valeur fausse sur ses voisines au lieu de la retirer ;
- les **jambes** ne sont jamais vues en entier (chevilles hors image sur 100 % des images, genoux 2 %, sous la table
  sur les plans larges) : le modèle les invente, d'où 172 pics par minute sur les jambes brutes.

**Correctif `robust_smooth.py`** (variante `robust` de `smooth_params.py`) :

1. **Valeurs aberrantes retirées** (filtre de Hampel : écart à la médiane d'une fenêtre courte, en unités de l'écart
   habituel de la chaîne, avec un seuil physique minimal pour ne pas toucher aux gestes) ;
2. **Parties cachées moins crues** : une partie (jambes, bras, poignets, doigts, par côté) est cachée si ses points
   2D sont hors image ou hors du masque SAM3 de la personne (la table cache les jambes) ; sa mesure compte 100 fois
   moins ;
3. **Lisseur de Kalman avant-arrière** (Rauch-Tung-Striebel, vitesse constante), une fréquence de coupure par partie ;
   la partie de chaque paramètre est trouvée sur MHR lui-même (première articulation qui bouge) ;
4. **Profondeur à part** : la translation caméra est lissée comme position dans l'image (`x/z`, `y/z`) et log de la
   profondeur, puis reconstruite : lisser la profondeur ne déplace plus l'avatar dans l'image.

| Mesure (30 images/s, 2 personnes) | Brut | Lissage actuel (lock+sg7) | **Robuste** |
|---|---|---|---|
| Pics par minute (squelette / profondeur) | 286 / 88 | 7,5 / 14,4 | **0 / 0** |
| Accélération p95 position du corps (mm) | 83,7 | 34,5 | **5,9** |
| … jambes / poignets / doigts (mm) | 90 / 71 / 60 | 24,5 / 19,2 / 15,8 | **1,8 / 7,4 / 4,4** |
| OKS / PCK@0,05 / PCK poignets (annotations) | 0,830 / 0,769 / 0,918 | 0,823 / 0,775 / 0,917 | 0,816 / 0,770 / **0,918** |

La précision 2D reste celle du brut (les poignets aussi : les gestes ne sont pas aplatis). Une profondeur coupée à
0,4 Hz faisait tomber l'OKS à 0,75 (la taille de l'avatar ne collait plus à l'image) : 2 Hz est gardé.

**60 images/s** : `run_tracks.py --offset 1` calcule les images impaires, `apply_people.py` les donne aux mêmes
personnes (vérifié : il refait `sam3d_people` à l'identique sur les images paires) ; le rendu interpole entre deux
images calculées au lieu de garder la pose (`--hold` pour l'ancien comportement). Les 9351 images impaires ont pris
2 h 11 (0,84 s par image).

| Mesure (60 images/s, 2 personnes) | Brut | lock+sg13 | **Robuste** |
|---|---|---|---|
| Pics par minute (squelette / profondeur) | 386 / 118 | 0,6 / 1,2 | **0 / 0** |
| Accélération p95 position du corps / jambes / poignets / doigts (mm) | 56 / 76 / 62 / 49 | | **1,5 / 0,4 / 1,8 / 1,1** |
| OKS / PCK@0,05 / PCK poignets | 0,830 / 0,770 / 0,920 | 0,824 / 0,776 / 0,922 | 0,816 / 0,769 / **0,922** |

À 60 images/s, une image fausse sur deux voisines justes est plus facile à repérer, et le brut a plus de pics par
minute (le même saut compte deux fois). Limite vue à l'œil : sur un geste très rapide des mains (image 8965), les
poignets lissés traînent jusqu'à environ 100 px derrière le brut pendant quelques images, puis le rejoignent
(médiane 22 px sur la seconde autour).

## 9. Visage : fait le 2026-11-20 (branche `feature/face-expression`)

- **Noms des 72 expressions MHR** : `docs/face-expressions.md` du dépôt MHR (`jawDrop`, `lipCornerPuller_L`,
  `eyesClosed_R`...). La conversion depuis les 52 blendshapes ARKit de MediaPipe est donc une **table nom à nom**
  (`ARKIT_TO_MHR`), pas un modèle appris.
- **Lecture** (`face_blendshapes.py`) : MediaPipe Face Landmarker sur un carré autour de chaque visage (marge 2,4 :
  avec 1,8 le détecteur manquait 50 % des visages de David). Visages placés par psifx (Holistic) ou, à défaut, par
  la tête de SAM 3D Body : couverture de 35 → 43 % des images de David (souvent de dos ou flou au premier plan) et
  48 → 73 % pour Marion.
- **Conversion** (`face_expression.py`) : visage de repos retiré par personne (20e centile : les lunettes se lisent
  comme un demi-clignement), lissage robuste (sans rejet sur les paupières : un clignement est un vrai pic),
  **mâchoire pilotée par l'écart des lèvres** (`jawOpen` ne bouge presque pas chez ces deux personnes), expressions
  bornées à [0, 1].
- **Rendu** : 32 niveaux d'ombre et une lumière de côté (au lieu de 6 niveaux face caméra) : le relief des lèvres se
  voit, pour 7 ms de plus par maillage.
- **Mesure de relecture** (MediaPipe relu sur le visage de l'avatar, corrélation avec l'original, extrait de 33 s) :

  | Blendshape | Sans expressions | Avec |
  |---|---|---|
  | Sourire (gauche) | 0,16 | **0,74** |
  | Moue | 0,31 | 0,61 |
  | Sourcils intérieurs | 0,03 | 0,42 |
  | Moyenne pondérée | 0,22 | 0,33 |

  Limites : les **clignements** ne passent pas (MHR n'a pas de globes oculaires), et la relecture sous-estime les
  fortes valeurs (MediaPipe ne trouve plus le visage d'un avatar très expressif) : le gain de la mâchoire a été réglé
  à l'œil (avec la relecture seule, l'avatar « criait »).

## 10. Chaîne des voix en une commande et deuxième vidéo (branche `fix/speaker-link`)

`speakers.py` enchaîne Light-ASD, diarisation à 6 voix, lien visage, empreintes de voix, `enhance` et sous-titres
(`people.json` facultatif : sinon une personne par piste SAM3).

| Vidéo | Vérité | Ancienne chaîne (lèvres) | **Nouvelle chaîne** | Voix comptées |
|---|---|---|---|---|
| Talk-show (60 i/s) | sous-titres faits main | 48,1 % des mots | **92,8 %** | 4 (juste) |
| Exemple (1 personne vue, 1 voix hors champ) | micros séparés | — | **100 %** du temps de parole | 2 (juste) |
| AVA (film doublé, 2 hommes) | AVA-ActiveSpeaker, visage par visage | 61,7 % | **87,8 %** | 3 (2 + 3,6 s hors champ) |

V3 n'est pas une vidéo indépendante (même talk-show, autre découpage à 25 i/s) : 37/40 répliques de référence justes
(2 répliques de l'animateur passées « hors champ »).

Sous-titres : texte de la couleur de la personne sur un bandeau noir à 70 % (lisible sur un avatar de même couleur).

Sources :
- Facial expression mapping, MHR : https://github.com/facebookresearch/MHR/blob/main/docs/face-expressions.md
- SAM-Body4D (lissage de Kalman des paramètres MHR, masques complétés pour l'occlusion) : https://arxiv.org/html/2512.08406v1
- World-Coordinate Human Motion Retargeting via SAM 3D Body (optimisation par fenêtre) : https://arxiv.org/html/2512.21573
- Lisseurs robustes aux valeurs aberrantes (Hampel, RTS) : https://apps.dtic.mil/sti/pdfs/ADA093543.pdf et https://arxiv.org/html/2410.20411v1
- DanceHMR (mains en vidéo, code non publié) : https://arxiv.org/pdf/2605.18102
- SAM 3D Body, points de contrôle DINOv3 / ViT-H : https://github.com/facebookresearch/sam-3d-body
