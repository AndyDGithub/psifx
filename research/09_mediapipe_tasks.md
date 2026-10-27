# MediaPipe : passage de l'API *Solutions* à l'API *Tasks*

Branche `feature/mediapipe-tasks` (depuis `dev`). Date : 2026-10-27.

## 1. Pourquoi

psifx utilisait `mediapipe.python.solutions.holistic`, l'API *Legacy Solutions*. Google ne la
maintient plus depuis 2023, et elle a disparu des versions récentes (0.10.32, 1.1.1). Rester sur
mediapipe 0.10.14 imposait protobuf < 5, alors que pyannote.audio 4 exige protobuf ≥ 5 (note 08).
Le portage lève ce conflit : mediapipe 1.1.1 ne dépend plus de protobuf du tout.

## 2. Ce qui change dans psifx

| Changement | Détail |
|---|---|
| Modèle | `HolisticLandmarker` en mode vidéo. Le bundle `holistic_landmarker.task` (14 Mo, public) est téléchargé dans `~/.cache/psifx/mediapipe` et sa somme SHA-256 est vérifiée. `MEDIAPIPE_HOLISTIC_PATH` ou `--model_path` permettent un usage hors ligne. |
| Squelette | Connexions tirées de `mediapipe.tasks.python.vision` : 35 arêtes pour le corps, 21 par main et 2556 pour le maillage du visage, comme avant. |
| Visage | Toujours 478 points. L'ancien code mettait 468 zéros quand le visage manquait, mais 478 points quand il était trouvé. |
| 3D | Nouvelle clé `pose_keypoints_3d` : 33 × `[x, y, z, visibilité]` en mètres, origine entre les hanches. Elle reprend le changement de `experiment/e2e-demo`. |
| Options | `--model_complexity` et `--smooth` sont ignorées (un seul modèle ; le mode vidéo lisse toujours), mais restent acceptées pour ne pas casser les scripts. Nouvelles options `--min_detection_confidence`, `--min_presence_confidence` et `--face_fallback` (voir §4). |
| Système | mediapipe 1.x charge `libEGL.so.1` et `libGLESv2.so.2`, d'où les paquets Ubuntu `libegl1` et `libgles2` ajoutés aux deux Dockerfiles. |

Tests : 84 tests unitaires (dont 7 pour MediaPipe) et les tests d'intégration MediaPipe passent.

## 3. Mesures

### 3.1 Vitesse (CPU, pistes SAM3 du test de bout en bout)

| Vidéo | Ancienne API | API Tasks |
|---|---|---|
| V1 (1 piste, 46 s) | 131 s | **68 s** |
| V2 AVA (2 pistes, 2 min) | 653 s | **335 s** |
| V3 talk-show (3 pistes, 5,5 min) | 1743 s | **992 s** |
| V4 danse (1 piste, 46 s) | 108 s | **43 s** |

L'API Tasks est 1,8 à 2,5 fois plus rapide.

### 3.2 Précision du corps (COCO val2017, une personne annotée par image)

| | OKS | PCK@0,2 | Images détectées | Temps |
|---|---|---|---|---|
| 8 images du test d'intégration, ancienne API | 0,647 | 0,968 | 100 % | — |
| 8 images du test d'intégration, API Tasks | **0,713** | **0,985** | 100 % | — |
| 500 images tirées au hasard, ancienne API | 0,329 | **0,538** | **78,2 %** | 47,5 s |
| 500 images tirées au hasard, API Tasks | **0,331** | 0,505 | 63,8 % | **17,7 s** |

L'OKS est équivalent. L'API Tasks détecte moins de personnes sur des images fixes enchaînées en
vidéo, mais elle est plus précise sur celles qu'elle trouve (OKS 0,52 contre 0,42 sur les images
détectées).

Baisser les seuils n'y change presque rien : avec 0,3 au lieu de 0,5, on détecte 67 % des images
et l'OKS est de 0,322. Sur nos vidéos, le corps est trouvé aussi souvent qu'avant :
- V1 : 100 % ;
- V2 : 99,7 → 97,2 % et 100 → 99,3 % ;
- V3 : identique ;
- V4 : 93,5 → 96,4 %.

L'écart médian avec l'ancienne API vaut 3 à 9 % de la hauteur du torse (15 % sur la danse V4).

### 3.3 Visage : la vraie régression

| Piste | Ancienne API | API Tasks | Tasks + repli |
|---|---|---|---|
| V1 | 100 % | 99,2 % | 99,8 % |
| V2, personne 0 | 92,3 % | 40,5 % | 85,2 % |
| V2, personne 1 | 67,4 % | 44,9 % | 61,8 % |
| V3, personne 0 | 40,4 % | 24,2 % | 36,1 % |
| V3, personne 1 | 65,1 % | 35,2 % | 59,9 % |
| V4 | 16,9 % | 6,9 % | 14,0 % |

L'ancien Holistic plaçait la zone du visage à partir de la pose et y lançait directement le modèle
de points du visage. Le Holistic de l'API Tasks lance d'abord un détecteur de visage, qui rate des
visages de taille moyenne, pourtant de face (vérifié sur une planche de 12 images de V2).

**Repli (`--face_fallback`)** : quand le visage manque, on prend un carré autour de la tête, défini
par les points 0 à 10 de la pose (nez, yeux, oreilles, bouche) et agrandi 3 fois. On le
redimensionne en 256 × 256 et on y lance le Face Landmarker en mode vidéo, qui suit le visage d'une
image à l'autre.

Résultats du balayage sur 600 images de V2 :

| Réglage | Visages trouvés |
|---|---|
| Holistic seul | 29 % |
| Cadre × 1,6 | 32 % |
| Cadre × 3 | 57 % |
| Cadre × 4 | 61 % |
| Cadre × 3, mode vidéo, seuil de présence 0,1 | 67 % |
| Cadre × 3, mode vidéo, seuil de présence 0,5 (retenu) | 59 % |
| Ancienne API | 73 % |

Les points retrouvés restent proches de ceux de l'ancienne API : 1 % de la hauteur du torse.

### 3.4 En aval : l'association voix ↔ personne par les lèvres

La diarisation et la transcription sont identiques ; seules les poses MediaPipe changent.

| Cas | Ancienne API | API Tasks | Tasks + repli |
|---|---|---|---|
| V1, diarisation 3.1 (micros) | 100 % | 100 % | 100 % |
| V2, diarisation 3.1 (vérité AVA) | 64,8 % | 54,5 % | 50,8 % |
| V2, community-1 auto (vérité AVA) | 88,8 % | **88,7 %** | 52,2 % |
| V3, 3.1 ou community-1 auto (étiquettes) | 97,7 % | 59,1 % | 59,1 % |
| V3, community-1 avec 2 locuteurs (étiquettes) | 97,7 % | **97,7 %** | 97,7 % |

**Lecture :**
- **Le repli dégrade le signal des lèvres.** Sur V2, le score de la personne 1 pour sa voix tombe
  de 0,90 à 0,03. Les visages en plus sont souvent petits ou de profil, et leur ouverture de bouche
  est bruitée. Pour les lèvres, mieux vaut moins de visages mais de bonne qualité. Le repli est donc
  **désactivé par défaut**, et reste disponible pour les analyses qui demandent la présence du
  visage.
- **La baisse sur V3 vient de MediaPipe, sur un point déjà fragile.** *(Corrigé le 2026-10-27 après le passage
  de `dev` ; une première lecture l'attribuait à la diarisation.)* Les deux versions de la diarisation coupent
  l'intervieweur en un groupe principal (≈ 136 s) et un petit groupe en gros plan (≈ 15 s). Le petit groupe lui est
  affecté en premier, et le groupe principal doit passer seul le seuil de 0,2 :

  | Diarisation | MediaPipe | Score du groupe principal | Attribution |
  |---|---|---|---|
  | 3.1 | ancienne API | 0,26 | 100 % |
  | 3.1 | Tasks | −0,15 | 63,6 % |
  | community-1 | ancienne API | 0,25 | 98,0 % |
  | community-1 | Tasks | −0,15 | 63,6 % |

  (Même transcription, celle du passage `dev`, et 39 étiquettes : d'où des valeurs un peu différentes du tableau
  ci-dessus. Script : `.e2e_outputs/scratch/v3_why/run.sh`.)

  Pendant les tours de l'intervieweur, souvent filmé en plan large ou de côté, son visage est trouvé sur 69 % des
  images avec l'ancienne API, contre 44 % avec Tasks. Avec le bon nombre de locuteurs, l'affectation un à un suffit et
  les trois versions donnent 97,7 %. À corriger : retrouver les visages manqués sans dégrader les lèvres, et rendre
  l'association robuste aux petits groupes de voix (future commande `identification`).

## 4. Décision

Adopter l'API Tasks (mediapipe 1.1.1), sans repli par défaut :
- elle est 2 fois plus rapide ;
- la précision du corps est équivalente sur COCO ;
- l'association voix ↔ personne est identique sur V1, V2 (avec community-1) et V3 (avec le bon
  nombre de locuteurs) ;
- elle lève le conflit protobuf qui bloquait pyannote 4.

Restent deux limites, à traiter ailleurs :
- le visage est moins souvent trouvé, ce qui fait tomber V3 à 63,6 % quand le nombre de locuteurs n'est pas donné
  (§3.4) ;
- l'association par les lèvres est sensible aux petits groupes de voix.

Pour reproduire :

```bash
research/mediapipe_tasks/run_videos.sh                       # Tasks API on the e2e masks (MP_SUFFIX=_fallback with --face_fallback)
python research/mediapipe_tasks/compare_archives.py .e2e_outputs/v2_ava/poses_mp .e2e_outputs/mp_tasks/v2_ava/poses_mp
python research/mediapipe_tasks/coco_accuracy.py prepare research/benchmark/data/annotations_trainval2017.zip .e2e_outputs/scratch/coco500
python research/mediapipe_tasks/coco_accuracy.py run .e2e_outputs/scratch/coco500 --tag tasks   # once per environment
DIAR=community1_auto research/mediapipe_tasks/downstream.sh mp_tasks v2_ava v3_talkshow
```

## Sources

- [MediaPipe Solutions guide : fin des *Legacy Solutions*](https://developers.google.com/edge/mediapipe/solutions/guide)
- [Holistic landmark detection guide for Python](https://developers.google.com/edge/mediapipe/solutions/vision/holistic_landmarker/python)
- [Holistic landmarks detection task guide](https://developers.google.com/edge/mediapipe/solutions/vision/holistic_landmarker)
- [mp.tasks.vision.holistic_landmarker (référence de l'API)](https://developers.google.com/edge/api/mediapipe/python/mp/tasks/vision/holistic_landmarker)
- [mediapipe 1.1.1 (PyPI)](https://pypi.org/project/mediapipe/1.1.1/)
