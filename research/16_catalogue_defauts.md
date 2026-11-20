# Catalogue des défauts de la reproduction en avatars (2026-11-20)

Objectif : rassembler les défauts de la chaîne avatar sur quatre scénarios différents avant de corriger, afin de
corriger les causes et non les symptômes. Tous les chiffres ci-dessous sont mesurés sur nos sorties, avec la même
chaîne pour toutes les vidéos (`scratch/avatar_chain.sh`). La chaîne enchaîne SAM 3D Body sur chaque image, puis
le lissage robuste, le visage, le recalage des mains sur RTMW, les plaques de décor et le rendu
`--background plate`.

| | Vidéo | Ce qu'elle teste |
|---|---|---|
| **TS** | Talk-show, 2 personnes, 118 plans, 60 im/s | Montage, zooms, gros plans |
| **V1** | `example/Video.mp4`, 1 personne, caméra fixe, 25 im/s, 46 s | Le cas le plus simple : référence |
| **V2** | AVA, 2 personnes, 1 plan avec caméra mobile lente, 25 im/s, 120 s | Caméra qui bouge, corps coupés par le cadre, manteaux |
| **V4** | Danse, personnage animé non humain (queue, jupe, cheveux), 608×1080, 24 im/s, 48 s | Mouvements rapides et complexes, corps non humain |

## 1. Mesures par scénario

Les erreurs 2D sont en tailles de torse (distance épaules–hanches), les erreurs des mains en tailles de main.
Elles sont mesurées sur les points RTMW de confiance supérieure à 0,5 (corps), et sur le modèle de main RTMPose
gardé à part pour les mains. « Rapide » désigne les 10 % d'images où les points 2D bougent le plus.

| Mesure | TS | V1 | V2 (p0 / p1) | V4 |
|---|---|---|---|---|
| Écart de taille avatar/2D dans un plan (écart-type) | < 4 % | 1,5 % | 7 % / 9 % | **13 %** |
| Variation de taille d'une image à l'autre (p95) | — | 0,5 % | 7 % / 8 % | **25 %** |
| Silhouette contre masque SAM3 : IoU, aire | 0,83–0,85 ; 1,00 | 0,88 ; 0,96 | 0,83 / 0,80 ; 0,93 / 0,86 | **0,54 ; 0,68** |
| Corps 2D : médiane, PCK@0,2 | OKS 0,82 | 0,05 ; 0,89 | 0,20 / 0,27 ; 0,50 / 0,44 | **0,34 ; 0,40** |
| Corps 2D, images rapides (brut → final) | — | 0,047 → 0,054 | 0,23 → 0,22 | **0,86 → 0,94** |
| Poignets et chevilles 2D (médiane) | PCK poignets 0,92 | 0,06 | **0,83 / 1,06** | 0,73 |
| Mains, médiane (brut → recalé) | 0,24 → 0,09 | 0,15 → 0,05 | 0,62 → 0,50 / **1,11 → 0,99** | 0,37 → 0,25 |
| Pics de mouvement par minute (brut → final) | 386 → 0 | 119 → 0 (profondeur) | 262 → 0,2 | **1367 → 293** |
| Sauts d'orientation > 45° par image (brut → final) | — | 0 → 0 | 0 → 0 | **27 → 0** |
| Visage lu par MediaPipe | — | 99,7 % | 87 % / **65 %** | **28 %** |
| Fond : plans avec plaque | 69/118 | 1/1 (après seuils) | **0/1** | 1/1 |

## 2. Défauts, classés par cause

Priorité : **A** = visible dans toute la vidéo ou fausse les mesures ; **B** = visible par moments ; **C** = mineur.

### 2.1 Les points 2D de référence ne sont pas fiables (cause commune à plusieurs défauts)

| # | Défaut | Où | Preuve | Prio |
|---|---|---|---|---|
| D1 | **Mains et poignets hors cadre détectés quand même** : RTMW les place au bord de l'image avec une confiance > 0,5 | V2 (et TS en gros plan) | Superposition des images 1200, 2000 et 2700 de V2 (`scratch/v2_overlay.jpg`) : les poignets verts sont collés au bas du cadre | **A** |
| D2 | Le recalage des mains suit ces faux points : sur V2 p1, l'erreur reste à 0,99 taille de main après recalage | V2 | `hand_refit` V2 | **A** |
| D3 | RTMW confond la queue et la jupe avec les jambes du personnage animé | V4 | `scratch/v4_overlay.jpg`, images 650 et 900 | B |

**Correctif proposé** : avant le recalage et les mesures, ignorer tout point 2D situé à moins d'une marge du bord
de l'image ou hors du masque SAM3 de la personne (dilaté). Même filtre dans `hand_reference.py`, `hand_refit.py`
et les diagnostics. Sans ce filtre, les mesures de V2 surestiment l'erreur de l'avatar.

### 2.2 Ce que SAM 3D Body produit image par image (les données)

| # | Défaut | Où | Preuve | Prio |
|---|---|---|---|---|
| D4 | **Taille fausse en gros plan et en insert** (jusqu'à ×2,35), à cause de la focale fixe supposée sur des cadrages serrés | TS (environ 8 % des images) | `size_diag2` TS | **A** |
| D5 | Le personnage « respire » en taille en caméra mobile : 7 à 9 % dans le plan, 7 à 8 % d'une image à l'autre (V1 : 1,5 % et 0,5 %) | V2 | `size_diag2v` | B |
| D6 | **Mouvement rapide** : brut très instable (bruit médian 71 mm, 1367 pics/min, 27 demi-tours en une image) ; le lissage laisse encore 293 pics/min | V4 | `motion_diag` V4 | **A** |
| D7 | Profondeur brute instable même en caméra fixe (119 pics/min), corrigée par le lissage | V1, TS | `motion_diag` | C (corrigé) |
| D8 | Jambes inventées quand elles sont hors cadre (100 % cachées dans V2) | V2 | `hidden share` | C (attendu) |

**Correctifs proposés** :
- **D4, D5** : transmettre au modèle une focale cohérente avec le cadrage. La focale propre au recadrage change
  avec le recadrage, et l'orientation globale ne se déduit pas correctement d'un recadrage seul (CLIFF, PersPose).
  On peut aussi estimer la focale par plan (W-HMR, Zolly), ou recaler l'échelle et la profondeur sur la hauteur du
  masque SAM3, puisque celle-ci est stable (IoU de 0,83 à 0,88 hors V4).
- **D6** : voir D9.

### 2.3 Lissage et recalage (notre traitement)

| # | Défaut | Où | Preuve | Prio |
|---|---|---|---|---|
| D9 | **Le lissage est en retard sur les gestes rapides** : sur les images rapides de V4, l'erreur 2D passe de 0,86 à 0,94 torse. Le lissage stabilise mais s'éloigne de l'image (déjà vu au §8 de la note 15 pour les mains) | V4, un peu V1 | `body_fidelity.py` | **A** pour la danse |
| D10 | Le recalage des mains réduit l'erreur mais ne corrige pas le reste du corps : rien ne recale le corps sur la 2D | V4, V2 | Corps brut ≈ final partout | B |

**Correctif proposé** : étendre le recalage 2D au corps entier, avec le même principe que le recalage des mains
(Geman-McClure, terme temporel). Il servirait de dernière étape après le lissage, avec un poids temporel adapté à
la vitesse, pour que le mouvement suive l'image sans revenir aux pics. Il ne doit être fait qu'**après D1**, sinon
il recalera aussi sur les faux points.

### 2.4 Forme du corps

| # | Défaut | Où | Preuve | Prio |
|---|---|---|---|---|
| D11 | Silhouette non humaine non représentable (queue, jupe, cheveux volumineux) : largeur 0,55 et IoU 0,54 | V4 | `silhouette_diagv` | C (limite du modèle humain) |
| D12 | Avatar plus étroit que la personne habillée (manteaux) : largeur 0,88–0,90 | V2 ; femme TS −13 % | `silhouette_diagv` | C |

Ce n'est pas un défaut de pose. Il est à documenter comme limite : MHR décrit un corps humain sans vêtements.

### 2.5 Visage

| # | Défaut | Où | Preuve | Prio |
|---|---|---|---|---|
| D13 | Visage lu sur 28 % des images seulement pour un visage animé (yeux et proportions hors de ce qu'apprend MediaPipe) | V4 | `face_blendshapes` | C (attendu) |
| D14 | Visage lu sur 65 % des images pour une personne souvent de profil ou de dos ; quand la tête de SAM 3D Body sert à placer la recherche, 2,7 % seulement sont lus | V2 p1 | `face_blendshapes` | B |
| D15 | Pas de clignements | TS | Visuel | C |

### 2.6 Fond (décor)

| # | Défaut | Où | Preuve | Prio |
|---|---|---|---|---|
| D16 | **Caméra mobile : fond gris sur toute la vidéo**. `decor_cameras` voit 21 coupes dans un plan unique (le panoramique ressemble à des coupes), il garde 2 morceaux qui commencent à 3,0 s et 36,6 s, et le rendu ne retrouve donc aucun début de plan | V2 | `decor_cameras.json`, `final_plate_own.log` | **A** pour V2 |
| D17 | Une plaque fixe ne peut de toute façon pas représenter un panoramique : il faut une plaque par image (mosaïque recalée par homographie, ou inpainting vidéo de type ProPainter) | V2 | Principe | A (même cause) |
| D18 | Halo et taches autour de l'avatar : le décor caché par la personne n'est jamais vu (12,9 % de V1) et l'inpainting OpenCV l'étale | V1, TS | Visuel | B |
| D19 | Plans sans plaque fiable en gris (49/118) | TS | §8 de la note 15 | Secondaire (accepté) |
| D20 | Les seuils de plaque doivent être réglés à la main pour un plan unique (`--min_plate_inliers 0`) | V1, V4 | Log V1 | C |

### 2.7 Rendu et outils

| # | Défaut | Où | Preuve | Prio |
|---|---|---|---|---|
| D21 | Dessin du maillage par lots : 0,85 % des pixels diffèrent d'un dessin triangle par triangle, uniquement sur des bords ; aucun trou | V1 | `holes_test.py` | C (vérifié, pas une cause) |
| D22 | `smooth_params.py` projette en 2D avec un centre fixe de 1920×1080 : faux pour V4, mais cela ne sert qu'au score d'annotations, non utilisé ici | Code | ligne 249 | C |

## 3. Ce qui a été vérifié et n'est pas une cause

- **Taille du personnage dans TS** : elle ne vient ni du rendu (silhouette d'aire 1,00), ni des plaques (qui
  correspondent à 2 % près). Elle vient du montage, des zooms et de D4.
- **Trous du maillage** : D21, négligeable.
- **Inversions avant/arrière** : le lissage les supprime toutes (V4 : 27 → 0).
- **Son** : corrigé dans `psifx/io/video.py` (branche `fix/visualization-audio`).

## 4. Ordre de correction proposé

1. **D1** : filtrer les points 2D hors cadre ou hors masque. C'est peu coûteux, cela rend les mesures justes et
   c'est la condition de D2 et de D10.
2. **D16** : détection de coupes robuste au mouvement de caméra ; rendu qui accepte un plan sans plaque. Ensuite
   **D17** : plaque par image en caméra mobile (mosaïque par homographie).
3. **D9 et D10** : recalage 2D du corps entier, avec un poids temporel adapté à la vitesse (la danse).
4. **D4 et D5** : focale par plan, ou échelle recalée sur le masque.
5. Le reste (C) est documenté comme limite dans le rapport.

## Sources

- PersPose, recadrage et intrinsèques de la caméra : https://arxiv.org/html/2508.17239v1
- W-HMR, focale estimée sans étiquettes ; revue de SPEC, CLIFF et Zolly : https://arxiv.org/pdf/2311.17460
- Zolly, focale et distorsion de perspective : https://arxiv.org/pdf/2303.13796
- ProPainter, inpainting vidéo : https://arxiv.org/abs/2309.03897
- VIP, suppression de personnes en vidéo réelle : https://arxiv.org/html/2504.03041v1

## 5. Correctifs appliqués (2026-11-20, une branche par correctif, empilées)

Les quatre correctifs sont empilés, chacun sur sa branche, dans le worktree `.worktrees/moving-plate`.
La chaîne qui les enchaîne est `scratch/avatar_chain2.sh`.

### 5.1 Points 2D hors cadre (`fix/offframe-keypoints`)

`keypoint_view.py` ne garde un point 2D que s'il est à la fois :
- dans l'image : hors des bandes noires, trouvées une fois par vidéo, et à plus de 1,5 % de la hauteur du bord ;
- sur le masque SAM3 de sa personne, dilaté de 12 px.

Ce filtre est utilisé par `hand_refit.py`, par la visibilité du lissage robuste et par les mesures.

| AVA | Avant | Après |
|---|---|---|
| Points de main RTMW gardés | 100 % | 49 % (51 % hors cadre) |
| Erreur des mains mesurée (taille de main), p0 / p1 | 0,50 / 0,99 | 0,10 / 0,40 |
| Erreur du corps mesurée (taille de torse), p0 / p1 | 0,20 / 0,27 | 0,05 / 0,12 |
| Mains cachées : déplacement entre l'ancien et le nouveau recalage | — | 78 / 88 px (médiane) |
| Mains vues : même déplacement | — | 0,3 / 4 px |

L'essentiel de l'erreur était **dans la mesure**, pas dans l'avatar : sur les points vus, l'ancien recalage fait
aussi bien que le nouveau. Le correctif sert surtout aux mains cachées. Le recalage les tirait vers le bord de
l'image ; elles restent maintenant là où le lissage les place.

### 5.2 Fond en caméra mobile (`fix/moving-camera-plate`)

- **Coupes trouvées à la pleine fréquence d'images**, comme le fait le rendu. Une coupe change l'image d'une image à
  l'autre ; un panoramique la change peu à chaque image. Avec 0,2 s entre deux images comparées, chaque panoramique
  passait pour une série de coupes.
  - AVA : 21 fausses coupes → 0.
  - Talk-show : 106 coupes, contre 115 avec l'ancienne règle.
- **Mosaïque par plan mobile.** Les images du plan sont enchaînées par des homographies calculées sur le décor, dans
  le repère de l'image du milieu (moins de dérive qu'à partir de la première). On en retire :
  - les personnes ;
  - les incrustations, c'est-à-dire ce qui reste fixe à l'écran alors que le décor, réaligné, change (logo) ;
  - les bandes noires.
  Le rendu découpe ensuite le fond de chaque image dans cette mosaïque.
- **AVA** : un seul plan (déplacement de 861 px), et le fond retrouvé sur toute la vidéo.
  - Écart médian au vrai décor : 3 niveaux de gris.
  - 93 % des pixels du décor sont à moins de 20 niveaux.
  - Défaut restant : les 2 premières secondes (panoramique rapide et flou), avec un écart de 43 niveaux.
- **Talk-show** : 7 petits plans mobiles (zooms, recadrages) reçoivent eux aussi une mosaïque.

### 5.3 Recalage du corps entier (`fix/body-refit`)

`body_refit.py` cale les 17 points COCO de l'avatar sur RTMW. Ces points sont ceux de MHR70, ceux que comparent les
annotations. Pour éviter de calculer le maillage, chaque point est attaché à une articulation par son décalage dans
le repère de celle-ci (0,3 cm d'erreur après un déplacement de 55 cm du coude). Le terme temporel est allégé là où
la 2D bouge vite.

Pour la mesure, une image sur 5 est retirée du recalage et l'erreur est mesurée sur ces images.

| | Lissé | **Recalé** | Brut |
|---|---|---|---|
| Talk-show, OKS / PCK@0,05 / PCK des poignets (annotations) | 0,816 / 0,768 / 0,922 | **0,841 / 0,790 / 0,931** | 0,830 / 0,769 / 0,920 |
| Talk-show, erreur sur les images retirées (torse), p0 / p1 | 0,063 / 0,055 | **0,030 / 0,032** | |
| V1, erreur sur les images retirées | 0,035 | **0,019** (tremblement égal au lissé) | |
| AVA, erreur sur les images retirées, p0 / p1 | 0,052 / 0,115 | **0,030 / 0,074** | |
| Danse animée, erreur sur les images retirées, toutes / rapides | 0,33 / 1,30 | **0,24 / 1,13** | |

Deux essais ont échoué, et leurs leçons sont gardées :
- **Articulations nues, tête libre, sans point du visage** : l'avatar se rapproche de RTMW mais la tête dérive.
  L'OKS tombe de 0,816 à 0,747. Être proche du détecteur ne veut pas dire être juste.
- **Poids temporel ajusté selon (fps/25)⁴** : à 60 images/s, le problème devient trop raide pour converger (OKS
  0,70). Le bruit du détecteur est propre à chaque image, donc le poids reste défini par image (10⁵).

Coût : le tremblement du talk-show passe de 0,004 à 0,011 torse par image², soit encore 14 fois moins que le brut.

### 5.4 Taille en gros plan (`fix/closeup-size`)

Le défaut annoncé (« taille fausse jusqu'à ×2,35 ») était **surtout un artefact de mesure**. En gros plan, RTMW
invente des épaules et des poignets hors du cadre ou collés au bord.
- Mesurée seulement sur les points vus, la dispersion entre plans de la personne 0 passe de 9 % à 4 %.
- Mesurée sur le seul visage (nez, yeux, oreilles), la taille de l'avatar correspond à l'image dans **tous** les
  plans, à 0,87–1,07 près. La tête est un peu trop petite (jusqu'à −13 %) dans les cadrages les plus serrés.

Recaler en plus la profondeur et la position (option `--depth`) ne change pas l'OKS (0,843) et triple le
tremblement : l'option est laissée désactivée. Les changements de taille visibles dans le talk-show viennent donc
du montage, pas de l'avatar.
