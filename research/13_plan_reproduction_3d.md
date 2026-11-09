# Plan : reproduire une vidéo avec des avatars 3D, et retirer l'apparence

Date : 2026-11-09. Statut : plan révisé après retour, à valider avec les encadrants. Point de départ : `dev` au tag
`tm-v1-optimisations`. Échéance de travail : fin décembre 2026. Le rapport s'écrit au fil du travail, idées abandonnées comprises.

## 1. Les objectifs

**Objectif de base : prouver que les données produites sont exploitables.** Une vidéo reproduite à la caméra d'origine,
avec un avatar par personne, le décor repris de la vidéo ou minimal, le son et les sous-titres. Le chercheur voit
directement ce que contiennent les données et ce qui leur manque. Pas de point de vue libre.

**Objectif de thèse proposé : retirer l'apparence en conservant le comportement.** L'avatar remplace le visage et le
corps, et on mesure deux choses en même temps :

- **Utilité :** le comportement mesuré sur la vidéo reproduite reste-t-il celui de l'original ?
- **Confidentialité :** peut-on encore reconnaître la personne ?

Le même travail sert donc les deux objectifs. Je n'ai trouvé aucune étude qui mesure ces deux axes pour un avatar 3D
sur des données d'interaction sociale : c'est la place que le travail peut prendre. Les méthodes publiées existent, mais
visent autre chose (section 3).

Le mot « anonymisé » est réservé à ce qu'une mesure justifie. Au sens du RGPD, des données restent personnelles tant
qu'une réidentification est raisonnablement probable (considérant 26). Le texte parle donc d'« apparence retirée ».

## 2. État mesuré

### 2.1 Maillages image par image (nuit du 6 novembre)

SAM 3D Body sur `ExtraitTalkShow.mp4` (5 min 12, 60 images/s, une image sur 2), masques SAM3 de la chaîne.
Coût : **0,69 s par image, soit 1 h 48**.

| Modèle dans la chaîne (200 images annotées) | OKS | PCK@0,1 |
|---|---|---|
| ViTPose | 0,893 | 0,937 |
| **SAM 3D Body** (points 2D réduits à COCO-17) | **0,849** | **0,904** |
| RTMO | 0,742 | 0,722 |
| MediaPipe | 0,690 | 0,794 |

| Piste | Images | Largeur d'épaules, cm (p5 / p50 / p95) | Vitesse relative au bassin, mm (p50 / p95) | Profondeur, m (p1 / p50 / p99) |
|---|---|---|---|---|
| 0 | 6 120 | 33,4 / 34,9 / 36,9 | 6,0 / 79,0 | 1,03 / 2,22 / **10,15** |
| 1 | 3 840 | 33,6 / 36,8 / 37,9 | 6,2 / 72,1 | 0,82 / 1,59 / 2,53 |
| 2 | 4 346 | **16,6** / 33,7 / 34,9 | 7,8 / 133,2 | 1,58 / 2,50 / **7,64** |

L'image voisine est utilisée quand l'image annotée est impaire. Les vitesses incluent les changements de plan, donc leur
p95 surestime le tremblement. SAM 3D Body n'a pas de score par point : les mesures de points fantômes ne s'appliquent pas.

**Conséquences :**

1. La profondeur est fausse : des personnes assises sont placées à 10 m, car la focale supposée est la même pour tous
   les plans.
2. La morphologie varie d'une image à l'autre.
3. Une personne porte plusieurs numéros : sur la planche de contrôle, la même femme est P1 en gros plan et P2 en plan
   large (à confirmer sur toute la vidéo).
4. `run_tracks.py` jette `shape_params`, `scale_params`, `body_pose_params`, `hand_pose_params`, `expr_params` et
   `global_rot`. Sans eux, ni verrouillage de la morphologie, ni lissage propre, ni morphologie neutre.

### 2.2 Points clés du décor (test du 9 novembre)

Script : `research/scene3d/decor_cameras.py`. SIFT d'OpenCV sur le décor seul (personnes masquées), homographie RANSAC
entre plans. Résultats dans `.e2e_outputs/extrait_talkshow/pipeline/decor_cameras/`.

| Mesure | Résultat |
|---|---|
| Plans d'au moins 1 s | 86 (médiane 2,4 s) |
| Points clés de décor par plan (p10 / p50 / p90) | 420 / 732 / 1 943 |
| Groupes de caméra | 19 ; les 9 plus longs couvrent 93 % de la durée |
| Points cohérents entre plans d'un même groupe (médiane) | 90 à 1 077 |
| Décalage du centre de l'image entre plans d'un groupe | Moins de 0,6 px à 960 de large pour 7 groupes de plusieurs plans sur 8, 3,6 px pour le dernier |
| Mouvement du décor dans un plan de plus de 5 s (p50 / p90) | 0,2 / 4,7 px, caméras quasi fixes |

**Lecture :**

- **Les points clés du décor fonctionnent pour retrouver la caméra.** Sur la planche, chaque grand groupe contient bien un seul cadrage.
  Une même caméra revient donc de nombreuses fois : le plan large revient 22 fois, un gros plan de l'homme 24 fois. Ses
  paramètres peuvent être estimés une fois, sur plusieurs dizaines de secondes d'images, au lieu d'être refaits à chaque coupe.
- **Le zoom se détecte aussi.** Les groupes 2, 3, 5 et 7 (la femme, cadrages voisins) sont reliés entre eux par 294 à
  372 points. Ils peuvent se ramener à un seul fond.
- **Piège trouvé : les incrustations.** Sans masquer les logos, un premier essai reliait les 86 plans en un seul groupe.
  Le dessin des correspondances montre que le grand logo d'introduction « SMALL TALK » se reliait au petit logo du coin :
  ce sont les mêmes lettres. Un masque simple des coins ne règle pas tout : le plan d'introduction, les bandeaux et les
  vidéos insérées restent à écarter.
- **Limite :** parmi les 20 liens les plus forts entre groupes, aucun ne relie, dans les groupes vérifiés sur la
  planche, un gros plan de la femme à un gros plan de l'homme. Les caméras se font face et ne voient pas le même décor.
  Sans point de vue libre, ce n'est plus un obstacle.
- **Une transition échappe à la détection.** Le groupe 6 est un plan unique dont le fond est un filé de zoom : le critère de
  coupe doit aussi voir les transitions animées.

**Fonds de scène.** La médiane du décor visible donne une image sans personne pour chaque caméra. Les fonds sont propres,
avec un trou à la place de la personne assise : de 14 à 24 % des pixels jamais vus pour les gros plans. L'avatar couvrira
une bonne partie de ce trou, et on complétera le reste (remplissage simple, puis fonds des autres cadrages reliés).
Les incrustations visibles dans certains fonds (groupe 0) sont à corriger.

### 2.3 Calculer moins d'images : simulation (test du 9 novembre)

Idée : le modèle ne calcule que certaines images, on devine les autres, et il recalcule si l'image change trop ou s'il y a une coupe.
Script : `research/scene3d/keyframe_simulation.py`. Simulation sans GPU, sur le talk-show : on cache des réponses déjà
calculées, on interpole les points à l'intérieur de chaque plan, puis on compare au modèle et aux annotations.

| Image calculée | Gain (par rapport à 1 sur 2) | OKS | Poignet, écart au modèle, mm (p50 / p95) |
|---|---|---|---|
| 1 sur 2 (référence) | – | 0,849 | – |
| 1 sur 4 | × 2,0 | 0,850 | 6,8 / 56 |
| 1 sur 6 | × 2,9 | 0,849 | 8,0 / 62 |
| **1 sur 10** | **× 4,7** | **0,848** | 9,5 / 72 |
| 1 sur 20 | × 8,7 | 0,843 | 13,5 / 94 |
| 1 sur 40 | × 15 | 0,821 | 19,4 / 132 |
| 1 sur 10, plus recalcul si l'image change (seuil haut) | × 3,4 | 0,849 | 8,3 / 63 |
| 1 sur 20, plus recalcul si l'image change (seuil haut) | × 3,6 | 0,849 | 11,1 / 74 |

**Lecture :**

- **L'espacement fixe tient bien jusqu'à 1 sur 10.** 1 sur 4 donne déjà 56 mm de p95 au poignet, ce qui indique surtout le
  tremblement du modèle lui-même.
- **Le recalcul selon la différence d'image n'apporte presque rien ici :** 1 sur 20 avec recalcul coûte comme 1 sur 7 environ, pour
  une erreur que 1 sur 10 fixe obtient à moindre coût. Une différence de pixels voit aussi la lumière et le décor, pas seulement le geste.
- **Les coupes sont déjà respectées** dans toutes ces lignes : aucune interpolation ne les traverse.

**Découpage récursif** (`research/scene3d/keyframe_recursive.py`). Les coupes sont trouvées d'avance, par différence d'image. La
dichotomie porte sur les gestes : on part d'une image sur 40, on calcule le milieu de chaque intervalle, et on ne redécoupe que si
le modèle contredit l'interpolation (corps : p95 des articulations autour du bassin, rotation globale retirée ; place : position du
bassin à l'image).

| Sur le talk-show | Gain | OKS | Poignet, p95 (mm) |
|---|---|---|---|
| 1 sur 10 fixe | × 4,7 | 0,848 | 72 |
| Récursif, seuil 120 mm | **× 5,0** | 0,846 | **56** |
| Récursif, seuil 80 mm | × 4,2 | 0,847 | 51 |
| Récursif, seuil 60 mm | × 3,4 | 0,847 | 47 |

Pour un gain voisin, les grosses erreurs au poignet baissent d'un quart. L'écart d'OKS (0,002) est dans le bruit de 200 images.

**Pire cas : caméra qui bouge, pas de coupe.** Chaque image n'est calculée qu'une fois et son résultat resservi : si tout est redécoupé,
on retrouve le coût normal (× 1,0), jamais plus, à part les tests eux-mêmes.

| Caméra synthétique, seuil 60 mm | Test du corps seul | Corps et place | Sans retirer la rotation |
|---|---|---|---|
| 30°/s, 300 px sur 8 s | × 3,7 | × 3,4 | × 3,0 |
| 90°/s, 300 px sur 4 s | × 3,5 | × 3,0 | × 2,5 |
| 90°/s, 600 px sur 2 s | × 3,6 | × 2,1 | × 2,0 |
| 180°/s, 600 px sur 1 s | × 3,4 | × 1,4 | × 1,3 |

- Le corps seul ne réagit presque pas à la caméra, une fois sa rotation retirée. C'est ce qui manque à une différence d'image.
- Le test de place, lui, déclenche : il faudra sans doute lui donner le mouvement de la caméra, tiré des points clés du décor (non testé).
- Avec un budget (jamais plus dense qu'une image sur 8), même si tout demande un redécoupage : × 2,5, OKS 0,849. Le gain minimum est alors garanti.
- Vraies vidéos, comparées à leurs archives à 1 image sur 5 : film V2 × 1,6 à × 1,8 ; danse V4 × 1,0, sans gain ni perte.

Cette caméra synthétique ne crée ni flou ni changement d'apparence : une vraie vidéo à caméra mobile reste à tester.

**Limites communes :** points interpolés et non paramètres MHR ; référence à 30 images/s, donc les gestes plus rapides ne sont pas vus ;
gain du modèle seul, sans le décodage vidéo ; une seule vidéo. Sur MEVID, interpoler lisse le mouvement et pourrait **gonfler
artificiellement la confidentialité** : à mesurer avec et sans.

## 3. Veille

### 3.1 Idée examinée et abandonnée : s'inspirer des VTubers

Idée du 2026-11-09 : un VTuber anime un modèle 3D avec une seule caméra, pourquoi pas nous ? Elle est écartée pour
deux raisons :

- **Les conditions sont trop différentes.** Un VTuber classique est proche de sa caméra, de face, et sa chaîne peut
  s'appuyer sur ces hypothèses. psifx doit rester fiable dans la majorité des situations : plans larges, trois-quarts,
  micro devant la bouche.
- **Les cas impressionnants ne viennent pas de la caméra.** Une VTubeuse comme Filian se déplace beaucoup dans l'espace
  avec un casque VR et des manettes : d'après le souvenir de l'auteur (non documenté dans mes sources), ce sont les
  capteurs qui portent le suivi. Les guides confirment que la webcam est le niveau le plus faible pour le corps entier.

**Ce qu'il en reste, indépendamment de cette piste :** séparer l'identité du mouvement, ce que MHR fait déjà. Un essai
des coefficients de visage de MediaPipe n'est plus prévu, sauf si les `expr_params` de MHR se révèlent insuffisants.

### 3.2 Autres briques

| Brique | Candidat | Ce qu'on sait | Décision |
|---|---|---|---|
| Modèle du corps | **MHR** | Apache-2.0 ; 45 paramètres d'identité, 204 de pose, 72 d'expression (FACS), rig FBX, conversion SMPL/SMPL-X | **Retenu** |
| Maillage | **SAM 3D Body** | Déjà testé. Licence SAM : redistribution sous les mêmes termes, citation obligatoire | **Retenu** |
| Cohérence temporelle | SAM-Body4D | MIT. Morphologie fixée, Kalman sur la pose et les mains. Aucune mesure chiffrée. 14,5 Go de pic par lots de 64 ; 26 min pour 90 images à 5 personnes avec la complétion des occlusions | **Reprendre la méthode**, sans la complétion ni l'environnement |
| Cohérence temporelle | Fenêtre glissante dans l'espace MHR (arXiv 2512.21573), `SAM3DBody_Smooth` | Même principe, avec une prudence sur les rotations rapides | À comparer au Kalman |
| Correspondance entre images | SIFT d'OpenCV ; SuperPoint + LightGlue | LightGlue mieux classé en relocalisation, licence de SuperPoint à vérifier. SIFT suffit déjà sur le talk-show | **SIFT**. LightGlue si V2 ou V4 l'exigent |
| Focale | MoGe (prévu par SAM 3D Body) | Médiane sur une caméra entière plutôt qu'image par image | Essai au jalon 3 |
| Vidéos montées, repère monde | HumanMM, PromptHMR-Vid, GVHMR, Human3R | Utiles pour une caméra mobile et un repère commun | **Hors périmètre**, sans point de vue libre |
| Avatar réaliste | ExAvatar, LHM, PERSONA, AHOY | Reproduisent l'apparence, donc l'identité | **Écarté**, à l'opposé de l'objectif |
| Rendu | OpenCV actuel, Blender (`bpy`), nvdiffrast, PyTorch3D | Aucun comparatif sur ce cas | Essai au jalon 4 |

### 3.3 Retirer l'apparence : l'existant

| Travail | Ce qu'il apporte | Ce que ça implique ici |
|---|---|---|
| MIRAGE (2026) | Démarche, posture et silhouette permettent la réidentification même sans le visage | Mesurer aussi le corps, pas seulement le visage |
| Pantomime (PoPETs 2026) | Le mouvement seul identifie ; leur anonymisation ramène l'identification à 10 % | Tension de fond : le comportement qu'on veut conserver est lui-même identifiant |
| ReGenHuman, GaitProtector (2026) | Anonymisation du corps entier ou de la démarche, avec mesure d'utilité | Méthodes de comparaison, cadre d'évaluation |
| NullFace, AnonNET, Kung et al. (WACV 2025) | Échange de visage conservant regard et expression | Point de comparaison : garde une vidéo réaliste |
| Avatars en réalité virtuelle sociale (IJHCS 2025) | Regard, parole et sourire proches du face-à-face avec des avatars génériques | Les avatars portent le comportement, mais le contexte diffère |
| VoicePrivacy 2024 | Trois mesures : EER (confidentialité), WER, reconnaissance d'émotion (utilité) | Modèle de double mesure à reprendre, et la voix reste identifiante |

## 4. Architecture

```
vidéo ─► plans ─► groupes de caméra (points clés du décor) ─► focale, fond de scène ─┐
  │                                                                                  ▼
  ├─► SAM3 ─► pistes ─► personnes fusionnées ─────────────────────────────────────► SCÈNE ─► vidéo reproduite
  │                                                                                  ▲  ├─► mesures d'utilité
  ├─► SAM 3D Body ─► paramètres MHR ─► lissage ─────────────────────────────┤  └─► mesures de confidentialité
  └─► audio ─► diarisation, transcription, voix ↔ personne ──────────────────────────┘
```

```
scene/
  scene.json         # plans [début, fin, caméra], caméras (focale, fond), personnes (pistes, morphologie)
  plates/<camera>.jpg
  people/<id>.npz    # images, pose, mains, expression, rotation et translation dans le repère caméra, statut
  speech.vtt
```

**Règles :**

- On garde les paramètres, pas les sommets : environ 1 à 2 Ko par image et par personne (estimation), contre 110 Ko.
- Chaque image dit son statut : `observé` ou `interpolé`. Les mesures ne comptent que l'observé.
- Le lissage ne traverse jamais une coupe.
- L'identité reste séparée du mouvement : pour partager, on remplace la morphologie et on ne publie pas le fichier d'origine.

**Niveaux d'apparence à comparer :**

| Niveau | Personne | Décor | Rôle |
|---|---|---|---|
| A0 | Vidéo d'origine | D'origine | Référence |
| A1 | Visage flouté | D'origine | Pratique courante, à battre |
| A2 | Avatar avec sa morphologie | Fond de scène | Reproduction fidèle : **objectif de base** |
| A3 | Avatar à morphologie moyenne | Fond de scène | Retire la silhouette |
| A4 | Avatar à morphologie moyenne | Décor minimal | Retire aussi le lieu |

## 5. Fil rouge et étapes

Ce travail change souvent de direction : un problème imprévu, une piste qui s'arrête, une nouvelle idée. Il n'y a donc
**pas de calendrier par semaine**, seulement un ordre et une échéance :

1. **Rendre les données exploitables :** paramètres, caméras, personnes, stabilité.
2. **Les montrer :** la vidéo reproduite, preuve de l'objectif de base.
3. **Mesurer :** ce qui reste du comportement, et ce qui reste de l'identité.
4. **Élargir :** autres vidéos, commandes psifx.

Si le temps manque, l'étape 4 saute d'abord. Les seuils sont des propositions, à ajuster sur les premières mesures. Une
branche et une note par étape, et le rapport avance en même temps : **ce qu'on a pensé, fait ou abandonné y est tracé au
moment où ça arrive**.

### Jalon 0 : garder les paramètres

- Enregistrer tous les paramètres MHR dans `run_tracks.py`, relancer le talk-show (1 h 48).
- Écrire `mesh_eval.py`, qui automatise les mesures de la section 2.1.

**Sortie :** sommets recalculés à moins de 1 mm ; OKS inchangé (0,849).

### Jalon 1 : plans, caméras, personnes

**Réduire les limites du décor**, du plus simple au plus coûteux :

| Limite observée | Piste simple | Si ça ne suffit pas |
|---|---|---|
| Logo relié à sa version agrandie | Trouver les zones identiques à travers **tous** les plans, puis les masquer, au lieu de boîtes écrites à la main | Vérifier que les points liés sont répartis dans l'image, pas tous sur un même graphisme |
| Plans d'incrustation (logo plein écran, vidéos insérées) | Plan sans personne suivie, ou sans lien vers les autres : le marquer `incrustation` et ne pas lui donner de fond | Classer le plan à l'aide de SAM3 |
| Transition animée (filé de zoom) | Mesurer le flou et le mouvement du décor dans le plan | Reprendre le détecteur de `reid.py` |
| Un cadrage trop proche d'un autre sépare une même caméra en plusieurs groupes | Relier les groupes dont l'homographie est un zoom (points déjà disponibles) | Clustering sur la matrice des liens |
| Trou du fond derrière la personne | Fusionner les fonds des cadrages reliés, puis remplissage OpenCV : l'avatar couvre le reste | Modèle d'inpainting vidéo (ProPainter : licence non commerciale, à éviter si possible) |
| Vérification à l'œil sur quelques plans seulement | Annoter à la main le groupe de chaque plan (86 plans, quelques minutes) | – |

**Personnes :** fusionner les pistes d'une même personne : apparence, morphologie, et deux pistes visibles dans la même
image ne peuvent pas être la même personne.

**Sortie :** plus de 95 % des plans dans le bon groupe et des images attribuées à la bonne personne, sur les annotations.

### Jalon 2 : cohérence temporelle et visage

- Morphologie et échelle par personne : médiane contre première image.
- Lissage plan par plan : Kalman contre fenêtre glissante.
- Visage : mesurer `expr_params`, en particulier avec le micro devant la bouche.
- Images clés (branche `optim/mesh-keyframes`, voir 2.3) : interpolation dans les paramètres MHR, temps réel mesuré. Comparer
  l'espacement fixe au découpage récursif avec un budget. Vérifier sur des gestes marqués à la main, et sur une vraie caméra mobile.

**Sortie :** accélération médiane divisée par au moins 2 et OKS en baisse de moins de 0,01. La borne sur l'OKS protège
contre un lissage qui effacerait les gestes.

### Jalon 3 : caméras et fonds

- Une focale par groupe de caméra (MoGe, médiane) et par facteur de zoom relié.
- Une profondeur stable par personne : l'échelle vient de la morphologie verrouillée.
- Fonds complétés dans le trou que l'avatar ne couvre pas.

**Sortie, par des invariants vérifiables sans vérité 3D :** bassin d'une personne assise stable à moins de 10 cm dans un
groupe ; plus aucune profondeur aberrante ; OKS du maillage reprojeté à 0,02 près.

### Jalon 4 : première vidéo reproduite

- Avatar d'une couleur par personne sur le fond de scène, son et sous-titres, à côté de l'original.
- Essai de rendu : OpenCV contre rasteriseur GPU ou Blender, sur 8 Go.

**Sortie :** vidéo complète du talk-show, vérifiée à l'œil sur chaque type de plan. **C'est la preuve de l'objectif de base.**

### Jalon 5 : utilité (A0 à A4)

Principe : passer les outils d'analyse sur la vidéo reproduite, puis comparer à l'original.

| Comportement | Mesure |
|---|---|
| Corps | ViTPose sur le rendu, OKS contre les 200 images annotées |
| Tête | Corrélation de l'orientation et des hochements |
| Gestes | Corrélation de la vitesse des poignets, rappel des gestes marqués à la main |
| Bouche | Corrélation de l'ouverture avec MediaPipe, activité pendant les tours de parole (sous-titres de référence) |
| Expressions | Comparaison des coefficients. OpenFace sur un avatar est un essai, pas une garantie |
| Interaction | Orientation mutuelle et synchronie, à l'intérieur des plans larges seulement |

### Jalon 6 : confidentialité

**Principe :** jouer l'attaquant. Un modèle de réidentification reçoit une vidéo modifiée et classe une galerie de
personnes connues. Le **Rank-1** est la part des cas où la bonne personne arrive en tête.

**Corpus : MEVID**, partie test (choix du 2026-11-09). Données MEVA sous CC BY 4.0, avec actrices et acteurs inscrits.

| Partie test | Valeur |
|---|---|
| Identités | 54 (52 dans les requêtes) |
| Pistes | 316 requêtes, 1 438 en galerie |
| Images | 1,19 million |
| Hasard | **Rank-1 de 1,7 %** (calculé sur ces listes, mêmes personne et caméra exclues) |
| Requêtes avec une autre tenue en galerie | 284 sur 316 |

Pourquoi MEVID plutôt que MARS, pourtant plus grand (1 261 identités) : chaque personne y porte environ 4 tenues. On peut
donc séparer ce qui vient des vêtements de ce qui vient du corps et du mouvement. Ses vidéos sont aussi peu contrôlées :
33 caméras, intérieur et extérieur, souvent en hauteur. Un premier échantillon (35 images, deux dossiers d'identité, donc peu
représentatif) montre des gens debout et assis, des visages souvent visibles, mais aussi des personnes coupées, des images
sombres et des images sans personne. Hauteur médiane : 179 px, 24 % sous 128 px. À remesurer sur toute la partie test.

**MARS en complément, seulement si un besoin apparaît.** Il offrirait une référence plus connue, mais pas de meilleurs
visages : c'est aussi de la surveillance. Avec une seule tenue par personne, retirer les vêtements y ferait chuter le score sans
rien dire du corps. Ses pistes sont aussi plus courtes (25 à 50 images), ce qui gêne l'attaque de démarche. Sa page officielle
reste à confirmer. On n'y va que si l'écart de référence de MEVID (52,5 %) se révèle trop étroit pour conclure.

**Niveaux :** N0 vidéo d'origine, N1 flou gaussien sur le visage, N2 avatar avec sa morphologie, N3 avatar à morphologie moyenne.

**Attaquants :**

| Attaquant | Ce qu'il a | Ce que ça mesure |
|---|---|---|
| Naïf | Galerie d'origine, modèle entraîné sur de vraies vidéos | Ce que l'apparence seule laisse passer. **Le plus faible : un score bas ne prouve rien à lui seul** |
| Informé | Galerie passée au même traitement | Ce qui reste commun aux avatars d'une même personne : morphologie et mouvement |
| Visage | Reconnaissance faciale, avec les photos d'inscription des personnes de MEVID (600 Mo) comme galerie | Ce que le visage laisse passer, surtout au niveau du flou |
| Démarche | Modèle de démarche (GaitBase d'OpenGait, licence à vérifier) | La signature du mouvement, qu'un modèle d'apparence ne voit pas |
| Adaptatif (si le temps le permet) | Modèle réentraîné sur des avatars | Le pire cas réaliste. En voix, ce type d'attaquant fait nettement mieux que les autres (VoicePrivacy 2020) |

**Déroulement :**

1. Modèle d'attaque : d'abord CAL, le meilleur publié sur MEVID, avec ses poids. **Retrouver ses 52,5 % de Rank-1 (mAP 27,1)
   avant toute autre mesure :** c'est la vérification que notre évaluation est juste. OSNet ensuite, en second avis.
2. Rendre une fenêtre de 32 images consécutives par piste : 1,9 h pour les requêtes et 8,7 h pour la galerie, à 0,69 s par image
   (à remesurer, les images sont plus petites). Tout rendre prendrait 39 h pour les seules requêtes.
3. Publier Rank-1, mAP et la courbe CMC, séparés entre même tenue et autre tenue, avec un intervalle de confiance par identité.

**Trois précautions de lecture :**

- **Un avatar raté donne aussi le hasard.** La confidentialité ne se lit jamais seule : il faut vérifier sur ces mêmes images que
  le maillage suit bien la personne (points reprojetés contre ViTPose sur l'original) et, si les images sont trop petites pour
  SAM 3D Body, le dire. C'est le risque principal de ce corpus.
- **Le hasard pour un modèle d'apparence ne prouve pas l'absence de fuite.** Il ne regarde pas le mouvement.
- **Un score de démarche au-dessus du hasard n'est pas un défaut à corriger.** C'est le prix de l'utilité : conserver le mouvement,
  c'est conserver ce qui identifie. Le résultat attendu est un compromis chiffré, et perturber le mouvement est une piste séparée.

**Hors de cette mesure :**

- Les personnes y sont surtout en déplacement ou occupées, elles ne conversent pas : le protocole n'évalue pas l'identification par
  les gestes d'une conversation. L'utilité reste mesurée sur le talk-show.
- La voix et le contenu de la parole identifient aussi. MEVID n'a pas de son utile ici : à documenter, pas à mesurer.
- Le flou du niveau N1 demande un détecteur de visage. Sur les petites images ou vues du dessus, il peut échouer souvent : c'est à
  mesurer, et les résultats sont à donner par taille d'image.
- Les images sans personne sont à écarter, avec la même règle pour tous les niveaux. Sinon, elles fausseraient la comparaison.

### Jalon 7 : élargir

- Tenue sur V1 (caméra fixe), V2 (film) et V4 (danse).
- Commandes psifx si le format est stable : `psifx video mesh sam3d_body inference`, `psifx scene build`, `psifx scene render`.

## 6. Risques

| Risque | Conséquence | Réponse |
|---|---|---|
| Le mouvement seul identifie la personne | L'objectif de confidentialité reste partiel | C'est un résultat à publier tel quel : on mesure, et on dit ce qui reste |
| Le micro cache la bouche | Lèvres fausses, utilité faible sur ce point | Mesure dédiée, statut par image |
| Lissage qui efface les gestes | Comportement faussé | Garde-fou OKS, gestes marqués à la main |
| Échéance de fin décembre, et direction qui peut encore changer | Objectif de thèse incomplet | Priorités du fil rouge (section 5) |
| Images MEVID trop petites pour SAM 3D Body | Confidentialité non mesurable, ou faussement bonne | Contrôle de qualité avant toute mesure. Repli : un corpus plus proche, comme Face-to-Face with Jimmy Fallon |
| Usage de vraies données | Question éthique et légale | Vidéos publiques pour le développement. Avis de l'institution avant toute donnée de participants |
| Dépendances : modèle à accès restreint, code non packagé | Installation fragile | Environnement `sam3d-body` figé, commit du clone noté |

## 7. Hors périmètre

- Point de vue libre et repère commun entre caméras.
- Avatars photoréalistes.
- Comportement généré pour les personnes hors champ.
- Caméras très mobiles.
- Anonymisation de la voix.

## 8. Décisions

**Prises le 2026-11-09 :**

1. Reproduction basique : décor minimal ou repris de la vidéo, sans point de vue libre.
2. But de la reproduction : prouver que les données sont exploitables et donner aux chercheurs un support visuel.
3. Retirer l'apparence est retenu comme piste d'objectif de thèse.
4. Échéance : fin décembre. Pas de calendrier par semaine, un fil rouge.
5. Piste VTuber abandonnée. Piste des points clés du décor à creuser.
6. Corpus de confidentialité : MEVID, pour ses tenues multiples et son côté non contrôlé. MARS seulement en complément, si besoin.

**Restent ouvertes :**

1. Confirmation de l'objectif de thèse à la réunion du 2026-11-09.

## Sources

- SAM 3D Body : https://github.com/facebookresearch/sam-3d-body, https://arxiv.org/abs/2602.15989
- MHR : https://github.com/facebookresearch/MHR, https://arxiv.org/pdf/2511.15586v3
- SAM-Body4D : https://arxiv.org/html/2512.08406v1, https://github.com/gaomingqi/sam-body4d,
  https://raw.githubusercontent.com/gaomingqi/sam-body4d/master/assets/doc/resources.md
- Lissage dans l'espace MHR : https://www.emergentmind.com/papers/2512.21573,
  https://docs.comfy.org/built-in-nodes/SAM3DBody_Smooth.md
- VTubers : https://mocaponline.com/blogs/mocap-news/vtuber-motion-capture-guide, https://jisaku.com/posts/vtuber-tracking-setup-guide
- Corpus : https://github.com/Kitware/MEVID, https://arxiv.org/abs/2211.04656, https://mevadata.org/, https://link.springer.com/content/pdf/10.1007%2F978-3-319-46466-4_52.pdf, https://arxiv.org/abs/2603.14794
- Attaque de démarche et attaquants : https://arxiv.org/abs/2211.06597, https://arxiv.org/pdf/2109.00648, https://arxiv.org/pdf/2005.08601
- Inpainting : https://ar5iv.labs.arxiv.org/html/2309.03897
- Images clés : https://arxiv.org/pdf/2007.15217
- Correspondance d'images : https://arxiv.org/pdf/2306.13643, https://arxiv.org/html/2602.08430v1
- Vidéos montées et repère monde : https://arxiv.org/abs/2503.07597, https://arxiv.org/pdf/2610.19729,
  https://openaccess.thecvf.com/content/CVPR2025/papers/Wang_PromptHMR_Promptable_Human_Mesh_Recovery_CVPR_2025_paper.pdf,
  https://openaccess.thecvf.com/content/CVPR2026/papers/Zhao_OnlineHMR_Video-based_Online_World-Grounded_Human_Mesh_Recovery_CVPR_2026_paper.pdf
- Avatars réalistes : https://lingtengqiu.github.io/LHM, https://mks0601.github.io/PERSONA, https://arxiv.org/html/2603.17975
- Rendu : https://pkhw2023-ship-it.github.io/medium-posts/posts/03-renderer-comparison-five-libraries/post.html
- Retirer l'apparence : https://arxiv.org/html/2609.24537v1, https://petsymposium.org/popets/2026/popets-2026-0053.pdf,
  https://arxiv.org/pdf/2606.14972, https://arxiv.org/pdf/2605.12431, https://arxiv.org/html/2503.08478v1,
  https://arxiv.org/html/2601.11635, https://openaccess.thecvf.com/content/WACV2025/papers/Kung_Face_Anonymization_Made_Simple_WACV_2025_paper.pdf,
  https://www.sciencedirect.com/science/article/pii/S107158192500045X, https://arxiv.org/pdf/2410.17098
- Voix et RGPD : https://arxiv.org/pdf/2404.02677, https://www.ucl.ac.uk/data-protection/data-protection/guidance/how/de-identifying-personal-data
