# Plan : reproduire une vidéo avec des avatars 3D, et retirer l'apparence

Date : 2026-11-09, révisé le 2026-11-18 après la réunion du 10 novembre (`14_reunion_2026-11-10.md`). Statut : objectif
discuté avec le professeur responsable. Point de départ : `dev` au tag
`tm-v1-optimisations`. Échéance de travail : fin décembre 2026. Le rapport s'écrit au fil du travail, idées abandonnées comprises.

## 1. Les objectifs

**Objectif de base : prouver que les données produites sont exploitables.** Une vidéo reproduite à la caméra d'origine,
avec un avatar par personne, le décor repris de la vidéo ou minimal, le son et les sous-titres. Le chercheur voit
directement ce que contiennent les données et ce qui leur manque. Pas de point de vue libre.

**Objectif de thèse : retirer l'apparence en conservant le comportement.** L'avatar remplace le visage et le
corps, et on mesure deux choses :

- **Utilité, en premier :** le comportement reste-t-il observable sur la vidéo avec avatars ?
- **Ce qui reste de l'identité, ensuite :** peut-on encore reconnaître la personne, en particulier par son mouvement ?

**Cas d'utilisation (proposé par le professeur le 2026-11-10) :** former de futurs psychologues. Un psychologue filme des
participants (par exemple avec des troubles de l'attention) ; la version avec avatars sert à apprendre à observer le
comportement, à analyser des situations et aux mises en situation. Les personnes en formation ne cherchent pas à reconnaître
les participants : une anonymisation complète n'est pas nécessaire. Le partage avec des collègues reste encadré par le RGPD
et la LPD suisse (consentement).

Le même travail sert donc les deux objectifs. Je n'ai trouvé aucune étude qui mesure ces deux axes pour un avatar 3D
sur des données d'interaction sociale : c'est la place que le travail peut prendre. Les méthodes publiées existent, mais
visent autre chose (section 3).

**Vocabulaire (décidé le 2026-11-10) :** ni « anonymisation » (impossible à garantir en gardant le comportement : la démarche
et les mouvements identifient), ni « pseudonymisation » (terme juridique : identifiants remplacés, clé de correspondance
gardée). On dit « retirer l'apparence » ou « remplacement par un avatar 3D ». Au sens du RGPD, des données restent
personnelles tant qu'une réidentification est raisonnablement probable (considérant 26) ; la LPD suisse raisonne de même.

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

### 2.4 Mesuré depuis le 9 novembre

**Paramètres (jalon 0, branche `feature/mesh-params`).** Talk-show complet : 9 352 images en 2 h 43. Les paramètres MHR
reconstruisent le maillage de leur propre calcul à 0,2 mm (médiane). Archives : 42 Mo au lieu de 1,3 Go de sommets. OKS 0,849.

**Deux calculs ne donnent pas les mêmes mains.** Sur 0,4 % des personnes-images, les mains diffèrent de plus de 50 mm (jusqu'à
environ 20 cm). Cause : le réseau principal tourne en bfloat16. Sur les images concernées, trois calculs en bfloat16 se
contredisent dans 33 % des cas, en float32 jamais (option `--fp32`). Le float32 coûte deux fois plus (1,50 s par image) et
donne le même OKS et le même tremblement : il n'achète que la reproductibilité. L'hypothèse « la main qui saute est celle du
décodeur du corps » est rejetée par les mesures.

**Images clés réelles (branche `optim/mesh-keyframes`).** Référence : calcul dense, 2 h 43, OKS 0,849.

| Réglage | Images calculées | Durée | OKS |
|---|---|---|---|
| Récursif, seuil 80 mm, budget 1 sur 8 | 18 % | 35 min | 0,847 |
| Récursif, seuil 80 mm | 24 % | 31 min | 0,847 |
| Récursif, seuil 40 mm | 43 % | 55 min | 0,848 |
| Mode `body` seul, toutes les images | 100 % | 34 min | 0,854 |
| **Mode `body` + images clés, seuil 40 mm** | – | **18 min** | **0,852** |

- Le seuil 40 mm bat l'espacement fixe 1 sur 4 (51 % des images) sur le poignet et le maillage, surtout pour la personne 2
  (maillage p95 : 66 contre 442 mm).
- Les gestes rapides restent mal interpolés (personne 2 : 320 à 370 mm au poignet).
- Le mode `body` est plus lisse et a un meilleur OKS, mais ses doigts ne sont pas mesurés.
- **Caméra qui bouge :** seule la vidéo AVA bouge (9 % des images). Seuil 80 mm : 20 % des images en 11 min, à peu près
  comme 1 sur 5 fixe. Les images clés vont bien là où la caméra bouge (16 % contre 9 %), mais l'erreur y reste deux fois plus
  haute. Pas encore de vidéo à caméra très mobile.

**MEVID, petites découpes (branche `experiment/mevid-small-crops`).** SAM 3D Body en mode `body`, découpe entière comme boîte,
sans masque, comparé à ViTPose sur la même découpe.

- Échantillon : 102 pistes × 8 images. La pose 2D tient (accord 0,91 vers 85 px, 0,95 au-dessus de 288 px). Le mode `full`
  donne les mêmes points que `body` à ces tailles, pour un coût triple.
- **Partie test entière : 1 754 pistes, fenêtre de 32 images au milieu, 55 291 découpes, 2 h 43** (0,18 s par découpe,
  ViTPose compris). Filtre : moyenne des scores ViTPose d'au moins 0,5, épaules et hanches au-dessus de 0,3, au moins 8
  découpes gardées par piste.

| Hauteur | Pistes | Découpes gardées | Échecs, sans → avec filtre | Accord avec ViTPose | Saccades (mm) | Morphologie dans / entre pistes |
|---|---|---|---|---|---|---|
| < 64 px | 2 | 100 % | 1,6 % → 1,6 % | 0,78 | 80 | 0,32 |
| 64–96 | 552 | 85 % | 8,9 % → 3,5 % | 0,91 | 39 | 0,15 |
| 96–128 | 202 | 84 % | 4,9 % → 1,0 % | 0,92 | 38 | 0,12 |
| 128–192 | 311 | 84 % | 8,4 % → 0,3 % | 0,91 | 22 | 0,07 |
| 192–288 | 376 | 92 % | 4,7 % → 0,8 % | 0,96 | 8 | 0,035 |
| > 288 | 311 | 92 % | 4,4 % → 1,0 % | 0,95 | 10 | 0,04 |

- **Restent utilisables : 1 560 pistes, les 54 personnes, 276 requêtes sur 316 et 1 284 pistes de galerie.**
- Le filtre écarte l'herbe, les images blanches, les voitures, les jambes seules et les têtes vues de dos ; il écarte à tort
  quelques personnes assises coupées à la taille, et il laisse passer les découpes à plusieurs personnes.
- Le mouvement 3D se dégrade sous 128 px (saccades quatre fois plus fortes), mais la morphologie reste propre à chaque
  personne : elle varie moins dans une piste qu'entre pistes, même petite. L'attaquant de morphologie a donc de quoi
  travailler.

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
2. **Les montrer :** la vidéo avec avatars, preuve de l'objectif de base **et produit du cas d'utilisation** (formation de
   psychologues, analyse de situations).
3. **Mesurer :** d'abord ce qui reste du comportement (utilité), puis ce qui reste de l'identité.
4. **Élargir :** autres vidéos, commandes psifx.

Si le temps manque, l'étape 4 saute d'abord. Les seuils sont des propositions, à ajuster sur les premières mesures. Une
branche et une note par étape, et le rapport avance en même temps : **ce qu'on a pensé, fait ou abandonné y est tracé au
moment où ça arrive**.

**Révision du 2026-11-18**, après la réunion du 10 novembre (`14_reunion_2026-11-10.md`). Les jalons 0 à 7 ci-dessous
remplacent ceux du 9 novembre. Ordre de travail : **E1, E2, E7 (avant la réunion du 24 novembre, avec E6.4 morphologie qui
ne coûte presque rien), puis E3, E4, E5, E6, E8.** Le téléchargement et l'extraction de MEVID pour E6.1 peuvent tourner en
fond plus tôt.

### E0. Garder les paramètres — fait (2026-11-09)

Voir 2.4. Les paramètres MHR reconstruisent le maillage de leur propre calcul (médiane 0,2 mm).

### E1. Noter — fait (2026-11-18)

Note de réunion (`14_reunion_2026-11-10.md`), cette révision, mémoire du fil rouge.

### E2. Vidéo avec avatars, version 1

**But :** la première vidéo du cas d'utilisation, et les exemples visuels que le rapport demande. Elle passe avant la
stabilité et les caméras : sans point de vue libre, la projection image par image de SAM 3D Body suffit à placer l'avatar.

**Branche :** `feature/avatar-render`. **Script :** `research/scene3d/render_avatars.py`.

- **Entrées :** archives de paramètres MHR (`sam3d_params`, dense, mode `full`, pour garder les mains : l'agitation des mains
  compte pour un psychologue), groupes de caméra et fonds (`decor_cameras/`), `subtitles.ass`, son d'origine.
- **Maillage :** reconstruit sur GPU par MHR à partir des paramètres, comme `mesh_eval.py` ; dessin par le peintre OpenCV de
  `render_video.py` (24 images/s mesurées), une couleur fixe par personne.
- **Niveaux** (section 4) : A0+ (avatar sur l'original, contrôle), A2 (fond de scène du groupe de caméra ; repli : fond
  gris si le plan n'a pas de fond), A3 (morphologie moyenne : paramètres d'identité et d'échelle remplacés par leur moyenne
  sur toutes les personnes), A4 (fond gris uniforme).
- **Sorties :** une vidéo par niveau avec son et sous-titres, une planche d'exemples (une image par groupe de caméra et par
  niveau) pour le rapport, temps de rendu par image.
- **Vérification :** à l'œil sur chaque groupe de caméra ; points 2D du maillage reprojeté identiques à ceux de l'archive
  (contrôle de la reconstruction).
- **Limites à tracer, pas à corriger ici :** personne qui change de couleur d'un plan à l'autre (E4), plans d'incrustation,
  trou du fond, tremblement (E3).

### E3. Stabilité

**But :** une vidéo regardable sans sauts : un psychologue ne doit pas prendre un tremblement du modèle pour un mouvement.

**Branche :** `optim/mesh-keyframes` (suite).

- **Morphologie et échelle par personne :** médiane sur toutes ses images, à la place de la valeur de chaque image.
- **Lissage à l'intérieur de chaque plan, jamais à travers une coupe :** filtre One Euro contre fenêtre glissante
  (Savitzky-Golay) sur les paramètres de pose, angles déroulés comme dans `run_keyframes.py`. Le Kalman de SAM-Body4D
  seulement si les deux échouent.
- **Choix de production à trancher sur ces mesures :** dense `full` (mains, 2 h 43 sur le talk-show) contre `body` + images
  clés (18 min, mains grossières). Les mains comptent pour le cas d'utilisation.
- **Mesures :** accélération médiane (divisée par 2 visé), OKS sur les 200 images annotées (baisse de moins de 0,01),
  erreur du poignet sur les 10 % de gestes les plus rapides (ne doit pas augmenter de plus de 20 %), et la vidéo.

### E4. Personnes et plans

**But :** une couleur et une morphologie par personne sur toute la vidéo, un fond propre.

- **Fusion des pistes d'une même personne :** distance entre morphologies médianes, et deux pistes visibles dans la même
  image ne peuvent pas être la même personne. Vérité : attribution des pistes annotée à la main (3 pistes, quelques minutes).
- **Plans d'incrustation** (logo plein écran, vidéos insérées) : marqués, sans fond ni avatar.
- **Trou du fond :** fusion des fonds des cadrages reliés, puis remplissage OpenCV ; l'avatar couvre le reste.
- **Focale par caméra (MoGe) : reportée.** Sans point de vue libre, elle ne change pas l'image. On n'y revient que si
  l'échelle de l'avatar saute visiblement à l'écran.

**Sortie :** plus de 95 % des images attribuées à la bonne personne.

### E5. Utilité : le comportement reste-t-il observable ?

**But :** chiffrer ce qu'un observateur perd en regardant l'avatar plutôt que l'original. C'est la mesure principale.

**Branche :** `eval/avatar-utility`. Principe inchangé : passer les outils d'analyse sur la vidéo rendue et comparer à
l'original, pour les niveaux A1 (visage flouté, la pratique courante) à A4.

| Comportement | Mesure |
|---|---|
| Corps | ViTPose sur le rendu, OKS contre les 200 images annotées |
| Gestes | Corrélation de la vitesse des poignets ; part des gestes rapides de l'original retrouvés sur le rendu |
| Tête | Corrélation de l'orientation et des hochements (MediaPipe sur l'original, tête du maillage sur le rendu) |
| Bouche | Corrélation de l'ouverture ; activité pendant les tours de parole (sous-titres de référence) |
| Expressions | Comparaison des coefficients. OpenFace sur un avatar est un essai, pas une garantie |
| Interaction | Orientation mutuelle et synchronie, dans les plans larges seulement |

**À décider avec les encadrants :** une petite étude de perception (3 à 5 personnes regardent des extraits originaux et avec
avatars et notent les comportements qu'elles voient). C'est la mesure la plus proche du cas d'utilisation, mais elle demande
des participants et un cadre.

### E6. Ce qui reste de l'identité (MEVID)

**But :** chiffrer ce que l'avatar laisse de l'identité, en particulier par le mouvement (le risque cité à la réunion), et la
fiabilité de la chaîne dans des conditions réelles. Ce n'est **pas** une preuve d'anonymisation.

**Branche :** `experiment/mevid-reid` (à partir de `experiment/mevid-small-crops`).

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
sombres et des images sans personne.

Hauteur des images, mesurée sur la première image des 1 754 pistes : médiane 159 px, p10 76 px, p90 367 px ; 32 % sous 96 px,
43 % sous 128 px. Les résultats seront donc donnés par taille, et SAM 3D Body est à contrôler d'abord sur les plus petites.

**MARS en complément, seulement si un besoin apparaît.** Il offrirait une référence plus connue, mais pas de meilleurs
visages : c'est aussi de la surveillance. Avec une seule tenue par personne, retirer les vêtements y ferait chuter le score sans
rien dire du corps. Ses pistes sont aussi plus courtes (25 à 50 images), ce qui gêne l'attaque de démarche. Sa page officielle
reste à confirmer. On n'y va que si l'écart de référence de MEVID (52,5 %) se révèle trop étroit pour conclure.

**E6.1 Référence d'apparence (CAL).** Code de l'article MEVID (dossier `CAL`) et poids publiés dans son tableau de résultats.
Extraire toute la partie test (1,19 million d'images, archive de 13 Go déjà téléchargée), **retrouver 52,5 % de Rank-1 et 27,1 %
de mAP avec leur protocole**. Puis refaire l'évaluation avec nos fenêtres de 32 images : c'est la référence N0 des mesures
suivantes, et l'écart entre les deux protocoles est à publier.

**E6.2 Rendu des fenêtres.** Paramètres déjà calculés (2.4). Rendu à la taille de la découpe, **sur fond gris** : garder le fond
de la découpe laisserait le lieu et la caméra identifier la piste. N2 morphologie propre, N3 morphologie moyenne. N1 (visage
flouté) demande un détecteur de visage : taux de détection donné par taille.

**E6.3 Contrôle de qualité, toujours à côté du score.** Filtre ViTPose (2.4), accord avec ViTPose, par taille de découpe. Un avatar
raté donne aussi le hasard (1,7 %).

**E6.4 Attaquants.**

| Attaquant | Ce qu'il a | Coût | Ce qu'il mesure |
|---|---|---|---|
| Morphologie | Plus proche voisin sur les paramètres d'identité MHR moyens de la piste, sans entraînement | Minutes : paramètres déjà là | Ce que N2 laisse et que N3 doit retirer |
| Apparence naïf | CAL entraîné sur de vraies vidéos, galerie d'origine, requêtes N1 à N3 | Après E6.1 | Ce que l'apparence seule laisse passer. Le plus faible |
| Apparence informé | CAL, galerie et requêtes au même niveau | Après E6.1 | Ce qui reste commun aux avatars d'une même personne |
| Mouvement, simple | Plus proche voisin sur des descripteurs de mouvement indépendants de la morphologie (angles des articulations, cadence, sur 32 images) | Une heure | Borne basse de la signature du mouvement |
| Mouvement, appris | Modèle de démarche sur squelettes (GaitGraph2 ou SkeletonGait d'OpenGait : poids et format d'entrée à vérifier), sinon entraîné sur la partie train de MEVID (30,5 Go, SAM 3D Body à passer dessus) | Jours | La signature du mouvement face à un attaquant réaliste |

**E6.5 Lecture.** Rank-1, mAP et courbe CMC ; même tenue contre autre tenue ; par taille de découpe ; intervalle de confiance
par rééchantillonnage des identités ; protocole MEVID (même personne et même caméra exclues). Un score de mouvement au-dessus
du hasard n'est pas un défaut à corriger : c'est le prix de l'utilité, et le résultat attendu est un compromis chiffré.

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

### E7. Rapport v7

**But :** la version à montrer à la réunion du 24 novembre, dans la structure demandée (`14_reunion_2026-11-10.md`, section 1).

**Branche :** `docs/report-v7`. PDF : `research/report/report-psifx-v7.pdf`, v6 conservé.

- **Structure :** contexte, objectifs et questions ; état de l'art et positionnement ; approche ; implémentation ;
  expériences et évaluation ; résultats et limites ; discussion et conclusion ; gestion de projet.
- **Vocabulaire :** « retirer l'apparence », plus jamais « anonymiser » ni « pseudonymiser » pour décrire le résultat.
- **État de l'art :** librairies (psifx et voisines), modèles, méthodes académiques (section 3.3) et solutions commerciales
  (brighter AI, et d'autres à rechercher), avec un tableau récapitulatif de ce qui est couvert ou non.
- **Schémas :** architecture de la chaîne (section 4), déroulement des images clés, architecture de SAM 3D Body et de MHR.
- **Résultats déjà mesurés (2.4) :** tableaux et graphiques.
- **Exemples visuels :** planche d'avatars (E2), planche MEVID par taille.
- **Gestion de projet :** fil rouge, branches, décisions et pistes abandonnées.

### E8. Élargir

- Tenue sur V1 (caméra fixe), V2 (film) et V4 (danse), et une vraie vidéo à caméra mobile.
- Commandes psifx si le format est stable : `psifx video mesh sam3d_body inference`, `psifx scene build`, `psifx scene render`.

## 6. Risques

| Risque | Conséquence | Réponse |
|---|---|---|
| Le mouvement seul identifie la personne | Pas de garantie d'anonymat | Attendu (réunion du 10 novembre) : on mesure et on dit ce qui reste ; le cas d'utilisation n'exige pas l'anonymat |
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

**Prises le 2026-11-10 (réunion) et le 2026-11-18 :**

7. Ni « anonymisation » ni « pseudonymisation » : « retirer l'apparence ».
8. Cas d'utilisation principal : formation de psychologues et analyse de situations avec la vidéo à avatars.
9. L'utilité passe avant la confidentialité ; la vidéo avec avatars (E2) passe avant la stabilité et les caméras.
10. La confidentialité devient « ce qui reste de l'identité », avec un attaquant de mouvement obligatoire.
11. Focale par caméra (MoGe) reportée : sans point de vue libre, elle ne change pas l'image.
12. Rapport v7 dans la structure du professeur pour la réunion du 2026-11-24.

**Restent ouvertes :**

1. Petite étude de perception pour l'utilité (E5) : à proposer aux encadrants.
2. Attaquant de mouvement appris : modèle publié ou entraînement sur la partie train de MEVID (E6.4).

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
- Réunion du 2026-11-10 et suites : voir les sources de `14_reunion_2026-11-10.md` ; MEVID, poids des modèles de référence :
  https://github.com/Kitware/MEVID ; démarche sur squelettes : https://github.com/tteepe/GaitGraph2/blob/main/README.md,
  https://arxiv.org/pdf/2311.13444.pdf, https://arxiv.org/pdf/2205.02692
