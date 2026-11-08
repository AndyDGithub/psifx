# Consignes d'annotation des poses (17 points COCO)

Le but est une référence fiable pour mesurer les modèles de psifx. Une image bien corrigée vaut mieux
que deux images corrigées vite. Les images ont été pré-annotées par ViTPose++ huge, un modèle que psifx
n'utilise pas : il faut donc **vérifier chaque point**, et ne pas valider un point parce qu'il semble
« presque juste ».

## Qui annoter

- **Toute personne dont le haut du corps fait au moins 1/10 de la hauteur de l'image** : les deux
  invités, les personnes de l'encart et du plateau.
- **Ne pas annoter** les personnes minuscules, floues ou presque entièrement cachées : supprime leur
  squelette, s'il y en a un. Les plans de foule n'ont souvent que 2 ou 3 personnes à garder.
- **Une personne sans squelette pré-annoté** : ajoute-la.

## Les trois états d'un point

| État | Quand | Dans CVAT |
|---|---|---|
| **Visible** | On voit l'articulation | point visible |
| **Caché, mais estimable** | Derrière le micro, la table, un bras, ou l'autre côté du corps, et sa position est évidente à quelques pixels près | point marqué occluded |
| **Absent** | Hors du cadre, ou impossible à placer sans inventer | point supprimé (outside) |

- **Hanches, genoux et chevilles sous la table : en général « absent ».** Le modèle les place souvent au bord
  de l'image ou sur la table. Ce sont des erreurs à supprimer, pas à corriger.
- **De profil**, l'oreille et l'œil du côté caché sont « absents », sauf si leur position est évidente.

## Où placer les points

- **Épaules, coudes, poignets, hanches, genoux, chevilles** : au centre de l'articulation, pas sur le bord du
  vêtement.
- **Yeux** : au centre de l'œil ; **oreilles** : au centre du conduit (ou du tragus) ; **nez** : la
  pointe.
- **Gauche et droite** sont ceux **de la personne**, pas de l'image : pour quelqu'un de face, son
  épaule gauche est à droite de l'image.

## Rythme

Environ 20 à 40 s par personne. Fais des pauses toutes les 30 à 40 images : la précision baisse avec
la fatigue. Si une image est ambiguë (flou de mouvement, fondu enchaîné), annote ce qui est sûr et
note le nom du fichier : on décidera ensemble de la garder ou non.

## Dans CVAT

- Ouvre http://localhost:8080 (identifiants dans `ext/cvat_login.txt`), tâche « ExtraitTalkShow poses »,
  puis son job. Les 200 images y sont avec les squelettes pré-annotés.
- Clic droit sur un point : le marquer **occluded** (caché mais estimable) ou **outside** (absent).
  Un point absent de la pré-annotation est déjà « outside » : pour l'ajouter, rends-le visible et
  place-le.
- Enregistre souvent (Ctrl+S). Pas besoin d'exporter à la fin : dis-moi que c'est fini, je récupère les
  annotations par l'API.
