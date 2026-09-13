# Résultats du benchmark — MediaPipe, RTMPose, RTMO, DETRPose, ViTPose, Sapiens2 (0.4B/0.8B/1B)

Date : 2026-09-28 (mise à jour — ajout de RTMO et DETRPose, cf. `04_modeles_optimisation.md` §1.1/§1.3)
Suite à `03_problematique_objectifs_protocole.md`. Résultats réels obtenus en exécutant `research/benchmark/` dans le devcontainer (RTX 4060, 8 Go). Code, CSV bruts et `pareto.png` dans `research/benchmark/results/`. Détails d'installation/débogage (réseau, CUDA, mémoire, OOM) dans `research/benchmark/README.md`.

**Ce que cette passe ajoute** : RTMO et DETRPose, les deux pistes "un-étage" identifiées en `04_modeles_optimisation.md` comme réponses possibles à deux limites déjà documentées ici (ViTPose dépend d'un détecteur externe ; Sapiens2 sature la mémoire GPU sur les scènes peuplées). 7 des 8 backends ont été réellement re-mesurés dans cette passe ; **Sapiens2-1B n'a pas pu être relancé** (voir §4) et ses valeurs de précision/vitesse sont reprises telles quelles de la passe précédente (colonne "1B" marquée *(repris)* dans le tableau — sa cellule de jerk est laissée vide plutôt que mélangée avec l'échelle de cette passe, cf. §4).

## 1. Tableau final

| Backend | OKS moyen (COCO, n=50) | PCK@0.2 | Détection COCO | FPS (GPU, vidéo) | Latence moyenne | Jerk normalisé (vidéo) | Sur le front de Pareto |
|---|---|---|---|---|---|---|---|
| MediaPipe (actuel) | 0.235 | 0.376 | 72% | 23.4 | 42.8 ms | 45.6 (le plus stable) | Oui |
| RTMPose (`"balanced"`) | 0.859 | 0.930 | 100% | 28.8 | 34.8 ms | 603.2 (le moins stable) | Oui |
| **RTMO (`"balanced"`, un-étage)** | 0.830 | 0.887 | 100% | **54.3 (le plus rapide)** | 18.4 ms | 147.2 | Oui |
| **DETRPose-M (un-étage)** | 0.832 | 0.910 | 96% | 13.3 | 75.5 ms | 287.7 | Oui |
| ViTPose | **0.890** | **0.968** | 100% | 21.1 | 47.4 ms | 398.5 | Oui |
| Sapiens2-0.4B (bf16) | 0.581 | 0.732 | 96% | 1.3 | 761.4 ms | 64.9 | Oui |
| Sapiens2-0.8B (bf16) | 0.574 | 0.726 | 94% | 1.0 | 1033.4 ms | 115.5 | Non — dominé par 0.4B |
| Sapiens2-1B (bf16) *(repris, non re-mesuré)* | 0.548 | 0.694 | 90% | 0.86 | 1164.7 ms | — | — |

Front de Pareto (précision × vitesse × stabilité, sur les 7 backends re-mesurés) : **mediapipe, rtmpose, rtmo, detrpose, vitpose, sapiens2-0.4b** — seul sapiens2-0.8b est dominé. Classement pondéré (poids égaux, normalisation min-max, voir `results/ranking.csv`) : **1. rtmo, 2. rtmpose, 3. vitpose, 4. mediapipe, 5. detrpose, 6. sapiens2-0.4b, 7. sapiens2-0.8b**. Voir `results/pareto.png`.

⚠️ Les valeurs absolues de vitesse (FPS/latence) et de jerk de cette passe **ne sont pas directement comparables** à celles de la passe précédente (documentées dans une version antérieure de ce fichier) — voir §4 pour les deux raisons distinctes (contention GPU concurrente, sensibilité du jerk à fps³). Les comparaisons **relatives entre backends de cette même passe** restent valides (même machine, même charge, mesurées dans la même exécution).

## 2. Lecture des résultats

### 2.1 Précision (répond à Q1)

MediaPipe reste très largement dominé (OKS 0.235, détection 72%) — confirme, avec nos propres chiffres, le consensus de la littérature indépendante (`02_*` §6.2). ViTPose reste le plus précis (OKS 0.890), suivi de RTMPose (0.859), puis DETRPose (0.832) et RTMO (0.830) qui sont quasiment à égalité entre eux. Les deux backends un-étage perdent donc 3 à 6 points d'OKS face aux approches top-down avec détecteur dédié (RTMPose/ViTPose) — un compromis précision/simplicité réel, pas gratuit, mais nettement plus favorable que ce que suggérait Sapiens2 (qui perd 25-30 points).

### 2.2 RTMO et DETRPose délivrent-ils ce que `04_modeles_optimisation.md` en attendait ?

**RTMO : oui, clairement.** `04_*` §1.1 pariait sur RTMO pour éliminer la dépendance de ViTPose à un détecteur externe tout en gardant une latence quasi insensible au nombre de personnes. Les chiffres le confirment : détection 100% (comme RTMPose/ViTPose), OKS à seulement 3 points de RTMPose, et surtout **le backend le plus rapide de toute la comparaison** (54.3 FPS, devant même RTMPose malgré son détecteur séparé). RTMO est le candidat qui change le plus le classement par rapport à `03_*` (passe précédente) : il est premier au classement pondéré.

**DETRPose : accuracy au rendez-vous, mais pas le gain de vitesse espéré ici.** `04_*` §1.3 citait des chiffres publiés très favorables (S : 4.99 ms, M : 7.01 ms, L : 9.50 ms sur V100/fp16/TensorRT). Mesuré ici en PyTorch eager (pas de fp16/TensorRT, GPU d'entrée de gamme 8 Go), DETRPose-M tourne à 75.5 ms/image (13.3 FPS) — un ordre de grandeur plus lent que les chiffres publiés, et plus lent que RTMPose *malgré* l'absence de détecteur séparé. Ce n'est pas un défaut de DETRPose en soi mais un rappel méthodologique déjà fait ailleurs dans ce TM (`02_*`, `03_*` §2.2) : **les chiffres publiés supposent une optimisation d'inférence (TensorRT/fp16) que cette passe ne fait pas** — DETRPose resterait un candidat sérieux à réévaluer après application de `04_*` §2.1/§2.2 (`torch.compile`/quantification), pas à écarter sur cette seule mesure brute.

**Sur le problème des scènes peuplées** : un test manuel avant le benchmark complet (image COCO à 26 personnes, `image_id` 7511 — celle qui fait OOM Sapiens2-0.8B/1B, §2.3 de la passe précédente) a d'abord semblé montrer que DETRPose ne détectait qu'une seule personne sur les 26, même à un seuil de confiance très bas (0.05) — une limite de capacité de requêtes différente de l'OOM mémoire de Sapiens2, mais un problème du même ordre pour les scènes denses. **Nuance importante en creusant `accuracy_raw.csv`** : ce benchmark évalue une seule personne annotée par image (protocole "un humain qualifiant par image", `research/benchmark/prepare_coco_subset.py`) — sur cette image précise, DETRPose trouve et pose correctement *la* personne annotée (OKS 0.989, quasi parfait), donc le sous-comptage des 25 autres personnes n'affecte pas le score agrégé ici. Le problème de capacité sur les scènes très peuplées reste réel (confirmé manuellement) mais n'est pas mesuré par ce protocole à une-personne-par-image — à garder en tête si psifx doit un jour traiter des scènes à many-personnes, pas juste du face-à-face.

### 2.3 Vitesse, mémoire (répond à Q3)

RTMO domine largement en vitesse (54.3 FPS), suivi de RTMPose (28.8) et MediaPipe (23.4), puis ViTPose (21.1) et DETRPose (13.3). Les trois tailles de Sapiens2 restent largement impraticables pour de la vidéo (0.4B : 1.3 FPS ; 0.8B : 1.0 FPS ; 1B : 0.86 FPS *(repris)*).

⚠️ **Toutes les vitesses mesurées dans cette passe sont plus basses que dans la passe précédente** (ex. MediaPipe 44.3→23.4 FPS, RTMPose 40.9→28.8 FPS, ViTPose 41.5→21.1 FPS) — pas une régression du code : le GPU (laptop RTX 4060, 8 Go) était partagé avec d'autres processus hôte pendant cette mesure (`nvidia-smi` a montré ~2 Go déjà utilisés par d'autres applications au moment du run). Le ralentissement est general à tous les backends de cette passe (donc les comparaisons *relatives* entre eux restent valides), mais les FPS absolus ne doivent pas être comparés directement à ceux de la version précédente de ce document, ni utilisés comme référence de production sans re-mesure sur un GPU dédié.

Le delta de mémoire GPU mesuré (`gpu_mem_delta_mb`) reste peu fiable pour Sapiens2 (même limite que documentée précédemment, §4).

### 2.4 Stabilité temporelle (répond à Q2)

RTMPose reste le moins stable (jerk le plus élevé), MediaPipe le plus stable (lissage intégré, cf. §4), et RTMO/DETRPose se situent dans la fourchette basse-moyenne. ⚠️ **Les valeurs absolues de jerk de cette passe sont ~1000-2000× plus élevées que celles publiées dans la version précédente de ce document** (ex. MediaPipe 0.0205 → 45.6). Vérifié à la main (`metrics.normalized_jerk_series`, formule et code inchangés) : avec un déplacement réel de ~1 px/frame et une vidéo à 25 fps, le calcul en trois différences successives divisées par `dt` produit un ordre de grandeur en dizaines une fois normalisé en px/s³ (25³ ≈ 15 600) — cohérent avec ce qui est mesuré ici. L'origine exacte de l'écart avec les anciennes valeurs n'a pas pu être établie (le code de `metrics.py` n'est pas versionné dans git, donc pas d'historique à diff — voir §4) ; à traiter comme un signal **relatif entre backends au sein d'une même passe**, pas comme une valeur absolue stable d'une passe à l'autre.

## 3. Réponse à la problématique (Q4) et recommandation

Arbitrage entre 7 candidats mesurés (+ Sapiens2-1B repris) :

- **Précision pure** (priorité identifiée en `01_*`/`02_*`) : **ViTPose** reste le meilleur choix (OKS 0.890), devant RTMPose (0.859).
- **Meilleur compromis précision/vitesse** : **RTMO** — 3 points d'OKS sous RTMPose, mais 2× plus rapide que RTMPose et ViTPose, sans détecteur externe à maintenir (élimine la limite documentée en `03_*` §4 de la passe précédente : "ViTPose dépend des boîtes du détecteur RTMPose"). Premier du classement pondéré (§1). **Nouveau candidat le plus intéressant issu de cette passe.**
- **DETRPose** : précision comparable à RTMO, mais nettement plus lent ici faute d'optimisation d'inférence (§2.2) — resterait à réévaluer après `torch.compile`/quantification (`04_*` §2.1/§2.2) avant de le retenir ou l'écarter définitivement.
- **Sapiens2, toutes tailles confondues** : toujours non praticable pour la vidéo (0.86 à 1.3 FPS) — conclusion inchangée par rapport à la passe précédente.
- **MediaPipe** : à garder si la stabilité temporelle ou la légèreté priment sur la précision (déjà en place dans psifx).

**Recommandation mise à jour** : le choix se resserre maintenant à deux candidats sérieux plutôt qu'un seul —
1. **ViTPose** si la précision est le critère dominant (recommandation inchangée de la passe précédente) ;
2. **RTMO** si un compromis précision/vitesse/simplicité (pas de détecteur séparé à intégrer/maintenir) est préférable — en particulier si psifx doit un jour traiter de la vidéo longue durée où la vitesse de RTMPose/ViTPose (~20-30 FPS mesurés ici, probablement plus bas encore sur un GPU moins dédié) devient un goulot d'étranglement.

Les deux sont Apache 2.0 (aucune friction de licence, `04_*` §1.1/§1.3), contrairement à Sapiens2. **À trancher avec le responsable/Matthew Vowels** plutôt qu'unilatéralement ici : le compromis précision (ViTPose) vs vitesse (RTMO) dépend d'un choix produit (traitement en léger différé où la précision prime, vs pipeline plus proche du temps réel) que ce benchmark ne peut pas arbitrer seul.

## 4. Limites de ce passage (transparence méthodologique)

- Échantillon COCO de 50 images — comparaison *relative* valide (même échantillon pour tous), valeurs absolues non comparables à val2017 complet.
- **Sapiens2-1B non re-mesuré dans cette passe** : sature la mémoire du GPU (8 Go) dès le chargement du modèle à cause de la contention avec d'autres processus hôte actifs au moment de cette exécution (~2 Go déjà utilisés hors de ce benchmark, confirmé via `nvidia-smi` côté hôte) — un budget qui suffisait lors de la passe précédente ne suffit plus ici. Ses valeurs de précision/vitesse sont reprises telles quelles de la passe précédente (le code/setup Sapiens2 n'a pas changé) ; sa cellule de jerk est laissée vide plutôt que comparée sur une échelle différente (voir point suivant).
- **Vitesse (FPS) et jerk de cette passe non directement comparables à la version précédente de ce document** : la vitesse à cause d'une contention GPU par d'autres processus hôte pendant la mesure (§2.3) ; le jerk parce que sa magnitude absolue s'est révélée ~1000-2000× différente sans qu'une cause de code ait pu être identifiée avec certitude (`metrics.py` n'étant pas versionné dans git avant cette session, pas d'historique pour diff — voir §2.4). Les deux métriques restent valides en comparaison *relative* entre backends **au sein de cette même passe**.
- DETRPose testé en taille M uniquement (pas de sweep S/M/L comme pour Sapiens2) — décision de portée délibérée, cf. `04_modeles_optimisation.md` §1.3 : le sweep de tailles avait été demandé explicitement pour Sapiens2, pas pour DETRPose.
- DETRPose testé en PyTorch eager, sans fp16/TensorRT — les chiffres de latence publiés (`04_*` §1.3) supposent cette optimisation ; à ne pas comparer directement (§2.2).
- Le sous-comptage de personnes par DETRPose sur les scènes très peuplées (confirmé manuellement sur l'image COCO à 26 personnes) n'est pas capturé par ce protocole (une seule personne annotée par image, `prepare_coco_subset.py`) — limite du protocole, pas une garantie que DETRPose gère bien les scènes peuplées.
- Jerk biaisé par le lissage intégré de MediaPipe (RTMPose/RTMO/DETRPose/ViTPose/Sapiens2 tournent frame par frame sans lissage) — non corrigé dans ce passage.
- ViTPose dépend des boîtes du détecteur RTMPose sur la vidéo (pas de détecteur propre) — son jerk vidéo hérite en partie de la stabilité de détection de RTMPose.
- **Mesure de mémoire GPU peu fiable pour Sapiens2/DETRPose** (sous-processus qui libère sa mémoire avant la mesure "après") — limite déjà documentée pour Sapiens2, s'applique de la même façon à DETRPose (même pattern subprocess).
- Sapiens2 testé avec son détecteur DETR par défaut uniquement, en bf16, dans un sous-processus Python 3.12 séparé — bugs d'intégration documentés dans `research/benchmark/README.md`.
- Méthodes anti-occlusion dédiées (DEKR) hors scope.
- Un seul type de vidéo testé (`example/data/Video.mp4`, 150 premières frames, ~2 personnes/frame maximum).
- Licence Sapiens2 toujours non validée avec le responsable (`01_*` §4.1) ; RTMO et DETRPose sont Apache 2.0 (déjà confirmé, `04_*` §1.1/§1.3), aucune validation requise sur ce point.

## Reproduire

```bash
cd research/benchmark
pip install rtmlib "onnxruntime-gpu==1.20.2"   # + setup Sapiens2 et DETRPose séparés (envs dédiés, checkpoints), voir README.md
python prepare_coco_subset.py --n 50
python run_accuracy.py --backends mediapipe,rtmpose,vitpose,sapiens2-0.4b,sapiens2-0.8b,sapiens2-1b,rtmo,detrpose
python run_video_benchmark.py --backends mediapipe,rtmpose,vitpose,sapiens2-0.4b,sapiens2-0.8b,sapiens2-1b,rtmo,detrpose --max-frames 150
python aggregate_results.py
```
