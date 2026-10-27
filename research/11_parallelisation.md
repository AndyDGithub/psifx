# Paralléliser les étapes longues

Branches `optim/mediapipe-parallel-people`, `optim/parallel-pipeline-stages` et `optim/sam3-adaptive-chunks`
(toutes depuis `dev`). Date : 2026-10-28. Machine : 20 cœurs, GPU de 8 Go.

## 1. Où passe le temps

Sur V3 (talk-show, 5 min 32 s), le suivi SAM3 prend 6 010 s sur environ 8 000 : 0,65 s par image en
1080p, limité par le calcul sur le GPU. Viennent ensuite MediaPipe (927 s, CPU seulement) et RTMO
(833 s, GPU). Ces deux derniers ne dépendent que des masques, mais tournaient l'un après l'autre.

## 2. Ce qui ne se parallélise pas : les chunks de SAM3

Le journal d'OOM (`CUDA OOM while processing frames … retrying as chunks of …`) laisse penser que les
chunks pourraient tourner en même temps. Ce n'est pas possible ici, pour deux raisons :
- **Mémoire :** un chunk de 75 à 150 images remplit déjà les 8 Go, et chaque processus de plus ajoute
  son propre contexte CUDA (0,5 à 1,2 Go). Deux chunks à la fois provoqueraient plus d'OOM, pas moins.
- **Ordre :** les identités d'un chunk viennent des masques de la dernière image du chunk précédent et
  de la galerie d'apparence. Le raccord est séquentiel par nature.

**Essai : garder, dans un plan, la taille de chunk qui a tenu** (`optim/sam3-adaptive-chunks`).
Mesuré sur les images 4100 à 5999 de la vidéo du 2026-10-22 (1080p, 60 i/s), avec les réglages
d'origine :

| Code | Tentatives en OOM | Temps |
|---|---|---|
| `dev` | 15 | 1 236 s |
| Taille gardée dans le plan | 7 | 1 228 s |

Les masques sont les mêmes (IoU moyen de 1,000 sur 6 pistes, de 0,999 et 0,995 sur les 2 autres, à
cause d'un découpage différent sur 55 images). **Le gain est de 0,6 % :** un OOM survient dès le début
du chunk et coûte environ 1 s. Ce changement n'est pas fusionné.

## 3. Ce qui se parallélise

### 3.1 MediaPipe : une personne par processus

Chaque personne a sa propre vidéo masquée et sa propre archive : rien n'est partagé. `multi-inference`
lance maintenant `--num_workers` personnes à la fois (par défaut un processus pour 4 cœurs, au plus un
par personne). Les processus sont créés en mode `spawn`, pas `fork`, pour qu'aucun enfant n'hérite des
threads ou de l'état CUDA du parent. Après une erreur, les personnes pas encore lancées sont annulées.

| V2 (2 personnes, 3 000 images) | Temps |
|---|---|
| 1 processus | 318 s |
| 2 processus | 212 s (× 1,5) |

Le gain n'atteint pas × 2 : la préparation de la vidéo masquée et MediaPipe utilisent déjà plusieurs
cœurs. Les archives sont identiques octet par octet (2 × 3 001 fichiers), y compris par rapport au
passage `dev` d'origine.

### 3.2 La chaîne : les étapes indépendantes en même temps

`run_pipeline.py` connaît maintenant les étapes dont chaque étape lit les sorties, et sait si elle
utilise le GPU. Une étape démarre dès que ses entrées sont prêtes. Une seule règle de partage : **une
seule étape GPU à la fois**, pour la même raison de mémoire qu'au §2. MediaPipe tourne donc pendant RTMO,
et l'extraction audio pendant SAM3.
- **Pas de threads :** l'ordonnanceur lance des sous-processus et les interroge (`Popen.poll`), dans un
  seul thread. Chaque sous-processus écrit dans son propre journal, jamais dans un tube, donc il ne peut
  pas se bloquer sur un tube plein.
- **Graphe vérifié :** chaque dépendance doit être une étape déjà déclarée, donc pas de cycle possible.
- **Erreur :** aucune nouvelle étape ne démarre. Celles qui tournent, indépendantes de l'étape en
  échec, se terminent, et ne laissent donc pas de sortie partielle. Ctrl-C les arrête.
- `--sequential` rétablit l'ancien ordre.

V2, à partir des masques SAM3 (le suivi et le filtrage sont sautés), avec MediaPipe en 2 processus :

| Ordonnancement | Temps | Facteur temps réel |
|---|---|---|
| Une étape après l'autre | 529 s | 4,3× |
| Étapes indépendantes en même temps | 395 s | 3,2× |

Toutes les sorties sont identiques : poses, RTTM, VTT, association, VTT enrichi. RTMO passe de 229 à
315 s quand MediaPipe occupe les cœurs : il prépare ses images sur le CPU.

Le facteur temps réel de `evaluate.py` additionnait les durées des étapes, ce qui le surestimait dès
que des étapes se chevauchent (5,65 au lieu de 3,2). Il se calcule maintenant sur l'union des
intervalles enregistrés dans `stage_spans.json`, sans compter les pauses entre deux reprises.

**Sur V3 (estimation à partir des temps mesurés, non mesurée de bout en bout) :** MediaPipe en
3 processus, en même temps que RTMO (lui-même ralenti d'environ 35 %), ramènerait les 1 760 s de ces deux
étapes à environ 1 100 à 1 200 s, soit un gain de 500 à 650 s sur environ 8 000. Le suivi SAM3 reste
l'essentiel du temps.

## 4. Suite possible

Pour accélérer vraiment SAM3, il faut réduire le calcul, pas le répartir : une résolution d'entrée plus
basse, ou une image sur deux avec interpolation des masques. Avec un deuxième GPU, deux vidéos pourraient
tourner en même temps, une par GPU.

## Sources

- Gestion des processus parallèles, `shutdown(cancel_futures=True)` :
  https://docs.python.org/3.11/library/concurrent.futures.html
- `spawn` plutôt que `fork` avec CUDA : https://discuss.pytorch.org/t/a-call-to-torch-cuda-is-available-makes-an-unrelated-multi-processing-computation-crash/4075/2
- Taille d'un contexte CUDA par processus : https://discuss.pytorch.org/t/cuda-context-size-per-process/79976
- Blocage de `Popen.wait()` sur un tube plein : https://docs.python.org/3.0/library/subprocess.html
- Fils d'exécution de MediaPipe : https://sftp.thewrap.com/post/boost-mediapipe-performance-control-thread
