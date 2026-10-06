# Benchmark pose estimation — MediaPipe vs RTMPose vs RTMO vs DETRPose vs ViTPose vs Sapiens2

Implémente le protocole décrit dans `../03_problematique_objectifs_protocole.md`. À exécuter dans le devcontainer (`psifx-env`).

## Installation (dépendances additionnelles, pas dans `requirements.txt` de psifx)

```bash
pip install rtmlib "onnxruntime-gpu==1.20.2"
```

⚠️ `onnxruntime-gpu` en version la plus récente (1.30.0 au moment de l'écriture) exige CUDA 13 et échoue silencieusement sur `CUDAExecutionProvider` dans ce devcontainer (CUDA 12.1/12.8) — il retombe sur CPU sans erreur bloquante, ce qui fausserait la comparaison de vitesse. `1.20.2` est la version testée et confirmée fonctionner avec `CUDAExecutionProvider` dans ce conteneur.

⚠️ Même avec `1.20.2`, `onnxruntime` ne trouve pas `libcudnn.so.9`/`libcublasLt.so` par défaut : ces `.so` existent bien (installés comme dépendances pip de `torch`, sous `site-packages/nvidia/{cudnn,cublas}/lib`) mais ce dossier n'est pas sur le `LD_LIBRARY_PATH` par défaut du conteneur. Sans ce correctif, RTMPose retombe silencieusement sur CPU (pas d'erreur bloquante, juste un warning facile à manquer dans les logs). Exporter avant de lancer les scripts :

```bash
export LD_LIBRARY_PATH="$LD_LIBRARY_PATH:$(python -c 'import nvidia.cudnn as m; print(m.__path__[0])')/lib:$(python -c 'import nvidia.cublas as m; print(m.__path__[0])')/lib"
```

⚠️ `nvidia.cudnn.__file__`/`nvidia.cublas.__file__` (utilisés dans une version précédente de cette commande) sont devenus `None` après une mise à jour de `nvidia-cudnn-cu12`/`nvidia-cublas-cu12` en paquet namespace (pas de `__init__.py`) -- `os.path.dirname(None)` lève `TypeError`, et le script retombe silencieusement sur CPU sans le remarquer si l'erreur n'est pas vérifiée. `__path__[0]` fonctionne dans les deux cas (paquet normal ou namespace).

⚠️ Ajouter ces chemins **à la fin** de `LD_LIBRARY_PATH`, pas au début : les préfixer casse le chargement CUDA de `torch` lui-même (`libc10_cuda.so: undefined symbol ... version libcudart.so.12`), car ça entre en collision avec les bibliothèques CUDA que `torch` embarque et résout normalement en premier. À la fin, ça ne comble que le trou d'`onnxruntime` sans perturber `torch` — nécessaire ici puisque MediaPipe/RTMPose (ONNXRuntime) et ViTPose (`transformers`/`torch`) tournent dans le même process Python.

`transformers` est déjà dans psifx (utilisé par SAM3) — vérifier que la version installée expose `VitPoseForPoseEstimation` (`python -c "from transformers import VitPoseForPoseEstimation"`) ; sinon `pip install -U transformers`.

### RTMO — déjà couvert par `rtmlib`, aucune installation supplémentaire

`RTMOBackend` réutilise `rtmlib.Body` (déjà installé pour RTMPose ci-dessus) avec `pose="rtmo"`, qui bascule la classe dans sa branche "un-étage" (confirmé en lisant `rtmlib/tools/solution/body.py` sur le dépôt source) : seul le modèle `RTMO` est instancié, pas de détecteur `YOLOX` séparé. Le poids ONNX (`rtmo-m_16xb16-600e_body7-640x640-...zip`, preset `"balanced"`, ~79 Mo) se télécharge automatiquement au premier appel, dans `~/.cache/rtmlib/hub/checkpoints/` — mêmes contraintes réseau (MTU) et `LD_LIBRARY_PATH` que RTMPose (ci-dessus), rien de spécifique à RTMO.

### DETRPose — environnement Python séparé

Apache 2.0 (`SebastianJanampa/DETRPose`, confirmé `04_modeles_optimisation.md` §1.3). Python 3.11 en principe compatible avec `psifx-env`, mais son `requirements.txt` pin `numpy==1.23.5` avec torch/torchvision non pinnés — **installé une fois par erreur directement dans `psifx-env`** pendant la mise en place de ce benchmark (voir avertissement ci-dessous) et ça a bien cassé numpy/scipy/opencv/contourpy pour mediapipe/rtmlib. `DETRPoseBackend` (`backends.py`) lance donc `detrpose_infer.py` en sous-processus dans un env dédié, même pattern que Sapiens2 — mais ici la raison est l'isolation du pin `numpy`, pas une incompatibilité de version Python.

```bash
# 1. Environnement dédié (avant tout pip install -- voir avertissement ci-dessous)
conda create -y -n detrpose-env python=3.11.9 pip

# 2. Cloner le repo (public, Apache 2.0)
git clone https://github.com/SebastianJanampa/DETRPose /home/psifx/detrpose

# 3. Installer DANS detrpose-env, jamais dans psifx-env
conda activate detrpose-env
pip install -r /home/psifx/detrpose/requirements.txt

# 4. Checkpoint (taille M uniquement -- voir 03_resultats.md §4 pour la justification de portée)
curl -L -o /home/psifx/detrpose/detrpose_hgnetv2_m.pth \
  https://github.com/SebastianJanampa/DETRPose/releases/download/model_weights/detrpose_hgnetv2_m.pth
```

⚠️ **`conda activate <env>` peut échouer silencieusement dans un `docker exec` non interactif** si l'environnement n'a pas fini de se créer/s'enregistrer avant l'appel suivant (`EnvironmentNameNotFound`), et le shell continue alors avec l'env actif précédent (`psifx-env`) sans faire échouer la commande -- `pip install -r requirements.txt` s'exécute alors silencieusement dans le mauvais environnement. Rencontré une fois ici : `numpy` de `psifx-env` a été downgradé de 2.0.2 à 1.23.5 (et `scipy`/`opencv-python`/`contourpy` remplacés) avant d'être remarqué. **Toujours vérifier l'environnement actif avant un `pip install` sensible** :
```bash
python -c "import sys; assert 'detrpose-env' in sys.executable, sys.executable"
```
Fix appliqué : `pip uninstall` des paquets ajoutés par erreur + réinstallation des versions exactes de `psifx-env` (numpy==2.0.2, scipy==1.17.1, opencv-python==4.11.0.86, contourpy==1.3.3, retrouvées dans les logs `pip install` du désinstall/réinstall).

`build_backend()` accepte `"detrpose"` et instancie `DETRPoseBackend()` (taille M uniquement, chemins par défaut `/home/psifx/detrpose/...`, `python_bin=/home/psifx/conda/envs/detrpose-env/bin/python`).

⚠️ **Sous-comptage sur les scènes très peuplées** : sur l'image COCO à 26 personnes (`image_id` 7511, celle qui fait OOM Sapiens2), DETRPose ne détecte qu'1 personne au seuil par défaut (0.5), et seulement 8 même à un seuil très bas (0.05) -- limite de capacité de requêtes du décodeur, différente de l'OOM mémoire de Sapiens2 mais du même ordre pratique pour les scènes denses. N'affecte pas le score agrégé de ce benchmark (une seule personne annotée par image, cf. `03_resultats.md` §2.2) mais à garder en tête pour un usage réel multi-personnes dense.

### Sapiens2 — environnement Python séparé

`sapiens2/pyproject.toml` exige **Python ≥3.12**, incompatible avec `psifx-env` (3.11). `Sapiens2Backend` (`backends.py`) lance `sapiens2_infer.py` en sous-processus dans un env dédié — même logique que l'intégration OpenFace dans psifx (runtime externe séparé, ponté par fichier).

```bash
# 1. Cloner le repo (fait manuellement, pas par ce script) puis le rendre visible au conteneur :
#    docker cp /chemin/vers/sapiens2 <container>:/home/psifx/sapiens2

# 2. Environnement dédié
conda create -y -n sapiens2-env python=3.12 pip
conda activate sapiens2-env
cd /home/psifx/sapiens2 && pip install -e .

# 3. Checkpoints (détecteur DETR + poids pose). Tailles testées : 0.4B, 0.8B, 1B.
#    5B exclu d'office : poids seuls ≈10 Go en bf16 > 8 Go de VRAM (RTX 4060), OOM garanti.
export HF_HUB_DISABLE_XET=1          # cf. avertissement réseau ci-dessous
export HF_HUB_DOWNLOAD_TIMEOUT=30
hf download facebook/detr-resnet-101-dc5 --local-dir /home/psifx/sapiens2_host/detector/detr-resnet-101-dc5
for size in 0.4b 0.8b 1b; do
  hf download "facebook/sapiens2-pose-${size}" --local-dir "/home/psifx/sapiens2_host/pose/sapiens2-pose-${size}"
  ln -sf "/home/psifx/sapiens2_host/pose/sapiens2-pose-${size}/sapiens2_${size}_pose.safetensors" \
         "/home/psifx/sapiens2_host/pose/sapiens2_${size}_pose.safetensors"   # chemin attendu par Sapiens2Backend
done
```

`build_backend()` accepte `"sapiens2-<size>"` (ex. `sapiens2-0.8b`) et instancie `Sapiens2Backend(model_size=size)` en conséquence ; `"sapiens2"` seul reste un alias pour `0.4b`.

⚠️ **Blocage réseau rencontré et sa cause** : les téléchargements HF de plusieurs centaines de Mo (poids du détecteur/modèle) restaient bloqués indéfiniment à 0 octet dans ce conteneur, alors que l'hôte (même réseau) les récupérait instantanément. Diagnostic : un problème de MTU sur le pont Docker par défaut (`docker0`, MTU 1500) combiné à une interface hôte WiFi — le handshake TLS démarre puis les paquets de taille normale sont perdus en silence (blackhole PMTU classique), alors que les petites requêtes passent. Fix appliqué : conteneur déplacé sur un réseau Docker dédié avec un MTU réduit, sans toucher à la configuration globale du démon (donc sans perturber d'autres conteneurs) :
```bash
docker network create --opt com.docker.network.driver.mtu=1400 psifx-bench-net
docker network connect psifx-bench-net <container>
docker network disconnect bridge <container>
```
`HF_HUB_DISABLE_XET=1` (protocole de transfert accéléré de HF, indépendamment concerné par le même type de blocage) et `HF_HUB_DOWNLOAD_TIMEOUT=30` (échoue vite plutôt que de bloquer indéfiniment, permettant une reprise) aident aussi. Si ça se reproduit : vérifier depuis l'hôte directement (`curl` vers le même hôte de CDN) pour confirmer que c'est bien spécifique au conteneur avant de chercher ailleurs.

⚠️ **OOM CUDA même sur le modèle 0.4B** (le plus petit) à résolution 1024×768 sur un GPU 8 Go en fp32 — `sapiens2_infer.py` charge donc le modèle et les activations en **bf16** (`model.to(torch.bfloat16)` + entrées castées avant chaque forward), ce qui suffit à tenir dans la mémoire disponible. `Sapiens2Backend` passe aussi `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` au sous-processus pour réduire la fragmentation (des avertissements `CUDACachingAllocator OOM` bénins peuvent malgré tout apparaître dans les logs — l'allocateur retente et réussit, ce n'est pas fatal tant que le script se termine avec un `results.json` valide).

⚠️ **Mur de capacité GPU sur les images à forte densité de personnes, aggravé par la taille du modèle** : toutes les boîtes détectées dans une image sont batchées en un seul passage forward (pour la vitesse) — une image à 26 personnes (présente dans l'échantillon COCO de test) OOM en bf16 dès 0.8B, et à 1B même en découpant récursivement le batch jusqu'à une seule personne à la fois (mur de capacité réel, pas un bug de découpage — le modèle + le détecteur DETR occupent déjà la quasi-totalité des 8 Go). `sapiens2_infer.py` gère ça en deux temps :
1. `run_forward_with_oom_retry()` : sur `OutOfMemoryError`, découpe le batch en deux et réessaie récursivement (avec `torch.cuda.empty_cache()` entre chaque sous-lot — nécessaire, pas juste au moment de l'échec, sinon les sous-lots suivants OOM aussi même à taille réduite).
2. Si même un lot d'une seule personne échoue, l'image est **abandonnée proprement** (0 instance enregistrée) plutôt que de faire planter tout le run — même principe que le retry par chunks déjà présent dans `psifx/video/tracking/sam3/tool.py`.

**Conséquence pour la lecture des résultats** : plus le modèle est gros, plus il "rate" d'images à forte densité — ce qui fait paraître les gros modèles moins précis dans les métriques agrégées brutes, alors qu'à qualité de détection égale (mêmes images), la précision ne baisse pas. Voir `03_resultats.md` §2.2 pour l'analyse complète et la comparaison corrigée.

## Exécution

```bash
cd research/benchmark
python prepare_coco_subset.py --n 50          # télécharge annotations + images uniquement (pas tout val2017)
python run_accuracy.py --backends mediapipe,rtmpose,vitpose,sapiens2-0.4b,sapiens2-0.8b,sapiens2-1b,rtmo,detrpose
python run_video_benchmark.py --backends mediapipe,rtmpose,vitpose,sapiens2-0.4b,sapiens2-0.8b,sapiens2-1b,rtmo,detrpose
python aggregate_results.py
```

Sorties dans `results/` : `accuracy_summary.csv`, `video_summary.csv`, `final_table.csv`, `ranking.csv`, `pareto.png`.

## Limites (voir aussi §3.5 du protocole)

- Échantillon COCO réduit (~50 images), pas tout val2017.
- ViTPose n'a pas de détecteur intégré : évalué avec les boîtes vérité-terrain sur COCO (évaluation "oracle-box", pratique standard), et avec les boîtes du détecteur de RTMPose sur `Video.mp4` (pas de vérité terrain disponible dessus) — donc `vitpose` nécessite `rtmpose` dans la même exécution de `run_video_benchmark.py`.
- Mémoire GPU mesurée via `nvidia-smi` (tout le process), pas un compteur par framework — plus juste pour comparer des runtimes différents (MediaPipe/ONNXRuntime/PyTorch) qu'un compteur `torch.cuda`.
- Suivi mono-personne sur la vidéo (premier·ère détecté·e par frame) : proxy de stabilité/vitesse, pas une évaluation multi-personnes complète.
- Sapiens2 testé en 0.4B/0.8B/1B (5B exclu, cf. ci-dessus). Testé en bf16 (nécessaire pour tenir en mémoire), pas fp32 — écart de précision possible mais non quantifié ici vs le fp32 natif du checkpoint.
- `gpu_mem_delta_mb` peu fiable pour Sapiens2 : mesuré avant/après l'appel `predict_all()`, qui englobe tout le cycle de vie du sous-processus — celui-ci a déjà libéré sa mémoire au moment de la mesure "après", donc le delta ne capture pas le pic réel pendant l'inférence.
- Sapiens2 et DETRPose tournent dans un sous-processus séparé (subprocess par backend, pas par image, `hasattr(backend, "predict_all")` dans `run_accuracy.py`/`run_video_benchmark.py`) : leur mesure de vitesse/latence a un profil différent de MediaPipe/RTMPose/RTMO/ViTPose (pas de partage de contexte CUDA avec le reste du benchmark, mais aussi pas d'overhead de changement de contexte inter-framework dans le même process comme pour les quatre autres).
- DETRPose testé en taille M uniquement, en PyTorch eager (pas de fp16/TensorRT) — les chiffres de latence publiés supposent cette optimisation, cf. `03_resultats.md` §2.2/§4.
- Licence Sapiens2 restrictive (cf. `01_exploration_modeles.md` §4.1) — ce benchmark technique n'implique pas une recommandation d'usage en production tant que la licence n'est pas validée avec le responsable. RTMO et DETRPose sont Apache 2.0, aucune validation requise sur ce point.

## Comparer les branches via la CLI psifx (`run_psifx_cli_benchmark.py`)

`run_accuracy.py` appelle chaque backend directement sur les images. `run_psifx_cli_benchmark.py` passe au contraire par
la vraie commande `psifx video pose <tool> single-inference` : les images COCO sont mises au format 640×640 (letterbox)
dans une vidéo, l'archive `.tar.gz` produite est notée en OKS/PCK (`psifx_eval.py`). Chaque exécution ajoute une ligne
à `results/psifx_cli/<branche>.csv` (un fichier par branche, donc pas de conflit au merge) avec le commit courant, ce qui permet de comparer les branches
modèle/optimisation entre elles sur exactement la même entrée et le même format de sortie.

```bash
python run_psifx_cli_benchmark.py --tool mediapipe               # 50 images (data/coco_subset)
python run_psifx_cli_benchmark.py --tool mediapipe --coco tests  # 8 images commitées dans tests/
```

Comparer la dernière mesure de chaque branche (lue directement dans les branches via `git show`, sans checkout ni merge) :

```bash
python compare_psifx_cli.py            # 50 images
python compare_psifx_cli.py --coco tests
```

`wall_seconds` inclut le chargement du modèle (ce que paie un utilisateur de psifx par vidéo), il n'est donc pas
comparable aux FPS de `run_video_benchmark.py`.

## Vitesse en régime établi (`time_psifx_pose.py`) — pour les branches d'optimisation

`wall_seconds` mélange chargement, compilation et inférence. `time_psifx_pose.py` charge l'outil psifx une fois, fait
2 passes d'échauffement sur les images COCO puis en chronomètre 3 (avec `torch.cuda.synchronize()`), et donne ms/image
et FPS. Les options d'optimisation passent par `--kwargs` (JSON, arguments du constructeur de l'outil) :

```bash
python time_psifx_pose.py --tool vitpose --device cuda --kwargs '{"fp16": true, "compile": true}'
python time_psifx_pose.py --tool rtmo --device cuda --kwargs '{"precision": "fp16"}'
```

RTMPose/RTMO sur GPU demandent `onnxruntime-gpu==1.20.2` et le `LD_LIBRARY_PATH` décrit en haut de ce fichier ; sans
lui, les outils psifx s'arrêtent avec une erreur explicite au lieu de retomber silencieusement sur CPU.
