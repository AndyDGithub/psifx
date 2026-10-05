# Optimisation ViTPose : `torch.compile` + fp16 (branche `optim/vitpose-compile-fp16`)

Date : 2026-10-07. GPU : RTX 4060 Laptop (8 Go, ~0.9 Go déjà occupés par d'autres applications). Piste §2.1 de
`04_modeles_optimisation.md`, appliquée à l'outil psifx `vitpose` (RT-DETR + ViTPose-base, `transformers` 5.3).

## Vitesse en régime établi (`time_psifx_pose.py`, 50 images COCO, 2 passes d'échauffement + 3 mesurées)

| Config | ms/image | FPS | Gain | Échauffement (s) | Pic GPU (Mo) |
|---|---|---|---|---|---|
| fp32 (référence) | 55.1 | 18.1 | — | 6 | 718 |
| `--fp16` | 56.9 | 17.6 | −3 % | 6 | 994 |
| `--compile` | 64.5 | 15.5 | −17 % | 122 | 656 |
| `--fp16 --compile` | 50.0 | 20.0 | **+10 %** | 139 | 652 |

## Précision (`run_psifx_cli_benchmark.py`, 50 images) : inchangée

OKS 0.850 / PCK 0.914 dans les trois configurations mesurées (fp32, `--fp16`, `--fp16 --compile`).

## Pourquoi le gain est faible : ViTPose n'est pas le goulot

Décomposition d'une image en fp32 (moyenne sur 100 images) :

| Étape | ms |
|---|---|
| Prétraitement détecteur (CPU) | 2.7 |
| **Détecteur RT-DETR (GPU)** | **26.7** |
| Post-traitement détecteur | 0.5 |
| **Prétraitement ViTPose (CPU, processeur HF)** | **12.8** |
| Modèle ViTPose (GPU) | 13.5 |
| Post-traitement ViTPose | 3.1 |

Le transformeur ViTPose, la cible de cette piste, ne pèse qu'environ 25 % du temps. Le backbone utilise déjà
SDPA (attention de type flash native de PyTorch, `_supports_sdpa = True`) : Flash Attention n'apporte donc rien de
plus. Compiler avec `dynamic=True` (nécessaire car la taille du batch dépend du nombre de personnes) coûte ~2 min à
chaque lancement et ralentit en fp32.

## Conclusion

- Le gain (+10 %) ne compense pas les ~2 min de compilation à chaque appel de la CLI, sauf sur des vidéos de plus de
  ~25 min à 18 FPS. **Peu intéressant pour psifx tel quel.**
- Les vrais leviers pour ViTPose sont **le détecteur** (RT-DETR, 50 % du temps : détecteur plus léger, ou n'en lancer
  un qu'une image sur N en suivant les boîtes) et **le prétraitement CPU** (24 % : le faire sur GPU en torch).
- Pour comparaison, RTMO (branche `feature/pose-rtmo`) n'a pas de détecteur séparé.
