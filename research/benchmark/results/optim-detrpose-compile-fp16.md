# Optimisation DETRPose : `torch.compile` + fp16 (branche `optim/detrpose-compile-fp16`)

Date : 2026-10-07. RTX 4060 Laptop (8 Go). DETRPose-M, PyTorch 2.14 dans son environnement séparé
(`detrpose-env`), lancé en sous-processus par l'outil psifx. Piste §2.1 de `04_modeles_optimisation.md`.

## Temps du modèle seul (mesuré dans le sous-processus, CUDA synchronisé, après échauffement)

| Appareil | Config | Modèle ms/image | vs fp32 GPU | OKS | PCK | Détection |
|---|---|---|---|---|---|---|
| CPU | fp32 | 297.9 | — | 0.840 | 0.922 | 98 % |
| GPU | fp32 | 23.9 | — | 0.840 | 0.922 | 98 % |
| GPU | `--fp16` | 35.8 | −33 % | 0.840 | 0.922 | 98 % |
| GPU | `--compile` | 15.4 | +55 % | 0.839 | 0.922 | 98 % |
| GPU | `--fp16 --compile` | **11.9** | **×2.0** | 0.840 | 0.922 | 98 % |

fp16 seul ralentit (autocast ajoute des conversions sans fusion) ; combiné à `torch.compile`, qui fusionne ces
conversions, il double la vitesse du modèle. La précision ne bouge pas.

## Temps de bout en bout : le pont par sous-processus domine

Une passe de 50 images vs 200 images (même appel, GPU) permet de séparer coût fixe et coût par image :

| Config | Coût fixe par vidéo | Bout en bout ms/image | dont modèle |
|---|---|---|---|
| fp32 | 2.6 s | 90.1 | 21.7 |
| `--fp16 --compile` | 8.2 s (compilation) | 101.6 | 14.4 |

~70 ms par image partent dans le pont, pas dans le modèle : écriture de chaque image en PNG côté psifx, relecture et
prétraitement PIL (redimensionnement, `ToTensor`) dans le sous-processus. **Accélérer le modèle ne change donc
presque rien pour l'utilisateur** tant que ce pont reste tel quel.

## Conclusion

- `--fp16 --compile` est l'optimisation de modèle la plus efficace mesurée ici (×2), mais son effet est noyé par le
  pont (~25 % du temps de bout en bout seulement).
- Le vrai levier pour DETRPose dans psifx : le pont. Pistes : transmettre les images brutes (tableaux numpy, pas de
  PNG), un sous-processus persistant alimenté par un pipe, prétraitement sur GPU en torch, ou installer DETRPose dans
  l'environnement psifx si le pin `numpy==1.23.5` peut être levé.

## Correction de mesure

`infer.py` mesurait le temps par image sans `torch.cuda.synchronize()` : seul le lancement des noyaux CUDA était
compté (exécution asynchrone). Corrigé dans cette branche. **Le même défaut existe dans
`research/benchmark/detrpose_infer.py`**, dont viennent les 13.3 FPS de DETRPose dans `03_resultats.md` : ce
chiffre est probablement surestimé.
