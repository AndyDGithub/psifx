# SAM 3D Body : un maillage 3D par personne, guidé par les masques SAM3

Branche `feature/sam3d-body` (depuis `dev`). Date : 2026-10-27. Scripts de recherche, pas encore de commande psifx.

## 1. Pourquoi

La 3D de psifx vient aujourd'hui de MediaPipe : 33 points du corps en mètres, sans maillage. SAM 3D Body (Meta,
CVPR 2026) reconstruit à partir d'une image le maillage complet du corps, des mains et des pieds (modèle MHR, environ
18 000 sommets) et 70 points 3D. Il accepte une boîte et un masque comme guides : les masques SAM3 que psifx produit
déjà lui disent quelle personne reconstruire.

## 2. Mise en place

| Point | Choix |
|---|---|
| Modèle | `facebook/sam-3d-body-vith` (ViT-H, 631 M de paramètres, accès restreint, licence SAM) |
| Code | Clone de `facebookresearch/sam-3d-body` (commit `b5c765a`), hors du dépôt psifx (`.e2e_outputs/ext/`) |
| Environnement | `sam3d-body`, clone de `psifx-dev` plus `pytorch-lightning yacs einops timm roma braceexpand webdataset omegaconf trimesh loguru rich pyrootutils dill` |
| Composants non chargés | Détecteur ViTDet (detectron2), segmenteur et estimateur de focale MoGe : les boîtes et les masques viennent de SAM3, et la focale vaut par défaut la diagonale de l'image |
| Sécurité | `model.ckpt` est chargé avec `torch.load(weights_only=True)` (le code amont utilise `weights_only=False`, qui peut exécuter du code). `mhr_model.pt` est un modèle TorchScript : il n'utilise que les opérations standard de PyTorch, sans bibliothèque compilée |

Scripts (`research/sam3d_body/`) :
- `run_tracks.py` : une image sur `--stride`, toutes les personnes de l'image en un seul appel ; sorties dans
  `<vidéo>/sam3d_body/<id>.npz` ;
- `contact_sheet.py` : les maillages projetés sur les images d'origine ;
- `compare_rtmo.py` : l'accord des points 2D (réduits à COCO-17) avec RTMO sur les mêmes pistes.

## 3. Mesures (GPU 8 Go, une image sur 5)

| Vidéo | Images traitées | Temps | Par image | Écart médian à RTMO (part du torse) | Points à moins de 0,2 torse |
|---|---|---|---|---|---|
| V1 (1 personne, une image sur 25) | 46 | 38 s | 0,82 s | 0,031 | 94,8 % |
| V2 film (2 personnes) | 600 | 632 s | 1,05 s | 0,032 / 0,047 | 97,3 % / 96,4 % |
| V3 talk-show (2 personnes + 1 piste) | 1662 | 1402 s | 0,84 s | 0,031 / 0,040 | 98,2 % / 93,2 % |
| V4 danse (humanoïde avec queue) | 230 | 130 s | 0,57 s | 0,067 | 85,0 % |

Pour toutes les images, il faudrait compter environ 5 fois plus de temps : un facteur temps réel d'environ 20× sur
V3, du même ordre que le reste de la chaîne.

**Lecture :**
- **2D :** sur les vidéos réelles, SAM 3D Body et RTMO s'accordent à 3 à 5 % du torse près. Les grands écarts
  viennent de membres cachés (cheville sous la jambe croisée en V1 : 0,61). Sans vérité terrain, on ne sait pas
  lequel des deux a raison.
- **Danse :** l'accord est plus faible, mais la planche montre un maillage plausible malgré la queue, la jupe et les
  mouvements amples. Un décalage apparaît de profil (image 160).
- **Visuel :** sur les quatre planches, le maillage épouse la personne, en plan large comme en gros plan, et
  derrière la table de V3.
- **Défaut de suivi, antérieur à ce test :** en V3, un danseur filmé dans un encart vidéo (image 2380) porte le
  numéro de piste de l'intervieweur, et reçoit donc son maillage. Cela vient du suivi SAM3, pas de SAM 3D Body.

## 4. Limites

- **Image par image :** aucun lissage temporel, donc le maillage peut trembler d'une image à l'autre ; non mesuré à
  une image sur 5.
- **Focale par défaut :** sans estimateur de focale (MoGe), la profondeur absolue (`cam_t`) est approximative. La
  forme et la pose du corps n'en dépendent pas.
- **Coût :** environ 0,6 à 1 s par image et par appel, contre environ 0,1 s pour RTMO.
- **Accès :** les poids sont en accès restreint et sous licence SAM, à vérifier avant toute redistribution.
- **Format :** les archives `.npz` ne sont pas encore au format des archives de pose psifx.

## 5. Suite possible

1. Mesurer la précision sur une vérité terrain 3D (par exemple 3DPW ou EMDB) avant d'en faire une commande.
2. Une commande `psifx video pose sam3d_body multi-inference --masks` qui écrive les 70 points 2D et 3D dans
   l'archive de pose habituelle, avec le maillage en option.
3. Utiliser le maillage dans le rendu de bout en bout, à la place du panneau 3D MediaPipe.

Pour reproduire (dans le conteneur, avec l'accès au modèle accordé) :

```bash
git clone https://github.com/facebookresearch/sam-3d-body.git .e2e_outputs/ext/sam-3d-body
hf download facebook/sam-3d-body-vith --local-dir .e2e_outputs/ext/ckpt/sam-3d-body-vith
python research/sam3d_body/run_tracks.py .e2e_outputs/dev_2026-10-27/v3_talkshow \
    --repo .e2e_outputs/ext/sam-3d-body --checkpoint .e2e_outputs/ext/ckpt/sam-3d-body-vith --stride 5 --save_vertices
python research/sam3d_body/contact_sheet.py .e2e_outputs/dev_2026-10-27/v3_talkshow --n 8
python research/sam3d_body/compare_rtmo.py .e2e_outputs/dev_2026-10-27/v3_talkshow
```

## Sources

- SAM 3D Body, dépôt et installation : https://github.com/facebookresearch/sam-3d-body
- Article : https://arxiv.org/abs/2602.15989
- Poids ViT-H : https://huggingface.co/facebook/sam-3d-body-vith
