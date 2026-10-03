# Optimisation RTMO : fp16 + INT8 (branche `optim/rtmo-fp16-int8`)

Date : 2026-10-07. Même matériel, même méthode et mêmes réglages que `optim-rtmpose-fp16-int8.md` (fp16 avec I/O
float32 ; INT8 statique QDQ S8S8 sur `Conv`/`MatMul`, calibré sur 32 images de `tests/integration/data/Multi.mp4`).
Un seul modèle ONNX ici (RTMO-m, sans détecteur séparé).

## Résultats (50 images COCO)

| Appareil | Précision | ms/image | FPS | vs fp32 même appareil | OKS | PCK | Détection |
|---|---|---|---|---|---|---|---|
| CPU | fp32 | 149.1 | 6.7 | — | 0.826 | 0.880 | 92 % |
| CPU | int8 | 258.2 | 3.9 | **−42 %** | **0.584** | 0.680 | 72 % |
| GPU | fp32 | 13.0 | 76.8 | — | 0.826 | 0.880 | 92 % |
| GPU | fp16 | 9.1 | 109.6 | **+43 %** | 0.826 | 0.880 | 92 % |
| GPU | int8 | 18.1 | 55.4 | **−28 %** | **0.580** | 0.662 | 70 % |

## Lecture

- **fp16 sur GPU : le meilleur gain mesuré de toutes les branches d'optimisation** (+43 %, 110 FPS), sans aucune
  perte de précision. RTMO étant entièrement convolutif (pas de détecteur ni de prétraitement lourd hors modèle),
  l'optimisation du modèle porte sur presque tout le temps de calcul, contrairement à ViTPose (25 %).
- **INT8 : à écarter**, pour les mêmes raisons que RTMPose (SiLU entre les convolutions → Q/DQ à chaque couche ;
  pas de noyaux INT8 réels pour le provider CUDA), et la perte de précision est encore plus forte (−24 points d'OKS,
  −20 points de détection) : le score de présence de RTMO souffre de la quantification.

## Bug rencontré et corrigé (sur `dev`)

`quant_pre_process` échouait sur RTMO (« Incomplete symbolic shape inference ») : l'inférence de formes symbolique
vise les transformeurs ; désactivée au profit de l'inférence de formes ONNX standard, comme le recommande la doc.
