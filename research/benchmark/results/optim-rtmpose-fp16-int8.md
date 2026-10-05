# Optimisation RTMPose : fp16 + INT8 (branche `optim/rtmpose-fp16-int8`)

Date : 2026-10-07. RTX 4060 Laptop (8 Go) + CPU avec AVX-VNNI. `onnxruntime-gpu` 1.20.2. Pistes §2.1/§2.2 de
`04_modeles_optimisation.md` appliquées aux deux modèles ONNX de l'outil psifx `rtmpose` (détecteur YOLOX-m +
RTMPose-m). `torch.compile` ne s'applique pas (modèles ONNX, déjà optimisés par onnxruntime).

- fp16 : `onnxruntime.transformers.float16`, entrées/sorties gardées en float32.
- INT8 : quantification statique QDQ S8S8, poids par canal, uniquement `Conv`/`MatMul` (cf. bug ci-dessous),
  calibrée sur 32 images de `tests/integration/data/Multi.mp4` (pas sur les images COCO évaluées).

## Résultats (50 images COCO ; vitesse via `time_psifx_pose.py`, précision via la CLI psifx)

| Appareil | Précision | ms/image | FPS | vs fp32 même appareil | OKS | PCK | Détection |
|---|---|---|---|---|---|---|---|
| CPU | fp32 | 274.5 | 3.6 | — | 0.873 | 0.935 | 98 % |
| CPU | int8 | 436.9 | 2.3 | **−59 %** | **0.647** | 0.824 | 98 % |
| GPU | fp32 | 24.6 | 40.7 | — | 0.873 | 0.935 | 98 % |
| GPU | fp16 | 22.1 | 45.3 | **+11 %** | 0.873 | 0.935 | 98 % |
| GPU | int8 | 32.3 | 31.0 | **−31 %** | **0.648** | 0.806 | 96 % |

## Lecture

- **fp16 sur GPU : petit gain gratuit** (+11 %), précision strictement identique. À garder.
- **INT8 : à écarter sur ce matériel**, plus lent ET nettement moins précis (−23 points d'OKS) :
  - plus lent même sur un CPU AVX-VNNI : l'activation SiLU (`Sigmoid` × `Mul`) entre chaque convolution reste en
    float, d'où une paire Quantize/Dequantize à chaque couche qui annule le gain des convolutions INT8 ;
  - sur GPU, le provider CUDA n'a pas de vrais noyaux INT8 pour ce format (la doc onnxruntime renvoie à TensorRT et
    aux Tensor Cores INT8) ;
  - la perte de précision vient des convolutions, pas de la tête SimCC : ne quantifier que `Conv` donne OKS 0.657.
    Pistes non testées : calibration par entropie/percentile, calibration sur des images proches de COCO, ou
    quantification-aware training.
- **Le GPU change tout** : 274 → 25 ms/image (×11) par rapport au CPU, bien plus que n'importe quelle optimisation.

## Bug rencontré et corrigé (sur `dev`)

Première version : tous les opérateurs quantifiés → **OKS 0, aucune détection**. La sortie `dets` de YOLOX concatène
coordonnées (0-640) et score (0-1) dans un seul tenseur ; une seule échelle INT8 pour les deux arrondit tous les
scores à 0. Corrigé en ne quantifiant que `Conv`/`MatMul` (test de non-régression ajouté). Second bug : YOLOX est en
opset 11, or la quantification par canal demande l'opset 13 → conversion automatique avant quantification.
