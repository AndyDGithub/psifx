# Pourquoi ViTPose est-il moins bon que prévu dans psifx ?

Date : 2026-10-13. Branche : `analysis/vitpose-gap`. Demande de la réunion du 2026-10-13.
Script : `benchmark/vitpose_gap.py` ; résultats : `benchmark/results/vitpose_gap_summary.txt` (moyennes et
intervalles) et `benchmark/results/vitpose_gap_per_image.csv` (OKS/PCK par image).

## 1. Le constat

Via la commande psifx (50 images COCO), ViTPose-B obtient OKS 0,850, sous RTMPose-m (0,873), alors que le banc de
recherche lui donnait 0,890 et que ViTPose avait la réputation d'être le plus précis.

## 2. Ce que dit vraiment la littérature (configurations exactes utilisées)

| Modèle | Entraînement | AP COCO val (publié) |
|---|---|---|
| ViTPose-B, décodeur simple (`usyd-community/vitpose-base-simple`) | COCO seul | 75,5 |
| RTMPose-m 256×192 (preset rtmlib `balanced`) | body7 (AIC, COCO, CrowdPose, MPII, sub-JHMDB, Halpe, PoseTrack18) | 74,9 |
| RTMO-m (preset rtmlib `balanced`) | body7 | 72,6 |

À taille « base/m », **la littérature ne prédit pas d'avantage net pour ViTPose** (0,6 AP d'écart). Les chiffres où
ViTPose domine concernent ses grandes variantes (L, H, G). L'attente « ViTPose est le plus précis » venait de ces
grandes versions et du banc de recherche, qui n'était pas à armes égales (§3.1).

## 3. Mesures, un facteur à la fois (50 images COCO, OKS moyen)

### 3.1 Le banc de recherche comparait deux choses différentes

| Configuration | ViTPose-B | RTMPose-m |
|---|---|---|
| Boîtes de la vérité terrain (pose seule) | 0,889 | 0,896 |
| Boîtes YOLOX, bonne personne | 0,849 | 0,858 |
| Boîtes RT-DETR, bonne personne | 0,862 | 0,859 |
| Réglage psifx (vidéo, détecteur propre, plus grande boîte) | 0,850 | 0,873 |

Le banc de recherche donnait à ViTPose les boîtes de la vérité terrain (0,890) et à RTMPose son propre détecteur
(0,859). **À boîtes égales, les deux modèles sont à égalité**, comme dans la littérature.

### 3.2 Effet de chaque facteur sur ViTPose-B (différences appariées, IC 95 % par bootstrap, 10 000 tirages)

| Facteur | Effet sur l'OKS | IC 95 % | Significatif ? |
|---|---|---|---|
| ViTPose vs RTMPose, réglage psifx | −0,022 | [−0,066 ; +0,008] | **Non** |
| ViTPose vs RTMPose, mêmes boîtes vérité terrain | −0,006 | [−0,038 ; +0,028] | Non |
| ViTPose vs RTMPose, mêmes boîtes YOLOX | −0,009 | [−0,034 ; +0,010] | Non |
| Boîtes RT-DETR plutôt que YOLOX | +0,013 | [−0,003 ; +0,038] | Non |
| Plus grande boîte plutôt que la bonne personne | −0,009 | [−0,027 ; +0,000] | Limite |
| Images passées par la vidéo H.264 plutôt que brutes | −0,002 | [−0,007 ; +0,001] | Non |
| *Flip test* (boîtes vérité terrain) | +0,006 | [+0,001 ; +0,012] | Oui, faible |

Effet global du détecteur pour ViTPose : de 0,889 (boîtes parfaites) à 0,853 (RT-DETR, plus grande boîte), soit
environ −0,036. **C'est le plus gros poste**, et il explique l'écart entre 0,890 (banc de recherche) et 0,850 (psifx).

### 3.3 Taille du modèle (boîtes vérité terrain, images brutes)

| Variante | Sans flip | Avec flip |
|---|---|---|
| ViTPose-B simple (psifx) | 0,889 | 0,896 |
| ViTPose-B classique | 0,883 | 0,895 |
| ViTPose++ S | 0,876 | 0,885 |
| ViTPose++ B | 0,890 | 0,901 |
| ViTPose++ L | 0,908 | 0,909 |
| ViTPose++ H | **0,913** | 0,912 |

## 4. Conclusion

1. **ViTPose n'est pas moins bon que prévu** : à taille égale, la littérature le place au niveau de RTMPose-m, et nos
   mesures aussi. L'écart psifx (0,850 contre 0,873) **n'est pas significatif** sur 50 images (IC qui contient 0).
2. **Le 0,890 du banc de recherche était optimiste** : ViTPose y recevait les boîtes de la vérité terrain. Dans psifx,
   le détecteur coûte environ 0,036 d'OKS, et le choix de la plus grande boîte environ 0,009.
3. **Sensibilité de la mesure** : RTMPose passe de 0,858 (images brutes) à 0,873 (images issues de la vidéo), alors
   que la différence de pixels est minime. Sur 50 images, des écarts de 0,01 à 0,02 sont du bruit. D'où l'intérêt du
   test sur un jeu complet (OCHuman, point 4 de la réunion).
4. **Leviers réels pour ViTPose dans psifx** : une variante plus grande (ViTPose++ L : +0,02 avec des boîtes
   parfaites), le *flip test* (+0,006, faible mais significatif), et un meilleur détecteur ou un meilleur choix de la
   personne suivie (jusqu'à +0,036).

## Sources

- [ViTPose (arXiv 2204.12484)](https://arxiv.org/pdf/2204.12484) — ViTPose-B simple 75,5 AP COCO val, détecteur 56 AP
- [MMPose — RTMPose, tableaux body7/Body8](https://raw.githubusercontent.com/open-mmlab/mmpose/main/projects/rtmpose/README.md) — RTMPose-m body7 74,9 AP
- [RTMO (arXiv 2312.07526)](https://arxiv.org/pdf/2312.07526) et modèle zoo MMPose — RTMO-m body7 72,6 AP
- [Transformers — ViTPose](https://huggingface.co/docs/transformers/v4.48.0/model_doc/vitpose) — variantes, ViTPose++ et `dataset_index`
