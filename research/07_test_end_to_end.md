# Test de bout en bout — de la vidéo aux données exploitables, puis re-rendu

Date : 2026-10-22, complété le 2026-10-26 (V3 après correction de SAM3, terme de visibilité). Branche `experiment/e2e-demo`. Matériel : RTX 4060 Laptop (8 Go), 62 Go de RAM.

**Objectif** : vérifier que psifx transforme des vidéos réelles et variées en données utilisables par un
chercheur, puis le prouver par un usage concret. On reconstruit une vidéo où chaque personne est un squelette
2D d'une couleur fixe, avec un panneau 3D simple et des sous-titres de la couleur de la personne qui parle.

## 1. Chaîne testée

`research/e2e/run_pipeline.py <dossier>` enchaîne les commandes psifx et les scripts de recherche. Chaque
étape est chronométrée dans `timings.json`. Une étape est sautée si ses sorties existent déjà, et la chaîne
s'arrête à la première erreur.

| Étape | Outil |
|---|---|
| Suivi des personnes | `psifx video tracking sam3 inference --text_prompt person` (un masque par personne) |
| Poses corps | `psifx video pose body multi-inference --model rtmo` (ViTPose en option) |
| Visage, mains, 3D | `psifx video pose mediapipe multi-inference` + nouvelle clé `pose_keypoints_3d` (commit ae402b4) |
| Audio | `psifx audio manipulation extraction / conversion / normalization`, Demucs en option (musique) |
| Qui parle quand | `psifx audio diarization pyannote inference` (`speaker-diarization-3.1`) |
| Transcription | `psifx audio transcription whisperx inference` (`large-v3-turbo`, multilingue) |
| Voix ↔ personne | `research/e2e/link_speakers.py` : mouvement des lèvres (MediaPipe) corrélé aux tours de parole, affectation hongroise, sinon « hors champ » |
| Sous-titres attribués | `psifx audio transcription enhance` avec le mapping des lèvres |
| Re-rendu | `research/e2e/render.py` → `render.mp4` (squelettes + 3D + sous-titres colorés) et `subtitles.ass` |
| Mesures | `research/e2e/evaluate.py` → `metrics.json`, `contact_sheet.jpg` ; `summarize.py` agrège les vidéos |

Les sorties restent dans `.e2e_outputs/<vidéo>/`, hors git.

## 2. Vidéos

| # | Vidéo | Difficulté visée | Vérité terrain |
|---|---|---|---|
| V1 | `example/data/Video.mp4`, 46 s | Cas connu : 1 personne visible, 1 voix hors champ | 1 micro-cravate par personne (stéréo G/D) |
| V2 | AVA-ActiveSpeaker, film `rJKeqfTlAeY`, 1020–1140 s (2 min) | 2 hommes, espagnol, plan fixe (aucune coupe, vérifié avec ffmpeg) | Boîtes de visage « parle / ne parle pas » annotées à la main |
| V3 | Talk-show Konbini « Small Talk », 5,5 min, 1080p | Français, 2 personnes, 112 changements de plan (un toutes les 3 s), incrustations graphiques | Aucune (qualitatif) |
| V4 | Short de danse, 48 s, 608×1080 | Personnage animé avec une queue, mouvements rapides, voix sous la musique | Paroles de la chanson (WER) |

AMI, prévu au départ, n'a pas pu être utilisé. Le site `groups.inf.ed.ac.uk` coupe la connexion TLS, et les
miroirs vidéo sur Hugging Face ont disparu. AVA-ActiveSpeaker le remplace pour l'association voix ↔ personne.

## 3. Résultats

| Vidéo | Durée (s) | Pistes SAM3 | Poses RTMO (% images) | Poses MediaPipe (% images) | Locuteurs diarisés | Texte attribué à une personne visible | Facteur temps réel |
|---|---|---|---|---|---|---|---|
| V1 exemple | 46 | 1 | 100 % | 100 % | 2 | 49 % | 21,3× |
| V2 AVA | 120 | 2 | 97 %, 100 % | 100 %, 100 % | 4 | 76 % | 24,5× |
| V3 talk-show | 333 | 3 (18 avant filtre) | 63 %, 70 %, 0 % | 63 %, 71 %, 0 % | 3 | 100 % | 27,8× |
| V4 danse | 48 | 1 | 87 % | 94 % | 1 | 0 % | 18,3× |

Facteur temps réel = temps de calcul total / durée de la vidéo. V3 : à cause du montage, les deux personnes ne
sont à l'écran que sur 64 et 72 % des images ; une pose est trouvée sur 98 % des images où elles le sont. La
troisième piste gardée est une main en très gros plan (plan récurrent, 15 % de la vidéo), sans pose.

**Temps par étape** (et multiple de la durée de la vidéo) :

| Vidéo | SAM3 | RTMO | ViTPose | MediaPipe (CPU) | Demucs | Diarisation | Transcription | Lèvres | Rendu |
|---|---|---|---|---|---|---|---|---|---|
| V1 | 624 s (13,6×) | 48 s (1,0×) | 97 s (2,1×) | 131 s (2,9×) | – | 19 s | 9 s | 1 s | 17 s |
| V2 | 1850 s (15,4×) | 241 s (2,0×) | – | 653 s (5,4×) | – | 11 s | 109 s | 3 s | 39 s |
| V3 | 6192 s (18,6×) | 887 s (2,7×) | – | 1743 s (5,2×) | – | 20 s | 178 s | 8 s | 123 s |
| V4 | 602 s (12,6×) | 31 s (0,7×) | 73 s (1,5×) | 108 s (2,3×) | 5 s | 7 s | 9 s | 1 s | 10 s |

**Association voix ↔ personne** :

- **V1, comparée aux micros** : 100 %. La voix du micro droit va bien à la personne visible (score +1,43) et
  celle du micro gauche est classée hors champ (−1,48).
- **V2, comparée à AVA** : sur 1784 instants où une personne parle, l'association est juste dans 65 % des cas.
  Elle ne répond que pour 72 % de ces instants ; quand elle répond, elle a raison à 90 %. Le hasard donnerait
  52 %. Les non-réponses viennent surtout de la diarisation : elle trouve 4 voix pour 2 personnes, et deux de
  ces voix sont classées hors champ.
- **V3, sans vérité terrain** : 40 segments de transcription étiquetés d'après le texte seul quand il ne laisse
  aucun doute (questions de l'intervieweur, invitée parlant d'elle au féminin ;
  `.e2e_outputs/v3_talkshow/reference_speakers.txt`).
  - Lèvres seules : 59 % du temps de ces segments attribué à la bonne personne (22 segments sur 40). La voix de
    l'intervieweur (41 % de la parole) part « hors champ » : ses lèvres, derrière une barbe et un micro, bougent à
    peine plus quand il parle (activité 0,048 contre 0,044 quand l'invitée parle, en gros plan).
  - Lèvres + visibilité : 98 % (39 sur 40). Le nouveau terme ajoute au score la part du temps de parole où la
    personne est à l'écran, moins cette part pendant que les autres parlent : dans une vidéo montée, la
    réalisation montre le plus souvent qui parle. Il est nul pour une caméra fixe où tout le monde reste visible :
    V1, V2 et V4 sont strictement inchangés.
  - Le lien de l'intervieweur reste juste au-dessus du seuil (0,26 pour 0,2) : fragile.

**Transcription (V4, chanson)** :

- sans Demucs, WhisperX ne détecte aucune parole et plante ;
- avec Demucs (voix seule), le WER vaut 0,58. « Get loose », répété dans le refrain, devient « can't lose » et
  pèse à lui seul environ 15 points.

**Stabilité des poses** (saccade normalisée du nez) :

| Vidéo | RTMO | ViTPose | MediaPipe |
|---|---|---|---|
| V1 (assis) | 76 | 137 | 51 |
| V2 | 146–152 | – | 78–86 |
| V4 (danse) | 2871 | 7852 | 2256 |

MediaPipe est le plus lisse, parce qu'il lisse dans le temps. ViTPose, qui traite chaque image seule, est le
plus saccadé.

## 4. Limites observées et causes

1. **SAM3 domine le coût**, à 13–19× le temps réel (19× sur la vidéo montée) : environ 0,5–0,75 s par image
   en 1080p. Avec les paquets de 300 images par défaut, la mémoire GPU déborde en 1080p ; `--chunk_size 64`
   règle le problème. Conséquence : 1 h de vidéo demanderait de 13 à 19 h de suivi sur ce portable.
2. **SAM3 et le montage (V3)** — *corrigé le 2026-10-26, voir §7.* psifx relie les identités d'un paquet au suivant en comparant les masques
   (IoU, `_map_chunk_object_ids` dans `psifx/video/tracking/sam3/tool.py`). Après un changement de plan, les
   masques ne se recouvrent plus, et chaque personne reçoit un nouvel identifiant. Le talk-show a ainsi atteint
   57 pistes pour 2 personnes à 76 % de la vidéo.
   - Des paquets courts multiplient les raccords, donc le problème.
   - psifx garde un encodeur ffmpeg ouvert par piste jusqu'à la fin, en 1080p sans perte (`-crf 0`), soit
     environ 800 Mo de RAM chacun.
   - Les 58 encodeurs occupaient 43 Go. Le noyau a tué VS Code faute de mémoire, ce qui a arrêté le conteneur.
   - `--max_num_objects` n'est pas une solution : au-delà de la limite, les personnes réapparues ne sont
     simplement plus suivies.
   - Il faudrait une ré-identification (apparence, visage) plutôt que la seule IoU, et fermer l'encodeur d'une
     piste disparue (ou écrire les masques en PNG/RLE).
3. **MediaPipe sur CPU** : 2–5× le temps réel, la deuxième étape la plus chère.
4. **Visage non humain (V4)** : MediaPipe ne trouve le visage animé que sur 7 % du temps chanté. Sans lèvres,
   pas d'association, et la chanteuse reste hors champ. Le corps est bien suivi (87–94 %), mais la queue est
   ignorée : aucun modèle ne la possède dans son squelette.
5. **Musique** : la détection de parole de WhisperX ne trouve rien sous la musique. Il faut séparer la voix
   avant (Demucs, environ 0,1× le temps réel).
6. **Sur-découpage de la diarisation** : V2 donne 4 voix pour 2 personnes. `link_speakers` rattache plusieurs
   voix à une même personne quand le score le permet ; sinon elles sont classées hors champ.
7. **Granularité de `enhance`** : chaque segment WhisperX reçoit un seul locuteur, celui qui recouvre le plus
   le segment. Une réplique rapide à l'intérieur d'un segment prend donc la couleur de l'autre personne (vu sur
   V3 : fin de réponse de l'invitée et question suivante dans un même segment).
8. **Lèvres peu lisibles (V3)** : barbe et micro devant la bouche. Compensé par la présence à l'écran (§3), qui
   ne sert qu'aux vidéos montées.

## 5. Bugs psifx trouvés (contournés, non corrigés)

| Bug | Symptôme | Contournement |
|---|---|---|
| `audio identification` : modèle par défaut speechbrain | `RemoteEntryNotFoundError` (`custom.py`) avec un `huggingface_hub` récent | `--model_names pyannote/embedding` |
| WhisperX sans parole détectée | `IndexError` dans le pipeline transformers au lieu d'une transcription vide | Séparer la voix (Demucs) |
| `--language` de la transcription | Option lue, mais jamais transmise à `pipeline.transcribe` | Langue auto-détectée |
| MediaPipe, nombre de points du visage | 468 zéros quand aucun visage n'est trouvé, 478 points sinon | **Corrigé** par le passage à l'API Tasks (note 09) |
| SAM3, une piste par plan + un encodeur ouvert par piste | Explosion de la RAM sur une vidéo montée (§4.2) | **Corrigé** dans psifx (§7) |

## 6. Ce qui mériterait de devenir une commande psifx

- **`psifx audio identification lips`** : l'association voix ↔ personne par les lèvres. Elle sort le même
  format que `identification pyannote`, donc `enhance` l'utilise sans changement. C'est la pièce qui manque
  quand il n'y a pas un micro par personne.
- **`psifx video visualization subtitles`** : sous-titres colorés par personne (incrustés et `.ass`), en
  réutilisant le VTT enrichi.
- **Une couleur par personne** dans les visualisations de pose, au lieu d'une couleur par articulation.
- **Une option de séparation de la voix** avant la transcription, pour les vidéos avec musique.
- **Côté SAM3** : la ré-identification au changement de plan et des encodeurs fermés à la disparition d'une
  piste. Il faut le faire avant de traiter des émissions montées.

## 7. Correction du suivi SAM3 (branche `fix/sam3-shot-changes`, mergée dans `dev`)

- **Détection des coupes** (différence entre vignettes 64×36 consécutives > 0,15 et > 3× le mouvement récent +
  0,02) : retrouve les 30 coupes `ffmpeg` de la première minute de V3 (+ 3 animations), aucune fausse coupe sur V1,
  V2, V4. Une session SAM3 par plan.
- **Ré-identification** : plongement DINOv2-small de la personne masquée, meilleure similarité cosinus ≥ 0,45 avec
  les plongements déjà vus de chaque personne ; deux pistes visibles ensemble ne partagent jamais un identifiant.
  Choix du descripteur sur 41 segments étiquetés : AUC 0,966 (DINOv2) contre 0,708 (histogramme de couleurs).
- **Mémoire** : encodeurs x264 en `-tune zerolatency`, 813 → 170 Mo chacun, toujours sans perte.
- **NMS et remplissage des trous** en PyTorch/SciPy quand le paquet `kernels` (code compilé téléchargé du Hub) est
  absent : la NMS seule ne change rien au nombre de pistes.

| Première minute de V3 | Pistes | Pistes mélangeant les 2 personnes | Temps |
|---|---|---|---|
| Avant | 16 | 4 | 1143 s |
| NMS seule | 16 | – | 1162 s |
| Coupes + ré-identification + NMS | 9 | 0 | 1032 s |

V3 complète : 18 pistes pour 113 plans, chaque personne en une seule piste pure sur 5,5 min. V1 : inchangé (1
piste, IoU moyenne 0,974 avec les anciens masques), +7 % de temps. Le pipeline écarte ensuite les pistes présentes
sur moins de 5 % des images (`filter_tracks.py`).

## 8. Nouveau passage depuis `dev` (2026-10-27)

Environnement neuf (`pip install -e ".[dev,rtm-gpu]"`) : pyannote `community-1`, MediaPipe « Tasks » (1.1.1) et
whisperx 3.8.7rc1. Les sorties sont dans `.e2e_outputs/dev_2026-10-27/<vidéo>/` (`render.mp4`, `contact_sheet.jpg`,
`metrics.json`).

| Vidéo | Mesure | Premier passage | `dev` |
|---|---|---|---|
| V1 | association voix–personne (micros) | 100 % | 100 % |
| V1 | facteur temps réel | 21,3× | 20,8× |
| V2 | locuteurs trouvés | 4 | **2** |
| V2 | bon locuteur actif (vérité AVA) | 64,8 % | **88,7 %** |
| V2 | sous-titres sur une personne visible | 75,8 % | **100 %** |
| V2 | facteur temps réel | 24,5× | 22,5× |
| V3 | attribution, nombre de locuteurs estimé | 97,7 % | 63,6 % |
| V3 | attribution, 2 locuteurs indiqués (`v3_talkshow_2spk`) | — | **98,0 %** |
| V3 | facteur temps réel | 27,8× | 24,1× |
| V4 | WER des paroles | 58,1 % | **55,2 %** |
| V4 | facteur temps réel | 18,3× | 17,7× |

**V3 :**
- sans nombre de locuteurs, la diarisation crée un petit groupe de 16 s pour l'intervieweur ;
- l'affectation un à un donne ce petit groupe à l'intervieweur, et son groupe principal (136 s) tombe sous le seuil, en
  « hors champ » ;
- la cause est MediaPipe Tasks : le visage de l'intervieweur n'est trouvé que sur 44 % des images de ses tours (69 %
  avec l'ancienne API), et le score passe de 0,25 à −0,15 ; la version de la diarisation ne joue aucun rôle (note 09, §3.4).

La transcription découpe autrement : les étiquettes de texte ont été replacées sur les nouveaux segments par leur
texte (39 sur 40 ; un segment coupé différemment est écarté).

## 9. Reproduire

```bash
# dans le conteneur, avec HF_TOKEN
pip install -r research/e2e/requirements.txt   # demucs, jiwer, yt-dlp
export HF_HUB_DISABLE_XET=1
python research/e2e/run_pipeline.py .e2e_outputs/v1_example --num_speakers 2 --stereo_mics --vitpose --chunk_size 64
python research/e2e/run_pipeline.py .e2e_outputs/v2_ava --chunk_size 64
python research/e2e/run_pipeline.py .e2e_outputs/v4_dance --separate_vocals --num_speakers 1 --vitpose --chunk_size 64
python research/e2e/run_pipeline.py .e2e_outputs/v3_talkshow --chunk_size 64   # add --num_speakers 2 if known
python research/e2e/summarize.py .e2e_outputs/v1_example .e2e_outputs/v2_ava .e2e_outputs/v3_talkshow .e2e_outputs/v4_dance
```

## Sources

- AVA-ActiveSpeaker : https://research.google.com/ava/download.html
- AMI Corpus : https://groups.inf.ed.ac.uk/ami/download/
- Multi-human Talking Video Dataset (non publié) : https://github.com/showlab/Multi-human-Talking-Video-Dataset, arXiv 2508.03050
- VoxConverse : https://github.com/joonson/voxconverse
- Demucs : https://github.com/facebookresearch/demucs
- pyannote speaker-diarization-3.1 : https://huggingface.co/pyannote/speaker-diarization-3.1
- WhisperX : https://github.com/m-bain/whisperX
