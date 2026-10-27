# Diarisation : pyannote `community-1` contre `3.1`

Branche `feature/diarization-community-1` (depuis `dev`). Date : 2026-10-27.

## 1. Question

pyannote a publié en 2025 `speaker-diarization-community-1`, qui demande pyannote.audio 4. Sur
AVA-AVD, son DER annoncé passe de 49,7 % (avec la 3.1) à 44,6 %. psifx utilise pyannote.audio
3.3.2 et la 3.1.

On veut savoir si community-1 rend la chaîne plus fiable sur nos vidéos, avec du bruit, et ce que
coûte la migration.

## 2. Ce qui change dans psifx

| Changement | Pourquoi |
|---|---|
| `pyannote_token_kwargs()` (`psifx/utils/huggingface.py`) | pyannote 4 a renommé `use_auth_token` en `token`. Les outils de diarisation et d'identification marchent avec les deux versions. |
| `_annotation()` dans l'outil de diarisation | pyannote 4 renvoie un `DiarizeOutput` (diarisation normale et diarisation *exclusive*), alors que pyannote 3 renvoie une `Annotation`. |
| Options `--min_speakers`, `--max_speakers`, `--exclusive` | Elles bornent le nombre de locuteurs quand on ne le connaît pas exactement. `--exclusive` ne garde qu'un locuteur à la fois, ce qui convient aux sous-titres. |
| Modèle par défaut : `default_model_name()` | community-1 avec pyannote 4, 3.1 sinon. L'ancien défaut de la CLI, `speaker-diarization@2.1.1` (2022), différait de celui de l'outil. |
| `PYANNOTE_METRICS_ENABLED=0` dans `psifx/__init__.py` | pyannote 4 **envoie par défaut** à `otel.pyannote.ai` le nom du modèle et la durée de chaque audio. psifx coupe cet envoi ; `PYANNOTE_METRICS_ENABLED=1` permet de le réactiver. |

Avec pyannote 3.3.2, le code modifié redonne exactement la même diarisation 3.1 sur V1. Les
85 tests unitaires passent avec les deux versions de pyannote, dont 6 nouveaux tests
(`tests/unit/audio/diarization/`).

## 3. Dépendances : le vrai coût

| Paquet | psifx | Avec pyannote 4 | Raison |
|---|---|---|---|
| pyannote.audio | 3.3.2 | 4.0.7 | — |
| whisperx | 3.7.4 | **3.8.7rc1** | La 3.7.4 exige pyannote < 4 et plante à l'exécution (`Inference(use_auth_token=…)`). Les versions 3.8.0 à 3.8.6 exigent huggingface-hub < 1.0, ce qui est incompatible avec transformers 5.3 (SAM3). La borne est levée dans 3.8.7rc1 ([PR #1375](https://github.com/m-bain/whisperX/pull/1375), testée avec hub 1.7.1). |
| torchcodec | — | 0.7.0 | Nouvelle dépendance de pyannote 4. La version 0.7 correspond à torch 2.8 ; pip choisirait la 0.17. |
| numpy | 2.0.2 | 2.2.6 | Exigé par whisperx 3.8 (≥ 2.1) et pyannote-metrics 4 (≥ 2.2.2). |
| matplotlib | 3.8.4 | 3.10.7 | pyannote 4 exige ≥ 3.10. |
| protobuf | 4.25.9 | **4.25.9** | **Conflit**, voir ci-dessous. |

**Le conflit protobuf.** La télémétrie de pyannote 4 (`opentelemetry-proto`) déclare
protobuf ≥ 5. mediapipe 0.10.14 déclare protobuf < 5, et **plante vraiment** avec protobuf 5.29 :
il n'arrive plus à lire son propre graphe (`Failed to parse: node {`).

On ne peut pas simplement monter mediapipe. Les versions récentes (0.10.32, 1.1.1) n'ont plus
l'API `mediapipe.python.solutions.holistic` qu'utilise psifx ; elles ne proposent que l'API
*Tasks* (`HolisticLandmarker`).

En pratique, pyannote 4 marche avec protobuf 4.25.9 tant que la télémétrie est coupée. Avec ces
versions installées à la main (`--no-deps`), on obtient :
- des poses MediaPipe identiques au bit près sur V1 ;
- la même transcription WhisperX, mot pour mot, mais avec des horodatages décalés de 0,02 à
  0,44 s (nouvel alignement de whisperx 3.8).

En revanche, `pip install psifx` avec pyannote 4 dans `requirements.txt` échoue à la résolution des
dépendances. **`requirements.txt` reste donc sur pyannote 3.3.2** dans cette branche. Le code
accepte les deux versions.

Recette de l'environnement de test (`psifx-pya4`, clone de `psifx-env`) :

```bash
pip install "pyannote.audio==4.0.7" -c constraints.txt   # torch 2.8, hub 1.7.1, transformers 5.3
pip install --no-deps whisperx==3.8.7rc1 numpy==2.2.6 matplotlib==3.10.7 torchcodec==0.7.0 protobuf==4.25.9 omegaconf
```

## 4. Protocole

- **Modèles** :
  - la 3.1 sous pyannote 4 (`legacy31`) ;
  - community-1 (`community1`) ;
  - la 3.1 sous pyannote 3.3.2, c'est-à-dire la sortie du test e2e (`e2e_legacy31_pya3`).
- **Réglages** :
  - nombre de locuteurs donné, ou estimé par le modèle (`_auto`, le cas réel) ;
  - sortie normale, ou exclusive (`_excl`).
- **Références** (`research/diarization/make_references.py`) :
  - **V1** : pseudo-référence tirée des deux micros. Une trame de 20 ms est attribuée au micro qui
    est au moins 6 dB plus fort que l'autre. Les chevauchements sont absents de cette référence,
    donc on compte avec un collar de 0,25 s.
  - **V3** : les 40 segments étiquetés d'après le texte (193 s). Seules ces zones sont comptées.
- **Bruit** (`make_noisy.py`) : V1 mélangée à 10, 5 et 0 dB de RSB (mesuré sur la parole) avec :
  - un bruit rose ;
  - la musique de V4, voix chantée comprise ;
  - un brouhaha (*babble*) : trois copies décalées du talk-show V3.
- **En aval** : l'association voix↔personne (`link_speakers.py`), `transcription enhance` puis
  `evaluate.py`, sur V2 (vérité AVA-ActiveSpeaker) et V3 (étiquettes de texte). L'image et la
  transcription restent celles du test e2e.

Pour reproduire :

```bash
python research/diarization/make_references.py .e2e_outputs
python research/diarization/make_noisy.py .e2e_outputs
python research/diarization/compare.py run .e2e_outputs --model pyannote/speaker-diarization-3.1 --exclusive
python research/diarization/compare.py run .e2e_outputs --model pyannote/speaker-diarization-community-1 --exclusive
python research/diarization/compare.py score .e2e_outputs
research/diarization/downstream.sh v1_example v2_ava v3_talkshow   # with the e2e-demo scripts
```

Les résultats bruts sont dans `research/diarization/results/`.

## 5. Résultats

### 5.1 Vidéos sans bruit ajouté

| Vidéo | Mesure | 3.1 (pya3, e2e) | 3.1 (pya4) | community-1 |
|---|---|---|---|---|
| V1 (2 pers.) | DER, nombre connu / auto | 8,4 % / — | 8,4 % / 8,4 % | 8,4 % / 8,4 % |
| V2 AVA (film) | locuteurs trouvés (auto) | 4 | 3 | **2** |
| V2 AVA | bon locuteur actif (instants AVA annotés) | 64,8 % | 72,4 % | **88,8 %** |
| V2 AVA | part des instants où une personne est désignée | 72,3 % | 80,0 % | **98,5 %** |
| V3 talk-show | DER sur les segments étiquetés, auto | 11,4 % | 11,4 % | 11,6 % |
| V3 talk-show | locuteurs trouvés (auto) | 3 | 3 | 3 |
| V3 talk-show | attribution des sous-titres (étiquettes) | 97,7 % | 97,7 % | 97,7 % |
| V4 danse | locuteurs trouvés | 1 | 1 | 1 |
| — | temps sur V3 (330 s d'audio, GPU) | — | 14,4 s | 15,5 s |

Sur V2, le gain ne vient pas d'une meilleure précision quand une personne est désignée (90 % dans
les trois cas). Il vient du nombre de locuteurs : la 3.1 crée des locuteurs en trop, que
l'association envoie « hors champ ». Les sous-titres correspondants perdent alors leur couleur.

### 5.2 V1 bruitée : DER, nombre de locuteurs estimé (cas réel)

| Bruit | RSB | 3.1 : DER (locuteurs) | community-1 : DER (locuteurs) |
|---|---|---|---|
| rose | 10 dB | 38,7 % (4) | **12,0 % (2)** |
| rose | 5 dB | 34,1 % (3) | **19,2 % (2)** |
| rose | 0 dB | 56,2 % (1) | 56,2 % (1) |
| musique | 10 dB | 48,2 % (4) | **20,8 % (2)** |
| musique | 5 dB | 36,1 % (2) | 36,1 % (2) |
| musique | 0 dB | 48,8 % (2) | 48,2 % (2) |
| brouhaha | 10 dB | 44,2 % (4) | **28,8 % (2)** |
| brouhaha | 5 dB | 81,2 % (5) | **71,7 % (2)** |
| brouhaha | 0 dB | 77,4 % (4) | **45,5 % (2)** |

Quand le nombre de locuteurs est donné (`--num_speakers 2`), les deux modèles sont à égalité,
sauf avec le brouhaha, où community-1 reste meilleur :
- 10 dB : 28,8 % contre 50,9 % ;
- 5 dB : 46,6 % contre 62,4 %.

**Lecture :**
- **Le gain porte sur l'estimation du nombre de locuteurs.** Quand le modèle doit l'estimer, la
  3.1 trouve 3 à 5 locuteurs pour 2 personnes dans 6 cas bruités sur 9 ; community-1 en trouve 2
  dans 8 cas sur 9.
- **La détection de parole est identique** : les deux modèles ont exactement les mêmes taux de
  parole manquée et de fausses alarmes. À 0 dB de bruit rose, 52 % de la parole est perdue par les
  deux. Pour gagner en robustesse au bruit fort, il faudra agir avant la diarisation (séparation
  de la voix, rehaussement), pas changer de modèle de diarisation.
- **Échec de community-1** : avec le brouhaha à 5 dB, il trouve bien 2 locuteurs mais mélange les
  deux voix réelles dans un même groupe (confusion de 28,6 %). Le sous-titre de V1 attribué à la
  personne visible tombe alors de 86 % (3.1) à 57 %. Ce score d'attribution est peu fiable sur V1 :
  une seule personne y est suivie, donc tout groupe classé « hors champ » compte juste pour le
  micro de gauche.
- **Sortie exclusive** : elle baisse le DER dans le bruit (musique 5 dB : 36,1 % → 14,2 %) et sur
  V3 (11,2 % → 9,5 %), parce qu'elle supprime les faux chevauchements. Mais nos références n'ont
  pas de chevauchement, ce qui la favorise. En aval, l'attribution des sous-titres ne change pas.
  Elle reste donc une option, désactivée par défaut.
- **Effet de pyannote 4 sur la 3.1** : passer à pyannote 4 modifie déjà la 3.1 (V2 : 4 → 3
  locuteurs, 64,8 % → 72,4 %). Ce pipeline télécharge en effet sa PLDA depuis le dépôt
  community-1. La comparaison « avant/après » pour l'utilisateur est donc la colonne pya3 contre
  community-1.

## 6. Bilan et décision à prendre

community-1 est meilleur ou égal à la 3.1 partout, sauf un cas (brouhaha à 5 dB, en attribution
sur V1). Il est surtout bien meilleur quand le nombre de locuteurs est inconnu, c'est-à-dire en
conditions réelles. Le temps de calcul est le même.

Le blocage vient du paquet, pas du modèle : pyannote 4 et mediapipe 0.10.14 ne peuvent pas être
déclarés ensemble. Il y a trois options :
1. **Porter l'estimation de poses MediaPipe vers l'API *Tasks*** (`HolisticLandmarker`, mediapipe
   ≥ 0.10.31), sur une branche à part. Cela lève le conflit protobuf, modernise MediaPipe, et
   permet de passer à pyannote 4 et whisperx 3.8.7 dans `requirements.txt`. C'est le plus propre,
   mais il faut revalider toutes les sorties MediaPipe.
2. Fusionner cette branche telle quelle : le code accepte pyannote 3 et 4, `requirements.txt`
   reste sur 3.3.2, et la recette du §3 permet d'utiliser community-1 à la main.
3. Attendre que pyannote rende la télémétrie optionnelle dans ses dépendances.

Autre bug psifx relevé : `audio identification pyannote` plante (`KeyError: 'SPEAKER_02'`) quand
la diarisation trouve plus de locuteurs qu'il n'y a de micros.

## Sources

- [pyannote/speaker-diarization-community-1 (Hugging Face)](https://huggingface.co/pyannote/speaker-diarization-community-1)
- [pyannote.audio, README : benchmark et télémétrie](https://github.com/pyannote/pyannote-audio)
- [pyannote.ai, Community-1](https://www.pyannote.ai/blog/community-1)
- [whisperX PR #1375 : borne huggingface-hub relâchée](https://github.com/m-bain/whisperX/pull/1375)
- [whisperx 3.8.7rc1 (PyPI)](https://pypi.org/project/whisperx/3.8.7rc1/)
- [MediaPipe Solutions guide : fin des *Legacy Solutions*](https://developers.google.com/edge/mediapipe/solutions/guide)
