# IA_Gait_interface – Interface Dash pour l'analyse de la marche (MediaPipe / XLSX / Vidéo)

Ce dépôt propose une application Dash dédiée à la visualisation, au filtrage et à l'analyse de la marche humaine à partir de vidéos (`.mp4`, `.mov`) ou de fichiers `.xlsx` contenant les landmarks MediaPipe Pose.  
L’outil permet de comparer un ou deux fichiers simultanément, d’appliquer différents filtres, de visualiser les squelettes, de calculer les angles articulaires, de segmenter la marche et d’extraire les paramètres spatio-temporels.

---

## 1. Présentation générale

L’application permet :

- d'importer un ou deux fichiers vidéo ou `.xlsx`,
- de lire ou extraire automatiquement les landmarks MediaPipe,
- d'appliquer différents filtres paramétrables,
- de visualiser les squelettes bruts et filtrés en 2D,
- de calculer les angles de hanche, de genou et de cheville,
- d’identifier automatiquement les événements de marche (HS/TO),
- d’afficher les cycles de marche normalisés (0–100 %),
- d’exporter les paramètres spatio-temporels dans un fichier Excel.

Un bouton Paramètres permet d’ajuster les caractéristiques d’entrée, dont la longueur du pied, utilisée pour le calcul de la longueur de pas, mais aussi les FPS, et les paramètres de la vidéo : le width et le height.

---

## 2. Formats de données supportés

### 2.1. Fichiers `.xlsx`  
Le fichier doit contenir les coordonnées MediaPipe Pose (33 landmarks × coordonnées x/y/z).

L’application recherche automatiquement les colonnes suivantes :

- `height`
- `width`
- `fps`

Si elles sont absentes, l’application crée des valeurs par défaut, modifiables dans l’interface.

### 2.2. Fichiers vidéo `.mp4` / `.mov`  
Si une vidéo est fournie, les landmarks MediaPipe Pose sont automatiquement extraits via `video_to_landmarks.py`.

---

## 3. Pipeline général de traitement

### 3.1. Extraction ou chargement des landmarks

Selon le type de fichier, les coordonnées MediaPipe sont :

- extraites d’une vidéo (`video_to_landmarks.py`),
- ou chargées depuis un fichier `.xlsx`.

---

3.2. Filtres et prétraitements disponibles

L’application propose plusieurs opérations de prétraitement ainsi que des filtres temporels afin d’améliorer la qualité des trajectoires MediaPipe.
Ces traitements peuvent être appliqués individuellement ou en chaîne, avec des paramètres ajustables directement dans l’interface Dash.

Prétraitements (normalisation des coordonnées)

Centrage des coordonnées (center_coords.py)
Recentre les coordonnées du squelette afin de supprimer toute translation globale au cours du mouvement.

Alignement en vue profil (align_profil.py)
Réoriente automatiquement le squelette pour obtenir une vue latérale cohérente, facilitant l’analyse de la marche.

Correction des longueurs segmentaires (correct_limb_lengths_stable.py)
Stabilise les longueurs des segments corporels pour minimiser les variations artificielles dues au tracking MediaPipe.

Filtres temporels (réduction du bruit)

Savitzky–Golay (smooth_savgol.py)
Filtre de lissage polynomial permettant de réduire le bruit tout en préservant la dynamique du signal.

Filtre de Kalman (kalman.py)
Méthode de filtrage optimal adaptée aux trajectoires instables ou bruitées, particulièrement efficace sur les données MediaPipe.

Butterworth (passe-bas)
Filtrage passe-bas classique via scipy.signal.butter et filtfilt, permettant d’éliminer les hautes fréquences indésirables.

Spline lissée (UnivariateSpline)
Interpolation lissée continue offrant une trajectoire régulière tout en respectant la tendance du mouvement.
---

### 3.3. Visualisation du squelette en 2D

L’application permet d’afficher :

- le squelette MediaPipe brut,
- le squelette filtré,
- la comparaison superposée de deux fichiers,
- l’évolution du squelette frame par frame.

---

### 3.4. Calcul des angles articulaires

Le module `gait_pipeline_core.py` calcule les angles suivants, pour chaque frame :

- hanche,
- genou,
- cheville.

Les angles peuvent être visualisés :

- sur toute la durée du signal,
- en synchronisation avec l’affichage du squelette,
- en comparaison entre deux fichiers (brut vs filtré).

---

### 3.5. Segmentation automatique de la marche (HS / TO)

L’application identifie automatiquement :

- les Heel Strikes (HS),
- les Toe Offs (TO).

Ces événements permettent :

- le découpage des cycles,
- la normalisation des angles entre 0 et 100 % du cycle de marche,

---

### 3.6. Paramètres spatio-temporels

Après segmentation, l’application extrait automatiquement les pas (Gauche/Droite) :

- durée du pas et du cycle,
- longueur de pas,
- cadence,
- phases stance et swing en %,
- Single et Double support %
- amplitude angulaire cheville,hanche,genou(ROM),
- statistiques globales (moyennes, écarts-types),
- Les angles cheville,hanche,genou frame par frame


Les résultats sont exportés dans :

```
gait_metrics.xlsx
```




