
# visualization.py
from __future__ import annotations
import numpy as np
import pandas as pd

# Sous-ensemble utile de connexions MediaPipe Pose
POSE_CONNECTIONS = [
    (11,13),(13,15), (12,14),(14,16),      # bras
    (11,12),                                # épaules
    (11,23),(12,24),                        # tronc
    (23,25),(25,27),(27,29),(27,31),       # jambe gauche
    (24,26),(26,28),(28,30),(28,32),       # jambe droite
# hanches
    (29,31),(30,32)                          # pieds
    
]

LANDMARKS = [
    # index = MediaPipe PoseLandmark index
    "NOSE", #0
    "LEFT_EYE_INNER","LEFT_EYE","LEFT_EYE_OUTER",
    "RIGHT_EYE_INNER","RIGHT_EYE","RIGHT_EYE_OUTER",
    "LEFT_EAR","RIGHT_EAR",
    "MOUTH_LEFT","MOUTH_RIGHT",
    "LEFT_SHOULDER","RIGHT_SHOULDER",  # 11,12
    "LEFT_ELBOW","RIGHT_ELBOW",        # 13,14
    "LEFT_WRIST","RIGHT_WRIST",        # 15,16
    "LEFT_PINKY","RIGHT_PINKY",
    "LEFT_INDEX","RIGHT_INDEX",
    "LEFT_THUMB","RIGHT_THUMB",
    "LEFT_HIP","RIGHT_HIP",            # 23,24
    "LEFT_KNEE","RIGHT_KNEE",          # 25,26
    "LEFT_ANKLE","RIGHT_ANKLE",        # 27,28
    "LEFT_HEEL","RIGHT_HEEL",          # 29,30
    "LEFT_FOOT_INDEX","RIGHT_FOOT_INDEX" # 31,32
]

NAME_TO_INDEX = {name:i for i,name in enumerate(LANDMARKS)}

def extract_coordinates(df: pd.DataFrame) -> np.ndarray:
    """Retourne un array (T, J, 2) en suivant l’ordre LANDMARKS pour les colonnes *_x/_y.
    S’il manque des colonnes, remplit par NaN.
    """
    T = len(df)
    J = len(LANDMARKS)
    out = np.full((T, J, 2), np.nan, dtype=float)
    for j, name in enumerate(LANDMARKS):
        cx, cy = f"{name}_x", f"{name}_y"
        if cx in df.columns and cy in df.columns:
            out[:, j, 0] = pd.to_numeric(df[cx], errors='coerce').to_numpy()
            out[:, j, 1] = pd.to_numeric(df[cy], errors='coerce').to_numpy()
    return out
