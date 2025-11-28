# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd
from pykalman import KalmanFilter

def kalman_smooth_coords(coords: np.ndarray) -> np.ndarray:
    
    """
    
    Applique un filtre de Kalman pour lisser une série de coordonnées 1D (df).
    
    Parameters
    ----------
    coords : np.ndarray
        Tableau 1D de coordonnées (ex: x ou y) à lisser.

    Returns
    -------
    np.ndarray
        Coordonnées lissées (même taille que l'entrée).
                             
    """
    
    
    n = len(coords)
    x_hat = np.zeros(n)
    P = np.zeros(n)

    # Initial values
    x_hat[0] = coords[0]
    P[0] = 1e5

    # Parameters
    GGt = np.nanvar(coords) * 0.1
    HHt = np.nanvar(np.diff(coords)) 

    for t in range(1, n):
        # Prediction
        x_pred = x_hat[t-1]
        P_pred = P[t-1] + HHt

        # Update
        K = P_pred / (P_pred + GGt)
        x_hat[t] = x_pred + K * (coords[t] - x_pred)
        P[t] = (1 - K) * P_pred

    return x_hat


def smooth_skeleton_kalman(df: pd.DataFrame, points: list, axes: list = ["_x", "_y"]) -> pd.DataFrame:
    
    """
    
    Applique un filtre de Kalman aux coordonnées x et y de chaque point du squelette.

    Parameters
    ----------
    df : pandas.DataFrame
        DataFrame contenant les coordonnées du squelette.
    points : list
        Liste des noms de points (sans suffixe '_x' ou '_y').
    axes : list
        Liste des axes à lisser (par défaut ['_x', '_y']).

    Returns
    -------
    pandas.DataFrame
        DataFrame avec coordonnées lissées.
        
    """
    
    
    df_smoothed = df.copy()

    for point in points:
        for suffix in ["_x", "_y"]:
            colname = point + suffix
            if colname in df_smoothed.columns:
                df_smoothed[colname] = kalman_smooth_coords(df_smoothed[colname].values)


    return df_smoothed



#====================================================================
#------------------------------- MAIN -------------------------------
#====================================================================

# test pour comparer les veleurs avec le code R de Frederic 

"""

df = pd.read_excel("C:/Users/m.brechenmacher/OneDrive - Institut/Documents/GaitAnalysis/correction_analyse_marche/AUDR_4kmh_reduced.xlsx")
df_smoothed = smooth_skeleton_kalman(df, points = ["NOSE", "LEFT_SHOULDER", "RIGHT_SHOULDER", "LEFT_ELBOW", "RIGHT_ELBOW",
                                             "LEFT_WRIST", "RIGHT_WRIST", "LEFT_HIP", "RIGHT_HIP", "LEFT_KNEE", "RIGHT_KNEE",
                                             "LEFT_ANKLE", "RIGHT_ANKLE", "LEFT_HEEL", "RIGHT_HEEL", "LEFT_FOOT_INDEX", "RIGHT_FOOT_INDEX"])
df_smoothed["LEFT_ANKLE_x"].head(20)

"""