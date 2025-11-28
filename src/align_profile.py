# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
import math
import re

def align_profile(df: pd.DataFrame, ref_size: float = None) -> pd.DataFrame:
    
    
    
    """
    
    Aligne chaque frame du squelette dans le plan sagittal (vue de profil) via rotation, orientation et mise 
    à l'échelle proportionnelle autour du barycentre des épaules et hanches.

    Parameters
    ----------
    df : pandas.DataFrame
        DataFrame contenant les colonnes de positions nommées sous la forme '<joint>_x' et '<joint>_y'.

    ref_size : float, optional
        Taille de référence (distance épaule-cheville) utilisée pour normaliser la hauteur. Si non fournie, 
        elle est calculée comme la médiane de cette distance sur toutes les frames valides.

    Returns
    -------
    pandas.DataFrame
        DataFrame transformé, avec les coordonnées x et y alignées par frame.
        
    """
    
    
    def rotate_z(x, y, angle_rad, cx, cy):
        x_shifted = x - cx
        y_shifted = y - cy
        rotated_x = x_shifted * math.cos(angle_rad) - y_shifted * math.sin(angle_rad)
        rotated_y = x_shifted * math.sin(angle_rad) + y_shifted * math.cos(angle_rad)
        return rotated_x + cx, rotated_y + cy

    def scale_coords(x, y, scale_factor, cx, cy):
        x_scaled = (x - cx) * scale_factor + cx
        y_scaled = (y - cy) * scale_factor + cy
        return x_scaled, y_scaled

    df = df.copy()
    coord_cols_x = [col for col in df.columns if re.search(r'_x$', col)]
    coord_cols_y = [col for col in df.columns if re.search(r'_y$', col)]

    if ref_size is None:
        sizes = [
            math.hypot(df.at[i, 'LEFT_SHOULDER_x'] - df.at[i, 'LEFT_ANKLE_x'],
                       df.at[i, 'LEFT_SHOULDER_y'] - df.at[i, 'LEFT_ANKLE_y'])
            if not pd.isna(df.at[i, 'LEFT_SHOULDER_y']) and not pd.isna(df.at[i, 'LEFT_ANKLE_y'])
            else np.nan
            for i in range(len(df))
        ]
        ref_size = np.nanmedian(sizes)

    for i in range(len(df)):
        # barycentre
        barycenter_x = np.nanmean([
            df.at[i, 'RIGHT_SHOULDER_x'], df.at[i, 'LEFT_SHOULDER_x'],
            df.at[i, 'RIGHT_HIP_x'], df.at[i, 'LEFT_HIP_x']
        ])
        barycenter_y = np.nanmean([
            df.at[i, 'RIGHT_SHOULDER_y'], df.at[i, 'LEFT_SHOULDER_y'],
            df.at[i, 'RIGHT_HIP_y'], df.at[i, 'LEFT_HIP_y']
        ])

        # PCA pour obtenir l'orientation du tronc
        mat = np.array([
            [df.at[i, 'LEFT_SHOULDER_x'] - barycenter_x, df.at[i, 'LEFT_SHOULDER_y'] - barycenter_y],
            [df.at[i, 'RIGHT_SHOULDER_x'] - barycenter_x, df.at[i, 'RIGHT_SHOULDER_y'] - barycenter_y],
            [df.at[i, 'LEFT_HIP_x'] - barycenter_x, df.at[i, 'LEFT_HIP_y'] - barycenter_y],
            [df.at[i, 'RIGHT_HIP_x'] - barycenter_x, df.at[i, 'RIGHT_HIP_y'] - barycenter_y]
        ])

        pca = PCA(n_components=2).fit(mat)
        angle_rad = math.atan2(pca.components_[0,1], pca.components_[0,0])
        if abs(pca.components_[0,1]) > abs(pca.components_[0,0]):
            angle_rad -= math.pi / 2

        # Appliquer la rotation
        for j in range(len(coord_cols_x)):
            x_col = coord_cols_x[j]
            y_col = coord_cols_y[j]
            x_rot, y_rot = rotate_z(df.at[i, x_col], df.at[i, y_col], -angle_rad, barycenter_x, barycenter_y)
            df.at[i, x_col] = x_rot
            df.at[i, y_col] = y_rot

        # Tête en haut
        if df.at[i, 'NOSE_y'] > barycenter_y:
            for j in range(len(coord_cols_x)):
                x_col = coord_cols_x[j]
                y_col = coord_cols_y[j]
                x_rot, y_rot = rotate_z(df.at[i, x_col], df.at[i, y_col], math.pi, barycenter_x, barycenter_y)
                df.at[i, x_col] = x_rot
                df.at[i, y_col] = y_rot

        # Mise à l'échelle
        current_size = math.hypot(df.at[i, 'LEFT_SHOULDER_x'] - df.at[i, 'LEFT_ANKLE_x'],
                                  df.at[i, 'LEFT_SHOULDER_y'] - df.at[i, 'LEFT_ANKLE_y'])
        if not pd.isna(current_size) and current_size > 0:
            scale_factor = ref_size / current_size
            for j in range(len(coord_cols_x)):
                x_col = coord_cols_x[j]
                y_col = coord_cols_y[j]
                x_scaled, y_scaled = scale_coords(df.at[i, x_col], df.at[i, y_col], scale_factor, barycenter_x, barycenter_y)
                df.at[i, x_col] = x_scaled
                df.at[i, y_col] = y_scaled

    return df



#====================================================================
#------------------------------- MAIN -------------------------------
#====================================================================

# test pour comparer les veleurs avec le code R de Frederic 


"""
df = pd.read_excel("C:/Users/m.brechenmacher/OneDrive - Institut/Documents/GaitAnalysis/correction_analyse_marche/AUDR_4kmh_reduced.xlsx")
df_align_profile = align_profile(df)
df_align_profile.head(10)

"""