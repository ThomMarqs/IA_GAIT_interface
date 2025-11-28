# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd

def correct_limb_lengths_stable(df: pd.DataFrame, num_ref_frames: int = 10, limit: bool = False) -> pd.DataFrame:
    
    """
    
    Ajuste les longueurs des segments du membre droit pour qu'elles correspondent à la moyenne des longueurs
    des segments du membre gauche sur les frames de reference avec equilibre des longueurs jambes gauche/droite.

    Parameters
    ----------
    df : pandas.DataFrame
        DataFrame avec colonnes pour les coordonnees des articulations, nommees 'LEFT_<segment>_x', 'LEFT_<segment>_y',
        'RIGHT_<segment>_x', 'RIGHT_<segment>_y'.

    num_ref_frames : int, optional
        Nombre de frames de reference utilisees pour calculer les longueurs moyennes (default: 10).

    limit : bool, optional
        Si True, limite la correction entre 90% et 110% de la longueur actuelle (default: False).

    Returns
    -------
    pandas.DataFrame
        DataFrame corrigé avec longueurs du membre droit ajustees.
        
        
    """
    
    # convertir les colonnes *_x et *_y en float pour éviter les conflits de type
    # j'ai rajouté cette partie car les test ne passaient pas pour soucis de type
    for col in df.columns:
        if col.endswith("_x") or col.endswith("_y"):
            df[col] = df[col].astype(float)


    def set_segment_length(x_start, y_start, x_end, y_end, target_length, limit=False):
        dx = x_end - x_start
        dy = y_end - y_start
        current_length = np.sqrt(dx ** 2 + dy ** 2)
        if current_length == 0:
            return x_end, y_end  # pas de division par zéro
        factor = target_length / current_length
        if limit:
            factor = min(max(factor, 0.9), 1.1)
        new_x_end = x_start + dx * factor
        new_y_end = y_start + dy * factor
        return new_x_end, new_y_end

    # difference de longueur des jambes gauche/droite pour chaque frame
    leg_diff = []
    for i in range(len(df)):
        left_leg = np.sqrt(
            (df.at[i, 'LEFT_HIP_x'] - df.at[i, 'LEFT_ANKLE_x']) ** 2 +
            (df.at[i, 'LEFT_HIP_y'] - df.at[i, 'LEFT_ANKLE_y']) ** 2
        )
        right_leg = np.sqrt(
            (df.at[i, 'RIGHT_HIP_x'] - df.at[i, 'RIGHT_ANKLE_x']) ** 2 +
            (df.at[i, 'RIGHT_HIP_y'] - df.at[i, 'RIGHT_ANKLE_y']) ** 2
        )
        leg_diff.append(abs(left_leg - right_leg))
    leg_diff = np.array(leg_diff)

    # selection des frames avec les plus faibles différences de longueur
    reference_frames = np.argsort(leg_diff)[:num_ref_frames]

    segments = [
        ("SHOULDER", "ELBOW"),
        ("ELBOW", "WRIST"),
        ("HIP", "KNEE"),
        ("KNEE", "ANKLE"),
        ("ANKLE", "HEEL"),
        ("ANKLE", "FOOT_INDEX")
    ]

    # calcul des longueurs moyennes des segments côté gauche sur frames de référence
    lengths_ref = {}
    for seg in segments:
        start_x = df.loc[reference_frames, f'LEFT_{seg[0]}_x']
        start_y = df.loc[reference_frames, f'LEFT_{seg[0]}_y']
        end_x = df.loc[reference_frames, f'LEFT_{seg[1]}_x']
        end_y = df.loc[reference_frames, f'LEFT_{seg[1]}_y']
        dist = np.sqrt((start_x - end_x) ** 2 + (start_y - end_y) ** 2)
        lengths_ref[f"{seg[0]}_{seg[1]}"] = dist.mean(skipna=True)

    df_corrected = df.copy()

    # correction des longueurs pour toutes les frames (membre droit)
    for i in range(len(df_corrected)):
        for seg in segments:
            seg_name = f"{seg[0]}_{seg[1]}"
            start_pt = f"RIGHT_{seg[0]}"
            end_pt = f"RIGHT_{seg[1]}"

            x_start = df_corrected.at[i, f"{start_pt}_x"]
            y_start = df_corrected.at[i, f"{start_pt}_y"]
            x_end = df_corrected.at[i, f"{end_pt}_x"]
            y_end = df_corrected.at[i, f"{end_pt}_y"]

            new_x_end, new_y_end = set_segment_length(
                x_start, y_start, x_end, y_end,
                lengths_ref[seg_name],
                limit=limit
            )

            df_corrected.at[i, f"{end_pt}_x"] = new_x_end
            df_corrected.at[i, f"{end_pt}_y"] = new_y_end

    return df_corrected



#====================================================================
#------------------------------- MAIN -------------------------------
#====================================================================

# test pour comparer les veleurs avec le code R de Frederic

"""
df = pd.read_excel("C:/Users/m.brechenmacher/OneDrive - Institut/Documents/GaitAnalysis/correction_analyse_marche/AUDR_4kmh_reduced.xlsx")
df_correct_limb_lengths_stable = correct_limb_lengths_stable(df,limit = False)
df_correct_limb_lengths_stable['RIGHT_ELBOW_x'].head(10)

"""