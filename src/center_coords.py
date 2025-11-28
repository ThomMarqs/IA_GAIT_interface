# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np

def center_coords(df: pd.DataFrame) -> pd.DataFrame:
    """

    Recentre les coordonnees par rapport au barycentre du tronc (epaules et hanches), puis les normalise dans l'intervalle [-100,100]

    Parameters
    ----------
    df : pandas.DataFrame
        Un DataFrame contenant des colonnes de coordonnees nommees sous la forme "<nom_segment>_x", "<nom_segment>_y", "<nom_segment>_z"
        Les colonnes suivantes doivent obligatoirement etre presentes : 
        LEFT_SHOULDER_x, RIGHT_SHOULDER_x, LEFT_HIP_x, RIGHT_HIP_x

    Returns 
    -------
    pandas.DataFrame
        Une copue du DataFrame original avec les coordonnees centrees et normalisees.
        Les colonnes barycentriques temporaires sont supprimees avant le retour

    Raises 
    ------
    KeyError
        Si l'une des colonnes necessaires au calcul du barycentre est absente


    """


    df = df.copy()

    # calcul du barycentre sur x, y, z
    df['barycenter_x'] = df[['LEFT_SHOULDER_x', 'RIGHT_SHOULDER_x', 'LEFT_HIP_x', 'RIGHT_HIP_x']].mean(axis=1, skipna=True)
    df['barycenter_y'] = df[['LEFT_SHOULDER_y', 'RIGHT_SHOULDER_y', 'LEFT_HIP_y', 'RIGHT_HIP_y']].mean(axis=1, skipna=True)
    df['barycenter_z'] = df[['LEFT_SHOULDER_z', 'RIGHT_SHOULDER_z', 'LEFT_HIP_z', 'RIGHT_HIP_z']].mean(axis=1, skipna=True)

    # colonnes de coordonnees x,y,z
    coord_cols = [col for col in df.columns if col.endswith(('_x', '_y', '_z'))]

    # recentrer les coordonnees par rapport au barycentre
    for col in coord_cols:
        axis = col[-1] 
        df[col] = df[col] - df[f'barycenter_{axis}']

    # normaliser dans l'intervalle [-100, 100]
    all_coords = df[coord_cols].values.flatten()
    max_abs = np.nanmax(np.abs(all_coords))
    if max_abs != 0:
        df[coord_cols] = df[coord_cols] / max_abs * 100

    # supprimer les colonnes barycentre
    df.drop(columns=['barycenter_x', 'barycenter_y', 'barycenter_z'], inplace=True)

    return df



#====================================================================
#------------------------------- MAIN -------------------------------
#====================================================================

# test pour comparer les veleurs avec le code R de Frederic 

"""

df = pd.read_excel("C:/Users/m.brechenmacher/OneDrive - Institut/Documents/GaitAnalysis/correction_analyse_marche/AUDR_4kmh_reduced.xlsx") 

df_centered = center_coords(df)

coord_cols = [col for col in df_centered.columns if col.endswith(('_x', '_y', '_z'))]
max_val = np.nanmax(np.abs(df_centered[coord_cols].values))
print("Max abs after normalization (ca doit etre 100 ou proche):", round(max_val, 2))

df_centered.to_excel("center_coords.xlsx", index=False)

print("Avant centrage :")
print(df[["RIGHT_HIP_x", "RIGHT_HIP_y"]].head())

print("\nApres centrage :")
df_centered = center_coords(df)
print(df_centered[["RIGHT_HIP_x", "RIGHT_HIP_y"]].head())

"""