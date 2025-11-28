import pandas as pd
from scipy.signal import savgol_filter
import re


def smooth_savgol(df, window_length=21, polyorder=2):
    
    """
    Applique un filtre de Savitzky-Golay pour lisser les coordonnées (x, y) des articulations.

    Ce filtre permet de réduire le bruit (données incohérentes ou parasites)
    en conservant la tendance globale du mouvement.

    Parameters
    ----------
    df : pandas.DataFrame
        Un DataFrame contenant des colonnes de coordonnées nommées sous la forme "<nom_segment>_x" ou "<nom_segment>_y"

    window_length : int, optional (default=21)
        Longueur de la fenêtre du filtre (doit être un entier impair supérieur à polyorder)

    polyorder : int, optional (default=2)
        Ordre du polynôme utilisé pour approximer la courbe locale

    Returns
    -------
    pandas.DataFrame
        Une copie du DataFrame original, avec les colonnes de coordonnées (x, y) lissées

    Notes
    -----
    - Les colonnes ne correspondant pas au motif `_(x|y)$` ne sont pas modifiées
    - Le filtre est appliqué indépendamment à chaque colonne concernée
    - Si window_length est trop grand par rapport au nombre de lignes, une erreur sera levée
    
    """

    df_sm = df.copy()
    
    for col in df.columns:
        if re.search(r'_(x|y)$', col):
            df_sm[col] = savgol_filter(df[col].astype(float), window_length=window_length, polyorder=polyorder)
    return df_sm


#====================================================================
#------------------------------- MAIN -------------------------------
#====================================================================

# test pour comparer les veleurs avec le code R de Frederic 

"""

df = pd.read_excel("C:/Users/m.brechenmacher/OneDrive - Institut/Documents/GaitAnalysis/correction_analyse_marche/AUDR_4kmh_reduced.xlsx") 
df_smooth = smooth_savgol(df, 21, 2)
df_smooth.head()

"""