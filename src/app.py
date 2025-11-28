import dash
from dash import dcc, html
from dash.dependencies import Input, Output, State, ALL, MATCH
import plotly.graph_objs as go
from plotly.subplots import make_subplots

import pandas as pd
import numpy as np
import base64
import io
import time
import logging
from dataclasses import dataclass, field
import tempfile
import os

from dash import dash_table

logging.basicConfig(level=logging.INFO, format='[%(asctime)s] %(levelname)s: %(message)s')

from scipy.signal import butter, filtfilt
from scipy.interpolate import UnivariateSpline

# === Modules locaux ===
from gait_pipeline_core import (
    compute_joint_angles, detect_gait_cycles, extract_valid_cycles,
    compute_outcomes_on_valid_cycles, build_frame_to_cycle_map
)
from visualization import extract_coordinates, POSE_CONNECTIONS
from center_coords import center_coords
from smooth_savgol import smooth_savgol
from align_profile import align_profile
from correct_limb_lengths_stable import correct_limb_lengths_stable
from kalman import smooth_skeleton_kalman

from video_to_landmarks import extract_landmarks_from_video


# --------------------------------------------------------------------------------------
# ÉTAT GLOBAL — 2 SOURCES
# --------------------------------------------------------------------------------------

@dataclass
class SourceState:
    df_raw: pd.DataFrame | None = None               # DataFrame brut (Excel)
    df_proc: pd.DataFrame | None = None              # DataFrame filtré
    coords_proc: np.ndarray | None = None            # (T, J, 2) filtré (coords_global)
    coords_proc_vis: np.ndarray | None = None        # filtré aligné sur brut (pour visu)
    coords_raw_norm: np.ndarray | None = None        # brut tel que lu (normalisé / pixels)
    coords_raw_px: np.ndarray | None = None          # brut en pixels
    angles_proc: pd.DataFrame | None = None          # angles sur df_proc
    angles_raw: pd.DataFrame | None = None           # angles sur df_raw_pixels
    n_frames: int = 0                                # nombre de frames
    pipe_cache: dict = field(default_factory=lambda: {
        "events": None,
        "cycles": None,
        "outcomes_detailed": None,
        "outcomes_summary": None
    })


SOURCES: dict[str, SourceState] = {
    "1": SourceState(),
    "2": SourceState(),
}

# Filtres disponibles (ordre de l’UI = ordre d’application)
FILTER_OPTIONS = [
    'Center Coordinates', 'Align Profile', 'Correct Limb Lengths',
    'Savitzky-Golay', 'Butterworth', 'Moving Mean', 'Univariate Spline', 'Kalman', 'Aucun'
]

app = dash.Dash(__name__, suppress_callback_exceptions=True)
app.title = "IA GAIT - Analyse de la Marche"
server = app.server

# === Bornes fixes d'affichage ===
X_RANGE_FIXED = None
Y_RANGE_FIXED = None
ANGLE_RANGES = {"hip": None, "knee": None, "ankle": None}

# === Référentiel fixe (anti-zoom) ===
GLOBAL_CENTER = None
GLOBAL_SCALE = None

LEFT_SHOULDER, RIGHT_SHOULDER = 11, 12
LEFT_ANKLE, RIGHT_ANKLE = 27, 28
LEFT_HIP, LEFT_KNEE = 23, 25
RIGHT_HIP, RIGHT_KNEE = 24, 26


# --------------------------------------------------------------------------------------
# Helpers génériques
# --------------------------------------------------------------------------------------
def _coords_is_empty(coords):
    if coords is None:
        return True
    if isinstance(coords, np.ndarray):
        return coords.size == 0
    if isinstance(coords, (list, tuple)):
        return len(coords) == 0
    return True


def _iter_frames(coords):
    if coords is None:
        return []
    if isinstance(coords, np.ndarray):
        if coords.ndim == 3 and coords.shape[-1] == 2:
            return [coords[t] for t in range(coords.shape[0])]
        if coords.ndim == 2 and coords.shape[-1] == 2:
            return [coords]
        return []
    if isinstance(coords, (list, tuple)):
        return list(coords)
    return []


def _file_params(df: pd.DataFrame):
    W = H = F = None
    if 'width' in df.columns:
        w = pd.to_numeric(df['width'], errors='coerce').median()
        W = float(w) if pd.notna(w) else None
    if 'height' in df.columns:
        h = pd.to_numeric(df['height'], errors='coerce').median()
        H = float(h) if pd.notna(h) else None
    if 'fps' in df.columns:
        f = pd.to_numeric(df['fps'], errors='coerce').median()
        F = int(f) if pd.notna(f) else None
    return W, H, F


# --------- Détection & dénormalisation 0..1 / [-1..1] pour overlay brut ---------
def detect_raw_norm_mode(coords):
    if _coords_is_empty(coords):
        return None
    arr = coords.reshape(-1, 2) if isinstance(coords, np.ndarray) else np.vstack(
        [f.reshape(-1, 2) for f in _iter_frames(coords)]
    )
    amin, amax = float(np.nanmin(arr)), float(np.nanmax(arr))
    if amin >= -1.05 and amax <= 1.05:
        if amin < -0.05:
            return 'neg1_1'
        if amin >= -0.01 and amax <= 1.01:
            return '0_1'
    return None


def denormalize_raw_coords(coords, width, height, mode=None):
    if _coords_is_empty(coords):
        return coords
    W, H = float(width or 1080.0), float(height or 1920.0)
    mode = detect_raw_norm_mode(coords) if mode is None else mode
    if mode not in ('0_1', 'neg1_1'):
        return coords
    scale = np.array([W, H], dtype=float)
    if isinstance(coords, np.ndarray):
        out = coords.astype(float)
        return out * scale if mode == '0_1' else (out * 0.5 + 0.5) * scale
    return [
        (f.astype(float) * scale) if mode == '0_1'
        else ((f.astype(float) * 0.5 + 0.5) * scale)
        for f in _iter_frames(coords)
    ]


# --------- Alignement VISU (échelle + rotation + translation) ---------
def _median_segment_length(coords, i, j):
    if _coords_is_empty(coords):
        return np.nan
    d = []
    for f in _iter_frames(coords):
        if max(i, j) < len(f):
            p, q = f[i], f[j]
            if not np.any(np.isnan([*p, *q])):
                d.append(float(np.linalg.norm(p - q)))
    return float(np.median(d)) if d else np.nan


def _bbox_diag_median(coords):
    if _coords_is_empty(coords):
        return np.nan
    d = []
    for f in _iter_frames(coords):
        valid = ~np.isnan(f).any(axis=1)
        pts = f[valid]
        if pts.size:
            xmn, ymn = np.min(pts, axis=0)
            xmx, ymx = np.max(pts, axis=0)
            d.append(float(np.hypot(xmx - xmn, ymx - ymn)))
    return float(np.median(d)) if d else np.nan


def _estimate_scale_to_raw(filt, raw):
    pairs = [
        (LEFT_SHOULDER, LEFT_ANKLE),
        (RIGHT_SHOULDER, RIGHT_ANKLE),
        (LEFT_HIP, LEFT_KNEE),
        (RIGHT_HIP, RIGHT_KNEE)
    ]
    ratios = []
    for i, j in pairs:
        mf = _median_segment_length(filt, i, j)
        mr = _median_segment_length(raw, i, j)
        if np.isfinite(mf) and np.isfinite(mr) and mf > 0:
            ratios.append(mr / mf)
    if ratios:
        return float(np.median(ratios))
    mf = _bbox_diag_median(filt)
    mr = _bbox_diag_median(raw)
    return (mr / mf) if (np.isfinite(mf) and np.isfinite(mr) and mf > 0) else 1.0


def _scale_coords(coords, s):
    if _coords_is_empty(coords) or not np.isfinite(s) or s == 1.0:
        return coords
    if isinstance(coords, np.ndarray):
        return coords.astype(float) * float(s)
    return [f.astype(float) * float(s) for f in _iter_frames(coords)]


def _estimate_rotation_to_raw(filt_px, raw_px):
    """Retourne un angle (rad) médian pour aligner le filtré sur le brut. 0 si indéterminé."""
    if _coords_is_empty(filt_px) or _coords_is_empty(raw_px):
        return 0.0
    ang = []
    F = _iter_frames(filt_px)
    R = _iter_frames(raw_px)
    for t in range(min(len(F), len(R))):
        X, Y = F[t], R[t]
        m = min(len(X), len(Y))
        X, Y = X[:m], Y[:m]
        valid = (~np.isnan(X).any(axis=1)) & (~np.isnan(Y).any(axis=1))
        if np.sum(valid) < 3:
            continue
        Xc = X[valid] - np.median(X[valid], axis=0)
        Yc = Y[valid] - np.median(Y[valid], axis=0)
        H = Xc.T @ Yc
        try:
            U, S, Vt = np.linalg.svd(H)
            Rm = Vt.T @ U.T
            if np.linalg.det(Rm) < 0:
                Vt[-1, :] *= -1
                Rm = Vt.T @ U.T
            theta = float(np.arctan2(Rm[1, 0], Rm[0, 0]))
            if np.isfinite(theta):
                ang.append(theta)
        except Exception:
            continue
    return float(np.median(ang)) if ang else 0.0


def _rotate_coords(coords, theta):
    if _coords_is_empty(coords) or abs(theta) < 1e-8:
        return coords
    c, s = np.cos(theta), np.sin(theta)
    Rm = np.array([[c, -s], [s, c]], dtype=float)
    if isinstance(coords, np.ndarray):
        out = coords.astype(float).copy()
        out[...] = out @ Rm.T
        return out
    return [f.astype(float) @ Rm.T for f in _iter_frames(coords)]


def _estimate_translation_to_raw(filt_px, raw_px):
    if _coords_is_empty(filt_px) or _coords_is_empty(raw_px):
        return np.array([0.0, 0.0])
    d = []
    F = _iter_frames(filt_px)
    R = _iter_frames(raw_px)
    for t in range(min(len(F), len(R))):
        X, Y = F[t], R[t]
        m = min(len(X), len(Y))
        X, Y = X[:m], Y[:m]
        valid = (~np.isnan(X).any(axis=1)) & (~np.isnan(Y).any(axis=1))
        if not np.any(valid):
            continue
        cX = np.median(X[valid], axis=0)
        cY = np.median(Y[valid], axis=0)
        if np.all(np.isfinite(cX)) and np.all(np.isfinite(cY)):
            d.append(cY - cX)
    return np.median(np.vstack(d), axis=0).astype(float) if d else np.array([0.0, 0.0])


def _translate_coords(coords, delta):
    if _coords_is_empty(coords):
        return coords
    dx, dy = float(delta[0]), float(delta[1])
    if isinstance(coords, np.ndarray):
        out = coords.astype(float).copy()
        out[..., 0] += dx
        out[..., 1] += dy
        return out
    trans = np.array([dx, dy], dtype=float)
    return [f.astype(float) + trans for f in _iter_frames(coords)]


def _frame_anchor(frame: np.ndarray, strategy: str = 'pelvis') -> np.ndarray | None:
    """Retourne le point repère d'une frame.
    - 'pelvis' : milieu des hanches (LEFT_HIP, RIGHT_HIP)
    Fallback : milieu des épaules, puis médiane des points valides.
    """
    try:
        if frame is None or frame.size == 0:
            return None
        if strategy == 'pelvis' and max(LEFT_HIP, RIGHT_HIP) < len(frame):
            lh, rh = frame[LEFT_HIP], frame[RIGHT_HIP]
            if not np.any(np.isnan([*lh, *rh])):
                return (lh + rh) / 2.0
        if max(LEFT_SHOULDER, RIGHT_SHOULDER) < len(frame):
            ls, rs = frame[LEFT_SHOULDER], frame[RIGHT_SHOULDER]
            if not np.any(np.isnan([*ls, *rs])):
                return (ls + rs) / 2.0
        valid = ~np.isnan(frame).any(axis=1)
        pts = frame[valid]
        if pts.size:
            return np.median(pts, axis=0)
    except Exception:
        pass
    return None


def _median_anchor(coords, strategy: str = 'pelvis') -> np.ndarray | None:
    """Médiane du point repère sur l'ensemble des frames (robuste)."""
    if _coords_is_empty(coords):
        return None
    anchors = []
    for fr in _iter_frames(coords):
        a = _frame_anchor(fr, strategy)
        if a is not None and np.all(np.isfinite(a)):
            anchors.append(a)
    if anchors:
        return np.median(np.vstack(anchors), axis=0)
    return None


def _anchor_delta_to_raw(filt_px, raw_px, strategy: str = 'pelvis') -> np.ndarray:
    """Delta de translation pour faire coïncider le repère (filt → brut)."""
    a_f = _median_anchor(filt_px, strategy)
    a_r = _median_anchor(raw_px, strategy)
    if a_f is None or a_r is None:
        return np.array([np.nan, np.nan], dtype=float)
    return (a_r - a_f).astype(float)


def prepare_coords_for_visualization():
    """Construit coords_proc_vis pour chaque source en alignant le filtré sur le BRUT (pixels)."""
    for key, state in SOURCES.items():
        if _coords_is_empty(state.coords_proc):
            state.coords_proc_vis = state.coords_proc
            continue
        if _coords_is_empty(state.coords_raw_px):
            state.coords_proc_vis = state.coords_proc
            continue
        s = _estimate_scale_to_raw(state.coords_proc, state.coords_raw_px)
        filt_px = _scale_coords(state.coords_proc, s)
        theta = _estimate_rotation_to_raw(filt_px, state.coords_raw_px)
        filt_px = _rotate_coords(filt_px, theta)
        dxy = _anchor_delta_to_raw(filt_px, state.coords_raw_px)
        if not np.all(np.isfinite(dxy)):
            dxy = _estimate_translation_to_raw(filt_px, state.coords_raw_px)
        state.coords_proc_vis = _translate_coords(filt_px, dxy)


def compute_fixed_display_frame_stats():
    """Calcule un centre & une échelle communs (pixels) sur BRUT + FILTRÉ-VISU des 2 sources."""
    global GLOBAL_CENTER, GLOBAL_SCALE
    stacks = []
    for state in SOURCES.values():
        stacks += _iter_frames(state.coords_proc_vis)
        stacks += _iter_frames(state.coords_raw_px)
    if not stacks:
        GLOBAL_CENTER = np.array([0.0, 0.0])
        GLOBAL_SCALE = 1.0
        return
    all_pts = np.concatenate([f.reshape(-1, 2) for f in stacks], axis=0)
    valid = ~np.isnan(all_pts).any(axis=1)
    GLOBAL_CENTER = np.nanmean(all_pts[valid], axis=0) if np.any(valid) else np.array([0.0, 0.0])
    dists = []
    for f in stacks:
        if max(LEFT_SHOULDER, LEFT_ANKLE) < len(f):
            p1, p2 = f[LEFT_SHOULDER], f[LEFT_ANKLE]
            if not np.any(np.isnan([*p1, *p2])):
                d = float(np.linalg.norm(p1 - p2))
                if d > 0:
                    dists.append(d)
    if dists:
        GLOBAL_SCALE = float(np.median(dists))
    else:
        if np.any(valid):
            xmn, ymn = np.nanmin(all_pts[valid], axis=0)
            xmx, ymx = np.nanmax(all_pts[valid], axis=0)
            GLOBAL_SCALE = float(max(1e-6, np.hypot(xmx - xmn, ymx - ymn)))
        else:
            GLOBAL_SCALE = 1.0


def transform_pose_fixed(pts: np.ndarray) -> np.ndarray:
    if pts is None or pts.size == 0 or GLOBAL_CENTER is None or GLOBAL_SCALE in (None, 0):
        return pts
    return (pts - GLOBAL_CENTER) / GLOBAL_SCALE


def compute_fixed_ranges():
    """Calcule les ranges communs X/Y + ranges d'angles à partir des 2 sources."""
    global X_RANGE_FIXED, Y_RANGE_FIXED, ANGLE_RANGES
    compute_fixed_display_frame_stats()
    stacks = []
    for state in SOURCES.values():
        stacks += _iter_frames(state.coords_proc_vis)
        stacks += _iter_frames(state.coords_raw_px)
    if stacks:
        all_x, all_y = [], []
        for f in stacks:
            tf = transform_pose_fixed(f)
            all_x.append(tf[:, 0].ravel())
            all_y.append(tf[:, 1].ravel())
        vx = np.concatenate(all_x) if all_x else np.array([])
        vy = np.concatenate(all_y) if all_y else np.array([])
        vx, vy = vx[~np.isnan(vx)], vy[~np.isnan(vy)]
        if vx.size and vy.size:
            xmn, xmx = float(np.min(vx)), float(np.max(vx))
            ymn, ymx = float(np.min(vy)), float(np.max(vy))
            xm, ym = (xmx - xmn) * 0.15, (ymx - ymn) * 0.15
            X_RANGE_FIXED = [xmn - xm, xmx + xm]
            Y_RANGE_FIXED = [-(ymx + ym), -(ymn - ym)]

    def _angle_range(joint):
        series = []
        for state in SOURCES.values():
            if state.angles_proc is not None:
                for c in [f"{joint}_L", f"{joint}_R"]:
                    if c in state.angles_proc.columns:
                        series.append(state.angles_proc[c].values)
            if state.angles_raw is not None:
                for c in [f"{joint}_L", f"{joint}_R"]:
                    if c in state.angles_raw.columns:
                        series.append(state.angles_raw[c].values)
        if not series:
            return None
        arr = np.concatenate(series)
        arr = arr[~np.isnan(arr)]
        if arr.size == 0:
            return None
        mn, mx = float(np.min(arr)), float(np.max(arr))
        margin = (mx - mn) * 0.10 if mx > mn else 10.0
        return [mn - margin, mx + margin]

    ANGLE_RANGES["hip"] = _angle_range("hip")
    ANGLE_RANGES["knee"] = _angle_range("knee")
    ANGLE_RANGES["ankle"] = _angle_range("ankle")


# --------------------------------------------------------------------------------------
# Helpers « filtres UI »
# --------------------------------------------------------------------------------------
def _apply_on_xy(df, func):
    df = df.copy()
    cols = [c for c in df.columns if c.endswith('_x') or c.endswith('_y')]
    for c in cols:
        s = pd.to_numeric(df[c], errors='coerce')
        if s.isna().all():
            continue
        v = s.interpolate(limit_direction='both').bfill().ffill().to_numpy(float)
        try:
            df[c] = func(v)
        except Exception:
            df[c] = v
    return df


def _butterworth(df, order=2, cutoff=4.0, fs=120.0):
    nyq = 0.5 * float(fs)
    normal = float(cutoff) / nyq if nyq > 0 else 0.1
    normal = max(min(normal, 0.99), 1e-6)
    b, a = butter(int(order), normal, btype='low', analog=False)
    return _apply_on_xy(df, lambda v: filtfilt(b, a, v))


def _moving_mean(df, window=5):
    df = df.copy()
    cols = [c for c in df.columns if c.endswith('_x') or c.endswith('_y')]
    for c in cols:
        s = pd.to_numeric(df[c], errors='coerce')
        df[c] = (s.rolling(int(window), min_periods=1, center=True)
                 .mean()
                 .bfill()
                 .ffill())
    return df


def _savgol(df, window=21, polyorder=2):
    return smooth_savgol(df, window_length=int(window), polyorder=int(polyorder))


def _unispline(df, s=0.5):
    def smooth_series(v):
        x = np.arange(len(v), dtype=float)
        try:
            return UnivariateSpline(x, v, s=float(s))(x)
        except Exception:
            return v

    return _apply_on_xy(df, smooth_series)


def _kalman_dispatch(df: pd.DataFrame):
    points = sorted({c[:-2] for c in df.columns if c.endswith('_x') and (f"{c[:-2]}_y" in df.columns)})
    try:
        return smooth_skeleton_kalman(df)
    except TypeError:
        try:
            return smooth_skeleton_kalman(df, points)
        except TypeError:
            try:
                return smooth_skeleton_kalman(df, points, 1e-5, 1e-2)
            except TypeError as e3:
                raise TypeError("smooth_skeleton_kalman() signature non supportée.") from e3


def _ensure_pixels(
    df_raw_in: pd.DataFrame,
    width_override: float | int | None = None,
    height_override: float | int | None = None,
    prefer_ui: bool = False
) -> pd.DataFrame:
    df = df_raw_in.copy()

    xcols = [c for c in df.columns if c.endswith('_x')]
    ycols = [c for c in df.columns if c.endswith('_y')]
    if not xcols or not ycols:
        return df

    vals = pd.concat([df[xcols].stack(), df[ycols].stack()], axis=0)
    vals = pd.to_numeric(vals, errors='coerce')
    vmin, vmax = float(vals.min(skipna=True)), float(vals.max(skipna=True))
    is_norm = np.isfinite(vmin) and np.isfinite(vmax) and (vmin >= -1.05) and (vmax <= 1.05)
    if not is_norm:
        return df  # déjà en pixels

    # Valeurs du fichier
    Wf = Hf = None
    if 'width' in df.columns and 'height' in df.columns:
        Wf = pd.to_numeric(df['width'], errors='coerce').bfill().ffill().median()
        Hf = pd.to_numeric(df['height'], errors='coerce').bfill().ffill().median()
        Wf = float(Wf) if pd.notna(Wf) else None
        Hf = float(Hf) if pd.notna(Hf) else None

    # Priorité
    if prefer_ui:
        W = float(width_override) if width_override else (Wf if Wf is not None else None)
        H = float(height_override) if height_override else (Hf if Hf is not None else None)
    else:
        W = Wf if Wf is not None else (float(width_override) if width_override else None)
        H = Hf if Hf is not None else (float(height_override) if height_override else None)

    # Défauts
    if W is None:
        W = 1080.0
    if H is None:
        H = 1920.0

    zero_one_like = (vmin >= -0.01) and (vmax <= 1.01)
    for c in xcols:
        v = pd.to_numeric(df[c], errors='coerce').to_numpy(dtype=float)
        df[c] = (v * W) if zero_one_like else ((v * 0.5 + 0.5) * W)
    for c in ycols:
        v = pd.to_numeric(df[c], errors='coerce').to_numpy(dtype=float)
        df[c] = (v * H) if zero_one_like else ((v * 0.5 + 0.5) * H)
    return df


def _effective_fs(df: pd.DataFrame, fs_from_ui: float | None) -> float:
    # 1) UI si dispo
    if fs_from_ui and fs_from_ui > 0:
        return float(int(round(fs_from_ui)))
    # 2) Fichier sinon
    if 'fps' in df.columns and pd.notna(df['fps']).any():
        f = pd.to_numeric(df['fps'], errors='coerce').median()
        if pd.notna(f) and f > 0:
            return float(int(round(f)))
    # 3) Défaut
    return 30.0


def apply_ui_filters_pipeline(
    df_raw_in: pd.DataFrame,
    filter_types,
    filter_params_children,
    fs_from_ui: float | None,
    width_override: float | int | None = None,
    height_override: float | int | None = None,
    prefer_ui: bool = False
):
    df = _ensure_pixels(
        df_raw_in,
        width_override=width_override,
        height_override=height_override,
        prefer_ui=prefer_ui
    )
    fs = _effective_fs(df_raw_in, fs_from_ui)

    for i, ftype in enumerate(filter_types or []):
        if not ftype or ftype == 'Aucun':
            continue
        params_i = {}
        try:
            children = (filter_params_children or [])[i]
            if isinstance(children, list):
                for child in children:
                    props = child.get('props', {}) if isinstance(child, dict) else {}
                    cid = props.get('id', {})
                    if isinstance(cid, dict) and cid.get('type') == 'filter-param':
                        k = cid.get('param')
                        v = props.get('value', None)
                        if k is not None:
                            params_i[k] = v
        except Exception:
            pass

        if ftype == 'Center Coordinates':
            df = center_coords(df)
        elif ftype == 'Align Profile':
            df = align_profile(df)
        elif ftype == 'Correct Limb Lengths':
            df = correct_limb_lengths_stable(df)
        elif ftype == 'Savitzky-Golay':
            df = _savgol(df, window=params_i.get('window', 21), polyorder=params_i.get('polyorder', 2))
        elif ftype == 'Butterworth':
            df = _butterworth(df, order=params_i.get('order', 2), cutoff=params_i.get('cutoff', 4.0), fs=fs)
        elif ftype == 'Moving Mean':
            df = _moving_mean(df, window=params_i.get('window', 5))
        elif ftype == 'Univariate Spline':
            df = _unispline(df, s=params_i.get('s', 0.5))
        elif ftype == 'Kalman':
            df = _kalman_dispatch(df)
    return df


# --------------------------------------------------------------------------------------
# UI
# --------------------------------------------------------------------------------------
header_block = html.Div([
    html.H1(
        "IA GAIT - Analyse de la Marche",
        style={
            'textAlign': 'center',
            'marginTop': '20px',
            'color': '#2c3e50',
            'fontWeight': 'bold'
        }
    )
])

upload_block = html.Div([

    # ========= SOURCE 1 =========
    html.Div([
        html.H3("Source 1", style={'marginBottom': '10px'}),

        # --- Boutons en ligne ---
        html.Div([
            dcc.Upload(
                id='upload-data-1',
                children=html.Button(
                    "Importer un fichier Excel",
                    style={
                        'backgroundColor': '#3498db',
                        'color': 'white',
                        'padding': '10px 20px',
                        'borderRadius': '8px',
                        'fontSize': '16px',
                        'cursor': 'pointer',
                    }
                ),
                multiple=False,
                accept='.xlsx',
                style={'marginRight': '10px'}
            ),

            dcc.Upload(
                id='upload-video-1',
                children=html.Button(
                    "Importer une vidéo (MP4/MOV)",
                    style={
                        'backgroundColor': '#8e44ad',
                        'color': 'white',
                        'padding': '10px 20px',
                        'borderRadius': '8px',
                        'fontSize': '16px',
                        'cursor': 'pointer',
                    }
                ),
                multiple=False,
                accept='video/mp4,video/quicktime'
            ),
        ], style={
            'display': 'flex',
            'flexDirection': 'row',
            'marginBottom': '10px'
        }),

        html.Div(
            id='video-conversion-status-1',
            style={'marginTop': '5px', 'color': '#8e44ad'}
        ),
    ], style={'flex': 1, 'paddingRight': '10px'}),


    # ========= SOURCE 2 =========
    html.Div([
        html.H3("Source 2", style={'marginBottom': '10px'}),

        # --- Boutons en ligne ---
        html.Div([
            dcc.Upload(
                id='upload-data-2',
                children=html.Button(
                    "Importer un fichier Excel",
                    style={
                        'backgroundColor': '#3498db',
                        'color': 'white',
                        'padding': '10px 20px',
                        'borderRadius': '8px',
                        'fontSize': '16px',
                        'cursor': 'pointer',
                    }
                ),
                multiple=False,
                accept='.xlsx',
                style={'marginRight': '10px'}
            ),

            dcc.Upload(
                id='upload-video-2',
                children=html.Button(
                    "Importer une vidéo (MP4/MOV)",
                    style={
                        'backgroundColor': '#8e44ad',
                        'color': 'white',
                        'padding': '10px 20px',
                        'borderRadius': '8px',
                        'fontSize': '16px',
                        'cursor': 'pointer',
                    }
                ),
                multiple=False,
                accept='video/mp4,video/quicktime'
            ),
        ], style={
            'display': 'flex',
            'flexDirection': 'row',
            'marginBottom': '10px'
        }),

        html.Div(
            id='video-conversion-status-2',
            style={'marginTop': '5px', 'color': '#8e44ad'}
        ),
    ], style={'flex': 1, 'paddingLeft': '10px'}),

], style={
    'marginBottom': 30,
    'padding': '20px',
    'backgroundColor': '#ecf0f1',
    'borderRadius': '12px',
    'display': 'flex',
    'flexDirection': 'row'
})

filters_block = html.Div([
    html.H3("Choix du nombre de filtres", style={'marginTop': '10px'}),
    dcc.Slider(
        id='filter-count',
        min=1,
        max=8,
        step=1,
        value=3,
        marks={i: str(i) for i in range(1, 9)},
        tooltip={"placement": "bottom", "always_visible": True}
    ),
    html.Div(id='filters-container', style={'marginTop': '20px'}),

    html.Label("Source à filtrer", style={'marginTop': '15px', 'fontWeight': 'bold'}),
    dcc.RadioItems(
        id='filter-target',
        options=[
            {'label': 'Source 1', 'value': '1'},
            {'label': 'Source 2', 'value': '2'},
            {'label': 'Les deux', 'value': 'both'}
        ],
        value='both',
        labelStyle={'display': 'inline-block', 'marginRight': '15px', 'marginTop': '5px'}
    ),

    html.Div([
        html.Button(
            "📊 Appliquer les filtres/paramètres",
            id='apply-filters',
            n_clicks=0,
            style={
                'marginTop': '20px',
                'backgroundColor': '#27ae60',
                'color': 'white',
                'padding': '10px 20px',
                'border': 'none',
                'borderRadius': '8px',
                'fontSize': '16px',
                'cursor': 'pointer'
            }
        ),
        html.Button(
            "⚙️ Paramètres",
            id='toggle-params-btn',
            n_clicks=0,
            style={
                'marginTop': '20px',
                'marginLeft': '15px',
                'backgroundColor': '#f39c12',
                'color': 'white',
                'padding': '10px 20px',
                'border': 'none',
                'borderRadius': '8px',
                'fontSize': '16px',
                'cursor': 'pointer'
            }
        ),
    ])
], style={
    'marginBottom': 30,
    'padding': '20px',
    'backgroundColor': '#ecf0f1',
    'borderRadius': '12px'
})

display_block = html.Div([
    html.H3("Paramètres des options d'affichage"),
    dcc.Checklist(
        id='display-options',
        options=[
            {'label': 'Afficher squelette non filtré', 'value': 'raw_skel'},
            {'label': 'Afficher angles non filtrés', 'value': 'raw_angles'},
            {'label': 'Afficher gauche', 'value': 'left'},
            {'label': 'Afficher droite', 'value': 'right'}
        ],
        value=['left', 'right'],
        labelStyle={'display': 'inline-block', 'marginRight': '15px', 'marginBottom': '10px'}
    ),

    html.Div([
        html.Div([
            html.Label("Frame Source 1"),
            dcc.Slider(
                id='frame-slider-1',
                min=0, max=0, step=1, value=0,
                marks={0: '0'},
                tooltip={"placement": "bottom", "always_visible": True}
            )
        ], style={'flex': 1, 'marginRight': '10px'}),

        html.Div([
            html.Label("Frame Source 2"),
            dcc.Slider(
                id='frame-slider-2',
                min=0, max=0, step=1, value=0,
                marks={0: '0'},
                tooltip={"placement": "bottom", "always_visible": True}
            )
        ], style={'flex': 1, 'marginLeft': '10px'}),
    ], style={'display': 'flex', 'marginTop': '10px'}),

    html.Div([
        html.Button(
            "▶️ Lancer (S1 + S2)",
            id='play-btn-1',
            n_clicks=0,
            style={
                'backgroundColor': '#2ecc71',
                'color': 'white',
                'marginRight': '10px',
                'border': 'none',
                'borderRadius': '5px',
                'padding': '8px 16px'
            }
        ),
        html.Button(
            "⏹️ Stop",
            id='stop-btn-1',
            n_clicks=0,
            style={
                'backgroundColor': '#e74c3c',
                'color': 'white',
                'border': 'none',
                'borderRadius': '5px',
                'padding': '8px 16px',
                'marginRight': '15px'
            }
        ),
        html.Button("⏮️ Précédent (S1)", id='prev-frame-1', n_clicks=0, style={'marginRight': '10px'}),
        html.Button("⏭️ Suivant (S1)", id='next-frame-1', n_clicks=0, style={'marginRight': '20px'})
    ], style={'marginTop': '15px'})
], style={
    'marginBottom': 30,
    'padding': '20px',
    'backgroundColor': '#ecf0f1',
    'borderRadius': '12px'
})

graphs_block = html.Div([
    html.Div([
        dcc.Graph(id='skeleton-graph', style={'height': '900px', 'width': '100%'})
    ], style={'flex': '1', 'minWidth': '500px', 'paddingRight': '20px'}),

    html.Div([
        dcc.Graph(id='hip-angle-graph', style={'height': '300px', 'width': '100%'}),
        dcc.Graph(id='knee-angle-graph', style={'height': '300px', 'width': '100%'}),
        dcc.Graph(id='ankle-angle-graph', style={'height': '300px', 'width': '100%'})
    ], style={
        'flex': '1',
        'minWidth': '450px',
        'display': 'flex',
        'flexDirection': 'column',
        'justifyContent': 'center',
        'alignItems': 'center'
    })
], style={
    'display': 'flex',
    'gap': '20px',
    'alignItems': 'flex-start',
    'marginBottom': '40px'
})


angles_selector_block = html.Div([

    html.H4(
        "Sélection des angles pour l’analyse des cycles",
        style={'marginTop': '25px', 'marginBottom': '10px', 'fontWeight': 'bold', 'color': '#2c3e50'}
    ),

    dcc.Checklist(
        id='angle-selection',
        options=[
            {'label': 'Hanche gauche (hip_L)', 'value': 'hip_L'},
            {'label': 'Hanche droite (hip_R)', 'value': 'hip_R'},
            {'label': 'Genou gauche (knee_L)', 'value': 'knee_L'},
            {'label': 'Genou droite (knee_R)', 'value': 'knee_R'},
            {'label': 'Cheville gauche (ankle_L)', 'value': 'ankle_L'},
            {'label': 'Cheville droite (ankle_R)', 'value': 'ankle_R'}
        ],
        value=['hip_L', 'hip_R', 'knee_L', 'knee_R', 'ankle_L', 'ankle_R'],
        labelStyle={'display': 'inline-block', 'marginBottom': '5px'}
    ),

    html.H4(
        "Mode d’affichage des cycles",
        style={'marginTop': '25px', 'marginBottom': '10px', 'fontWeight': 'bold', 'color': '#2c3e50'}
    ),

    dcc.RadioItems(
        id='cycle-display-mode',
        options=[
            {'label': 'Moyenne ± Écart-type', 'value': 'mean_sd'},
            {'label': 'Tous les cycles', 'value': 'all_cycles'}
        ],
        value='mean_sd',
        labelStyle={'display': 'block', 'marginBottom': '5px'}
    )

], style={
    'marginBottom': '30px',
    'padding': '20px',
    'backgroundColor': '#ecf0f1',
    'borderRadius': '12px',
    'width': '90%',
    'margin': 'auto'
})

# Deux blocs cycles
cycles_block = html.Div([
    html.H3(
        "Cinématique des membres inférieurs – S1 & S2",
        style={
            'marginBottom': '10px',
            'color': '#2c3e50',
            'fontWeight': 'bold'
        }
    ),

    html.Button(
        "Lancer l'analyse",
        id='analyze-cycles-both',
        n_clicks=0,
        style={
            'backgroundColor': '#8e44ad',
            'color': 'white',
            'padding': '10px 20px',
            'borderRadius': '8px',
            'marginBottom': '20px',
            'cursor': 'pointer',
            'fontSize': '16px',
            'border': 'none',
            'fontWeight': 'bold'
        }
    ),

    dcc.Graph(
        id='normalized-cycle-plot-both',
        style={'height': '600px'}
    )

], style={
    'width': '90%',
    'margin': 'auto',
    'marginBottom': '30px',
    'padding': '20px',
    'backgroundColor': '#ecf0f1',
    'borderRadius': '12px'
})

# Deux blocs spatio-temporels
spatiotemp_block_1 = html.Div([
    html.H3("Paramètres spatio-temporels"),
    html.Button(
        "Lancer l'analyse S1",
        id='extract-gait-btn-1',
        n_clicks=0,
        style={
            'backgroundColor': '#2980b9',
            'color': 'white',
            'padding': '10px 20px',
            'borderRadius': '8px',
            'marginBottom': '20px'
        }
    ),
html.Button(
    "Exporter les données S1",
    id="export-xlsx-1",
    n_clicks=0,
    style={
        'backgroundColor': '#16a085',
        'color': 'white',
        'padding': '10px 20px',
        'borderRadius': '8px',
        'marginBottom': '20px',
        'cursor': 'pointer'
    }
),
dcc.Download(id="download-xlsx-1"),

dcc.Download(id='download-gait-1'),
dcc.Download(id='download-angles-1'),

    html.Div(id='gait-table-container-1')
], style={'width': '90%', 'margin': 'auto', 'marginBottom': '30px'})

spatiotemp_block_2 = html.Div([
    html.H3("Paramètres spatio-temporels"),
    html.Button(
        "Lancer l'analyse S2",
        id='extract-gait-btn-2',
        n_clicks=0,
        style={
            'backgroundColor': '#2980b9',
            'color': 'white',
            'padding': '10px 20px',
            'borderRadius': '8px',
            'marginBottom': '20px'
        }
    ),
    html.Button(
    "Exporter les données S2",
    id="export-xlsx-2",
    n_clicks=0,
    style={
        'backgroundColor': '#16a085',
        'color': 'white',
        'padding': '10px 20px',
        'borderRadius': '8px',
        'marginBottom': '20px',
        'cursor': 'pointer'
    }
),
dcc.Download(id="download-xlsx-2"),

    html.Div(id='gait-table-container-2')
], style={'width': '90%', 'margin': 'auto', 'marginBottom': '40px'})

debug_block = html.Div([
    html.H3("🛠️ Debug Log"),
    html.Pre(
        id='debug-output',
        style={
            'whiteSpace': 'pre-wrap',
            'color': '#c0392b',
            'backgroundColor': '#f5f5f5',
            'padding': '10px',
            'borderRadius': '5px',
            'fontSize': '14px'
        }
    )
], style={'width': '90%', 'margin': 'auto'})

sidebar_block = html.Div([
    html.Div([
        html.H3("⚙️ Paramètres", style={'marginBottom': '20px', 'fontWeight': 'bold'}),
        html.Label("Framerate (Hz)"),
        dcc.Input(
            id='input-framerate',
            type='number',
            value=120,
            step=1,
            style={'width': '100%', 'marginBottom': '12px'}
        ),
        html.Label("Longueur du fémur (cm)"),
        dcc.Input(
            id='input-femur',
            type='number',
            value=44.0,
            step=0.1,
            style={'width': '100%', 'marginBottom': '12px'}
        ),
        html.Label("Largeur image (px)"),
        dcc.Input(
            id='input-width',
            type='number',
            value=1080,
            step=1,
            style={'width': '100%', 'marginBottom': '12px'}
        ),
        html.Label("Hauteur image (px)"),
        dcc.Input(
            id='input-height',
            type='number',
            value=1920,
            step=1,
            style={'width': '100%', 'marginBottom': '12px'}
        ),
        html.Button(
            "❌ Fermer",
            id='close-params-btn',
            n_clicks=0,
            style={
                'backgroundColor': '#e74c3c',
                'color': 'white',
                'padding': '8px 16px',
                'border': 'none',
                'borderRadius': '6px',
                'width': '100%',
                'marginTop': '10px'
            }
        )
    ])
], id='params-sidebar', style={
    'position': 'fixed',
    'top': '0',
    'left': '0',
    'width': '300px',
    'height': '100%',
    'backgroundColor': '#ecf0f1',
    'boxShadow': '2px 0 5px rgba(0,0,0,0.2)',
    'zIndex': '1000',
    'padding': '20px',
    'overflowY': 'auto',
    'transition': 'transform 0.3s ease-in-out',
    'transform': 'translateX(-100%)'
})

app.layout = html.Div([
    header_block,
    upload_block,
    filters_block,
    display_block,
    dcc.Interval(id='play-interval-1', interval=1000 / 30, n_intervals=0, disabled=True),
    dcc.Store(id='sidebar-open', data=False),
    dcc.Store(id='debug-log', data=""),
    dcc.Store(id='refresh-visuals', data=0),

    graphs_block,
    angles_selector_block,
    # cycles_block_1,
    # cycles_block_2,
    cycles_block,
    spatiotemp_block_1,
    spatiotemp_block_2,
    debug_block,
    sidebar_block
])


# --------------------------------------------------------------------------------------
# Callbacks : vidéo → Excel
# --------------------------------------------------------------------------------------
def _handle_video_to_excel(video_content, video_filename):
    if video_content is None:
        raise dash.exceptions.PreventUpdate
    header, encoded = video_content.split(',')
    video_bytes = base64.b64decode(encoded)
    temp_video_path = os.path.join(tempfile.gettempdir(), video_filename)
    with open(temp_video_path, 'wb') as f:
        f.write(video_bytes)
    temp_excel_path = os.path.join(
        tempfile.gettempdir(),
        f"landmarks_from_{os.path.splitext(video_filename)[0]}.xlsx"
    )
    extract_landmarks_from_video(temp_video_path, temp_excel_path)
    with open(temp_excel_path, 'rb') as f:
        excel_bytes = f.read()
        excel_b64 = base64.b64encode(excel_bytes).decode('utf-8')
    excel_content = (
        "data:application/vnd.openxmlformats-officedocument.spreadsheetml.sheet;base64," +
        excel_b64
    )
    excel_filename = os.path.basename(temp_excel_path)
    return excel_content, excel_filename, f"✔ Conversion terminée : {excel_filename}"


@app.callback(
    Output('upload-data-1', 'contents'),
    Output('upload-data-1', 'filename'),
    Output('video-conversion-status-1', 'children'),
    Input('upload-video-1', 'contents'),
    State('upload-video-1', 'filename'),
    prevent_initial_call=True
)
def handle_video_upload_1(video_content, video_filename):
    try:
        return _handle_video_to_excel(video_content, video_filename)
    except Exception as e:
        return None, None, f"❌ Erreur conversion (Source 1) : {e}"


@app.callback(
    Output('upload-data-2', 'contents'),
    Output('upload-data-2', 'filename'),
    Output('video-conversion-status-2', 'children'),
    Input('upload-video-2', 'contents'),
    State('upload-video-2', 'filename'),
    prevent_initial_call=True
)
def handle_video_upload_2(video_content, video_filename):
    try:
        return _handle_video_to_excel(video_content, video_filename)
    except Exception as e:
        return None, None, f"❌ Erreur conversion (Source 2) : {e}"


# --------------------------------------------------------------------------------------
# Callbacks : UI des filtres
# --------------------------------------------------------------------------------------
@app.callback(
    Output({'type': 'filter-params', 'index': MATCH}, 'children'),
    Input({'type': 'filter-type', 'index': MATCH}, 'value')
)
def update_filter_params(filter_type):
    if not filter_type or filter_type in ['Center Coordinates', 'Align Profile', 'Correct Limb Lengths', 'Aucun']:
        return []
    if filter_type == 'Butterworth':
        return [
            html.Label("Order"),
            dcc.Input(id={'param': 'order', 'type': 'filter-param'}, type='number', value=2),
            html.Label("Cutoff (Hz)"),
            dcc.Input(id={'param': 'cutoff', 'type': 'filter-param'}, type='number', value=4.0)
        ]
    if filter_type == 'Savitzky-Golay':
        return [
            html.Label("Window Length"),
            dcc.Input(id={'param': 'window', 'type': 'filter-param'}, type='number', value=21),
            html.Label("Polynomial Order"),
            dcc.Input(id={'param': 'polyorder', 'type': 'filter-param'}, type='number', value=2)
        ]
    if filter_type == 'Moving Mean':
        return [
            html.Label("Window Size"),
            dcc.Input(id={'param': 'window', 'type': 'filter-param'}, type='number', value=5)
        ]
    if filter_type == 'Univariate Spline':
        return [
            html.Label("Smoothing Factor (s)"),
            dcc.Input(id={'param': 's', 'type': 'filter-param'}, type='number', value=0.5)
        ]
    if filter_type == 'Kalman':
        return []
    return []


# --------------------------------------------------------------------------------------
# Traitement d'une source
# --------------------------------------------------------------------------------------
def process_source_from_excel_content(
    source_key: str,
    contents: str,
    filename: str,
    filter_types,
    filter_params_children,
    ui_width: int | None,
    ui_height: int | None,
    ui_fps: float | None,
    ui_femur: float
):
    """Décodage Excel, prétraitement, angles, events, cycles, outcomes pour une source."""
    global SOURCES

    t0 = time.time()
    state = SOURCES[source_key]

    if contents is None:
        return 0, f"\n⚠️ Aucun contenu pour la source {source_key}.", ui_width, ui_height, ui_fps

    # Décodage Excel
    try:
        _, content_string = contents.split(',')
    except ValueError:
        return 0, f"\n❌ Contenu Excel invalide pour la source {source_key}.", ui_width, ui_height, ui_fps

    decoded = base64.b64decode(content_string)
    df_raw = pd.read_excel(io.BytesIO(decoded))
    state.df_raw = df_raw

    # Paramètres du fichier
    Wf, Hf, Ff = _file_params(df_raw)

    # Dimensions pour la dénormalisation / filtres
    W_px = ui_width or Wf or 1080
    H_px = ui_height or Hf or 1920

    # Pipeline filtrée
    df_proc = apply_ui_filters_pipeline(
        df_raw,
        filter_types,
        filter_params_children,
        ui_fps,
        width_override=W_px,
        height_override=H_px,
        prefer_ui=True
    )
    state.df_proc = df_proc

    # Coords filtrées
    coords_proc = extract_coordinates(df_proc)
    state.coords_proc = coords_proc
    state.n_frames = coords_proc.shape[0] if coords_proc is not None else 0

    # Coords brutes
    coords_raw_norm = extract_coordinates(df_raw)
    state.coords_raw_norm = coords_raw_norm
    state.coords_raw_px = denormalize_raw_coords(coords_raw_norm, W_px, H_px)

    # Angles
    state.angles_proc = compute_joint_angles(df_proc, return_dataframe=True)
    df_raw_px = _ensure_pixels(
        df_raw,
        width_override=W_px,
        height_override=H_px,
        prefer_ui=True
    )
    state.angles_raw = compute_joint_angles(df_raw_px, return_dataframe=True)

    # Gait events / cycles / outcomes
    fps_eff = _effective_fs(df_raw, ui_fps)
    events = detect_gait_cycles(df_proc, framerate=fps_eff, cutoff=5)
    cycles = extract_valid_cycles(df_proc, events, framerate=fps_eff)

    angles_df = state.angles_proc.copy()
    angles_df.insert(0, 'Cycle', build_frame_to_cycle_map(cycles, len(df_proc)))
    outcomes_summary, outcomes_detailed = compute_outcomes_on_valid_cycles(
        df_proc,
        cycles,
        events,
        framerate=fps_eff,
        femur_length_cm=ui_femur,
        angles_df=angles_df
    )
    state.pipe_cache.update({
        "events": events,
        "cycles": cycles,
        "outcomes_detailed": outcomes_detailed,
        "outcomes_summary": outcomes_summary
    })

    msg = (
        f"\n📁 Source {source_key} chargée : {filename} | "
        f"Frames : {state.n_frames} | width={W_px} height={H_px} fps={fps_eff} "
        f"⏱️ {time.time() - t0:.3f} s"
    )
    return state.n_frames, msg, W_px, H_px, fps_eff


def reprocess_source_with_filters(
    source_key: str,
    filter_types,
    filter_params_children,
    ui_width: int | None,
    ui_height: int | None,
    ui_fps: float | None,
    ui_femur: float
):
    """Réapplique les filtres à une source déjà chargée (df_raw déjà présent)."""
    global SOURCES

    state = SOURCES[source_key]
    if state.df_raw is None:
        return 0, f"\n⚠️ Source {source_key} non chargée, impossible de réappliquer les filtres.", ui_width, ui_height, ui_fps

    t0 = time.time()

    df_raw = state.df_raw

    # Paramètres du fichier
    Wf, Hf, Ff = _file_params(df_raw)

    W_px = ui_width or Wf or 1080
    H_px = ui_height or Hf or 1920

    df_proc = apply_ui_filters_pipeline(
        df_raw,
        filter_types,
        filter_params_children,
        ui_fps,
        width_override=W_px,
        height_override=H_px,
        prefer_ui=True
    )
    state.df_proc = df_proc

    coords_proc = extract_coordinates(df_proc)
    state.coords_proc = coords_proc
    state.n_frames = coords_proc.shape[0] if coords_proc is not None else 0

    coords_raw_norm = extract_coordinates(df_raw)
    state.coords_raw_norm = coords_raw_norm
    state.coords_raw_px = denormalize_raw_coords(coords_raw_norm, W_px, H_px)

    state.angles_proc = compute_joint_angles(df_proc, return_dataframe=True)
    df_raw_px = _ensure_pixels(
        df_raw,
        width_override=W_px,
        height_override=H_px,
        prefer_ui=True
    )
    state.angles_raw = compute_joint_angles(df_raw_px, return_dataframe=True)

    fps_eff = _effective_fs(df_raw, ui_fps)
    events = detect_gait_cycles(df_proc, framerate=fps_eff, cutoff=5)
    cycles = extract_valid_cycles(df_proc, events, framerate=fps_eff)

    angles_df = state.angles_proc.copy()
    angles_df.insert(0, 'Cycle', build_frame_to_cycle_map(cycles, len(df_proc)))
    outcomes_summary, outcomes_detailed = compute_outcomes_on_valid_cycles(
        df_proc,
        cycles,
        events,
        framerate=fps_eff,
        femur_length_cm=ui_femur,
        angles_df=angles_df
    )
    state.pipe_cache.update({
        "events": events,
        "cycles": cycles,
        "outcomes_detailed": outcomes_detailed,
        "outcomes_summary": outcomes_summary
    })

    msg = (
        f"\n[Apply] Source {source_key} | width={W_px} height={H_px} "
        f"fps={fps_eff} | frames={state.n_frames} "
        f"| temps={time.time() - t0:.3f}s"
    )
    return state.n_frames, msg, W_px, H_px, fps_eff


# --------------------------------------------------------------------------------------
# Callback principal : upload + filtres pour les 2 sources
# --------------------------------------------------------------------------------------
@app.callback(
    Output('frame-slider-1', 'max'),
    Output('frame-slider-2', 'max'),
    Output('debug-log', 'data'),
    Output('filters-container', 'children'),
    Output('input-width', 'value'),
    Output('input-height', 'value'),
    Output('input-framerate', 'value'),
    Output('refresh-visuals', 'data'),
    Input('upload-data-1', 'contents'),
    Input('upload-data-2', 'contents'),
    Input('apply-filters', 'n_clicks'),
    Input('filter-count', 'value'),
    Input('filter-target', 'value'),
    State('upload-data-1', 'filename'),
    State('upload-data-2', 'filename'),
    State({'type': 'filter-type', 'index': ALL}, 'value'),
    State({'type': 'filter-params', 'index': ALL}, 'children'),
    State('debug-log', 'data'),
    State('input-width', 'value'),
    State('input-height', 'value'),
    State('input-framerate', 'value'),
    State('input-femur', 'value'),
    prevent_initial_call=True
)
def handle_all_inputs(
    contents1, contents2, n_clicks_apply, filter_count, filter_target,
    filename1, filename2, filter_types, filter_params_children,
    debug_log, ui_width, ui_height, ui_fps, ui_femur
):
    ctx = dash.callback_context
    if not ctx.triggered:
        raise dash.exceptions.PreventUpdate
    trigger = ctx.triggered[0]['prop_id'].split('.')[0]

    # UI dynamique des filtres
    filters_ui = [
        html.Div([
            html.Label(f"Étape {i + 1}"),
            dcc.Dropdown(
                id={'type': 'filter-type', 'index': i},
                options=[{'label': f, 'value': f} for f in FILTER_OPTIONS],
                placeholder=f"Choisir filtre {i + 1}"
            ),
            html.Div(id={'type': 'filter-params', 'index': i})
        ], style={'marginBottom': '20px'})
        for i in range(filter_count)
    ]

    # Si on change juste le nombre de filtres
    if trigger == 'filter-count':
        return dash.no_update, dash.no_update, debug_log, filters_ui, dash.no_update, dash.no_update, dash.no_update, dash.no_update

    # Lecture des paramètres
    ui_width = int(ui_width) if ui_width else 1080
    ui_height = int(ui_height) if ui_height else 1920
    ui_fps = float(ui_fps) if ui_fps else None
    ui_femur = float(ui_femur) if ui_femur else 44.0

    max1 = SOURCES["1"].n_frames - 1 if SOURCES["1"].n_frames > 0 else 0
    max2 = SOURCES["2"].n_frames - 1 if SOURCES["2"].n_frames > 0 else 0

    # Upload Excel source 1
    if trigger == 'upload-data-1':
        if contents1 is None:
            return max1, max2, debug_log, filters_ui, ui_width, ui_height, ui_fps, dash.no_update
        n_frames, msg, W_px, H_px, eff_fps = process_source_from_excel_content(
            "1", contents1, filename1, filter_types, filter_params_children,
            ui_width, ui_height, ui_fps, ui_femur
        )
        prepare_coords_for_visualization()
        compute_fixed_ranges()
        debug_log += msg
        max1 = n_frames - 1 if n_frames > 0 else 0
        return max1, max2, debug_log, filters_ui, W_px, H_px, eff_fps, time.time()

    # Upload Excel source 2
    if trigger == 'upload-data-2':
        if contents2 is None:
            return max1, max2, debug_log, filters_ui, ui_width, ui_height, ui_fps, dash.no_update
        n_frames, msg, W_px, H_px, eff_fps = process_source_from_excel_content(
            "2", contents2, filename2, filter_types, filter_params_children,
            ui_width, ui_height, ui_fps, ui_femur
        )
        prepare_coords_for_visualization()
        compute_fixed_ranges()
        debug_log += msg
        max2 = n_frames - 1 if n_frames > 0 else 0
        return max1, max2, debug_log, filters_ui, W_px, H_px, eff_fps, time.time()

    # Apply filters — réapplique les filtres à la/les source(s) ciblée(s)
    if trigger == 'apply-filters':
        msg_all = ""

        if filter_target in ['1', 'both'] and SOURCES["1"].df_raw is not None:
            n_frames1, msg1, W1, H1, F1 = reprocess_source_with_filters(
                "1", filter_types, filter_params_children,
                ui_width, ui_height, ui_fps, ui_femur
            )
            max1 = n_frames1 - 1 if n_frames1 > 0 else 0
            ui_width, ui_height, ui_fps = W1, H1, F1
            msg_all += msg1

        if filter_target in ['2', 'both'] and SOURCES["2"].df_raw is not None:
            n_frames2, msg2, W2, H2, F2 = reprocess_source_with_filters(
                "2", filter_types, filter_params_children,
                ui_width, ui_height, ui_fps, ui_femur
            )
            max2 = n_frames2 - 1 if n_frames2 > 0 else 0
            ui_width, ui_height, ui_fps = W2, H2, F2
            msg_all += msg2

        prepare_coords_for_visualization()
        compute_fixed_ranges()
        debug_log += msg_all
        return max1, max2, debug_log, filters_ui, ui_width, ui_height, ui_fps, time.time()

    raise dash.exceptions.PreventUpdate


# --------------------------------------------------------------------------------------
# Sliders / Play pour les 2 sources
# --------------------------------------------------------------------------------------
@app.callback(
    Output('frame-slider-1', 'value'),
    Output('frame-slider-2', 'value'),
    [
        Input('play-interval-1', 'n_intervals'),
        Input('prev-frame-1', 'n_clicks'),
        Input('next-frame-1', 'n_clicks')
    ],
    [
        State('frame-slider-1', 'value'),
        State('frame-slider-1', 'max'),
        State('frame-slider-2', 'value'),
        State('frame-slider-2', 'max')
    ]
)
def update_frames(n_intervals, prev_clicks, next_clicks,
                  current_frame_1, max_val_1,
                  current_frame_2, max_val_2):
    ctx = dash.callback_context
    if not ctx.triggered:
        raise dash.exceptions.PreventUpdate
    trigger = ctx.triggered[0]['prop_id'].split('.')[0]

    # Défauts si pas de frames
    if max_val_1 is None:
        max_val_1 = 0
    if max_val_2 is None:
        max_val_2 = 0
    if current_frame_1 is None:
        current_frame_1 = 0
    if current_frame_2 is None:
        current_frame_2 = 0

    # Lecture automatique : on avance les deux
    if trigger == 'play-interval-1':
        if max_val_1 > 0:
            new_f1 = (current_frame_1 + 1) % (max_val_1 + 1)
        else:
            new_f1 = 0
        if max_val_2 > 0:
            new_f2 = (current_frame_2 + 1) % (max_val_2 + 1)
        else:
            new_f2 = 0
        return new_f1, new_f2

    # Navigation manuelle : on ne touche qu'à S1
    if trigger == 'prev-frame-1':
        new_f1 = max(0, current_frame_1 - 1)
        return new_f1, current_frame_2
    if trigger == 'next-frame-1':
        new_f1 = min(max_val_1, current_frame_1 + 1)
        return new_f1, current_frame_2

    return current_frame_1, current_frame_2


@app.callback(
    Output('play-interval-1', 'disabled'),
    [
        Input('play-btn-1', 'n_clicks'),
        Input('stop-btn-1', 'n_clicks'),
        Input('prev-frame-1', 'n_clicks'),
        Input('next-frame-1', 'n_clicks')
    ]
)
def toggle_play_1(n_play, n_stop, n_prev, n_next):
    ctx = dash.callback_context
    if not ctx.triggered:
        raise dash.exceptions.PreventUpdate
    trigger = ctx.triggered[0]['prop_id'].split('.')[0]
    if trigger in ['prev-frame-1', 'next-frame-1']:
        return True
    return n_play <= n_stop


@app.callback(
    Output('debug-output', 'children'),
    Input('debug-log', 'data')
)
def display_debug(debug_data):
    return debug_data


# --------------------------------------------------------------------------------------
# VISU : squelette + cinématique (3 graphes : hip / knee / ankle)
# --------------------------------------------------------------------------------------
@app.callback(
    Output('skeleton-graph', 'figure'),
    Output('hip-angle-graph', 'figure'),
    Output('knee-angle-graph', 'figure'),
    Output('ankle-angle-graph', 'figure'),
    Input('frame-slider-1', 'value'),
    Input('frame-slider-2', 'value'),
    Input('display-options', 'value'),
    Input('angle-selection', 'value'),
    Input('refresh-visuals', 'data')
)
def update_visuals(frame_idx1, frame_idx2, options, angle_selection, _refresh_token):

    # 1) SQUELETTE COMBINÉ
    fig_skel = go.Figure()

    def draw_source(state: SourceState, frame_idx: int, color_filt: str, color_raw: str, label: str):
        if state.coords_proc_vis is None or state.n_frames == 0:
            return
        frames_filt = _iter_frames(state.coords_proc_vis)
        if frame_idx >= len(frames_filt):
            return
        pts_filt = transform_pose_fixed(frames_filt[frame_idx])
        fig_skel.add_trace(go.Scatter(
            x=pts_filt[:, 0], y=-pts_filt[:, 1],
            mode='markers',
            marker=dict(color=color_filt, size=6),
            name=f'{label} filtré'
        ))
        for s, e in POSE_CONNECTIONS:
            x0, y0 = pts_filt[s]
            x1, y1 = pts_filt[e]
            if not np.any(np.isnan([x0, y0, x1, y1])):
                fig_skel.add_trace(go.Scatter(
                    x=[x0, x1], y=[-y0, -y1],
                    mode='lines',
                    line=dict(color=color_filt),
                    showlegend=False
                ))

        if 'raw_skel' in options and state.coords_raw_px is not None:
            frames_raw = _iter_frames(state.coords_raw_px)
            if frame_idx < len(frames_raw):
                pts_raw = transform_pose_fixed(frames_raw[frame_idx])
                fig_skel.add_trace(go.Scatter(
                    x=pts_raw[:, 0], y=-pts_raw[:, 1],
                    mode='markers',
                    marker=dict(color=color_raw, size=5),
                    name=f'{label} brut'
                ))
                for s, e in POSE_CONNECTIONS:
                    x0, y0 = pts_raw[s]
                    x1, y1 = pts_raw[e]
                    if not np.any(np.isnan([x0, y0, x1, y1])):
                        fig_skel.add_trace(go.Scatter(
                            x=[x0, x1], y=[-y0, -y1],
                            mode='lines',
                            line=dict(color=color_raw, dash='dot'),
                            showlegend=False
                        ))

    draw_source(SOURCES["1"], frame_idx1 or 0, 'blue', 'lightblue', 'Source 1')
    draw_source(SOURCES["2"], frame_idx2 or 0, 'green', 'lightgreen', 'Source 2')

    fig_skel.update_layout(
        title=f"Squelette – Frame S1={frame_idx1} / S2={frame_idx2}",
        xaxis=dict(visible=False, range=X_RANGE_FIXED, fixedrange=True),
        yaxis=dict(visible=False, range=Y_RANGE_FIXED, scaleanchor='x', fixedrange=True),
        height=900,
        width=600,
        showlegend=True,
        margin=dict(l=0, r=0, t=40, b=0),
        uirevision='skeleton'
    )

    # 2) CINÉMATIQUE : 3 GRAPHES (hip / knee / ankle)
    show_raw_angles = 'raw_angles' in options

    s1_sel = set([f"S1_{x}" for x in angle_selection])
    s2_sel = set([f"S2_{x}" for x in angle_selection])


    # mapping joint -> fig, key, label
    joints = [
        ('hip', 'hip-angle-graph', 'Hanche'),
        ('knee', 'knee-angle-graph', 'Genou'),
        ('ankle', 'ankle-angle-graph', 'Cheville')
    ]

    # couleurs par articulation / source
    color_map = {
        'hip': {'1': '#1f77b4', '2': '#2ca02c'},
        'knee': {'1': '#d62728', '2': '#ff7f0e'},
        'ankle': {'1': '#9467bd', '2': '#8c564b'}
    }

    figs = {}

    for joint_key, fig_id, joint_label in joints:
        fig = go.Figure()
        all_vals = []

        # Source 1
        st1 = SOURCES["1"]
        if st1.angles_proc is not None:
            for side, side_label in [('L', 'Gauche'), ('R', 'Droite')]:
                code = f"S1_{joint_key}_{side}"
                if code in s1_sel:
                    col = f"{joint_key}_{side}"
                    if col in st1.angles_proc.columns:
                        y = pd.to_numeric(st1.angles_proc[col], errors='coerce').to_numpy(dtype=float)
                        x = np.arange(len(y))
                        fig.add_trace(go.Scatter(
                            x=x,
                            y=y,
                            name=f"S1 {joint_label} {side_label} (filtré)",
                            line=dict(color=color_map[joint_key]['1'])
                        ))
                        all_vals.append(y)
                    if show_raw_angles and st1.angles_raw is not None and col in st1.angles_raw.columns:
                        y_raw = pd.to_numeric(st1.angles_raw[col], errors='coerce').to_numpy(dtype=float)
                        x_raw = np.arange(len(y_raw))
                        fig.add_trace(go.Scatter(
                            x=x_raw,
                            y=y_raw,
                            name=f"S1 {joint_label} {side_label} (brut)",
                            line=dict(color=color_map[joint_key]['1'], dash='dot')
                        ))
                        all_vals.append(y_raw)

        # Source 2
        st2 = SOURCES["2"]
        if st2.angles_proc is not None:
            for side, side_label in [('L', 'Gauche'), ('R', 'Droite')]:
                code = f"S2_{joint_key}_{side}"
                if code in s2_sel:
                    col = f"{joint_key}_{side}"
                    if col in st2.angles_proc.columns:
                        y = pd.to_numeric(st2.angles_proc[col], errors='coerce').to_numpy(dtype=float)
                            # note: this indentation
                        x = np.arange(len(y))
                        fig.add_trace(go.Scatter(
                            x=x,
                            y=y,
                            name=f"S2 {joint_label} {side_label} (filtré)",
                            line=dict(color=color_map[joint_key]['2'])
                        ))
                        all_vals.append(y)
                    if show_raw_angles and st2.angles_raw is not None and col in st2.angles_raw.columns:
                        y_raw = pd.to_numeric(st2.angles_raw[col], errors='coerce').to_numpy(dtype=float)
                        x_raw = np.arange(len(y_raw))
                        fig.add_trace(go.Scatter(
                            x=x_raw,
                            y=y_raw,
                            name=f"S2 {joint_label} {side_label} (brut)",
                            line=dict(color=color_map[joint_key]['2'], dash='dot')
                        ))
                        all_vals.append(y_raw)

        # Lignes verticales pour frames S1/S2
        fig.add_vline(x=frame_idx1 or 0, line=dict(color='black', dash='dot', width=1))
        fig.add_vline(x=frame_idx2 or 0, line=dict(color='gray', dash='dot', width=1))

        # Range verticale
        if all_vals:
            arr = np.concatenate(all_vals)
            arr = arr[~np.isnan(arr)]
            if arr.size:
                mn, mx = float(np.min(arr)), float(np.max(arr))
                margin = (mx - mn) * 0.10 if mx > mn else 10.0
                y_range = [mn - margin, mx + margin]
            else:
                y_range = [0, 180]
        else:
            # fallback éventuel : utiliser ANGLE_RANGES si dispo
            if ANGLE_RANGES.get(joint_key) is not None:
                y_range = ANGLE_RANGES[joint_key]
            else:
                y_range = [0, 180]

        fig.update_layout(
            title=f"Angle {joint_label}",
            xaxis_title="Frame",
            yaxis_title="Angle (°)",
            height=280,
            margin=dict(t=40, b=20),
            template='simple_white',
            yaxis=dict(range=y_range, fixedrange=True),
            uirevision=f'angles-{joint_key}'
        )

        figs[fig_id] = fig

    return fig_skel, figs['hip-angle-graph'], figs['knee-angle-graph'], figs['ankle-angle-graph']


# --------------------------------------------------------------------------------------
# Normalisation des cycles (par source)
# --------------------------------------------------------------------------------------
def _resample_series_to_n_points(y: np.ndarray, n_points: int) -> np.ndarray:
    if len(y) == 0:
        return np.array([])
    x_old = np.linspace(0.0, 1.0, num=len(y))
    x_new = np.linspace(0.0, 1.0, num=n_points)
    return np.interp(x_new, x_old, y)


def normalize_angle_cycles_simple(
    angles_df: pd.DataFrame,
    cycles_indices: list[list[int]],
    n_points: int,
    joint: str,
    side: str
) -> pd.DataFrame:
    rows = []
    for cid, idxs in enumerate(cycles_indices, start=1):
        idxs = [i for i in idxs if 0 <= i < len(angles_df)]
        if not idxs:
            continue
        y = pd.to_numeric(
            angles_df[f"{joint}_{side}"].iloc[idxs],
            errors='coerce'
        ).to_numpy(dtype=float)
        y = y[~np.isnan(y)]
        if y.size == 0:
            continue
        y_res = _resample_series_to_n_points(y, n_points)
        rows += [
            {
                "phase": k / (n_points - 1),
                "angle": float(val),
                "joint": joint,
                "side": side,
                "cycle": cid
            }
            for k, val in enumerate(y_res)
        ]
    return pd.DataFrame(rows)


def _run_gait_cycle_analysis_for_source(source_key: str, selected_angles, display_mode):
    state = SOURCES[source_key]
    if state.df_proc is None or state.angles_proc is None:
        return go.Figure()
    if state.pipe_cache["cycles"] is None or state.pipe_cache["events"] is None:
        return go.Figure()

    cycles_df = state.pipe_cache["cycles"]
    events = state.pipe_cache["events"]
    if cycles_df.empty:
        return go.Figure(layout=dict(title="Aucun cycle valide détecté."))

    # Cycles de pas droite : Start_HS_R → End_HS_R
    step_cycles_R = [
        list(range(int(r["Start_HS_R"]), int(r["End_HS_R"]) + 1))
        for _, r in cycles_df.iterrows()
    ]

    # Cycles de pas gauche basés sur HS_L successifs
    hsL = cycles_df["HS_L"].dropna().astype(int).to_list()
    step_cycles_L = [
        list(range(hsL[i], hsL[i + 1] + 1))
        for i in range(len(hsL) - 1)
    ]

    n_points = 100
    frames = []
    for ak in selected_angles:
        joint, side = ak.split('_')
        frames.append(
            normalize_angle_cycles_simple(
                state.angles_proc,
                step_cycles_R if side == 'R' else step_cycles_L,
                n_points,
                joint,
                side
            )
        )
    if not frames:
        return go.Figure(layout=dict(title="Aucune série sélectionnée."))

    angle_cycles = pd.concat(frames, ignore_index=True)
    if angle_cycles.empty:
        return go.Figure(layout=dict(title="Erreur : normalisation vide."))

    fig = go.Figure()
    for ak in selected_angles:
        joint, side = ak.split('_')
        subset = angle_cycles[
            (angle_cycles['joint'] == joint) &
            (angle_cycles['side'] == side)
        ]
        if subset.empty:
            continue
        color = 'red' if side == 'R' else 'blue'
        if display_mode == 'mean_sd':
            g = subset.groupby('phase')
            mean = g['angle'].mean()
            std = g['angle'].std()
            fig.add_trace(go.Scatter(
                x=mean.index * 100,
                y=mean,
                mode='lines',
                name=f"{joint.capitalize()} {side} (S{source_key})",
                line=dict(color=color)
            ))
            fig.add_trace(go.Scatter(
                x=list(mean.index * 100) + list((mean.index * 100)[::-1]),
                y=list(mean + std) + list((mean - std)[::-1]),
                fill='toself',
                opacity=0.15,
                line=dict(width=0),
                showlegend=False
            ))
        else:
            for cid in subset['cycle'].unique():
                cd = subset[subset['cycle'] == cid]
                fig.add_trace(go.Scatter(
                    x=cd['phase'] * 100,
                    y=cd['angle'],
                    name=f"{joint.capitalize()} {side} - Cycle {cid} (S{source_key})",
                    mode='lines',
                    opacity=0.5,
                    line=dict(color=color)
                ))

    # Ajout des TO moyens (swing / phase moyenne du toe-off dans le cycle)
    for side in ['L', 'R']:
        key = f'TO_{side}'
        to_events = events.get(key, [])
        steps = step_cycles_L if side == 'L' else step_cycles_R
        to_phases = []
        for cyc in steps:
            if not cyc:
                continue
            inside = [idx for idx in to_events if cyc[0] <= idx <= cyc[-1]]
            if inside:
                to_frame = inside[0]
                start, end = cyc[0], cyc[-1]
                if end > start:
                    to_phases.append((to_frame - start) / (end - start) * 100)
        if to_phases:
            mean_phase = float(np.mean(to_phases))
            fig.add_shape(
                type='line',
                x0=mean_phase,
                x1=mean_phase,
                y0=0,
                y1=1,
                xref='x',
                yref='paper',
                line=dict(color='gray', dash='dash', width=2)
            )
            fig.add_annotation(
                x=mean_phase,
                y=1.02,
                xref='x',
                yref='paper',
                text=f"TO_{side} : {mean_phase:.1f}%",
                showarrow=False
            )

    fig.update_layout(
        title=f"Angles articulaires normalisés par cycle – Source {source_key}",
        xaxis_title="Phase du cycle (%)",
        yaxis_title="Angle (°)",
        height=600,
        template="simple_white"
    )
    return fig


@app.callback(
    Output('normalized-cycle-plot-1', 'figure'),
    Input('analyze-cycles-1', 'n_clicks'),
    State('angle-selection', 'value'),
    State('cycle-display-mode', 'value'),
    prevent_initial_call=True
)
def run_gait_cycle_analysis_1(n_clicks, selected_angles, display_mode):
    return _run_gait_cycle_analysis_for_source("1", selected_angles, display_mode)


@app.callback(
    Output('normalized-cycle-plot-2', 'figure'),
    Input('analyze-cycles-2', 'n_clicks'),
    State('angle-selection', 'value'),
    State('cycle-display-mode', 'value'),
    prevent_initial_call=True
)
def run_gait_cycle_analysis_2(n_clicks, selected_angles, display_mode):
    return _run_gait_cycle_analysis_for_source("2", selected_angles, display_mode)



@app.callback(
    Output('normalized-cycle-plot-both', 'figure'),
    Input('analyze-cycles-both', 'n_clicks'),
    State('angle-selection', 'value'),
    State('cycle-display-mode', 'value'),
    prevent_initial_call=True
)
def run_gait_cycle_analysis_both(n_clicks, selected_angles, display_mode):
    fig = go.Figure()

    # Analyse S1
    fig_s1 = _run_gait_cycle_analysis_for_source("1", selected_angles, display_mode)
    for trace in fig_s1.data:
        fig.add_trace(trace)

    # Analyse S2
    fig_s2 = _run_gait_cycle_analysis_for_source("2", selected_angles, display_mode)
    for trace in fig_s2.data:
        fig.add_trace(trace)

    fig.update_layout(
        title="Cycles normalisés",
        xaxis_title="Phase du cycle (%)",
        yaxis_title="Angle (°)",
        template="simple_white",
        height=600
    )
    return fig

# --------------------------------------------------------------------------------------
# Spatio-temporel (par source)
# --------------------------------------------------------------------------------------
def _build_gait_table_and_bars(gait: pd.DataFrame):
    if gait is None or gait.empty:
        return html.Div("❌ Aucun cycle valide détecté.", style={'color': 'red'})

    gait = gait.copy()

    # Arrondi des colonnes numériques
    for col in gait.columns:
        if pd.api.types.is_float_dtype(gait[col]):
            gait[col] = gait[col].round(2)

    # Agrégation G/D
    mean_std = gait.groupby('G/D').agg({
        'Pas (cm)': ['mean', 'std'],
        'Rythme[pas/m]': ['mean', 'std'],
        'TStance%': ['mean', 'std'],
        'TSwing%': ['mean', 'std'],
        'TPas': ['mean', 'std']
    }).reset_index()
    mean_std.columns = ['G/D'] + [f"{c[0]}_{c[1]}" for c in mean_std.columns[1:]]

    table = dash_table.DataTable(
        columns=[{"name": c, "id": c} for c in gait.columns],
        data=gait.to_dict('records'),
        style_table={
            'overflowX': 'auto',
            'overflowY': 'scroll',
            'maxHeight': '600px',
            'border': '1px solid #ccc'
        },
        style_cell={
            'textAlign': 'center',
            'padding': '6px',
            'minWidth': '80px',
            'fontFamily': 'Arial'
        },
        style_header={
            'backgroundColor': '#2c3e50',
            'color': 'white',
            'fontWeight': 'bold'
        },
        style_data={
            'backgroundColor': '#ecf0f1',
            'color': '#2c3e50'
        },
        page_size=20
    )

    def bars(df):
        fig = make_subplots(
            rows=1,
            cols=5,
            subplot_titles=[
                "Longueur de pas (cm)",
                "Cadence (pas/min)",
                "Temps d'appui (%)",
                "Temps de swing (%)",
                "Temps d’un pas (s)"
            ],
            horizontal_spacing=0.07
        )
        metrics = [
            ("Pas (cm)", 1, 1),
            ("Rythme[pas/m]", 1, 2),
            ("TStance%", 1, 3),
            ("TSwing%", 1, 4),
            ("TPas", 1, 5)
        ]
        color = {'G': 'blue', 'D': 'red'}
        for m, row, col in metrics:
            for side in ['G', 'D']:
                mc, sc = f"{m}_mean", f"{m}_std"
                if mc in df.columns and sc in df.columns:
                    sub = df.loc[df['G/D'] == side]
                    if sub.empty:
                        continue
                    fig.add_trace(
                        go.Bar(
                            x=[side],
                            y=[sub[mc].values[0]],
                            marker_color=color[side],
                            error_y=dict(
                                type='data',
                                array=[sub[sc].values[0]],
                                visible=True
                            ),
                            width=0.4,
                            showlegend=False
                        ),
                        row=row,
                        col=col
                    )
        fig.update_layout(
            height=420,
            width=1500,
            title="Comparaison spatio-temporelle G/D (moyenne ± écart-type)",
            title_x=0.5,
            template="plotly_white",
            margin=dict(t=80)
        )
        return fig

    return html.Div([
        html.H5(
            "📋 Paramètres spatio-temporels détaillés (pas à pas)",
            style={'marginBottom': '10px'}
        ),
        table,
        html.Br(),
        dcc.Graph(figure=bars(mean_std))
    ])


@app.callback(
    Output('gait-table-container-1', 'children'),
    Input('extract-gait-btn-1', 'n_clicks'),
    prevent_initial_call=True
)
def extract_gait_metrics_1(n_clicks):
    state = SOURCES["1"]
    if state.pipe_cache["outcomes_detailed"] is None:
        return html.Div("⚠️ Aucune donnée prétraitée disponible (Source 1).", style={'color': 'red'})
    try:
        gait = state.pipe_cache["outcomes_detailed"].copy()
        return _build_gait_table_and_bars(gait)
    except Exception as e:
        return html.Div(f"Erreur lors du calcul des paramètres (Source 1) : {e}", style={'color': 'red'})


@app.callback(
    Output('gait-table-container-2', 'children'),
    Input('extract-gait-btn-2', 'n_clicks'),
    prevent_initial_call=True
)
def extract_gait_metrics_2(n_clicks):
    state = SOURCES["2"]
    if state.pipe_cache["outcomes_detailed"] is None:
        return html.Div("⚠️ Aucune donnée prétraitée disponible (Source 2).", style={'color': 'red'})
    try:
        gait = state.pipe_cache["outcomes_detailed"].copy()
        return _build_gait_table_and_bars(gait)
    except Exception as e:
        return html.Div(f"Erreur lors du calcul des paramètres (Source 2) : {e}", style={'color': 'red'})


# --------------------------------------------------------------------------------------
# Sidebar (paramètres)
# --------------------------------------------------------------------------------------
@app.callback(
    Output('params-sidebar', 'style'),
    Output('sidebar-open', 'data'),
    Input('toggle-params-btn', 'n_clicks'),
    Input('close-params-btn', 'n_clicks'),
    State('sidebar-open', 'data'),
    prevent_initial_call=True
)
def toggle_sidebar(open_clicks, close_clicks, is_open):
    ctx = dash.callback_context
    if not ctx.triggered:
        raise dash.exceptions.PreventUpdate

    trig = ctx.triggered[0]['prop_id'].split('.')[0]

    # calcul du nouvel état (ouvert/fermé)
    if trig == 'toggle-params-btn':
        show = not is_open
    else:
        show = False

    base_style = {
        'position': 'fixed',
        'top': '0',
        'left': '0',
        'width': '300px',
        'height': '100%',
        'backgroundColor': '#ecf0f1',
        'boxShadow': '2px 0 5px rgba(0,0,0,0.2)',
        'zIndex': '1000',
        'padding': '20px',
        'overflowY': 'auto',
        'transition': 'transform 0.3s ease-in-out'
    }
    if show:
        base_style['transform'] = 'translateX(0%)'
    else:
        base_style['transform'] = 'translateX(-100%)'

    return base_style, show


def write_xlsx(buffer, sheets: dict[str, pd.DataFrame]):
    """Écrit un fichier Excel avec plusieurs onglets dans un buffer binaire."""
    with pd.ExcelWriter(buffer, engine="openpyxl") as writer:
        for sheet_name, df in sheets.items():
            df.to_excel(writer, index=False, sheet_name=sheet_name)
    buffer.seek(0)

@app.callback(
    Output("download-xlsx-1", "data"),
    Input("export-xlsx-1", "n_clicks"),
    prevent_initial_call=True
)
def export_xlsx_source_1(n_clicks):

    state = SOURCES["1"]

    if state.pipe_cache["outcomes_detailed"] is None or state.angles_proc is None:
        return dash.no_update

    # === Sheet 1 : Spatiotemporel ===
    spatio = state.pipe_cache["outcomes_detailed"].copy()

    # === Sheet 2 : Angles ===
    angles = state.angles_proc.copy()
    angles.insert(0, "frame", range(len(angles)))

    # === Sheet 3 : Metadata ===
    meta = pd.DataFrame([{
        "source": "S1",
        "fps": state.df_proc["fps"].median() if "fps" in state.df_proc else None,
        "width": state.df_proc["width"].median() if "width" in state.df_proc else None,
        "height": state.df_proc["height"].median() if "height" in state.df_proc else None,
        "n_frames": state.n_frames
    }])

    # === Assemblage Excel ===
    return dcc.send_bytes(
        lambda buffer: write_xlsx(buffer, {
            "Spatiotemporal": spatio,
            "Angles": angles,
            "Metadata": meta
        }),
        filename="gait_full_source1.xlsx"
    )
@app.callback(
    Output("download-xlsx-2", "data"),
    Input("export-xlsx-2", "n_clicks"),
    prevent_initial_call=True
)
def export_xlsx_source_2(n_clicks):

    state = SOURCES["2"]

    if state.pipe_cache["outcomes_detailed"] is None or state.angles_proc is None:
        return dash.no_update

    spatio = state.pipe_cache["outcomes_detailed"].copy()

    angles = state.angles_proc.copy()
    angles.insert(0, "frame", range(len(angles)))

    meta = pd.DataFrame([{
        "source": "S2",
        "fps": state.df_proc["fps"].median() if "fps" in state.df_proc else None,
        "width": state.df_proc["width"].median() if "width" in state.df_proc else None,
        "height": state.df_proc["height"].median() if "height" in state.df_proc else None,
        "n_frames": state.n_frames
    }])

    return dcc.send_bytes(
        lambda buffer: write_xlsx(buffer, {
            "Spatiotemporal": spatio,
            "Angles": angles,
            "Metadata": meta
        }),
        filename="gait_full_source2.xlsx"
    )

# --------------------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------------------
if __name__ == "__main__":
    app.run(debug=True)
