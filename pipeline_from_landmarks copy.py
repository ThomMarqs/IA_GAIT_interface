from __future__ import annotations
from typing import Optional
import numpy as np
import pandas as pd

from center_coords import center_coords
from smooth_savgol import smooth_savgol
from align_profile import align_profile
from correct_limb_lengths_stable import correct_limb_lengths_stable
from kalman import smooth_skeleton_kalman

from gait_pipeline_core import (
    detect_gait_cycles,
    extract_valid_cycles,
    compute_outcomes_on_valid_cycles,
    compute_joint_angles,
    build_frame_to_cycle_map,
)

def _ensure_pixels(df_raw: pd.DataFrame) -> pd.DataFrame:
    df = df_raw.copy()
    if 'width' in df.columns and 'height' in df.columns:
        xcols = [c for c in df.columns if c.endswith('_x')]
        ycols = [c for c in df.columns if c.endswith('_y')]
        if xcols and ycols:
            vals = pd.concat([df[xcols].stack(), df[ycols].stack()], axis=0)
            try:
                mx = float(pd.to_numeric(vals, errors='coerce').max())
            except Exception:
                mx = np.inf
            if np.isfinite(mx) and mx <= 1.05:  # normalisé 0..1
                W = pd.to_numeric(df['width'], errors='coerce').fillna(method='ffill').fillna(method='bfill').to_numpy(float)
                H = pd.to_numeric(df['height'], errors='coerce').fillna(method='ffill').fillna(method='bfill').to_numpy(float)
                for c in xcols:
                    df[c] = pd.to_numeric(df[c], errors='coerce').to_numpy(float) * W
                for c in ycols:
                    df[c] = pd.to_numeric(df[c], errors='coerce').to_numpy(float) * H
    return df

def preprocess_like_video_pipeline(df_raw: pd.DataFrame, drop_meta: bool = False) -> pd.DataFrame:
    df = _ensure_pixels(df_raw)
    df = center_coords(df)
    df = smooth_savgol(df, window_length=21, polyorder=2)
    df = align_profile(df)
    df = correct_limb_lengths_stable(df)
    df = smooth_skeleton_kalman(df, [
        "NOSE","LEFT_SHOULDER","RIGHT_SHOULDER","LEFT_ELBOW","RIGHT_ELBOW","LEFT_WRIST","RIGHT_WRIST",
        "LEFT_HIP","RIGHT_HIP","LEFT_KNEE","RIGHT_KNEE","LEFT_ANKLE","RIGHT_ANKLE",
        "LEFT_HEEL","RIGHT_HEEL","LEFT_FOOT_INDEX","RIGHT_FOOT_INDEX",
    ])
    if drop_meta:
        for c in ("fps","width","height"):
            if c in df.columns:
                df.drop(columns=c, inplace=True)
    return df

def run_pipeline_from_landmarks(df_raw_in: pd.DataFrame, femur_length_cm: float = 44.0, framerate: Optional[float] = None) -> dict:
    df_proc = preprocess_like_video_pipeline(df_raw_in, drop_meta=False)

    if framerate is None:
        if 'fps' in df_raw_in.columns and pd.notna(df_raw_in['fps']).any():
            fps = float(pd.to_numeric(df_raw_in['fps'], errors='coerce').median())
            fps = fps if np.isfinite(fps) and fps > 0 else 30.0
        else:
            fps = 30.0
    else:
        fps = float(framerate)

    events = detect_gait_cycles(df_proc, framerate=fps, cutoff=5)
    cycles = extract_valid_cycles(df_proc, events, framerate=fps)

    angles_df = compute_joint_angles(df_proc)
    f2c = build_frame_to_cycle_map(cycles, len(df_proc))
    angles_df.insert(0, 'Cycle', f2c)

    outcomes_summary, outcomes_detailed = compute_outcomes_on_valid_cycles(
        df_proc, cycles, events, framerate=fps, femur_length_cm=femur_length_cm, angles_df=angles_df
    )

    return {
        'df_proc': df_proc,
        'events': events,
        'cycles': cycles,
        'angles_df': angles_df,
        'outcomes_summary': outcomes_summary,
        'outcomes_detailed': outcomes_detailed,
        'fps': fps,
    }
