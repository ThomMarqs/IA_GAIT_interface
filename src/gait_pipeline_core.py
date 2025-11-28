from __future__ import annotations
from typing import Dict, List, Tuple, Optional, Sequence
import re
import numpy as np
import pandas as pd
from scipy.signal import butter, filtfilt

# ---------------- Helpers filtrage bas‑passe ----------------
def apply_lowpass_filter(data, cutoff_freq, sampling_rate, order: int = 4):
    nyquist = 0.5 * sampling_rate
    normal_cutoff = cutoff_freq / nyquist
    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    return filtfilt(b, a, pd.to_numeric(data, errors='coerce').fillna(method='ffill').fillna(method='bfill').values)

# ---------------- Helpers colonnes ----------------
def find_column(columns, pattern):
    for col in columns:
        if re.search(pattern, col, re.IGNORECASE):
            return col
    raise ValueError(f"Colonne non trouvée pour le motif : {pattern}")

# ---------------- Détection événements ----------------
def detect_gait_cycles(df: pd.DataFrame, framerate: float, cutoff: float = 5) -> Dict[str, List[int]]:
    left_knee_x  = find_column(df.columns, r"LEFT_KNEE.*x")
    right_knee_x = find_column(df.columns, r"RIGHT_KNEE.*x")
    left_foot_x  = find_column(df.columns, r"LEFT_FOOT_INDEX.*x")
    left_heel_x  = find_column(df.columns, r"LEFT_HEEL.*x")
    left_hip_x   = find_column(df.columns, r"LEFT_HIP.*x")
    right_foot_x = find_column(df.columns, r"RIGHT_FOOT_INDEX.*x")
    right_heel_x = find_column(df.columns, r"RIGHT_HEEL.*x")
    right_hip_x  = find_column(df.columns, r"RIGHT_HIP.*x")

    init_idx = min(120, len(df)-1)
    init_diff = df[left_knee_x].iloc[init_idx] - df[right_knee_x].iloc[init_idx]
    pos_1, pos_2 = (left_knee_x, right_knee_x) if init_diff > 0 else (right_knee_x, left_knee_x)
    sens_marche = df[left_heel_x].iloc[init_idx] - df[left_foot_x].iloc[init_idx]

    position_diff = df[pos_1] - df[pos_2]
    position_diff_filtered = pd.Series(apply_lowpass_filter(position_diff, cutoff, framerate), index=position_diff.index)

    crossings = np.where(np.diff(np.sign(position_diff_filtered)))[0]
    crossings = np.insert(crossings, 0, 0)
    neg_cross = crossings[position_diff_filtered.iloc[crossings] < 0]
    pos_cross = crossings[position_diff_filtered.iloc[crossings] > 0]

    df = df.copy()
    df["LEFT_HIP_FOOT"] = df[left_hip_x] - df[left_foot_x]
    df["RIGHT_HIP_FOOT"] = df[right_hip_x] - df[right_foot_x]
    df["LEFT_HIP_FOOT_filtered"] = apply_lowpass_filter(df["LEFT_HIP_FOOT"], cutoff, framerate)
    df["RIGHT_HIP_FOOT_filtered"] = apply_lowpass_filter(df["RIGHT_HIP_FOOT"], cutoff, framerate)

    if sens_marche < 0:
        df["LEFT_HIP_FOOT_filtered"] *= -1
        df["RIGHT_HIP_FOOT_filtered"] *= -1

    def detect_peaks(crossings_idx, signal, find_max=True):
        indices = []
        for i in range(1, len(crossings_idx)):
            seg = pd.Series(signal, index=range(len(signal))).iloc[crossings_idx[i-1]:crossings_idx[i]]
            if seg.empty: continue
            idx = int(seg.idxmax()) if find_max else int(seg.idxmin())
            indices.append(idx)
        return indices

    events = {
        "HS_L": detect_peaks(neg_cross, df["LEFT_HIP_FOOT_filtered"], True),
        "TO_L": detect_peaks(pos_cross, df["LEFT_HIP_FOOT_filtered"], False),
        "HS_R": detect_peaks(pos_cross, df["RIGHT_HIP_FOOT_filtered"], True),
        "TO_R": detect_peaks(neg_cross, df["RIGHT_HIP_FOOT_filtered"], False),
    }
    return events

# ---------------- Cycles valides ----------------
def extract_valid_cycles(
    df: pd.DataFrame,
    events: Dict[str, List[int]],
    framerate: float,
    min_stride_s: float = 0.1,
    max_stride_s: float = 6.0,
) -> pd.DataFrame:
    def _su(v): return sorted(set(int(x) for x in (v or []) if pd.notna(x)))
    hsr, hsl, tol, tor = _su(events.get("HS_R")), _su(events.get("HS_L")), _su(events.get("TO_L")), _su(events.get("TO_R"))

    rows = []
    for i in range(len(hsr)-1):
        s, e = hsr[i], hsr[i+1]
        stride = (e - s)/float(framerate)
        if not (min_stride_s <= stride <= max_stride_s):
            continue
        toL_c = [t for t in tol if s < t < e]
        hsL_c = [h for h in hsl if s < h < e]
        toR_c = [t for t in tor if s < t < e]
        if not (toL_c and hsL_c and toR_c):
            continue
        toL, hsL, toR = min(toL_c), min(hsL_c), max(toR_c)
        ordered = sorted([toL, hsL, toR])
        toL, hsL, toR = ordered[0], ordered[1], ordered[2]
        if not (s < toL < hsL < toR < e):
            continue
        rows.append({"Cycle": len(rows)+1, "Start_HS_R": s, "TO_L": toL, "HS_L": hsL, "TO_R": toR, "End_HS_R": e, "Stride_s": stride})

    return pd.DataFrame(rows, columns=["Cycle","Start_HS_R","TO_L","HS_L","TO_R","End_HS_R","Stride_s"])

# ---------------- Mapping frame→cycle ----------------
def build_frame_to_cycle_map(cycles: pd.DataFrame, n_frames: int) -> np.ndarray:
    f2c = np.full(n_frames, np.nan, dtype=float)
    if cycles is None or cycles.empty:
        return f2c
    for r in cycles.itertuples(index=False):
        s, e, c = int(r.Start_HS_R), int(r.End_HS_R), int(r.Cycle)
        s = max(0, s); e = min(n_frames - 1, e)
        f2c[s:e+1] = c
    return f2c

# ---------------- Angles 2D ----------------
def compute_joint_angles(df: pd.DataFrame, return_dataframe: bool = True) -> pd.DataFrame:
    def angle(a, b, c):
        v1 = np.array([a["x"]-b["x"], a["y"]-b["y"]])
        v2 = np.array([c["x"]-b["x"], c["y"]-b["y"]])
        den = np.linalg.norm(v1)*np.linalg.norm(v2)
        if den == 0: return np.nan
        cosv = np.dot(v1, v2)/den
        ang = np.degrees(np.arccos(np.clip(cosv, -1.0, 1.0)))
        return 180 - ang

    def hip_angle(shoulder, hip, knee):
        ba = np.array([shoulder["x"]-hip["x"], shoulder["y"]-hip["y"]])
        bc = np.array([knee["x"]-hip["x"], knee["y"]-hip["y"]])
        den = np.linalg.norm(ba)*np.linalg.norm(bc)
        if den == 0: return np.nan
        cosv = np.dot(ba, bc)/den
        ang = np.degrees(np.arccos(np.clip(cosv, -1.0, 1.0)))
        cross = ba[0]*bc[1] - ba[1]*bc[0]
        out = (360 - ang) - 180
        return -out if cross > 0 else out

    def ankle_angle(knee, ankle, heel, foot):
        def inter(p1, p2, p3, p4):
            x1,y1,x2,y2 = p1["x"],p1["y"],p2["x"],p2["y"]
            x3,y3,x4,y4 = p3["x"],p3["y"],p4["x"],p4["y"]
            den = (x1-x2)*(y3-y4)-(y1-y2)*(x3-x4)
            if den == 0: return None
            px = ((x1*y2-y1*x2)*(x3-x4)-(x1-x2)*(x3*y4-y3*x4))/den
            py = ((x1*y2-y1*x2)*(y3-y4)-(y1-y2)*(x3*y4-y3*x4))/den
            return {"x":px,"y":py}
        p = inter(knee, ankle, heel, foot)
        if p is None: return np.nan
        v1 = np.array([knee["x"]-p["x"], knee["y"]-p["y"]])
        v2 = np.array([foot["x"]-p["x"], foot["y"]-p["y"]])
        den = np.linalg.norm(v1)*np.linalg.norm(v2)
        if den == 0: return np.nan
        ang = np.degrees(np.arccos(np.clip(np.dot(v1, v2)/den, -1.0, 1.0)))
        return 90 - ang

    names = [
        "LEFT_HIP","LEFT_KNEE","LEFT_ANKLE","RIGHT_HIP","RIGHT_KNEE","RIGHT_ANKLE",
        "LEFT_SHOULDER","RIGHT_SHOULDER","LEFT_FOOT_INDEX","RIGHT_FOOT_INDEX","LEFT_HEEL","RIGHT_HEEL",
    ]
    out = {k: [] for k in ["knee_L","knee_R","ankle_L","ankle_R","hip_L","hip_R"]}
    for i in range(len(df)):
        try:
            pts = {n: {"x": df[f"{n}_x"].iat[i], "y": df[f"{n}_y"].iat[i]} for n in names}
            out["knee_L"].append(angle(pts["LEFT_HIP"], pts["LEFT_KNEE"], pts["LEFT_ANKLE"]))
            out["knee_R"].append(angle(pts["RIGHT_HIP"], pts["RIGHT_KNEE"], pts["RIGHT_ANKLE"]))
            out["ankle_L"].append(ankle_angle(pts["LEFT_KNEE"], pts["LEFT_ANKLE"], pts["LEFT_HEEL"], pts["LEFT_FOOT_INDEX"]))
            out["ankle_R"].append(ankle_angle(pts["RIGHT_KNEE"], pts["RIGHT_ANKLE"], pts["RIGHT_HEEL"], pts["RIGHT_FOOT_INDEX"]))
            out["hip_L"].append(hip_angle(pts["LEFT_SHOULDER"], pts["LEFT_HIP"], pts["LEFT_KNEE"]))
            out["hip_R"].append(hip_angle(pts["RIGHT_SHOULDER"], pts["RIGHT_HIP"], pts["RIGHT_KNEE"]))
        except Exception:
            for k in out: out[k].append(np.nan)
    df_angles = pd.DataFrame(out, index=df.index)
    return df_angles if return_dataframe else df_angles

# ---------------- Outcomes sur cycles valides ----------------
def compute_outcomes_on_valid_cycles(
    df: pd.DataFrame,
    cycles: pd.DataFrame,
    events: Dict[str, List[int]],
    framerate: float = 30.0,
    femur_length_cm: float = 44.0,
    angles_df: Optional[pd.DataFrame] = None,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    if cycles is None or cycles.empty:
        return pd.DataFrame(), pd.DataFrame()

    def _lp(data, cutoff_freq, sampling_rate, order=4):
        nyquist = 0.5 * sampling_rate
        normal_cutoff = cutoff_freq / nyquist
        b, a = butter(order, normal_cutoff, btype="low", analog=False)
        v = pd.to_numeric(data, errors='coerce').interpolate(limit_direction='both').fillna(method='bfill').fillna(method='ffill')
        return filtfilt(b, a, v.values)

    if "LEFT_HIP_FOOT_filtered" not in df.columns or "RIGHT_HIP_FOOT_filtered" not in df.columns:
        df = df.copy()
        df["LEFT_HIP_FOOT_filtered"]  = _lp(df["LEFT_HIP_x"]  - df["LEFT_FOOT_INDEX_x"],  5, framerate)
        df["RIGHT_HIP_FOOT_filtered"] = _lp(df["RIGHT_HIP_x"] - df["RIGHT_FOOT_INDEX_x"], 5, framerate)

    femur_lengths = np.sqrt((df["RIGHT_HIP_x"] - df["RIGHT_KNEE_x"])**2 + (df["RIGHT_HIP_y"] - df["RIGHT_KNEE_y"])**2)
    mean_femur_px = float(np.nanmean(femur_lengths))

    cyc = cycles.sort_values("Start_HS_R").reset_index(drop=True)
    detailed_rows = []

    # Strides droits
    for r in cyc.itertuples(index=False):
        s, toL, hsL, toR, e = int(r.Start_HS_R), int(r.TO_L), int(r.HS_L), int(r.TO_R), int(r.End_HS_R)
        stride_time = (e - s) / framerate
        if stride_time < 0.5 or stride_time > 3: continue
        step_time_R = (hsL - s) / framerate
        cadence_R  = 60.0 / step_time_R if step_time_R > 0 else 0.0
        stance_R   = ((toR - s)   / framerate) / stride_time * 100.0
        swing_R    = ((e - toR)   / framerate) / stride_time * 100.0
        single_sup = ((hsL - toL) / framerate) / stride_time * 100.0
        double_sup = ((toL - s)   / framerate) / stride_time * 100.0
        heel_dx = np.abs(df["RIGHT_HEEL_x"].iloc[hsL:e+1].values - df["LEFT_HEEL_x"].iloc[hsL:e+1].values)
        toe_dx  = np.abs(df["RIGHT_FOOT_INDEX_x"].iloc[hsL:e+1].values - df["LEFT_FOOT_INDEX_x"].iloc[hsL:e+1].values)
        sl_R_px = float(np.nanmax(np.fmax(heel_dx, toe_dx)))
        sl_R_fem = sl_R_px / mean_femur_px if mean_femur_px > 0 else np.nan
        sl_R_cm  = sl_R_fem * femur_length_cm if np.isfinite(sl_R_fem) else np.nan
        if angles_df is not None and not angles_df.empty:
            rom_knee_R  = float(np.nanmax(angles_df["knee_R"].iloc[s:e])  - np.nanmin(angles_df["knee_R"].iloc[s:e]))
            rom_hip_R   = float(np.nanmax(angles_df["hip_R"].iloc[s:e])   - np.nanmin(angles_df["hip_R"].iloc[s:e]))
            rom_ankle_R = float(np.nanmax(angles_df["ankle_R"].iloc[s:e]) - np.nanmin(angles_df["ankle_R"].iloc[s:e]))
        else:
            rom_knee_R = rom_hip_R = rom_ankle_R = np.nan
        detailed_rows.append({
            "StartIdx": s, "EndIdx": e, "G/D": "D", "TPas": float(step_time_R), "Rythme[pas/m]": float(cadence_R),
            "Pas (cm)": float(sl_R_cm), "StrideTime/Cycle": float(stride_time), "SingleSupport%": float(single_sup),
            "DoubleSupport%": float(double_sup), "TStance%": float(stance_R), "TSwing%": float(swing_R),
            "Step Length (× fémur)": float(sl_R_fem), "ROM_genou (°)": float(rom_knee_R),
            "ROM_hanche (°)": float(rom_hip_R), "ROM_cheville (°)": float(rom_ankle_R),
        })

    # Strides gauches (entre deux cycles consécutifs)
    for i in range(len(cyc)-1):
        r_i, r_j = cyc.iloc[i], cyc.iloc[i+1]
        hsL_i, hsL_j = int(r_i.HS_L), int(r_j.HS_L)
        sR_i, eR_i, toR_i, toL_j = int(r_i.Start_HS_R), int(r_i.End_HS_R), int(r_i.TO_R), int(r_j.TO_L)
        start, end = hsL_i, hsL_j
        stride_time = (end - start) / framerate
        if stride_time < 0.5 or stride_time > 3: continue
        to_opp, hs_opp, to_same = toR_i, eR_i, toL_j
        if not (start < to_opp < hs_opp < to_same < end):
            continue
        step_time_L = (hs_opp - start) / framerate
        cadence_L   = 60.0 / step_time_L if step_time_L > 0 else 0.0
        stance_L    = ((to_same - start) / framerate) / stride_time * 100.0
        swing_L     = ((end - to_same)   / framerate) / stride_time * 100.0
        single_sup  = ((hs_opp - to_opp) / framerate) / stride_time * 100.0
        double_sup  = ((to_opp - start)  / framerate) / stride_time * 100.0
        heel_dx = np.abs(df["RIGHT_HEEL_x"].iloc[sR_i:hsL_i+1].values - df["LEFT_HEEL_x"].iloc[sR_i:hsL_i+1].values)
        toe_dx  = np.abs(df["RIGHT_FOOT_INDEX_x"].iloc[sR_i:hsL_i+1].values - df["LEFT_FOOT_INDEX_x"].iloc[sR_i:hsL_i+1].values)
        sl_L_px = float(np.nanmax(np.fmax(heel_dx, toe_dx)))
        sl_L_fem = sl_L_px / mean_femur_px if mean_femur_px > 0 else np.nan
        sl_L_cm  = sl_L_fem * femur_length_cm if np.isfinite(sl_L_fem) else np.nan
        if angles_df is not None and not angles_df.empty:
            rom_knee_L  = float(np.nanmax(angles_df["knee_L"].iloc[start:end])  - np.nanmin(angles_df["knee_L"].iloc[start:end]))
            rom_hip_L   = float(np.nanmax(angles_df["hip_L"].iloc[start:end])   - np.nanmin(angles_df["hip_L"].iloc[start:end]))
            rom_ankle_L = float(np.nanmax(angles_df["ankle_L"].iloc[start:end]) - np.nanmin(angles_df["ankle_L"].iloc[start:end]))
        else:
            rom_knee_L = rom_hip_L = rom_ankle_L = np.nan
        detailed_rows.append({
            "StartIdx": start, "EndIdx": end, "G/D": "G", "TPas": float(step_time_L), "Rythme[pas/m]": float(cadence_L),
            "Pas (cm)": float(sl_L_cm), "StrideTime/Cycle": float(stride_time), "SingleSupport%": float(single_sup),
            "DoubleSupport%": float(double_sup), "TStance%": float(stance_L), "TSwing%": float(swing_L),
            "Step Length (× fémur)": float(sl_L_fem), "ROM_genou (°)": float(rom_knee_L),
            "ROM_hanche (°)": float(rom_hip_L), "ROM_cheville (°)": float(rom_ankle_L),
        })

    if not detailed_rows:
        return pd.DataFrame(), pd.DataFrame()

    gait_detailed = pd.DataFrame(detailed_rows).sort_values("StartIdx").reset_index(drop=True)
    gait_detailed.insert(0, "Index", gait_detailed.index + 1)

    def summarize(vals: Sequence[float]) -> Dict[str, float]:
        arr = pd.to_numeric(pd.Series(vals), errors='coerce').dropna().values
        if arr.size == 0:
            return {"Minimum":0.0,"Maximum":0.0,"Average":0.0,"Standard Deviation":0.0,"Coefficient of Variation (%)":0.0}
        avg = float(arr.mean()); std = float(arr.std())
        return {"Minimum": float(arr.min()), "Maximum": float(arr.max()), "Average": avg,
                "Standard Deviation": std, "Coefficient of Variation (%)": float((std/avg*100) if avg!=0 else 0.0)}

    metrics = {
        "Step Time (s)": summarize(gait_detailed["TPas"]),
        "Cadence (steps/min)": summarize(gait_detailed["Rythme[pas/m]"]),
        "Stride Time (s)": summarize(gait_detailed["StrideTime/Cycle"]),
        "Stance Time (%)": summarize(gait_detailed["TStance%"]),
        "Swing Time (%)": summarize(gait_detailed["TSwing%"]),
        "Single Support (%)": summarize(gait_detailed["SingleSupport%"]),
        "Double Support (%)": summarize(gait_detailed["DoubleSupport%"]),
        "Step Length (× fémur)": summarize(gait_detailed["Step Length (× fémur)"]),
        "Pas (cm)": summarize(gait_detailed["Pas (cm)"]),
        "ROM_genou (°)": summarize(gait_detailed["ROM_genou (°)"]),
        "ROM_hanche (°)": summarize(gait_detailed["ROM_hanche (°)"]),
        "ROM_cheville (°)": summarize(gait_detailed["ROM_cheville (°)"]),
    }
    gait_summary = pd.DataFrame(metrics).T
    return gait_summary, gait_detailed

