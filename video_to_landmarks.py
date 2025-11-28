# video_to_landmarks.py
import cv2
import mediapipe as mp
import pandas as pd
import numpy as np
import os

# 33 noms MediaPipe Pose, en MAJUSCULES comme demandé
MP_LANDMARK_NAMES = [
    "NOSE",
    "LEFT_EYE_INNER", "LEFT_EYE", "LEFT_EYE_OUTER",
    "RIGHT_EYE_INNER", "RIGHT_EYE", "RIGHT_EYE_OUTER",
    "LEFT_EAR", "RIGHT_EAR",
    "MOUTH_LEFT", "MOUTH_RIGHT",
    "LEFT_SHOULDER", "RIGHT_SHOULDER",
    "LEFT_ELBOW", "RIGHT_ELBOW",
    "LEFT_WRIST", "RIGHT_WRIST",
    "LEFT_PINKY", "RIGHT_PINKY",
    "LEFT_INDEX", "RIGHT_INDEX",
    "LEFT_THUMB", "RIGHT_THUMB",
    "LEFT_HIP", "RIGHT_HIP",
    "LEFT_KNEE", "RIGHT_KNEE",
    "LEFT_ANKLE", "RIGHT_ANKLE",
    "LEFT_HEEL", "RIGHT_HEEL",
    "LEFT_FOOT_INDEX", "RIGHT_FOOT_INDEX"
]


def extract_landmarks_from_video(
    video_path: str,
    output_path: str = "temp_outputs/video_landmarks.xlsx",
    model_complexity: int = 2
):
    """
    Extrait les 33 landmarks MediaPipe Pose (x,y,z,visibility)
    + ajoute frame, time_s, fps, width, height.
    FORMAT EXACTEMENT IDENTIQUE À L’EXEMPLE FOURNI PAR L’UTILISATEUR.
    """

    mp_pose = mp.solutions.pose
    pose = mp_pose.Pose(
        static_image_mode=False,
        model_complexity=model_complexity,
        enable_segmentation=False,
        smooth_landmarks=True
    )

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Impossible d’ouvrir la vidéo : {video_path}")

    width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps    = cap.get(cv2.CAP_PROP_FPS)
    frame_index = 0

    rows = []

    while True:
        ok, frame = cap.read()
        if not ok:
            break

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        res = pose.process(frame_rgb)

        # Infos générales
        row = {
            "frame": frame_index,
            "time_s": frame_index / fps if fps > 0 else np.nan,
            "fps": fps,
            "width": width,
            "height": height
        }

        # Ajout des 33 landmarks *4 colonnes* comme dans ton fichier
        if res.pose_landmarks:
            for name, lm in zip(MP_LANDMARK_NAMES, res.pose_landmarks.landmark):
                row[f"{name}_x"] = lm.x
                row[f"{name}_y"] = lm.y
                row[f"{name}_z"] = lm.z
                row[f"{name}_visibility"] = lm.visibility
        else:
            for name in MP_LANDMARK_NAMES:
                row[f"{name}_x"] = np.nan
                row[f"{name}_y"] = np.nan
                row[f"{name}_z"] = np.nan
                row[f"{name}_visibility"] = np.nan

        rows.append(row)
        frame_index += 1

    cap.release()
    pose.close()

    df = pd.DataFrame(rows)

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df.to_excel(output_path, index=False)

    return output_path


if __name__ == "__main__":
    video_file = r"C:\Users\t.marques\OneDrive - Institut\Bureau\IA Gait\Data test\RF_1.mp4"
    output_file = r"C:\Users\t.marques\OneDrive - Institut\Bureau\IA Gait\Pipeline\AIGait_interface_test\src\interface2\tmp.xlsx"
    extract_landmarks_from_video(video_file, output_file)

# import pandas as pd
# from scipy.io import loadmat

# # Chargement du .mat
# mat = loadmat(
#     r"C:\Users\t.marques\Downloads\test.mat",
#     struct_as_record=False,
#     squeeze_me=True
# )

# angles = mat["res_var_angle_t"]

# # Extraction des champs dans un dict
# results = {field: getattr(angles, field) for field in angles._fieldnames}

# # Construction du DataFrame final
# df = pd.DataFrame()

# for name, arr in results.items():
#     df[f"{name}_x"] = arr[:, 0]
#     df[f"{name}_y"] = arr[:, 1]
#     df[f"{name}_z"] = arr[:, 2]

# # Export CSV
# output_path = r"C:\Users\t.marques\Downloads\angles_export.csv"
# df.to_csv(output_path, index=False)

# print("CSV bien enregistré dans :", output_path)
