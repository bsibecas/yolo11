import cv2
import numpy as np
from ultralytics import YOLO

# Cargar modelo YOLOv11
model = YOLO("yolo11n.pt")

# Ruta del vídeo
input_video_path = "./comma_small.mp4"
output_video_path = "./tracked_comma_vehicles.mp4"

# Ejecutar tracking
results = model.track(input_video_path, show=False, stream=False)

# Obtener nombres de clase
class_names = list(model.names.values()) if isinstance(model.names, dict) else model.names

# Revisar las clases disponibles
print("Clases del modelo:")
for i, name in enumerate(class_names):
    print(f"{i}: {name}")

# Clases deseadas (vehículos)
vehicle_classes = {'car', 'bus', 'truck', 'motorbike', 'bicycle'}  # ajusta según lo que imprima arriba

results_dfs = []

for res in results:
    df = res.to_df()

    if 'cls' in df.columns:
        df['class_name'] = df['cls'].apply(lambda x: class_names[int(x)])
        df['confidence'] = df['conf']  # Asegúrate que 'conf' existe
        df = df[(df['class_name'].isin(vehicle_classes)) & (df['confidence'] > 0.4)]

    results_dfs.append(df)

track_paths = []

for res in results_dfs:
    if res.empty or 'track_id' not in res.columns:
        track_paths.append(({}, [], []))
        continue

    tracks = res['track_id'].astype(str).values
    bboxes = res['box'].values
    class_names_this_frame = res['class_name'].values
    bboxes = [dict(bbox) for bbox in bboxes]

    frame_tracks = {}
    for track, bbox in zip(tracks, bboxes):
        center = (
            int((bbox['x1'] + bbox['x2']) / 2),
            int((bbox['y1'] + bbox['y2']) / 2)
        )
        frame_tracks[f'track_{track}'] = center
    track_paths.append((frame_tracks, bboxes, tracks, class_names_this_frame))

# Consolidar trayectorias
merged_tracks = {}
for frame_tracks, _, _, _ in track_paths:
    for track_id, center in frame_tracks.items():
        if track_id not in merged_tracks:
            merged_tracks[track_id] = []
        merged_tracks[track_id].append(center)

# Preparar vídeo de salida
cap = cv2.VideoCapture(input_video_path)
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = cap.get(cv2.CAP_PROP_FPS)
fourcc = cv2.VideoWriter_fourcc(*'XVID')
out = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))

frame_idx = 0
while cap.isOpened():
    ret, frame = cap.read()
    if not ret or frame_idx >= len(track_paths):
        break

    frame_tracks, bboxes, tracks, class_names_this_frame = track_paths[frame_idx]

    for bbox, track, class_name in zip(bboxes, tracks, class_names_this_frame):
        x1, y1, x2, y2 = map(int, [bbox['x1'], bbox['y1'], bbox['x2'], bbox['y2']])
        center = (int((x1 + x2) / 2), int((y1 + y2) / 2))

        # Bounding box
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

        # ID y clase
        cv2.putText(frame, f'{class_name} ID {track}', (x1, y1 - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 2)

        # Trayectoria
        track_id_str = f'track_{track}'
        if track_id_str in merged_tracks:
            points = merged_tracks[track_id_str]
            for i in range(1, len(points)):
                if points[i - 1] and points[i]:
                    cv2.line(frame, points[i - 1], points[i], (255, 0, 0), 2)

    out.write(frame)
    frame_idx += 1

cap.release()
out.release()
print("✅ Vídeo generado solo con vehículos detectados:", output_video_path)
