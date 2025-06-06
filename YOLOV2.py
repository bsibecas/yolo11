import cv2
import numpy as np
from ultralytics import YOLO

# Cargar modelo YOLOv11
model = YOLO("yolo11n.pt")

# Ruta del vídeo de entrada y salida
input_video_path = "./comma_small.mp4"
output_video_path = "./tracked_comma_vehicles.mp4"

# Ejecutar tracking
results = model.track(input_video_path, show=False, stream=False)

# Convertir resultados a DataFrame y filtrar solo vehículos
vehicle_classes = {'car', 'bus', 'truck', 'motorbike'}  # Clases deseadas
results_dfs = []

for res in results:
    df = res.to_df()
    if 'class_name' in df.columns:
        df = df[df['class_name'].isin(vehicle_classes)]
    results_dfs.append(df)

track_paths = []

for res in results_dfs:
    if res.empty:
        track_paths.append(({}, [], []))
        continue

    tracks = res['track_id'].astype(str).values
    bboxes = res['box'].values
    bboxes = [dict(bbox) for bbox in bboxes]

    frame_tracks = {}
    for track, bbox in zip(tracks, bboxes):
        center = (
            int((bbox['x1'] + bbox['x2']) / 2),
            int((bbox['y1'] + bbox['y2']) / 2)
        )
        frame_tracks[f'track_{track}'] = center
    track_paths.append((frame_tracks, bboxes, tracks))

# Consolidar trayectorias por track_id
merged_tracks = {}
for frame_tracks, _, _ in track_paths:
    for track_id, center in frame_tracks.items():
        if track_id not in merged_tracks:
            merged_tracks[track_id] = []
        merged_tracks[track_id].append(center)

# Procesar vídeo de entrada para generar salida
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

    frame_tracks, bboxes, tracks = track_paths[frame_idx]

    for bbox, track in zip(bboxes, tracks):
        x1, y1, x2, y2 = map(int, [bbox['x1'], bbox['y1'], bbox['x2'], bbox['y2']])
        center = (int((x1 + x2) / 2), int((y1 + y2) / 2))

        # Dibujar bounding box e ID
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(frame, f'ID {track}', (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        # Dibujar trayectoria
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
print("✅ Video generado solo con detección de vehículos: tracked_comma_vehicles.avi")