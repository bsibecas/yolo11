import cv2
import numpy as np
from ultralytics import YOLO
from collections import Counter

model = YOLO("yolo11n.pt")

input_video_path = "./comma_small.mp4"
output_video_path = "./tracked_comma_output.mp4"

# Clases de vehículos por nombre
vehicle_class_names = ['car', 'truck', 'bus', 'motorbike', 'bicycle']

# Mostrar todas las clases del modelo
print("\n📋 Todas las clases del modelo:")
print(model.names)

# Mapear a índices de clase (cls)
vehicle_class_ids = [i for i, name in model.names.items() if name in vehicle_class_names]

if not vehicle_class_ids:
    print("🚨 No se encontraron IDs de vehículos en model.names")

print("🔎 Clases en el modelo:")
for k, v in model.names.items():
    print(f"{k}: {v}")

print("✅ IDs de clases de vehículos:", vehicle_class_ids)

# Mostrar el mapeo de ID a nombre para vehículos
print("\n🔍 Mapeo de IDs de vehículos:")
for i in vehicle_class_ids:
    print(f"{i}: {model.names[i]}")

# Procesamiento con YOLO
results = model.track(input_video_path, show=False, stream=False)

results_dfs = []
all_classes = []

for frame_index, res in enumerate(results):
    df = res.to_df()
    if df is not None and not df.empty:
        print(f"\n📦 Frame {frame_index} - todas las detecciones:")
        print(df[['name', 'cls']])

        all_classes.extend(df['cls'].tolist())

        if 'cls' in df.columns:
            df_filtered = df[df['cls'].isin(vehicle_class_ids)]
            print(f"🚗 Frame {frame_index} - después del filtro de vehículos:")
            print(df_filtered[['name', 'cls']])
            df = df_filtered
        results_dfs.append(df)
    else:
        results_dfs.append(None)

# Conteo total de clases detectadas
print("\n📊 Conteo total de clases detectadas en todo el video:")
counter = Counter(all_classes)
for cls_id, count in counter.items():
    print(f"{model.names[cls_id]} ({cls_id}): {count}")

# Seguimiento de trayectorias
track_paths = []
global_track_id = 0

for res in results_dfs:
    if res is None or res.empty:
        track_paths.append(({}, [], []))
        continue

    if 'track_id' in res.columns:
        tracks = res['track_id'].astype(str).values
    else:
        tracks = [str(global_track_id + i) for i in range(len(res))]
        global_track_id += len(res)

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

# Consolidar trayectorias
merged_tracks = {}
for frame_tracks, _, _ in track_paths:
    for track_id, center in frame_tracks.items():
        if track_id not in merged_tracks:
            merged_tracks[track_id] = []
        merged_tracks[track_id].append(center)

# Preparar salida de video
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

        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(frame, f'ID {track}', (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

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
print("✅ Video de vehículos generado:", output_video_path)
