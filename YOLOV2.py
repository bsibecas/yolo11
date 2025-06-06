import cv2
import numpy as np
from ultralytics import YOLO

model = YOLO("yolo11n.pt")  # Sigue usando yolo11n, aunque puede que no tenga tracking

input_video_path = "./comma_small.mp4"
output_video_path = "./tracked_comma_output.mp4"

results = model.track(input_video_path, show=False, stream=False)

results_dfs = [res.to_df() for res in results]
track_paths = []

# Generador de ID en caso de que no haya 'track_id'
global_track_id = 0

for res in results_dfs:
    if res is None or res.empty:
        track_paths.append(({}, [], []))
        continue

    # Si 'track_id' no existe, generar IDs únicos por objeto
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

# Leer vídeo original y preparar salida
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
print("✅ Video con tracking (simulado) generado:", output_video_path)
