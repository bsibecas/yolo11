from ultralytics import YOLO

# Load an official or custom model
model = YOLO("yolov11n.pt")  # Load an official Detect model

# Perform tracking with the model
results = model.track(
    "./comma_small.mp4",
    show=False, stream=False
)

results_dfs = [res.to_df() for res in results]
track_paths = []

for res in results_dfs:
    tracks = res['track_id'].astype(str).values
    bboxes = res['box'].values
    bboxes = [dict(bbox) for bbox in bboxes]

    frame_tracks = {}
    for track, bbox in zip(tracks, bboxes):
        center = (bbox['x2'] + bbox['x1']) / 2, (bbox['y2'] + bbox['y1']) / 2
        center = (bbox['x1'], bbox['y1'])
        frame_tracks[f'track_{track}'] = center
    track_paths.append(frame_tracks)

# Consolidate all tracks with the same name
merged_tracks = {}

for frame_tracks in track_paths:
    for track_id, center in frame_tracks.items():
        if track_id not in merged_tracks:
            merged_tracks[track_id] = []
        merged_tracks[track_id].append(center)

# Display the merged tracks
merged_tracks
