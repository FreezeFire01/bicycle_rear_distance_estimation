# all_frames_check.py
import cv2, ast, pandas as pd, os, subprocess, json

VIDEO_PATH = "/GARMIN_VIDEOS/2026-03-30/MP_needed/GRMN0383.MP4"
CSV_PATH   = "activity_20260330_clean.csv"
OUT_DIR    = "all_frames_check_0383"
os.makedirs(OUT_DIR, exist_ok=True)

df = pd.read_csv(CSV_PATH)
df['timestamp'] = pd.to_datetime(df['timestamp'], utc=True)
df['ranges_parsed'] = df['radar_ranges'].apply(lambda s: [v for v in __import__('ast').literal_eval(str(s)) if v and v > 0] if pd.notna(s) else [])

result = subprocess.run(
    ["ffprobe", "-v", "quiet", "-print_format", "json",
     "-show_format", "-show_streams", VIDEO_PATH],
    capture_output=True, text=True
)
info = json.loads(result.stdout)
video_start = None
fps = 29.97
for source in [info.get("format", {}).get("tags", {})] + \
               [s.get("tags", {}) for s in info.get("streams", [])]:
    if "creation_time" in source:
        video_start = pd.to_datetime(source["creation_time"], utc=True)
for stream in info.get("streams", []):
    if stream.get("codec_type") == "video":
        n, d = stream.get("r_frame_rate", "30/1").split("/")
        fps = float(n) / float(d)

duration = float(info["format"].get("duration", 30))
total_frames = int(duration * fps)
video_end = video_start + pd.Timedelta(seconds=duration)

# FIT záznamy pre toto video
in_video = df[
    (df['timestamp'] >= video_start) &
    (df['timestamp'] <= video_end)
].copy()
in_video['offset_sec'] = (in_video['timestamp'] - video_start).dt.total_seconds()
in_video['frame_num']  = (in_video['offset_sec'] * fps).astype(int)

# Vytvor lookup: frame → radar dáta
frame_to_radar = {}
for _, row in in_video.iterrows():
    fn = int(row['frame_num'])
    frame_to_radar[fn] = row['ranges_parsed']

cap = cv2.VideoCapture(VIDEO_PATH)
print(f"Extrahujem {total_frames} framov ...")

for fn in range(total_frames):
    cap.set(cv2.CAP_PROP_POS_FRAMES, fn)
    ret, frame = cap.read()
    if not ret:
        break

    sec = fn / fps
    # Nájdi najbližší FIT záznam
    closest_fn = min(frame_to_radar.keys(), key=lambda x: abs(x - fn))
    radar = frame_to_radar[closest_fn] if abs(closest_fn - fn) <= 30 else []

    has_car = len(radar) > 0
    color   = (0, 255, 0) if has_car else (100, 100, 100)

    cv2.putText(frame, f"frame={fn:04d}  sec={sec:.2f}",
                (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
    cv2.putText(frame, f"radar={radar}",
                (10, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

    cars_tag = f"_CAR{'_'.join(str(d)+'m' for d in radar)}" if has_car else "_nocar"
    fname    = f"{OUT_DIR}/f{fn:04d}_sec{sec:.2f}{cars_tag}.jpg"
    cv2.imwrite(fname, frame)

cap.release()
print(f"✅ Hotovo! {total_frames} framov v '{OUT_DIR}/'")
print(f"Súbory s '_CAR' v názve = radar videl auto")
print(f"Hľadaj PRVÝ frame kde vidíš auto vzadu v obraze")