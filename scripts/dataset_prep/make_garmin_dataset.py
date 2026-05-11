"""
Použitie:
  python make_garmin_dataset.py \
    --csv activity_20260412.csv \
    --video-dir /home/jozef/Documents/FIIT/5thSemester/GARMIN_VIDEOS/2026-04-12/MP \
    --output-dir /home/jozef/Documents/FIIT/5thSemester/BP1/garmin_dataset \
    --offset -1.30 \
    --date 20260227
"""

import argparse
import csv
import json
import os
import shutil
import subprocess
import sys
from datetime import datetime, timedelta
from pathlib import Path



#parsing
def parse_fit_records(csv_path: str) -> list[dict]:
    records = []
    with open(csv_path, 'r') as f:
        reader = csv.reader(f)
        header = next(reader)
        idx = {col: header.index(col) for col in [
            'timestamp', 'radar_current', 'radar_ranges', 'radar_speeds',
            'position_lat', 'position_long', 'enhanced_speed',
        ]}
        for row in reader:
            if row[0] != 'record':
                continue
            ts_str = row[idx['timestamp']]
            if not ts_str:
                continue
            ts = datetime.strptime(ts_str, "%Y-%m-%d %H:%M:%S")
            ranges = parse_tuple(row[idx['radar_ranges']])
            speeds = parse_tuple(row[idx['radar_speeds']])
            non_zero = [r for r in ranges if r > 0]
            lat_raw = row[idx['position_lat']]
            lon_raw = row[idx['position_long']]
            spd_raw = row[idx['enhanced_speed']]
            records.append({
                'timestamp': ts,
                'num_cars': len(non_zero),
                'closest_m': min(non_zero) if non_zero else None,
                'ranges': ranges,
                'speeds': speeds,
                'lat': float(lat_raw) * (180.0 / 2 ** 31) if lat_raw else None,
                'lon': float(lon_raw) * (180.0 / 2 ** 31) if lon_raw else None,
                'speed_kmh': float(spd_raw) * 3.6 if spd_raw else None,
            })
    records.sort(key=lambda r: r['timestamp'])
    return records


def parse_tuple(s: str) -> list[int]:
    if not s:
        return []
    result = []
    for val in s.strip('()').split(','):
        val = val.strip()
        if val in ('None', ''):
            result.append(0)
        else:
            try:
                result.append(int(float(val)))
            except ValueError:
                result.append(0)
    return result

#video helper
def get_clip_start_utc(mp4_path: str) -> datetime:
    cmd = ['ffprobe', '-v', 'quiet', '-print_format', 'json',
           '-show_format', str(mp4_path)]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"ffprobe failed: {result.stderr}")
    meta = json.loads(result.stdout)
    ct = meta['format']['tags']['creation_time'].replace('Z', '+00:00')
    return datetime.fromisoformat(ct).replace(tzinfo=None)


def get_fps(mp4_path: str) -> float:
    cmd = ['ffprobe', '-v', 'quiet', '-print_format', 'json',
           '-show_streams', '-select_streams', 'v:0', str(mp4_path)]
    result = subprocess.run(cmd, capture_output=True, text=True)
    meta = json.loads(result.stdout)
    fps_str = meta['streams'][0]['r_frame_rate']
    if '/' in fps_str:
        n, d = fps_str.split('/')
        return float(n) / float(d)
    return float(fps_str)


def extract_all_frames(mp4_path: str, temp_dir: str) -> int:
    os.makedirs(temp_dir, exist_ok=True)
    existing = [f for f in os.listdir(temp_dir) if f.endswith('.jpg')]
    if len(existing) > 800:
        return len(existing)
    cmd = ['ffmpeg', '-y', '-i', str(mp4_path),
           '-qmin', '1', '-q:v', '2',
           os.path.join(temp_dir, 'f%05d.jpg')]
    subprocess.run(cmd, capture_output=True, text=True)
    return len([f for f in os.listdir(temp_dir) if f.endswith('.jpg')])



#fit matching
def find_nearest_fit(fit_records, target_time):
    timestamps = [r['timestamp'] for r in fit_records]
    lo, hi = 0, len(timestamps) - 1
    while lo < hi:
        mid = (lo + hi) // 2
        if timestamps[mid] < target_time:
            lo = mid + 1
        else:
            hi = mid
    best_idx = lo
    best_diff = abs((timestamps[lo] - target_time).total_seconds())
    if lo > 0:
        diff_prev = abs((timestamps[lo - 1] - target_time).total_seconds())
        if diff_prev < best_diff:
            best_idx = lo - 1
            best_diff = diff_prev
    return fit_records[best_idx], best_diff


def interpolate_distance(fit_records, target_time, car_index=0):
    timestamps = [r['timestamp'] for r in fit_records]
    lo, hi = 0, len(timestamps) - 1
    while lo < hi:
        mid = (lo + hi) // 2
        if timestamps[mid] < target_time:
            lo = mid + 1
        else:
            hi = mid
    if lo == 0:
        i1, i2 = 0, min(1, len(fit_records) - 1)
    elif lo >= len(fit_records):
        i1, i2 = len(fit_records) - 2, len(fit_records) - 1
    else:
        i1, i2 = lo - 1, lo
    r1, r2 = fit_records[i1], fit_records[i2]
    if car_index >= len(r1['ranges']) or car_index >= len(r2['ranges']):
        return None
    d1, d2 = r1['ranges'][car_index], r2['ranges'][car_index]
    if d1 == 0 or d2 == 0:
        return None
    dt_total = (r2['timestamp'] - r1['timestamp']).total_seconds()
    if dt_total == 0:
        return float(d1)
    alpha = (target_time - r1['timestamp']).total_seconds() / dt_total
    return d1 + (d2 - d1) * alpha



#pipeline
def process_clip(mp4_path, fit_records, images_dir, date_prefix, offset_s, fps=29.97):
    clip_name = Path(mp4_path).stem  # e.g. GRMN0073
    clip_start = get_clip_start_utc(mp4_path)

    # Temp dir for all 900 frames
    temp_dir = os.path.join('/tmp', f'garmin_frames_{clip_name}')
    total_frames = extract_all_frames(mp4_path, temp_dir)

    duration = total_frames / fps
    num_seconds = int(duration)

    labels = []

    for sec in range(num_seconds):
        # Real time for this second (with offset correction)
        real_time = clip_start + timedelta(seconds=sec) - timedelta(seconds=offset_s)

        # Pick exact frame
        frame_idx = round(sec * fps) + 1
        src_file = f'f{frame_idx:05d}.jpg'
        src_path = os.path.join(temp_dir, src_file)

        if not os.path.exists(src_path):
            continue

        # Output name: 20260227_GRMN0073_s00.jpg
        out_name = f"{date_prefix}_{clip_name}_s{sec:02d}.jpg"
        dst_path = os.path.join(images_dir, out_name)
        shutil.copy2(src_path, dst_path)

        # Match to FIT
        fit_rec, sync_off = find_nearest_fit(fit_records, real_time)
        closest_interp = interpolate_distance(fit_records, real_time, 0)

        labels.append({
            'image': out_name,
            'clip': clip_name,
            'frame_index': frame_idx,
            'second_in_clip': sec,
            'timestamp_utc': real_time.strftime("%Y-%m-%d %H:%M:%S"),
            'fit_timestamp': fit_rec['timestamp'].strftime("%Y-%m-%d %H:%M:%S"),
            'sync_offset_s': round(sync_off, 3),
            'num_cars': fit_rec['num_cars'],
            'closest_m': fit_rec['closest_m'],
            'closest_m_interp': round(closest_interp, 1) if closest_interp else None,
            'all_ranges': str(fit_rec['ranges']),
            'all_speeds': str(fit_rec['speeds']),
            'lat': round(fit_rec['lat'], 6) if fit_rec['lat'] else None,
            'lon': round(fit_rec['lon'], 6) if fit_rec['lon'] else None,
            'cyclist_speed_kmh': round(fit_rec['speed_kmh'], 1) if fit_rec['speed_kmh'] else None,
        })

    return labels


def main():
    parser = argparse.ArgumentParser(description='Garmin → YOLO Dataset Generator')
    parser.add_argument('--csv', required=True, help='activity.csv (FIT export)')
    parser.add_argument('--video-dir', required=True, help='Priečinok s GRMN*.MP4')
    parser.add_argument('--output-dir', default='./garmin_dataset', help='Výstupný priečinok')
    parser.add_argument('--offset', type=float, default=-1.15,
                        help='Sync offset (default: -1.15, empiricky nameraný)')
    parser.add_argument('--date', default=None,
                        help='Dátum jazdy pre naming (napr. 20260227). Ak nie je zadaný, vezme sa z FIT.')
    parser.add_argument('--clip', default=None, help='Spracuj len jeden klip')
    args = parser.parse_args()

    # Parse FIT
    print("Parsing FIT records...")
    fit_records = parse_fit_records(args.csv)
    print(f"  {len(fit_records)} records: {fit_records[0]['timestamp']} → {fit_records[-1]['timestamp']}")

    # Date prefix
    if args.date:
        date_prefix = args.date
    else:
        date_prefix = fit_records[0]['timestamp'].strftime("%Y%m%d")
    print(f"  Date prefix: {date_prefix}")

    # Find clips
    video_dir = Path(args.video_dir)
    if args.clip:
        clips = [video_dir / args.clip]
    else:
        clips = sorted(video_dir.glob('GRMN*.MP4'))

    print(f"  Found {len(clips)} clips")

    # Prepare output
    images_dir = os.path.join(args.output_dir, 'images')
    os.makedirs(images_dir, exist_ok=True)

    # Process all clips
    all_labels = []
    fps = None

    for i, mp4_path in enumerate(clips):
        if not mp4_path.exists():
            print(f"{mp4_path.name} not found, skipping")
            continue

        clip_name = mp4_path.stem

        if fps is None:
            fps = get_fps(str(mp4_path))
            print(f"  FPS: {fps:.2f}")

        print(f"  [{i + 1}/{len(clips)}] {clip_name}...", end='', flush=True)

        try:
            labels = process_clip(
                str(mp4_path), fit_records, images_dir,
                date_prefix, args.offset, fps
            )
            all_labels.extend(labels)

            cars = sum(1 for l in labels if l['num_cars'] > 0)
            print(f" {len(labels)} frames, {cars} with cars")
        except Exception as e:
            print(f" ERROR: {e}")
            continue

    # Write master metadata CSV
    if all_labels:
        meta_path = os.path.join(args.output_dir, 'metadata.csv')
        fieldnames = list(all_labels[0].keys())

        # Load existing metadata if present
        existing_images = set()
        existing_rows = []
        if os.path.exists(meta_path):
            with open(meta_path, 'r') as f:
                reader = csv.DictReader(f)
                for row in reader:
                    existing_images.add(row['image'])
                    existing_rows.append(row)

        # Merge: keep existing + add new
        new_labels = [l for l in all_labels if l['image'] not in existing_images]
        all_rows = existing_rows + [{k: str(v) for k, v in l.items()} for l in new_labels]

        with open(meta_path, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(all_rows)

        if existing_rows:
            print(
                f"\nMetadata: {meta_path} (existujúcich: {len(existing_rows)}, nových: {len(new_labels)}, celkom: {len(all_rows)})")
        else:
            print(f"\nMetadata: {meta_path} ({len(all_labels)} riadkov)")

    # Write sync info (per-date, doesn't overwrite other dates)
    sync_info = {
        'date': date_prefix,
        'fit_csv': args.csv,
        'offset_s': args.offset,
        'offset_meaning': 'Video starts 1.15s AFTER creation_time. '
                          'Measured by matching 6m→0m radar disappearance '
                          'event in GRMN0081 (FIT 13:55:06) to frame 536.',
        'fps': fps,
        'frame_selection': 'frame_index = round(second * fps) + 1',
        'fit_timezone': 'UTC (verified via timestamp_correlation: local=UTC+3600s=CET)',
        'mp4_timezone': 'UTC (creation_time suffix Z)',
        'precision': '±1s (FIT 1Hz sampling + creation_time floor rounding)',
        'total_clips': len(clips),
        'total_frames': len(all_labels),
        'frames_with_cars': sum(1 for l in all_labels if l['num_cars'] > 0),
    }
    sync_path = os.path.join(args.output_dir, f'sync_info_{date_prefix}.json')
    with open(sync_path, 'w') as f:
        json.dump(sync_info, f, indent=2)
    print(f"Sync info: {sync_path}")

    # Print summary
    total = len(all_labels)
    with_cars = sum(1 for l in all_labels if l['num_cars'] > 0)

    print(f"DATASET SUMMARY")
    print(f"  Images:         {total}")
    print(f"  With cars:      {with_cars} ({100 * with_cars / max(total, 1):.1f}%)")
    print(f"  Without cars:   {total - with_cars}")
    print(f"  Location:       {args.output_dir}/images/")
    print(f"  Metadata:       {args.output_dir}/metadata.csv")

if __name__ == '__main__':
    main()