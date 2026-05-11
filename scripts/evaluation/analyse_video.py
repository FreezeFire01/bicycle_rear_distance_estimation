"""
Pouzitie:
  python analyse_video.py --video sample.mp4 \\
      --yolo final_models/yolo_garmin.pt \\
      --regressor final_models/rf_garmin_pixel.pkl \\
      --feature-type pixel

    priklad:
    python analyse_video.py \
    --video /home/jozef/Documents/FIIT/5thSemester/BP1/scripts/evaluation/GRMN0723.MP4 \
    --yolo /home/jozef/Documents/FIIT/5thSemester/BP1/final_models/garmin_only_yolo11n_v3/weights/best.pt \
    --regressor /home/jozef/Documents/FIIT/5thSemester/BP1/final_models/rf_garmin_fov.pkl \
    --feature-type fov

  Typy features (--feature-type):
    full   = 7 features s cx, cy (default rf_garmin_full)
    pixel  = 6 features s offset_x (rf_garmin_pixel)
    fov    = 6 features FOV-invariant uhlove (rf_garmin_fov, rf_nuscenes_fov)

  Ak --feature-type nie je zadany, skript ho odhadne podla nazvu suboru
  alebo podla poctu features v RF modeli.
"""

import argparse
import cv2
import joblib
import numpy as np
from pathlib import Path
from ultralytics import YOLO

GARMIN_FOV_H = 122
GARMIN_FOV_V = 68.6


def parse_args():
    parser = argparse.ArgumentParser(description='Vehicle distance estimation preview')
    parser.add_argument('--video', required=True, help='Cesta k video súboru alebo cislo kamery')
    parser.add_argument('--yolo', required=True, help='Cesta k YOLO modelu (.pt)')
    parser.add_argument('--regressor', required=True, help='Cesta k RF regresoru (.pkl)')
    parser.add_argument('--feature-type', choices=['full', 'pixel', 'fov', 'auto'],
                        default='auto',
                        help='Typ features (auto = detekcia z nazvu/poctu)')
    parser.add_argument('--conf', type=float, default=0.3, help='YOLO confidence threshold')
    parser.add_argument('--save', type=str, default=None, help='Voliteľné: uložiť výstup')
    return parser.parse_args()


def detect_feature_type(rf_path, n_features, override=None):
    if override and override != 'auto':
        return override

    name = Path(rf_path).name.lower()

    # Detekcia podla nazvu suboru
    if 'fov' in name:
        return 'fov'
    if 'pixel' in name:
        return 'pixel'
    if 'full' in name:
        return 'full'

    if n_features == 7:
        return 'full'
    if n_features == 6:
        return 'pixel'

    raise ValueError(f'Neviem detegovat typ pre {n_features} features')


def extract_features(x1, y1, x2, y2, img_w, img_h, ftype):
    # Geometrické základy
    w_px = x2 - x1
    h_px = y2 - y1
    cx_px = x1 + w_px / 2.0
    cy_px = y1 + h_px / 2.0

    # Normalizovane (frakcie obrazu)
    cx = cx_px / img_w
    cy = cy_px / img_h
    w_n = w_px / img_w
    h_n = h_px / img_h
    area_n = w_n * h_n
    aspect = w_px / (h_px + 1e-6)
    y_bottom = y2 / img_h
    offset_x_n = cx - 0.5

    if ftype == 'full':
        # 7 features: cx, cy, w, h, area, aspect, y_bottom
        return np.array([[cx, cy, w_n, h_n, area_n, aspect, y_bottom]],
                        dtype=np.float32)

    elif ftype == 'pixel':
        # 6 features: w, h, area, aspect, y_bottom, offset_x
        return np.array([[w_n, h_n, area_n, aspect, y_bottom, offset_x_n]],
                        dtype=np.float32)

    elif ftype == 'fov':
        # 6 features FOV-invariant: w_angle, h_angle, area_angle, aspect, y_angle, offset_angle
        w_angle = w_n * GARMIN_FOV_H
        h_angle = h_n * GARMIN_FOV_V
        area_angle = w_angle * h_angle
        y_angle = (y_bottom - 0.5) * GARMIN_FOV_V
        offset_angle = (cx - 0.5) * GARMIN_FOV_H
        return np.array([[w_angle, h_angle, area_angle, aspect, y_angle, offset_angle]],
                        dtype=np.float32)

    raise ValueError(f'Neznámy typ features: {ftype}')


def main():
    args = parse_args()

    yolo_path = Path(args.yolo)
    rf_path = Path(args.regressor)

    if not yolo_path.exists():
        print(f'YOLO model neexistuje: {yolo_path}')
        return
    if not rf_path.exists():
        print(f'RF regresor neexistuje: {rf_path}')
        return

    print(f'Loading YOLO: {yolo_path}')
    det_model = YOLO(str(yolo_path))

    print(f'Loading regressor: {rf_path}')
    reg = joblib.load(rf_path)
    n_features = reg.n_features_in_

    ftype = detect_feature_type(rf_path, n_features, args.feature_type)
    print(f'Features expected: {n_features}')
    print(f'Feature type: {ftype.upper()}')

    video_src = args.video
    if video_src.isdigit():
        video_src = int(video_src)
    elif not Path(video_src).exists():
        print(f'Video neexistuje: {video_src}')
        return

    cap = cv2.VideoCapture(video_src)
    if not cap.isOpened():
        print(f'Nepodarilo sa otvoriť video')
        return

    # Optional output writer
    writer = None
    if args.save:
        fps = cap.get(cv2.CAP_PROP_FPS) or 30
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        writer = cv2.VideoWriter(args.save, fourcc, fps, (w, h))
        print(f'Saving to: {args.save}')

    win = "Bicycle Distance Estimation"
    cv2.namedWindow(win, cv2.WINDOW_NORMAL)
    print('Stlač Q alebo ESC pre ukončenie')

    while True:
        ok, frame = cap.read()
        if not ok:
            break

        results = det_model.predict(frame, conf=args.conf, imgsz=(832, 480),
                                    verbose=False)
        h, w = frame.shape[:2]
        boxes = results[0].boxes

        if boxes is not None and len(boxes) > 0:
            xyxy = boxes.xyxy.cpu().numpy()

            for (x1, y1, x2, y2) in xyxy:
                X = extract_features(x1, y1, x2, y2, w, h, ftype)
                dist_m = reg.predict(X)[0]

                # Draw
                cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)),
                              (0, 0, 255), 2)
                cv2.putText(frame, f"{dist_m:.1f} m",
                            (int(x1), max(int(y1) - 10, 10)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

        # Info overlay
        cv2.putText(frame, f"Model: {rf_path.name} ({ftype})",
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)

        if writer is not None:
            writer.write(frame)

        cv2.imshow(win, frame)
        k = cv2.waitKey(1) & 0xFF
        if k == 27 or k == ord('q'):
            break

    cap.release()
    if writer is not None:
        writer.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()