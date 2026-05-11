import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from dist_yolo import DistYOLO


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data',
        default='/home/jozef/Documents/FIIT/5thSemester/BP1/nuscenes_dist_dataset/dataset.yaml')
    parser.add_argument('--cfg',
        default='/home/jozef/Documents/FIIT/5thSemester/ultralytics/ultralytics/cfg/models/11/dist-yolo11n.yaml')
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--batch', type=int, default=16)
    parser.add_argument('--imgsz', type=int, default=832)
    parser.add_argument('--name', default='dist_yolo_nuscenes_full')
    parser.add_argument('--device', default='0')
    parser.add_argument('--patience', type=int, default=10,
                        help='nuScenes konverguje rychlejsie')
    args = parser.parse_args()

    print("DIST-YOLO TRAINING - NUSCENES")
    print(f"  Config:   {args.cfg}")
    print(f"  Data:     {args.data}")
    print(f"  Epochs:   {args.epochs}")
    print(f"  Batch:    {args.batch}")
    print(f"  Img size: {args.imgsz}")
    print(f"  Name:     {args.name}")

    model = DistYOLO(args.cfg, task='detect')

    results = model.train(
        data=args.data,
        epochs=args.epochs,
        batch=args.batch,
        imgsz=args.imgsz,
        name=args.name,
        device=args.device,

        mosaic=0.0,
        mixup=0.0,
        cutmix=0.0,
        copy_paste=0.0,

        fliplr=0.5,
        hsv_h=0.015,
        hsv_s=0.7,
        hsv_v=0.4,

        # Optimizácia
        optimizer='SGD',
        lr0=0.01,
        patience=args.patience,

        workers=8,
    )

if __name__ == '__main__':
    main()