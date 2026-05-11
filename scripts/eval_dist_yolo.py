import argparse
import sys
from pathlib import Path
import numpy as np
import torch
import yaml
import cv2
from torchvision.ops import nms as tv_nms

sys.path.insert(0, '/home/jozef/Documents/FIIT/5thSemester/BP1')


def letterbox(img, new_shape=(832, 832), color=(114, 114, 114)):
    shape = img.shape[:2]
    if isinstance(new_shape, int):
        new_shape = (new_shape, new_shape)
    r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
    new_unpad = (int(round(shape[1] * r)), int(round(shape[0] * r)))
    dw = (new_shape[1] - new_unpad[0]) / 2
    dh = (new_shape[0] - new_unpad[1]) / 2
    if shape[::-1] != new_unpad:
        img = cv2.resize(img, new_unpad, interpolation=cv2.INTER_LINEAR)
    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    img = cv2.copyMakeBorder(img, top, bottom, left, right,
                             cv2.BORDER_CONSTANT, value=color)
    return img, r, (dw, dh)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', required=True)
    parser.add_argument('--data', required=True, help='Cesta k dataset.yaml')
    parser.add_argument('--max-distance', type=float, default=90.0)
    parser.add_argument('--conf', type=float, default=0.25)
    parser.add_argument('--iou-thr', type=float, default=0.5)
    parser.add_argument('--split', default='test', choices=['train', 'val', 'test'])
    args = parser.parse_args()


    print(f"  Model:        {args.model}")
    print(f"  Data:         {args.data}")
    print(f"  Split:        {args.split}")
    print(f"  MAX_DISTANCE: {args.max_distance} m")

    # Nacitaj YAML
    with open(args.data) as f:
        ds_cfg = yaml.safe_load(f)

    base_path = Path(ds_cfg.get('path', ''))
    split_path = ds_cfg.get(args.split, f'{args.split}/images')

    if Path(split_path).is_absolute():
        images_dir = Path(split_path)
    else:
        images_dir = base_path / split_path

    labels_dir = Path(str(images_dir).replace('/images', '/labels'))
    nc = ds_cfg.get('nc', 2)

    print(f"  Images: {images_dir}")
    print(f"  Labels: {labels_dir}")

    if not images_dir.exists():
        print(f"Adresar neexistuje: {images_dir}")
        return

    print(f"  Images count: {len(list(images_dir.glob('*.jpg')))}")
    print(f"  Detected nc from YAML: {nc}")

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    ckpt = torch.load(args.model, map_location=device, weights_only=False)
    model = ckpt['model'].float().to(device)
    model.eval()

    # Eval
    print("\n[2] Inference + distance MAE...")

    distance_errors = []
    n_predictions = 0
    n_matches = 0
    expected_total = 4 + nc + 1

    with torch.no_grad():
        for img_file in sorted(images_dir.glob('*.jpg')):
            lbl_file = labels_dir / (img_file.stem + '.txt')
            if not lbl_file.exists():
                continue

            gt_data = []
            with open(lbl_file) as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) >= 6:
                        try:
                            gt_data.append({
                                'cls': int(parts[0]),
                                'cx': float(parts[1]), 'cy': float(parts[2]),
                                'w': float(parts[3]), 'h': float(parts[4]),
                                'distance_m': float(parts[5]) * args.max_distance,
                            })
                        except ValueError:
                            continue

            if not gt_data:
                continue

            img_bgr = cv2.imread(str(img_file))
            if img_bgr is None:
                continue
            H_orig, W_orig = img_bgr.shape[:2]

            img_lb, ratio, (dw, dh) = letterbox(img_bgr, (832, 832))
            img_rgb = cv2.cvtColor(img_lb, cv2.COLOR_BGR2RGB)
            img_t = torch.from_numpy(img_rgb).float().to(device) / 255.0
            img_t = img_t.permute(2, 0, 1).unsqueeze(0)

            preds = model(img_t)
            pred_tensor = preds[0] if isinstance(preds, (list, tuple)) else preds
            if pred_tensor.dim() != 3:
                continue
            if pred_tensor.shape[1] == expected_total:
                pred_tensor = pred_tensor.permute(0, 2, 1)

            box_data = pred_tensor[0]
            cls_conf, cls_id = box_data[:, 4:4 + nc].max(dim=1)
            mask = cls_conf > args.conf

            if mask.sum() == 0:
                continue

            filt = box_data[mask]
            cls_filt = cls_id[mask]
            conf_filt = cls_conf[mask]
            dist_filt = filt[:, 4 + nc]

            cx_p = filt[:, 0];
            cy_p = filt[:, 1]
            w_p = filt[:, 2];
            h_p = filt[:, 3]
            x1 = cx_p - w_p / 2;
            y1 = cy_p - h_p / 2
            x2 = cx_p + w_p / 2;
            y2 = cy_p + h_p / 2
            boxes_for_nms = torch.stack([x1, y1, x2, y2], dim=1)

            keep = tv_nms(boxes_for_nms, conf_filt, iou_threshold=0.45)
            final_boxes = boxes_for_nms[keep]
            final_dist = dist_filt[keep] * args.max_distance
            final_cls = cls_filt[keep]

            final_boxes_np = final_boxes.clone().cpu().numpy()
            final_boxes_np[:, [0, 2]] -= dw
            final_boxes_np[:, [1, 3]] -= dh
            final_boxes_np /= ratio
            final_boxes_np[:, 0::2] = np.clip(final_boxes_np[:, 0::2], 0, W_orig)
            final_boxes_np[:, 1::2] = np.clip(final_boxes_np[:, 1::2], 0, H_orig)
            pred_distances = final_dist.cpu().numpy()
            n_predictions += len(final_boxes_np)

            for gt in gt_data:
                gt_x1 = (gt['cx'] - gt['w'] / 2) * W_orig
                gt_y1 = (gt['cy'] - gt['h'] / 2) * H_orig
                gt_x2 = (gt['cx'] + gt['w'] / 2) * W_orig
                gt_y2 = (gt['cy'] + gt['h'] / 2) * H_orig

                best_iou = 0
                best_pd = None
                for i in range(len(final_boxes_np)):
                    px1, py1, px2, py2 = final_boxes_np[i]
                    ix1 = max(gt_x1, px1);
                    iy1 = max(gt_y1, py1)
                    ix2 = min(gt_x2, px2);
                    iy2 = min(gt_y2, py2)
                    inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
                    gt_a = (gt_x2 - gt_x1) * (gt_y2 - gt_y1)
                    p_a = (px2 - px1) * (py2 - py1)
                    u = gt_a + p_a - inter
                    iou_val = inter / (u + 1e-9)
                    if iou_val > best_iou:
                        best_iou = iou_val
                        best_pd = pred_distances[i]

                if best_iou > args.iou_thr and best_pd is not None:
                    err = abs(gt['distance_m'] - float(best_pd))
                    distance_errors.append({
                        'gt': gt['distance_m'],
                        'pred': float(best_pd),
                        'err': err,
                    })
                    n_matches += 1

    print(f"  Predictions: {n_predictions}")
    print(f"  Matches:     {n_matches}")

    if not distance_errors:
        print("\nZiadne matches!")
        return

    errors = np.array([d['err'] for d in distance_errors])
    gts = np.array([d['gt'] for d in distance_errors])

    print(f"\n  Total MAE:   {errors.mean():.2f} m")
    print(f"  RMSE:        {np.sqrt((errors ** 2).mean()):.2f} m")

    print("\n  Per-bin MAE:")
    for lo, hi in [(0, 20), (20, 40),(0,50), (40, 60), (60, 90)]:
        mask = (gts >= lo) & (gts < hi)
        if mask.sum() > 0:
            print(f"    {lo}-{hi}m: MAE={errors[mask].mean():.2f}m (n={int(mask.sum())})")

    # MAE pri ≤50m
    mask = gts <= 50
    if mask.sum() > 0:
        print(f"\n  MAE ≤50m: {errors[mask].mean():.2f}m (n={int(mask.sum())})")


if __name__ == '__main__':
    main()