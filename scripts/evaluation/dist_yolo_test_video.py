import sys
sys.path.insert(0, '/BP1')

import cv2
import numpy as np
import torch
from torchvision.ops import nms as tv_nms
from dist_yolo import DistYOLO

MODEL_PATH = "/BP1/runs/detect/dist_yolo_garmin_v2/weights/best.pt"
VIDEO_PATH = "/GARMIN_VIDEOS/2026-04-05/MP/GRMN0723.MP4"
MAX_DISTANCE = 90.0
CONF_THR = 0.25
NC = 2
CLASS_NAMES = ['car', 'truck']
IMGSZ = 832

yolo = DistYOLO(MODEL_PATH, task='detect')
device = 'cuda' if torch.cuda.is_available() else 'cpu'
model = yolo.model.float().to(device)
model.eval()


def letterbox(img, new_shape=IMGSZ, color=(114, 114, 114)):
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


def color_for_distance(d):
    if d < 15:   return (0, 0, 255)
    elif d < 30: return (0, 165, 255)
    elif d < 50: return (0, 255, 255)
    else:        return (0, 255, 0)


cap = cv2.VideoCapture(VIDEO_PATH)
win = "Dist-YOLO preview (q/ESC = quit, s = screenshot)"
cv2.namedWindow(win, cv2.WINDOW_NORMAL)

while True:
    ok, frame = cap.read()
    if not ok:
        break

    H, W = frame.shape[:2]

    img_lb, ratio, (dw, dh) = letterbox(frame, IMGSZ)
    img_rgb = cv2.cvtColor(img_lb, cv2.COLOR_BGR2RGB)
    img_t = torch.from_numpy(img_rgb).float().to(device) / 255.0
    img_t = img_t.permute(2, 0, 1).unsqueeze(0)

    with torch.no_grad():
        preds = model(img_t)

    pred = preds[0] if isinstance(preds, (list, tuple)) else preds

    expected_total = 4 + NC + 1
    if pred.dim() == 3 and pred.shape[1] == expected_total:
        pred = pred.permute(0, 2, 1)

    box_data = pred[0]

    cls_conf, cls_id = box_data[:, 4:4+NC].max(dim=1)
    mask = cls_conf > CONF_THR

    if mask.sum() > 0:
        filt = box_data[mask]
        cls_filt = cls_id[mask]
        conf_filt = cls_conf[mask]
        dist_filt = filt[:, 4 + NC] * MAX_DISTANCE

        cx, cy, w, h = filt[:, 0], filt[:, 1], filt[:, 2], filt[:, 3]
        x1 = cx - w / 2; y1 = cy - h / 2
        x2 = cx + w / 2; y2 = cy + h / 2
        boxes_nms = torch.stack([x1, y1, x2, y2], dim=1)

        keep = tv_nms(boxes_nms, conf_filt, iou_threshold=0.45)
        final_boxes = boxes_nms[keep].cpu().numpy()
        final_dist = dist_filt[keep].cpu().numpy()
        final_cls = cls_filt[keep].cpu().numpy()

        final_boxes[:, [0, 2]] -= dw
        final_boxes[:, [1, 3]] -= dh
        final_boxes /= ratio
        final_boxes[:, 0::2] = np.clip(final_boxes[:, 0::2], 0, W)
        final_boxes[:, 1::2] = np.clip(final_boxes[:, 1::2], 0, H)

        for i in range(len(final_boxes)):
            x1, y1, x2, y2 = final_boxes[i].astype(int)
            d_m = float(final_dist[i])
            cls_name = CLASS_NAMES[int(final_cls[i])]
            color = color_for_distance(d_m)

            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            cv2.putText(frame,
                        f"{cls_name} {d_m:.1f}m",
                        (x1, max(y1 - 10, 10)),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.7, color, 2)

    cv2.imshow(win, frame)
    k = cv2.waitKey(1) & 0xFF
    if k == 27 or k == ord('q'):
        break
    elif k == ord('s'):
        import time
        fname = f"/tmp/distyolo_screenshot_{int(time.time())}.jpg"
        cv2.imwrite(fname, frame)
        print(f"Screenshot saved: {fname}")

cap.release()
cv2.destroyAllWindows()