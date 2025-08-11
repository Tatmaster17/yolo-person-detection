import time
from ultralytics import YOLO
import torch
import numpy as np


model = YOLO("best.pt")


CONF_THR = 0.35   # .25, 0.30, 0.35, 0.4, 0.5
IOU_NMS = 0.45
MAX_DET = 100
MIN_BOX_AREA_RATIO = 1e-4  # минимальная площадь бокса относительно изображения, убрать очень мелкие
IMG_SIZE = 1024

def _normalize_box(x1, y1, x2, y2, w_img, h_img):
    xc = ((x1 + x2) / 2.0) / w_img
    yc = ((y1 + y2) / 2.0) / h_img
    w = (x2 - x1) / w_img
    h = (y2 - y1) / h_img
    return xc, yc, w, h

def predict(images: list, image_ids: list = None) -> list:
    all_predictions = []
    use_gpu = 0 if torch.cuda.is_available() else "cpu"

    if image_ids is None:
        # если id не переданы — составим из индексов
        image_ids = [str(i) for i in range(len(images))]

    for img, img_id in zip(images, image_ids):
        t0 = time.time()
        results = model.predict(
            source=img,
            imgsz=IMG_SIZE,
            conf=CONF_THR,    # фильтр confid
            iou=IOU_NMS,      # NMS IoU
            max_det=MAX_DET,
            device=use_gpu,
            verbose=False
        )
        time_spent = time.time() - t0

        # ultralytics predict возвращает список результатов (по одному на изображение)
        res = results[0]
        w_img, h_img = int(res.orig_shape[1]), int(res.orig_shape[0])

        image_preds = []
        # если нет боксов добавляем пустую запись
        if len(res.boxes) == 0:
            image_preds.append({
                'image_id': str(img_id),
                'label': 0,
                'xc': None,
                'yc': None,
                'w': None,
                'h': None,
                'w_img': w_img,
                'h_img': h_img,
                'score': None,
                'time_spent': float(time_spent)
            })
            all_predictions.append(image_preds)
            continue

        for box in res.boxes:
            cls = int(box.cls.cpu().numpy()[0])
            conf = float(box.conf.cpu().numpy()[0])
            # Отсекаем нелюдей (если класс person==0)
            if cls != 0:
                continue

            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().tolist()

            # фильтр по минимальной площади (во избежать мелких шумов)
            box_area = max(0.0, (x2 - x1) * (y2 - y1))
            if box_area / (w_img * h_img) < MIN_BOX_AREA_RATIO:
                continue

            xc, yc, w, h = _normalize_box(x1, y1, x2, y2, w_img, h_img)

            # двойная проверка диапазонов
            if not (0 <= xc <= 1 and 0 <= yc <= 1 and 0 < w <= 1 and 0 < h <= 1):
                continue

            image_preds.append({
                'image_id': str(img_id),
                'label': 0,
                'xc': float(xc),
                'yc': float(yc),
                'w': float(w),
                'h': float(h),
                'w_img': int(w_img),
                'h_img': int(h_img),
                'score': float(conf),
                'time_spent': float(time_spent)
            })

        # если после фильтрации пусто добавить пустую запись с time_spent
        if not image_preds:
            image_preds.append({
                'image_id': str(img_id),
                'label': 0,
                'xc': None,
                'yc': None,
                'w': None,
                'h': None,
                'w_img': w_img,
                'h_img': h_img,
                'score': None,
                'time_spent': float(time_spent)
            })

        all_predictions.append(image_preds)

    return all_predictions
