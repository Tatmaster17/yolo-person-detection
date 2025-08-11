#!/bin/bash

yolo task=detect \
    mode=train \
    model=/workspace/training/yolo8x_finetune_heavy/weights/best.pt \
    data=/workspace/training/02_part_dataset/data.yaml \
    epochs=7 \
    imgsz=1024 \
    batch=8 \
    amp=True \
    workers=12 \
    shear=5 \
    project=/workspace/training/yolo8x_finetune_heavy2soft \
    lr0=0.00015 \
    cos_lr=True \
    erasing=0.15 \
    mosaic=0.4 \
    mixup=0.05 \
    degrees=5 \
    translate=0.1 \
    scale=0.85 \
    cache=True