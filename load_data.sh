#!/bin/bash

# ------------------ OCR Model ------------------
# EasyORC_custom
echo "EasyORC_custom"
gdown 1mZ_nfXYpmV91gPYCtGdG89Qh3x0qPfTQ -O ./src/weights/ocr/
# LPRNet
echo "LPRNet"
gdown 1rzfZ5UB_8LJ0pg9s9jHVu4GCCnKHEriB -O ./src/weights/ocr/

# ------------------ Detect model ------------------
# FasterRCNN
echo "FasterRCNN"
gdown 1wEvIJs7quYkljh0uzSRkmwn-4SZ5RJHb -O ./src/weights/detect/
# YoloV5
echo "YoloV5"
gdown 1Tyz5YJGjzPGlivp4DoJyegkzU2IUmcvo -O ./src/weights/detect/
# Pnet
echo "Pnet"
gdown 1z5IrXUrEJ3O2Kehu5JjQKp5iLZMUmAne -O ./src/weights/detect/
# Onet
echo "Onet"
gdown 1fAjcVAeE9Hz_otMF2kHeuTW0I8rFwdGM -O ./src/weights/detect/

# ------------------ Transform model ------------------
# STN
echo "STN"
gdown 1cnuUkpxBkMThqdFvR4QPDu2zP53ziaBS -O ./src/weights/transform/
