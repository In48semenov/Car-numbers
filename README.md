# Car number
**Detection of car numbers and their recognition**

# Dataset
1. [Kaggle - VKCV 2022 Contest_02](https://www.kaggle.com/competitions/vkcv2022-contest-02-carplates/data)
2. [Kaggle - Nomeroff Russian License plates](https://www.kaggle.com/datasets/evgrafovmaxim/nomeroff-russian-license-plates?select=autoriaNumberplateOcrRu-2021-09-01)
# Pipeline
![Pipeline](./data/pipeline_v2.png)

# Project structure

Our solution consists of three main parts:
1. **Detector** - selection of a bounding box on a given image
2. **Transformation** - transformation of a dedicated car number in order to improve the accuracy of number recognition
3. **Plate recognition** - number recognition

There is a strict interface for each of the parts. Each part is independent of the other. The final pipeline consists of the sequential application of each part.

```
.
├── data
├── notebooks               <- Jupyter notebooks
├── README.md               <- The top-level README for developers using this project
├── requirements.txt        <- The requirements file for reproducing the analysis environment
├── result                  <- Empty folder for saving results
├── src
│   ├── configs             <- Folder that contains the parameters needed to run
│   ├── models              <- Folder that contains the description of the model
│   ├── predict
│   │   ├── backend.py      <- Carplates recognition pipeline
│   │   ├── detection.py    <- Implementations for number detection part
│   │   ├── ocr.py          <- Implementations for OCR part
│   │   └── transform.py    <- Implementations for transform part 
│   ├── utils
│   └── weights             <- Serialized models
└── tutorial.ipynb          <- Demonstration work
```

# Results 
**Plate detector**
|Model|mAP 0.5|
|:------|:-------|
|YOLOv5|0.991|
|FasterRCNN|0.946|
|MTCNN|0.952|

**Plate recognition**
|Model|Accuracy|
|:------|:-------|
|EasyOCR(default)|0.003|
|EasyOCR(custom)|0.854|
|LPRNEt|0.751|

# Usage
We tested three different detection models, three text recognition models and transform model.You can combine them however you like. Be  careful with experiments, look at the results.


To demonstrate the results of the pipeline, you can use a [tutorial.ipynb](./tutorial.ipynb) Before using it, you need to install the project dependencies:
```
pip install -r requirements.txt 
```
[Faster RCNN](https://drive.google.com/file/d/14nrZrry97JVd2AypwCQ6v_U5BYEbUGO3/view?usp=sharing) weights must be inserted into src/weights/detect/FasterRCNN_v1_last.pth

# Reference 
1. [MTCNN](https://arxiv.org/abs/1604.02878v1)
1. [LPRNet](https://arxiv.org/abs/1806.10447)
1. [Spatial Transformer Layer](https://arxiv.org/abs/1506.02025)
1. [LPRNet Pytorch](https://github.com/sirius-ai/LPRNet_Pytorch)
1. [License Plate Detection Pytorch](https://github.com/xuexingyu24/License_Plate_Detection_Pytorch)
1. [YOLO V5](https://github.com/ultralytics/yolov5)
1. [Object Detection Metrics](https://github.com/rafaelpadilla/Object-Detection-Metrics)