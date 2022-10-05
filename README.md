# Car number
**Detection of car numbers and their recognition**

# Dataset
We used [data](https://disk.yandex.ru/d/NANSgQklgRElog) from Kaggle [competition](https://www.kaggle.com/competitions/vkcv2022-contest-02-carplates/data).

# Pipeline
![Pipeline](https://github.com/In48semenov/Car-numbers/blob/master/data/pipeline.png)

 

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
**We tested three different detection models, three text recognition models and transform model.You can combine them however you like.
Be  careful with experiments, look at the results.\
First of all you need:**

pip install -r requirements.txt 

*Check tutorial.ipynb, file contains full guide.*
